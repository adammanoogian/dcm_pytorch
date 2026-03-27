"""Guide factory, SVI runner, and posterior extraction for Pyro DCM models.

Provides shared inference infrastructure for all three DCM variants:
task-based, spectral, and regression. The guide factory wraps
``AutoNormal`` with appropriate initialization, the SVI runner handles
``ClippedAdam`` with gradient clipping, learning rate decay, and NaN
detection, and the posterior extraction helper simplifies retrieval
of variational parameters.

References
----------
04-RESEARCH.md -- Pyro patterns, pitfalls, and configuration.
"""

from __future__ import annotations

from typing import Any, Callable

import math
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.optim import ClippedAdam


def create_guide(
    model: Callable[..., Any],
    init_scale: float = 0.01,
) -> AutoNormal:
    """Create an AutoNormal mean-field guide for a Pyro model.

    Creates a diagonal Gaussian (mean-field) variational guide using
    Pyro's ``AutoNormal``. Each latent variable gets independent
    ``loc`` and ``scale`` variational parameters.

    The ``init_scale`` parameter controls the initial width of guide
    distributions. A small value (0.01) starts the guide tight around
    zero, preventing ODE blow-up from extreme initial ``A_free``
    samples during SVI (see 04-RESEARCH.md Pitfall 1).

    Parameters
    ----------
    model : callable
        Pyro model function (e.g., ``task_dcm_model``,
        ``spectral_dcm_model``, ``rdcm_model``).
    init_scale : float, optional
        Initial scale for guide distributions. Default 0.01.

    Returns
    -------
    AutoNormal
        Pyro ``AutoNormal`` guide instance.

    Notes
    -----
    ``AutoNormal`` creates independent Normal variational distributions
    for each latent variable, with learnable ``loc`` (mean) and
    ``scale`` (standard deviation) parameters. This is a mean-field
    approximation that ignores posterior correlations but is fast and
    numerically stable.

    The ``init_scale=0.01`` default is critical for ODE-based models
    (task DCM, spectral DCM): starting with larger scales can produce
    A matrices with large positive eigenvalues, causing ODE blow-up
    during the first SVI steps.
    """
    return AutoNormal(model, init_scale=init_scale)


def run_svi(
    model: Callable[..., Any],
    guide: Callable[..., Any],
    model_args: tuple[Any, ...],
    num_steps: int = 2000,
    lr: float = 0.01,
    clip_norm: float = 10.0,
    lr_decay_factor: float = 0.01,
    num_particles: int = 1,
) -> dict[str, Any]:
    """Run SVI optimization for a Pyro model/guide pair.

    Trains the variational guide to approximate the posterior using
    stochastic variational inference with ``ClippedAdam`` optimizer,
    ``Trace_ELBO`` loss, gradient clipping, and exponential learning
    rate decay.

    Parameters
    ----------
    model : callable
        Pyro model function.
    guide : callable
        Pyro guide function (from ``create_guide`` or custom).
    model_args : tuple
        Positional arguments passed to both model and guide.
    num_steps : int, optional
        Number of SVI optimization steps. Default 2000.
    lr : float, optional
        Initial learning rate. Default 0.01.
    clip_norm : float, optional
        Maximum gradient norm for clipping. Default 10.0.
    lr_decay_factor : float, optional
        Decay learning rate to this fraction of initial over the
        full training run. Default 0.01 (decay to 1% of initial lr).
    num_particles : int, optional
        Number of ELBO particles for gradient estimation. Default 1.

    Returns
    -------
    dict
        Keys:

        - ``'losses'``: list of float, ELBO loss at each step.
        - ``'final_loss'``: float, loss at last step.
        - ``'num_steps'``: int, number of steps completed.

    Raises
    ------
    RuntimeError
        If ELBO becomes NaN at any step.

    Notes
    -----
    - ``pyro.clear_param_store()`` is called at the start to ensure
      a fresh optimization (see 04-RESEARCH.md Pitfall 6).
    - Learning rate decay: ``lrd = lr_decay_factor ** (1 / num_steps)``,
      applied per-step multiplicatively by ``ClippedAdam``.
    - Gradient clipping via ``clip_norm`` prevents exploding gradients
      from ODE-based models (see 04-RESEARCH.md Pitfall 1).
    """
    pyro.clear_param_store()

    # Per-step multiplicative LR decay
    lrd = lr_decay_factor ** (1.0 / max(num_steps, 1))

    optimizer = ClippedAdam({
        "lr": lr,
        "clip_norm": clip_norm,
        "lrd": lrd,
    })

    elbo = Trace_ELBO(
        num_particles=num_particles,
        vectorize_particles=(num_particles > 1),
    )

    svi = SVI(model, guide, optimizer, loss=elbo)

    losses: list[float] = []
    for step in range(num_steps):
        loss = svi.step(*model_args)
        losses.append(loss)

        if math.isnan(loss):
            msg = f"NaN ELBO at step {step}"
            raise RuntimeError(msg)

    return {
        "losses": losses,
        "final_loss": losses[-1],
        "num_steps": num_steps,
    }


def extract_posterior_params(
    guide: AutoNormal,
    model_args: tuple[Any, ...],
) -> dict[str, Any]:
    """Extract posterior parameters from a trained AutoNormal guide.

    Retrieves the variational median (posterior mode approximation)
    and all learned guide parameters (locs and scales) from Pyro's
    parameter store.

    Parameters
    ----------
    guide : AutoNormal
        Trained ``AutoNormal`` guide instance.
    model_args : tuple
        Arguments to the model (needed for ``guide.median()``).

    Returns
    -------
    dict
        Keys:

        - ``'median'``: dict mapping site names to median values.
        - ``'params'``: dict mapping parameter store keys to tensors.

    Notes
    -----
    Call this after ``run_svi`` completes. The median values
    approximate the posterior mode under the mean-field assumption.
    The params dict contains the raw ``AutoNormal_loc`` and
    ``AutoNormal_scale`` parameters for each latent variable.
    """
    # Get median values from guide
    median = guide.median(*model_args)

    # Get all parameters from the param store
    param_store = pyro.get_param_store()
    params = {
        name: param_store[name].detach().clone()
        for name in param_store
    }

    return {
        "median": median,
        "params": params,
    }
