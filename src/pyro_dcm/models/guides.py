"""Guide factory, SVI runner, and posterior extraction for Pyro DCM models.

Provides shared inference infrastructure for all three DCM variants:
task-based, spectral, and regression. The guide factory supports six
Pyro ``AutoGuide`` types via a string-based registry, the SVI runner
handles ``ClippedAdam`` with gradient clipping, learning rate decay,
and NaN detection, and the posterior extraction helper simplifies
retrieval of variational parameters.

References
----------
04-RESEARCH.md -- Pyro patterns, pitfalls, and configuration.
10-RESEARCH.md -- Guide variant init_scale asymmetry and blocklists.
"""

from __future__ import annotations

from typing import Any, Callable

import math
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import (
    AutoDelta,
    AutoGuide,
    AutoIAFNormal,
    AutoLaplaceApproximation,
    AutoLowRankMultivariateNormal,
    AutoMultivariateNormal,
    AutoNormal,
)
from pyro.optim import ClippedAdam


GUIDE_REGISTRY: dict[str, type[AutoGuide]] = {
    "auto_delta": AutoDelta,
    "auto_normal": AutoNormal,
    "auto_lowrank_mvn": AutoLowRankMultivariateNormal,
    "auto_mvn": AutoMultivariateNormal,
    "auto_iaf": AutoIAFNormal,
    "auto_laplace": AutoLaplaceApproximation,
}
"""Mapping of guide type keys to Pyro AutoGuide classes."""

_INIT_SCALE_GUIDES: set[str] = {
    "auto_normal",
    "auto_lowrank_mvn",
    "auto_mvn",
}
"""Guide types that accept an ``init_scale`` constructor argument."""

MEAN_FIELD_GUIDES: set[str] = {"auto_delta", "auto_normal"}
"""Guide types compatible with ``TraceMeanField_ELBO``."""

_MAX_REGIONS: dict[str, int] = {
    "auto_mvn": 7,
}
"""Maximum allowed ``n_regions`` per guide type (inclusive)."""


def create_guide(
    model: Callable[..., Any],
    *,
    guide_type: str = "auto_normal",
    init_scale: float = 0.01,
    n_regions: int | None = None,
    **kwargs: Any,
) -> AutoGuide:
    """Create a Pyro AutoGuide for a DCM model.

    Factory that instantiates one of six supported guide types with
    appropriate constructor arguments. Handles ``init_scale`` asymmetry
    (only passed to guides that accept it) and enforces an N-based
    blocklist to prevent memory explosion with full-covariance guides
    on large models.

    Parameters
    ----------
    model : callable
        Pyro model function (e.g., ``task_dcm_model``,
        ``spectral_dcm_model``, ``rdcm_model``).
    guide_type : str, optional
        Guide variant key. One of ``'auto_delta'``,
        ``'auto_normal'`` (default), ``'auto_lowrank_mvn'``,
        ``'auto_mvn'``, ``'auto_iaf'``, ``'auto_laplace'``.
    init_scale : float, optional
        Initial scale for guide distributions. Only passed to
        ``auto_normal``, ``auto_lowrank_mvn``, and ``auto_mvn``.
        Default 0.01.
    n_regions : int or None, optional
        Number of brain regions. Used for blocklist enforcement.
        When provided, ``auto_mvn`` is blocked at ``n_regions >= 8``
        to prevent memory explosion. Default None (no check).
    **kwargs
        Extra keyword arguments forwarded to the guide constructor.
        Useful overrides:

        - ``rank`` (int): for ``auto_lowrank_mvn``, default 2.
        - ``num_transforms`` (int): for ``auto_iaf``, default 2.
        - ``hidden_dim`` (int): for ``auto_iaf``, default 20.

    Returns
    -------
    AutoGuide
        Pyro guide instance of the requested type.

    Raises
    ------
    ValueError
        If ``guide_type`` is not in ``GUIDE_REGISTRY``, or if
        ``n_regions`` exceeds the blocklist limit for the requested
        guide type.

    Notes
    -----
    The ``init_scale=0.01`` default is critical for ODE-based models
    (task DCM, spectral DCM): starting with larger scales can produce
    A matrices with large positive eigenvalues, causing ODE blow-up
    during the first SVI steps (see 04-RESEARCH.md Pitfall 1).

    ``AutoDelta``, ``AutoIAFNormal``, and ``AutoLaplaceApproximation``
    do not accept ``init_scale``; it is silently ignored for those
    guide types.

    Examples
    --------
    >>> from pyro_dcm.models import task_dcm_model, create_guide
    >>> guide = create_guide(task_dcm_model, init_scale=0.01)
    >>> iaf = create_guide(task_dcm_model, guide_type='auto_iaf')
    """
    # Validate guide_type
    if guide_type not in GUIDE_REGISTRY:
        valid = sorted(GUIDE_REGISTRY.keys())
        msg = (
            f"Unknown guide_type {guide_type!r}. "
            f"Available: {valid}"
        )
        raise ValueError(msg)

    # Blocklist check
    if n_regions is not None and guide_type in _MAX_REGIONS:
        max_n = _MAX_REGIONS[guide_type]
        if n_regions > max_n:
            msg = (
                f"guide_type {guide_type!r} is blocked for "
                f"n_regions={n_regions} (max {max_n}). "
                f"Use 'auto_lowrank_mvn' instead."
            )
            raise ValueError(msg)

    # Build constructor kwargs
    ctor_kwargs: dict[str, Any] = {}

    if guide_type in _INIT_SCALE_GUIDES:
        ctor_kwargs["init_scale"] = init_scale

    if guide_type == "auto_lowrank_mvn":
        ctor_kwargs["rank"] = kwargs.pop("rank", 2)

    if guide_type == "auto_iaf":
        ctor_kwargs["num_transforms"] = kwargs.pop(
            "num_transforms", 2,
        )
        ctor_kwargs["hidden_dim"] = kwargs.pop("hidden_dim", 20)

    # Pass remaining kwargs through
    ctor_kwargs.update(kwargs)

    return GUIDE_REGISTRY[guide_type](model, **ctor_kwargs)


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

    Examples
    --------
    >>> from pyro_dcm.models import task_dcm_model, create_guide, run_svi
    >>> guide = create_guide(task_dcm_model, init_scale=0.01)
    >>> result = run_svi(
    ...     task_dcm_model, guide,
    ...     model_args=(bold, stimulus, a_mask, c_mask, t_eval, TR, dt),
    ...     num_steps=500, lr=0.005,
    ... )
    >>> print(f"Final loss: {result['final_loss']:.2f}")
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
    guide: AutoGuide,
    model_args: tuple[Any, ...],
) -> dict[str, Any]:
    """Extract posterior parameters from a trained AutoGuide.

    Retrieves the variational median (posterior mode approximation)
    and all learned guide parameters (locs and scales) from Pyro's
    parameter store.

    Parameters
    ----------
    guide : AutoGuide
        Trained Pyro guide instance (any type from
        ``GUIDE_REGISTRY``).
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

    Examples
    --------
    >>> from pyro_dcm.models import create_guide, run_svi, extract_posterior_params
    >>> # After running SVI:
    >>> posterior = extract_posterior_params(guide, model_args)
    >>> A_median = posterior['median']['A']
    >>> A_free_loc = posterior['params']['AutoNormal.locs.A_free']
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
