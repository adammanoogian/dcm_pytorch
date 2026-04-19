"""Guide factory, SVI runner, and posterior extraction for Pyro DCM models.

Provides shared inference infrastructure for all three DCM variants:
task-based, spectral, and regression. The guide factory supports six
Pyro ``AutoGuide`` types via a string-based registry, the SVI runner
handles ``ClippedAdam`` with gradient clipping, learning rate decay,
NaN detection, and three ELBO variants (Trace, TraceMeanField, Renyi),
and the posterior extraction helper simplifies retrieval of
variational parameters.

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
from pyro.infer import (
    Predictive,
    RenyiELBO,
    SVI,
    Trace_ELBO,
    TraceMeanField_ELBO,
)
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

ELBO_REGISTRY: dict[str, type] = {
    "trace_elbo": Trace_ELBO,
    "tracemeanfield_elbo": TraceMeanField_ELBO,
    "renyi_elbo": RenyiELBO,
}
"""Mapping of ELBO type keys to Pyro ELBO classes."""


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
        - ``hidden_dim`` (int or list[int]): for ``auto_iaf``,
          default ``[20]``. An int is wrapped in a list.

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
        hidden_dim = kwargs.pop("hidden_dim", [20])
        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim]
        ctor_kwargs["hidden_dim"] = hidden_dim

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
    elbo_type: str = "trace_elbo",
    guide_type: str | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run SVI optimization for a Pyro model/guide pair.

    Trains the variational guide to approximate the posterior using
    stochastic variational inference with ``ClippedAdam`` optimizer,
    configurable ELBO loss, gradient clipping, and exponential learning
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
    elbo_type : str, optional
        ELBO objective key. One of ``'trace_elbo'`` (default),
        ``'tracemeanfield_elbo'``, or ``'renyi_elbo'``.
        ``'renyi_elbo'`` uses ``alpha=0.5`` and forces
        ``num_particles >= 2``.
    guide_type : str or None, optional
        Guide type key (e.g., ``'auto_normal'``). Used for
        validation: ``'tracemeanfield_elbo'`` requires a mean-field
        guide (``'auto_delta'`` or ``'auto_normal'``). Default None
        (no validation).
    model_kwargs : dict[str, Any] or None, optional
        Extra keyword arguments forwarded to ``svi.step(*model_args,
        **model_kwargs)`` at every step and to
        ``guide.laplace_approximation`` (if applicable). Required for
        models that expose keyword-only parameters (e.g.,
        ``task_dcm_model``'s ``b_masks`` and ``stim_mod`` kwargs in the
        bilinear branch -- v0.3.0+). Default ``None`` -> empty dict ->
        bit-exact equivalent to the pre-v0.3.0 signature for all linear
        callers (``task_svi.py``, ``spectral_svi.py``, ``rdcm_vb.py``,
        ``amortized_*.py``).

    Returns
    -------
    dict
        Keys:

        - ``'losses'``: list of float, ELBO loss at each step.
        - ``'final_loss'``: float, loss at last step.
        - ``'num_steps'``: int, number of steps completed.
        - ``'guide'``: (only for ``guide_type='auto_laplace'``)
          Post-Laplace ``AutoMultivariateNormal`` guide.

    Raises
    ------
    ValueError
        If ``elbo_type`` is not in ``ELBO_REGISTRY``, or if
        ``'tracemeanfield_elbo'`` is used with a non-mean-field guide.
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
    # Validate elbo_type
    if elbo_type not in ELBO_REGISTRY:
        valid = sorted(ELBO_REGISTRY.keys())
        msg = (
            f"Unknown elbo_type {elbo_type!r}. "
            f"Available: {valid}"
        )
        raise ValueError(msg)

    # Mean-field guard
    if (
        elbo_type == "tracemeanfield_elbo"
        and guide_type is not None
        and guide_type not in MEAN_FIELD_GUIDES
    ):
        msg = (
            f"TraceMeanField_ELBO requires a mean-field guide "
            f"(auto_delta or auto_normal), got {guide_type!r}. "
            f"Use 'trace_elbo' or 'renyi_elbo' instead."
        )
        raise ValueError(msg)

    pyro.clear_param_store()

    # Per-step multiplicative LR decay
    lrd = lr_decay_factor ** (1.0 / max(num_steps, 1))

    optimizer = ClippedAdam({
        "lr": lr,
        "clip_norm": clip_norm,
        "lrd": lrd,
    })

    # Build ELBO object
    if elbo_type == "renyi_elbo":
        renyi_particles = max(num_particles, 2)
        elbo = RenyiELBO(
            alpha=0.5,
            num_particles=renyi_particles,
            vectorize_particles=(renyi_particles > 1),
        )
    else:
        elbo_cls = ELBO_REGISTRY[elbo_type]
        elbo = elbo_cls(
            num_particles=num_particles,
            vectorize_particles=(num_particles > 1),
        )

    svi = SVI(model, guide, optimizer, loss=elbo)

    kw: dict[str, Any] = model_kwargs or {}
    losses: list[float] = []
    for step in range(num_steps):
        loss = svi.step(*model_args, **kw)
        losses.append(loss)

        if math.isnan(loss):
            msg = f"NaN ELBO at step {step}"
            raise RuntimeError(msg)

    # Post-process AutoLaplaceApproximation
    post_guide = None
    if guide_type == "auto_laplace":
        post_guide = guide.laplace_approximation(*model_args, **kw)

    result: dict[str, Any] = {
        "losses": losses,
        "final_loss": losses[-1],
        "num_steps": num_steps,
    }
    if post_guide is not None:
        result["guide"] = post_guide
    return result


def extract_posterior_params(
    guide: AutoGuide,
    model_args: tuple[Any, ...],
    model: Callable[..., Any] | None = None,
    num_samples: int = 1000,
) -> dict[str, Any]:
    """Extract posterior parameters via Predictive-based sampling.

    Draws ``num_samples`` from the trained guide using
    ``pyro.infer.Predictive``, then computes per-site mean, std, and
    raw samples. Works identically for all six guide types in
    ``GUIDE_REGISTRY`` (including ``AutoDelta``, which returns
    ``std=0`` for all sites).

    Parameters
    ----------
    guide : AutoGuide
        Trained Pyro guide instance (any type from
        ``GUIDE_REGISTRY``).
    model_args : tuple
        Positional arguments passed to the model/guide.
    model : callable or None, optional
        Pyro model function. If ``None``, uses ``guide.model``
        (all ``AutoGuide`` subclasses store the model). Default
        ``None`` preserves backward compatibility.
    num_samples : int, optional
        Number of posterior samples to draw. Default 1000.

    Returns
    -------
    dict
        Per-site dicts with keys ``'mean'``, ``'std'``, ``'samples'``,
        plus a top-level ``'median'`` key mapping site names to their
        mean values for backward compatibility.

        Example structure::

            {
                "A_free": {
                    "mean": Tensor,
                    "std": Tensor,
                    "samples": Tensor,  # (num_samples, ...)
                },
                "C": { ... },
                "median": {"A_free": Tensor, "C": Tensor},
            }

    Notes
    -----
    Call this after ``run_svi`` completes. The ``'median'`` key
    provides backward compatibility with code that previously used
    ``guide.median()`` -- the values are sample means (which
    approximate medians for symmetric posteriors).

    For ``AutoDelta`` guides, all samples are identical point
    estimates, so ``std`` is exactly zero.

    **Bilinear task DCM sites (v0.3.0+):** When the guide trains on
    ``task_dcm_model`` in bilinear mode (non-empty ``b_masks``), per-
    modulator parameters appear in the returned dict under keys
    ``B_free_0``, ``B_free_1``, ..., ``B_free_{J-1}`` (raw free
    parameters; ``mean`` is the per-modulator posterior median
    approximation). The masked, parameterized stacked B matrix may
    also appear under key ``B`` (shape ``(J, N, N)``; see
    ``pyro_dcm.forward_models.neural_state.parameterize_B``) when the
    underlying ``Predictive`` call returns deterministic sites.
    Whether ``Predictive(return_sites=None)`` includes deterministic
    sites depends on the Pyro version; pass
    ``return_sites=[..., 'B']`` explicitly to guarantee the masked
    tensor appears across versions. Compute per-modulator medians
    either as ``posterior["B_free_j"]["mean"]`` (raw; always
    available) or ``posterior["B"]["mean"][j]`` (masked; available
    when B is requested or included by default). Closes MODEL-05
    (``.planning/REQUIREMENTS.md``).

    Examples
    --------
    >>> from pyro_dcm.models import create_guide, run_svi, extract_posterior_params
    >>> # After running SVI:
    >>> posterior = extract_posterior_params(guide, model_args)
    >>> A_mean = posterior['A_free']['mean']
    >>> A_std = posterior['A_free']['std']
    >>> A_median_compat = posterior['median']['A_free']

    Bilinear task DCM:

    >>> # After SVI on task_dcm_model with b_masks=[mask_0], stim_mod=mod:
    >>> posterior = extract_posterior_params(guide, model_args)
    >>> B_raw = posterior['B_free_0']['mean']           # (N, N), always available
    >>> # Masked (J, N, N) tensor: available when Predictive includes
    >>> # deterministic sites (Pyro 1.9+ default) or via explicit return_sites.
    >>> if 'B' in posterior:
    ...     B_masked = posterior['B']['mean']               # (J, N, N)
    ...     B_for_modulator_0 = posterior['B']['mean'][0]   # (N, N)
    """
    if model is None:
        model = guide.model

    predictive = Predictive(
        model,
        guide=guide,
        num_samples=num_samples,
        return_sites=None,
    )

    with torch.no_grad():
        samples = predictive(*model_args)

    result: dict[str, Any] = {}
    median_dict: dict[str, torch.Tensor] = {}

    for site_name, tensor in samples.items():
        if tensor.is_complex():
            # Complex sites (e.g. predicted_csd) -- compute
            # statistics on real/imag parts separately.
            site_mean = tensor.mean(dim=0)
            site_std = tensor.real.float().std(dim=0)
        else:
            site_mean = tensor.float().mean(dim=0)
            site_std = tensor.float().std(dim=0)
        result[site_name] = {
            "mean": site_mean,
            "std": site_std,
            "samples": tensor,
        }
        median_dict[site_name] = site_mean

    result["median"] = median_dict

    return result
