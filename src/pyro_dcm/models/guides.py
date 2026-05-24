"""Guide factory, SVI runner, and posterior extraction for Pyro DCM models.

Provides shared inference infrastructure for all three DCM variants:
task-based, spectral, and regression. The guide factory supports six
Pyro ``AutoGuide`` types via a string-based registry, the SVI runner
handles ``ClippedAdam`` with gradient clipping, learning rate decay,
NaN detection, and three ELBO variants (Trace, TraceMeanField, Renyi),
and the posterior extraction helper simplifies retrieval of
variational parameters.

Multi-start SVI (``n_restarts > 1``) addresses ELBO landscape
multi-modality (pitfall LC11, Langdon & Engel 2025) by running
independent SVI optimizations and selecting the best by final ELBO.

References
----------
04-RESEARCH.md -- Pyro patterns, pitfalls, and configuration.
10-RESEARCH.md -- Guide variant init_scale asymmetry and blocklists.
20-RESEARCH.md -- Latent circuit DCM pitfalls (LC11: multi-start SVI).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import Any

import pyro
import torch
from pyro.infer import (
    SVI,
    Predictive,
    RenyiELBO,
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


_svi_logger = logging.getLogger("pyro_dcm.svi")


def _build_elbo(
    elbo_type: str,
    num_particles: int,
) -> Any:
    """Build a Pyro ELBO loss object.

    Parameters
    ----------
    elbo_type : str
        ELBO type key from ``ELBO_REGISTRY``.
    num_particles : int
        Number of ELBO particles for gradient estimation.

    Returns
    -------
    ELBO
        Pyro ELBO instance.
    """
    if elbo_type == "renyi_elbo":
        renyi_particles = max(num_particles, 2)
        return RenyiELBO(
            alpha=0.5,
            num_particles=renyi_particles,
            vectorize_particles=(renyi_particles > 1),
        )
    elbo_cls = ELBO_REGISTRY[elbo_type]
    return elbo_cls(
        num_particles=num_particles,
        vectorize_particles=(num_particles > 1),
    )


def _run_single_svi(
    model: Callable[..., Any],
    guide: Callable[..., Any],
    model_args: tuple[Any, ...],
    num_steps: int,
    lr: float,
    clip_norm: float,
    lr_decay_factor: float,
    num_particles: int,
    elbo_type: str,
    guide_type: str | None,
    model_kwargs: dict[str, Any],
    *,
    catch_nan: bool = False,
) -> dict[str, Any]:
    """Execute a single SVI optimization loop.

    Parameters
    ----------
    model : callable
        Pyro model function.
    guide : callable
        Pyro guide function.
    model_args : tuple
        Positional arguments for model/guide.
    num_steps : int
        Number of SVI steps.
    lr : float
        Initial learning rate.
    clip_norm : float
        Maximum gradient norm for clipping.
    lr_decay_factor : float
        Decay learning rate to this fraction over the run.
    num_particles : int
        Number of ELBO particles.
    elbo_type : str
        ELBO type key.
    guide_type : str or None
        Guide type key (for auto_laplace post-processing).
    model_kwargs : dict
        Extra keyword arguments for ``svi.step``.
    catch_nan : bool, optional
        If True, catch NaN ELBO and return ``final_loss=inf``
        instead of raising. Default False.

    Returns
    -------
    dict
        Keys: ``'losses'``, ``'final_loss'``, ``'num_steps'``,
        and optionally ``'guide'`` (for auto_laplace).
    """
    lrd = lr_decay_factor ** (1.0 / max(num_steps, 1))
    optimizer = ClippedAdam({
        "lr": lr,
        "clip_norm": clip_norm,
        "lrd": lrd,
    })
    elbo = _build_elbo(elbo_type, num_particles)
    svi = SVI(model, guide, optimizer, loss=elbo)

    losses: list[float] = []
    for step in range(num_steps):
        try:
            loss = svi.step(*model_args, **model_kwargs)
        except RuntimeError:
            if catch_nan:
                return {
                    "losses": losses,
                    "final_loss": float("inf"),
                    "num_steps": len(losses),
                }
            raise
        losses.append(loss)

        if math.isnan(loss):
            if catch_nan:
                return {
                    "losses": losses,
                    "final_loss": float("inf"),
                    "num_steps": len(losses),
                }
            msg = f"NaN ELBO at step {step}"
            raise RuntimeError(msg)

    post_guide = None
    if guide_type == "auto_laplace":
        post_guide = guide.laplace_approximation(
            *model_args, **model_kwargs,
        )

    result: dict[str, Any] = {
        "losses": losses,
        "final_loss": losses[-1],
        "num_steps": num_steps,
    }
    if post_guide is not None:
        result["guide"] = post_guide
    return result


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
    n_restarts: int = 1,
    guide_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Run SVI optimization for a Pyro model/guide pair.

    Trains the variational guide to approximate the posterior using
    stochastic variational inference with ``ClippedAdam`` optimizer,
    configurable ELBO loss, gradient clipping, and exponential learning
    rate decay.

    When ``n_restarts > 1``, runs multiple independent SVI
    optimizations with fresh guide instances from ``guide_factory``
    and selects the result with lowest final ELBO loss. This
    addresses ELBO landscape multi-modality (pitfall LC11, Langdon &
    Engel 2025). NaN restarts are assigned ``final_loss=inf`` and
    skipped; if ALL restarts produce NaN, raises ``RuntimeError``.

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
    n_restarts : int, optional
        Number of independent SVI optimizations. Default 1 preserves
        backward compatibility (single run, identical return dict).
        When ``> 1``, runs ``n_restarts`` independent SVI loops,
        each starting from a fresh guide instance via
        ``guide_factory()``, and returns the result with the lowest
        ``final_loss``.
    guide_factory : callable or None, optional
        Zero-argument callable that returns a fresh guide instance.
        Required when ``n_restarts > 1``. Each restart calls
        ``guide_factory()`` to create an independently initialized
        guide. When ``None`` and ``n_restarts > 1``, raises
        ``ValueError``. Default None.

    Returns
    -------
    dict
        When ``n_restarts <= 1`` (default):

        - ``'losses'``: list of float, ELBO loss at each step.
        - ``'final_loss'``: float, loss at last step.
        - ``'num_steps'``: int, number of steps completed.
        - ``'guide'``: (only for ``guide_type='auto_laplace'``)
          Post-Laplace ``AutoMultivariateNormal`` guide.

        When ``n_restarts > 1``, additional keys:

        - ``'n_restarts'``: int, number of restarts requested.
        - ``'best_restart_idx'``: int, index of the best restart.
        - ``'all_restarts'``: list of dicts, per-restart results
          with keys ``'losses'``, ``'final_loss'``, ``'restart'``,
          ``'num_steps'``.
        - ``'guide_factory'``: callable, the guide factory for
          creating a guide matching the restored param store.

    Raises
    ------
    ValueError
        If ``elbo_type`` is not in ``ELBO_REGISTRY``, if
        ``'tracemeanfield_elbo'`` is used with a non-mean-field guide,
        or if ``n_restarts > 1`` and ``guide_factory is None``.
    RuntimeError
        If ELBO becomes NaN at any step (single restart), or if
        ALL restarts produce NaN (multi-restart).

    Notes
    -----
    - ``pyro.clear_param_store()`` is called at the start of each
      SVI run to ensure fresh optimization (see 04-RESEARCH.md
      Pitfall 6).
    - Learning rate decay: ``lrd = lr_decay_factor ** (1 / num_steps)``,
      applied per-step multiplicatively by ``ClippedAdam``.
    - Gradient clipping via ``clip_norm`` prevents exploding gradients
      from ODE-based models (see 04-RESEARCH.md Pitfall 1).
    - Multi-start SVI: after all restarts, the param store is
      restored to the best restart's state. Callers can immediately
      use ``extract_posterior_params`` on a ``guide_factory()``-
      created guide with the restored param store.

    Examples
    --------
    Single restart (backward compatible):

    >>> from pyro_dcm.models import task_dcm_model, create_guide, run_svi
    >>> guide = create_guide(task_dcm_model, init_scale=0.01)
    >>> result = run_svi(
    ...     task_dcm_model, guide,
    ...     model_args=(bold, stimulus, a_mask, c_mask, t_eval, TR, dt),
    ...     num_steps=500, lr=0.005,
    ... )
    >>> print(f"Final loss: {result['final_loss']:.2f}")

    Multi-start SVI (10 restarts):

    >>> from functools import partial
    >>> gf = partial(create_guide, task_dcm_model, init_scale=0.01)
    >>> result = run_svi(
    ...     task_dcm_model, gf(),
    ...     model_args=(bold, stimulus, a_mask, c_mask, t_eval, TR, dt),
    ...     num_steps=500, lr=0.005,
    ...     n_restarts=10, guide_factory=gf,
    ... )
    >>> print(f"Best ELBO: {result['final_loss']:.2f}")
    >>> print(f"Best restart: {result['best_restart_idx']}")
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

    kw: dict[str, Any] = model_kwargs or {}

    # ------------------------------------------------------------------
    # Single-restart path: identical to pre-Phase-20 behavior
    # ------------------------------------------------------------------
    if n_restarts <= 1:
        pyro.clear_param_store()
        return _run_single_svi(
            model, guide, model_args, num_steps, lr,
            clip_norm, lr_decay_factor, num_particles,
            elbo_type, guide_type, kw,
        )

    # ------------------------------------------------------------------
    # Multi-restart path
    # ------------------------------------------------------------------
    if guide_factory is None:
        msg = (
            "guide_factory is required when n_restarts > 1. "
            "Provide a zero-argument callable that returns a "
            "fresh guide instance (e.g., "
            "functools.partial(create_guide, model, "
            "guide_type='auto_normal', init_scale=0.01))."
        )
        raise ValueError(msg)

    all_restarts: list[dict[str, Any]] = []
    all_states: list[dict[str, Any]] = []

    for restart_idx in range(n_restarts):
        pyro.clear_param_store()
        current_guide = guide_factory()

        restart_result = _run_single_svi(
            model, current_guide, model_args, num_steps, lr,
            clip_norm, lr_decay_factor, num_particles,
            elbo_type, guide_type, kw,
            catch_nan=True,
        )

        restart_result["restart"] = restart_idx
        all_restarts.append(restart_result)
        all_states.append(pyro.get_param_store().get_state())

        _svi_logger.info(
            "Multi-start SVI: restart %d/%d, final ELBO: %.2f",
            restart_idx + 1,
            n_restarts,
            restart_result["final_loss"],
        )

    # Select best restart by minimum final_loss
    finite_restarts = [
        (i, r) for i, r in enumerate(all_restarts)
        if r["final_loss"] != float("inf")
    ]

    if not finite_restarts:
        msg = (
            f"All {n_restarts} SVI restarts produced NaN ELBO. "
            f"Consider adjusting learning rate, init_scale, or "
            f"model parameterization."
        )
        raise RuntimeError(msg)

    best_idx = min(
        finite_restarts,
        key=lambda x: x[1]["final_loss"],
    )[0]

    _svi_logger.info(
        "Multi-start SVI: best restart %d with ELBO %.2f",
        best_idx,
        all_restarts[best_idx]["final_loss"],
    )

    # Restore best restart's param store state
    pyro.clear_param_store()
    pyro.get_param_store().set_state(all_states[best_idx])

    best = all_restarts[best_idx]
    result: dict[str, Any] = {
        "losses": best["losses"],
        "final_loss": best["final_loss"],
        "num_steps": best["num_steps"],
        "n_restarts": n_restarts,
        "best_restart_idx": best_idx,
        "all_restarts": all_restarts,
        "guide_factory": guide_factory,
    }
    if "guide" in best:
        result["guide"] = best["guide"]
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
