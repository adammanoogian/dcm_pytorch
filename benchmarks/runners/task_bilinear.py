"""Task-bilinear DCM SVI benchmark runner (v0.3.0 Phase 16).

Implements the simulate -> fit-bilinear -> fit-linear-baseline -> measure loop for
task-based bilinear DCM recovery. For each seed, ``run_task_bilinear_svi``
generates (or loads) a 3-region bilinear ground-truth fixture with asymmetric
V1->V5->SPL modulatory hierarchy (B[1,0]=0.4, B[2,1]=0.3), runs Pyro SVI on the
bilinear task_dcm_model, then runs the bit-exact linear baseline (b_masks=None
short-circuit, MODEL-04) on the SAME fixture for RECOV-03 comparative A-RMSE.
Returns per-seed posteriors and timing.

Downstream: plan 16-02 consumes a_rmse_bilinear_list / a_rmse_linear_list /
posterior_list for RECOV-03..08 metric computation + forest-plot + acceptance-gate
table. Plan 16-03 extends this runner with a stimulus_mod_factory kwarg for the
v0.3.1 HGF hook.

References
----------
.planning/phases/16-bilinear-recovery-benchmark/16-CONTEXT.md
.planning/phases/16-bilinear-recovery-benchmark/16-RESEARCH.md Section 2, 6
.planning/REQUIREMENTS.md RECOV-01, RECOV-02
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np
import pyro
import torch

from benchmarks.config import BenchmarkConfig
from benchmarks.fixtures import load_fixture
from benchmarks.metrics import compute_rmse, compute_summary_stats
from pyro_dcm.forward_models.neural_state import parameterize_A
from pyro_dcm.models import (
    create_guide,
    extract_posterior_params,
    run_svi,
    task_dcm_model,
)
from pyro_dcm.simulators.task_simulator import (
    make_block_stimulus,
    make_epoch_stimulus,
    make_random_stable_A,
    simulate_task_dcm,
)
from pyro_dcm.utils.ode_integrator import PiecewiseConstantInput

# Phase 16 ground-truth constants (16-CONTEXT.md topology + 16-RESEARCH Sec 3).
_DURATION = 200.0
_TR = 2.0
_DT_SIM = 0.01
_DT_MODEL = 0.5
_SNR = 3.0
_EPOCH_TIMES = [20.0, 65.0, 110.0, 155.0]
_EPOCH_DURATIONS = [12.0] * 4
_EPOCH_AMPLITUDES = [1.0] * 4
_B_10 = 0.4
_B_21 = 0.3
_C_00 = 0.5
_DRIVING_N_BLOCKS = 5
_DRIVING_BLOCK_DURATION = 20.0
_DRIVING_REST_DURATION = 20.0

# Bilinear init-scale retry policy.
# Cluster job 54901072 observed 3/10 seeds NaN at SVI step 0 with
# init_scale=0.005: the initial posterior sample + ground truth combine to
# overflow the bilinear ODE forward model before any gradient update. A one-
# shot retry at half the init_scale collapses the initial posterior tightly
# enough around the prior mean to avoid the overflow on nearly all such seeds
# (Pitfall B1/B6). The original scale is preferred when it works so that
# retries do not bias the posterior-variance estimates that feed RECOV-06.
_BILINEAR_INIT_SCALE = 0.005
_BILINEAR_INIT_SCALE_RETRY = 0.001


# ---------------------------------------------------------------------------
# HGF forward-compat factory hook (16-CONTEXT.md lock-in)
# ---------------------------------------------------------------------------

# Type alias for the factory contract (Plan 16-03 L9).
# Factories accept a seed (int) and return a breakpoint dict with 'times' (K,)
# and 'values' (K, J) tensors. v0.3.1 HGF factories will use the same shape;
# the factory is injected at runtime at the runner's call site, not stored in
# BenchmarkConfig (keeps .npz reproducibility path clean per research Section
# 5.3).
StimulusModFactory = Callable[[int], dict[str, torch.Tensor]]


def make_sinusoid_mod_factory(
    duration: float = 200.0,
    dt: float = 0.01,
    frequency: float = 0.05,
    amplitude: float = 0.5,
) -> StimulusModFactory:
    """Build a mock sinusoidal stimulus-modulator factory (Phase 16 placeholder).

    Returns a closure that accepts a seed and produces a deterministic
    sinusoidal modulator breakpoint dict. **Not physiologically meaningful.**
    Used exclusively to exercise the factory-hook plumbing in Phase 16 tests
    (16-CONTEXT.md: "exercised by a placeholder mock factory... the
    indirection is proven wired, not a theoretical API").

    v0.3.1 will add an HGF-trajectory factory that uses the SAME signature;
    the runner does not need changes because the factory contract is fixed
    by :data:`StimulusModFactory`.

    Parameters
    ----------
    duration : float, optional
        Total simulation duration. Default 200.0 (matches Phase 16 locked
        value).
    dt : float, optional
        Simulation time step. Default 0.01.
    frequency : float, optional
        Sinusoid frequency in Hz. Default 0.05 (period = 20s; roughly one
        "epoch-equivalent" per cycle).
    amplitude : float, optional
        Sinusoid amplitude. Default 0.5 (< 1.0 to keep Gershgorin bound safe
        per plan 16-01 L2 rationale).

    Returns
    -------
    Callable[[int], dict[str, torch.Tensor]]
        Factory closure. Calling with any int seed returns::

            {
                "times":  torch.Tensor of shape (K,),   # float64
                "values": torch.Tensor of shape (K, 1), # float64; J=1
            }

    Notes
    -----
    The factory is deterministic given ``duration``, ``dt``, ``frequency``,
    ``amplitude``; the ``seed`` input is ignored numerically but retained in
    the closure signature to match the v0.3.1 HGF factory, which WILL use
    the seed to sample belief-trajectory random draws.

    References
    ----------
    .planning/phases/16-bilinear-recovery-benchmark/16-CONTEXT.md HGF hook
    lock-in.
    .planning/phases/16-bilinear-recovery-benchmark/16-RESEARCH.md Section
    5.2.
    """
    def _factory(seed: int) -> dict[str, torch.Tensor]:
        # Seed discipline preserved for future noisy-factory variants.
        torch.manual_seed(seed)
        times = torch.arange(0, duration, dt, dtype=torch.float64)
        values = (
            amplitude
            * torch.sin(2.0 * torch.pi * frequency * times)
        ).unsqueeze(-1)
        return {"times": times, "values": values}
    return _factory


def _make_bilinear_ground_truth(
    n_regions: int, seed_i: int,
) -> dict[str, Any]:
    """Construct ground-truth A, C, B, b_mask, stim, stim_mod for a seed.

    Matches generate_task_bilinear_fixtures exactly so the inline-generation
    path is bit-identical to the .npz-loading path when fixtures_dir=None
    (research Section 1.3 reproducibility guarantee).

    Parameters
    ----------
    n_regions : int
        Number of brain regions.
    seed_i : int
        Per-seed random seed.

    Returns
    -------
    dict
        Ground-truth tensors and driving/modulatory inputs for this seed.
    """
    torch.manual_seed(seed_i)
    A_true = make_random_stable_A(n_regions, density=0.5, seed=seed_i)
    C = torch.zeros(n_regions, 1, dtype=torch.float64)
    C[0, 0] = _C_00
    b_mask_0 = torch.zeros(n_regions, n_regions, dtype=torch.float64)
    b_mask_0[1, 0] = 1.0
    b_mask_0[2, 1] = 1.0
    B_true = torch.zeros(1, n_regions, n_regions, dtype=torch.float64)
    B_true[0, 1, 0] = _B_10
    B_true[0, 2, 1] = _B_21
    stim = make_block_stimulus(
        n_blocks=_DRIVING_N_BLOCKS,
        block_duration=_DRIVING_BLOCK_DURATION,
        rest_duration=_DRIVING_REST_DURATION,
        n_inputs=1,
    )
    stim_mod_dict = make_epoch_stimulus(
        event_times=_EPOCH_TIMES,
        event_durations=_EPOCH_DURATIONS,
        event_amplitudes=_EPOCH_AMPLITUDES,
        duration=_DURATION,
        dt=_DT_SIM,
        n_inputs=1,
    )
    stim_mod = PiecewiseConstantInput(
        stim_mod_dict["times"], stim_mod_dict["values"],
    )
    sim = simulate_task_dcm(
        A_true, C, stim,
        duration=_DURATION, dt=_DT_SIM, TR=_TR, SNR=_SNR,
        seed=seed_i, solver="rk4",
        B_list=[B_true[0]],
        stimulus_mod=stim_mod,
    )
    return {
        "A_true": A_true,
        "C": C,
        "B_true": B_true,
        "b_mask_0": b_mask_0,
        "stim": stim,
        "stimulus": sim["stimulus"],
        "stim_mod": stim_mod,
        "bold": sim["bold"],
    }


def _make_bilinear_ground_truth_with_factory(
    n_regions: int,
    seed_i: int,
    stim_mod_factory: StimulusModFactory,
) -> dict[str, Any]:
    """Ground-truth builder using a custom stim_mod factory (Plan 16-03 L10).

    Identical to :func:`_make_bilinear_ground_truth` except the stim_mod
    breakpoint dict comes from the provided factory instead of
    :func:`make_epoch_stimulus`. ``A`` / ``C`` / ``B`` / ``b_mask`` /
    driving stimulus remain seed-deterministic per Phase 16 ground-truth
    constants (plan 16-01).

    Parameters
    ----------
    n_regions : int
        Number of brain regions.
    seed_i : int
        Per-seed index; drives ``torch.manual_seed`` and ``A`` random
        generation.
    stim_mod_factory : StimulusModFactory
        Custom factory injected by the caller. Called as
        ``factory(seed=seed_i)``.

    Returns
    -------
    dict
        Same keys as :func:`_make_bilinear_ground_truth`: ``A_true``,
        ``C``, ``B_true``, ``b_mask_0``, ``stim`` (block-design dict),
        ``stimulus`` (PiecewiseConstantInput), ``stim_mod``
        (PiecewiseConstantInput built from factory output), ``bold``.
    """
    torch.manual_seed(seed_i)
    A_true = make_random_stable_A(n_regions, density=0.5, seed=seed_i)
    C = torch.zeros(n_regions, 1, dtype=torch.float64)
    C[0, 0] = _C_00
    b_mask_0 = torch.zeros(n_regions, n_regions, dtype=torch.float64)
    b_mask_0[1, 0] = 1.0
    b_mask_0[2, 1] = 1.0
    B_true = torch.zeros(1, n_regions, n_regions, dtype=torch.float64)
    B_true[0, 1, 0] = _B_10
    B_true[0, 2, 1] = _B_21
    stim = make_block_stimulus(
        n_blocks=_DRIVING_N_BLOCKS,
        block_duration=_DRIVING_BLOCK_DURATION,
        rest_duration=_DRIVING_REST_DURATION,
        n_inputs=1,
    )
    # Factory-generated stim_mod (L10: custom factory bypasses fixture cache).
    stim_mod_dict = stim_mod_factory(seed=seed_i)
    if (
        not isinstance(stim_mod_dict, dict)
        or "times" not in stim_mod_dict
        or "values" not in stim_mod_dict
    ):
        keys = (
            list(stim_mod_dict.keys())
            if isinstance(stim_mod_dict, dict)
            else "N/A"
        )
        raise TypeError(
            f"stimulus_mod_factory must return dict with 'times' and "
            f"'values' keys; got {type(stim_mod_dict).__name__} with keys "
            f"{keys}"
        )
    stim_mod = PiecewiseConstantInput(
        stim_mod_dict["times"], stim_mod_dict["values"],
    )
    sim = simulate_task_dcm(
        A_true, C, stim,
        duration=_DURATION, dt=_DT_SIM, TR=_TR, SNR=_SNR,
        seed=seed_i, solver="rk4",
        B_list=[B_true[0]],
        stimulus_mod=stim_mod,
    )
    return {
        "A_true": A_true,
        "C": C,
        "B_true": B_true,
        "b_mask_0": b_mask_0,
        "stim": stim,
        "stimulus": sim["stimulus"],
        "stim_mod": stim_mod,
        "bold": sim["bold"],
    }


def _load_or_make_fixture(
    i: int, config: BenchmarkConfig,
) -> dict[str, Any]:
    """Fixture dispatch: .npz load if fixtures_dir set, else inline generate.

    Research Section 1.3: load_fixture is generic -- reads every .npz key via
    ``data.files`` loop. Phase 16 bilinear fields (B_true, b_mask_0,
    stim_mod_times, stim_mod_values, J) load automatically with no loader
    changes.

    Parameters
    ----------
    i : int
        Dataset index (0-based).
    config : BenchmarkConfig
        Benchmark configuration.

    Returns
    -------
    dict
        Ground-truth tensors + PiecewiseConstantInput driving/modulator.
    """
    n_regions = config.n_regions
    seed_i = config.seed + i
    if config.fixtures_dir is not None:
        data = load_fixture(
            "task_bilinear", n_regions, i, config.fixtures_dir,
        )
        A_true = data["A_true"]
        C = data["C"]
        B_true = data["B_true"]
        b_mask_0 = data["b_mask_0"]
        stimulus = PiecewiseConstantInput(
            data["stimulus_times"], data["stimulus_values"],
        )
        stim_mod = PiecewiseConstantInput(
            data["stim_mod_times"], data["stim_mod_values"],
        )
        bold = data["bold"]
        return {
            "A_true": A_true, "C": C, "B_true": B_true, "b_mask_0": b_mask_0,
            "stimulus": stimulus, "stim_mod": stim_mod, "bold": bold,
        }
    return _make_bilinear_ground_truth(n_regions, seed_i)


def _fit_and_extract(
    model_args: tuple[Any, ...],
    model_kwargs: dict[str, Any],
    guide_type: str,
    init_scale: float,
    num_steps: int,
    elbo_type: str,
) -> tuple[dict[str, Any], float]:
    """Fit one SVI run and extract posterior samples.

    Parameters
    ----------
    model_args : tuple
        Positional args for ``task_dcm_model``.
    model_kwargs : dict
        Keyword args (e.g., ``b_masks``, ``stim_mod``). Empty dict means
        linear short-circuit.
    guide_type : str
        Guide type key forwarded to ``create_guide``.
    init_scale : float
        Guide init scale (passed to AutoNormal family).
    num_steps : int
        SVI step count.
    elbo_type : str
        ELBO type key forwarded to ``run_svi``.

    Returns
    -------
    tuple[dict, float]
        (posterior_dict, elapsed_seconds). posterior_dict has keys for
        each sample site: {'mean', 'std', 'samples'}. Also carries
        'final_losses' (last 20 SVI losses) as a convergence diagnostic.
    """
    pyro.clear_param_store()
    guide = create_guide(
        task_dcm_model, guide_type=guide_type, init_scale=init_scale,
    )
    t0 = time.time()
    svi_result = run_svi(
        task_dcm_model, guide, model_args,
        num_steps=num_steps, lr=0.005,
        clip_norm=10.0, lr_decay_factor=0.01,
        elbo_type=elbo_type, guide_type=guide_type,
        model_kwargs=model_kwargs if model_kwargs else None,
    )
    elapsed = time.time() - t0
    # Predictive wraps bilinear kwargs via functools.partial.
    if model_kwargs:
        model_for_pred = partial(task_dcm_model, **model_kwargs)
    else:
        model_for_pred = task_dcm_model
    posterior = extract_posterior_params(
        guide, model_args, model=model_for_pred, num_samples=200,
    )
    # Capture final-losses tail for per-seed convergence diagnostics
    # (FIX 5 per orchestrator revision: Phase 15 NaN-safe guard zeros
    # predicted_bold, so a finite loss does NOT imply a meaningful posterior
    # -- expose the loss tail so downstream consumers can spot flat-lined
    # seeds).
    losses = svi_result.get("losses", []) if isinstance(svi_result, dict) else []
    posterior["final_losses"] = list(losses[-20:]) if losses else []
    return posterior, elapsed


def _posterior_to_numpy(
    posterior: dict[str, Any], B_true: torch.Tensor,
) -> dict[str, Any]:
    """Convert torch-tensor posterior dict to JSON-serializable numpy dict.

    Keeps per-site mean, std, and first 100 samples (downsample from 200 to
    control on-disk size; 100 samples target 95% CI quantile estimation
    variance ~7% per element at 10 seeds x 7 null elements = 70 pairs,
    keeping pooled-coverage noise below the RECOV-06 85% threshold margin).
    Also carries B_true for the RECOV metric helpers and final_losses for
    per-seed convergence diagnostics (FIX 3 per orchestrator revision).

    Note: ``B_free_0.samples`` stores the RAW unmasked ``B_free_0`` Pyro
    site draws (shape ``(S, N, N)``) from ``task_dcm_model.py``
    ``pyro.sample(f"B_free_{j}", ...)``. Masked-out elements retain their
    prior ``N(0, 1)`` posterior since the likelihood cannot constrain them
    -- their 95% CI covers zero by construction. This is the CORRECT
    source for RECOV-06 coverage_of_zero (which tests null elements); the
    alternative ``B`` deterministic site would have zero-ed null elements
    and give tautological 100% coverage.

    Parameters
    ----------
    posterior : dict
        Posterior dict from ``extract_posterior_params``.
    B_true : torch.Tensor
        Ground-truth B tensor for this seed, shape ``(J, N, N)``.

    Returns
    -------
    dict
        JSON-serializable per-site dict (lists of floats).
    """
    out: dict[str, Any] = {"B_true": B_true.cpu().numpy().tolist()}
    # Preserve final_losses as diagnostics (flat array of floats).
    if "final_losses" in posterior:
        out["final_losses"] = list(posterior["final_losses"])
    for site, vals in posterior.items():
        if site in ("median", "final_losses"):
            continue  # backward-compat key / diagnostics; handled above
        if not isinstance(vals, dict):
            continue
        site_out: dict[str, Any] = {}
        for key in ("mean", "std", "samples"):
            if key not in vals:
                continue
            v = vals[key]
            if isinstance(v, torch.Tensor):
                if key == "samples":
                    v = v[:100]
                site_out[key] = v.detach().cpu().numpy().tolist()
            else:
                site_out[key] = v
        out[site] = site_out
    return out


def _fit_bilinear_with_retry(
    model_args: tuple[Any, ...],
    model_kwargs: dict[str, Any],
    num_steps: int,
    elbo_type: str,
) -> tuple[dict[str, Any], float, float]:
    """Fit the bilinear task-DCM SVI; retry once at halved init_scale on NaN.

    Wraps :func:`_fit_and_extract` with a one-shot retry policy triggered by
    ``run_svi``'s ``RuntimeError("NaN ELBO at step {step}")`` when the NaN
    occurs at step 0. Non-zero-step NaNs are re-raised unchanged (they
    indicate divergence after learning has started, which is a substantively
    different failure mode than init-scale overflow).

    Returns ``(posterior, elapsed_seconds, init_scale_used)`` so the runner
    can record the retry decision per seed. ``init_scale_used`` is either
    :data:`_BILINEAR_INIT_SCALE` (default) or
    :data:`_BILINEAR_INIT_SCALE_RETRY` (retry taken).
    """
    try:
        posterior, elapsed = _fit_and_extract(
            model_args, model_kwargs,
            guide_type="auto_normal",
            init_scale=_BILINEAR_INIT_SCALE,
            num_steps=num_steps, elbo_type=elbo_type,
        )
        return posterior, elapsed, _BILINEAR_INIT_SCALE
    except RuntimeError as err:
        if "NaN ELBO at step 0" not in str(err):
            raise
        print(
            f"  NaN at step 0 with init_scale={_BILINEAR_INIT_SCALE}; "
            f"retrying once with init_scale={_BILINEAR_INIT_SCALE_RETRY}"
        )
        posterior, elapsed = _fit_and_extract(
            model_args, model_kwargs,
            guide_type="auto_normal",
            init_scale=_BILINEAR_INIT_SCALE_RETRY,
            num_steps=num_steps, elbo_type=elbo_type,
        )
        return posterior, elapsed, _BILINEAR_INIT_SCALE_RETRY


def run_task_bilinear_svi(
    config: BenchmarkConfig,
    *,
    stimulus_mod_factory: StimulusModFactory | None = None,
) -> dict[str, Any]:
    """Run task-bilinear DCM SVI recovery benchmark.

    For each dataset: (1) generate or load bilinear fixture; (2) fit bilinear
    ``task_dcm_model``; (3) fit bit-exact linear baseline (``b_masks=None``,
    MODEL-04) on the SAME fixture; (4) extract posteriors for both;
    (5) compute per-seed a_rmse for both and record wall time. Closes
    RECOV-01 and RECOV-02.

    Guide: AutoNormal with ``init_scale=0.005`` (L2) for bilinear;
    ``init_scale=0.01`` (task_svi.py default) for linear baseline. SVI
    steps: ``config.n_svi_steps`` (default: 500 quick, 1500 full per L4).

    Parameters
    ----------
    config : BenchmarkConfig
        Benchmark configuration. Uses n_datasets, n_svi_steps, seed,
        n_regions, quick, fixtures_dir, elbo_type.
    stimulus_mod_factory : StimulusModFactory or None, optional
        Custom stim_mod breakpoint-dict factory (Plan 16-03 L9). Default
        ``None`` -> Phase 16 default 4x12s epoch schedule via
        :func:`make_epoch_stimulus`. Non-None: the runner calls
        ``factory(seed=seed_i)`` per seed and substitutes the result into
        the bilinear fit (``A`` / ``C`` / ``B`` ground truth still
        seed-deterministic). When factory is non-None, the runner
        BYPASSES the ``config.fixtures_dir`` cache for stim_mod
        specifically (Plan 16-03 L10): factories are test/sweep artifacts
        and do NOT participate in .npz reproducibility. v0.3.1 HGF
        factories (``make_hgf_factory``) will plug in here unchanged.

    Returns
    -------
    dict
        Keys:

        - ``a_rmse_bilinear_list``, ``a_rmse_linear_list`` : list[float]
        - ``time_bilinear_list``, ``time_linear_list`` : list[float]
        - ``posterior_list`` : list[dict] per-seed posteriors with keys
          ``A_free``, ``B_free_0``, ``B`` (if available), ``C``,
          ``noise_prec``; each value is a dict with ``mean``, ``std``,
          ``samples`` (numpy arrays for JSON serializability)
        - ``b_true_list`` : list[list[float]] per-seed B_true flattened
        - ``a_true_list``, ``a_inferred_bilinear_list``,
          ``a_inferred_linear_list`` : list[list[float]]
        - ``n_success``, ``n_failed`` : int
        - ``metadata`` : dict with variant, method, n_regions, duration,
          etc. ``metadata['stimulus_mod_factory']`` records either
          ``'default_epochs'`` (when ``stimulus_mod_factory`` is None) or
          ``'custom'``.

    Notes
    -----
    Wraps each seed's SVI in ``try/except (RuntimeError, ValueError,
    AssertionError)``; n_failed increments on failures; requires
    ``n_success >= config.n_datasets // 2`` else returns
    ``{"status": "insufficient_data", ...}`` per task_svi.py:278-284.
    """
    N = config.n_regions
    num_steps = config.n_svi_steps

    a_rmse_bilinear_list: list[float] = []
    a_rmse_linear_list: list[float] = []
    time_bilinear_list: list[float] = []
    time_linear_list: list[float] = []
    posterior_list: list[dict[str, Any]] = []
    b_true_list: list[list[float]] = []
    a_true_list: list[list[float]] = []
    a_inferred_bilinear_list: list[list[float]] = []
    a_inferred_linear_list: list[list[float]] = []
    init_scale_used_bilinear_list: list[float] = []
    n_failed = 0

    # Silence pyro_dcm.stability WARNING spam during bilinear early-SVI draws
    # (research Section 2.4; D4 stability monitor is log-only, never raises).
    stability_logger = logging.getLogger("pyro_dcm.stability")
    prev_stability_level = stability_logger.level
    stability_logger.setLevel(logging.ERROR)

    try:
        for i in range(config.n_datasets):
            seed_i = config.seed + i
            print(
                f"Running dataset {i + 1}/{config.n_datasets} "
                f"(seed {seed_i})..."
            )
            try:
                torch.manual_seed(seed_i)
                np.random.seed(seed_i)
                pyro.set_rng_seed(seed_i)
                pyro.enable_validation(False)

                # Plan 16-03 L10: factory-hook dispatch.
                if stimulus_mod_factory is not None:
                    # Custom factory -> bypass fixture cache for stim_mod;
                    # always generate inline with the seed.
                    data = _make_bilinear_ground_truth_with_factory(
                        config.n_regions, seed_i, stimulus_mod_factory,
                    )
                else:
                    # Default path: plan 16-01 fixture-or-inline dispatch.
                    data = _load_or_make_fixture(i, config)

                # Model args (positional, shared between bilinear and linear).
                a_mask = torch.ones(N, N, dtype=torch.float64)
                c_mask = torch.zeros(N, 1, dtype=torch.float64)
                c_mask[0, 0] = 1.0
                t_eval = torch.arange(
                    0, _DURATION, _DT_MODEL, dtype=torch.float64,
                )
                model_args = (
                    data["bold"], data["stimulus"],
                    a_mask, c_mask, t_eval, _TR, _DT_MODEL,
                )

                # --- BILINEAR FIT (L2: auto_normal, init_scale=0.005) ---
                # One-shot retry at _BILINEAR_INIT_SCALE_RETRY on NaN at step 0
                # rescues cluster-observed init-scale overflow seeds
                # (Pitfall B1/B6); init_scale_used_bi is recorded per seed so
                # the SUMMARY can flag retries.
                bilinear_kwargs = {
                    "b_masks": [data["b_mask_0"]],
                    "stim_mod": data["stim_mod"],
                }
                posterior_bi, t_bi, init_scale_used_bi = (
                    _fit_bilinear_with_retry(
                        model_args, bilinear_kwargs,
                        num_steps=num_steps, elbo_type=config.elbo_type,
                    )
                )
                A_bi = parameterize_A(posterior_bi["A_free"]["mean"])
                a_rmse_bi = compute_rmse(data["A_true"], A_bi)

                # --- LINEAR BASELINE FIT (L3: same fixture, b_masks=None) ---
                # model_kwargs={} -> empty dict -> b_masks=None linear
                # short-circuit per MODEL-04 (Phase 15 L3 bit-exact lock).
                posterior_lin, t_lin = _fit_and_extract(
                    model_args, model_kwargs={},
                    guide_type="auto_normal", init_scale=0.01,
                    num_steps=num_steps, elbo_type=config.elbo_type,
                )
                A_lin = parameterize_A(posterior_lin["A_free"]["mean"])
                a_rmse_lin = compute_rmse(data["A_true"], A_lin)

                a_rmse_bilinear_list.append(a_rmse_bi)
                a_rmse_linear_list.append(a_rmse_lin)
                time_bilinear_list.append(t_bi)
                time_linear_list.append(t_lin)
                a_true_list.append(data["A_true"].flatten().tolist())
                a_inferred_bilinear_list.append(A_bi.flatten().tolist())
                a_inferred_linear_list.append(A_lin.flatten().tolist())
                b_true_list.append(data["B_true"].flatten().tolist())
                init_scale_used_bilinear_list.append(init_scale_used_bi)
                posterior_list.append(
                    _posterior_to_numpy(posterior_bi, data["B_true"]),
                )

                print(
                    f"  a_rmse_bi={a_rmse_bi:.4f}, "
                    f"a_rmse_lin={a_rmse_lin:.4f}, "
                    f"t_bi={t_bi:.1f}s, t_lin={t_lin:.1f}s, "
                    f"init_scale_bi={init_scale_used_bi}"
                )
            except (RuntimeError, ValueError, AssertionError) as e:
                print(f"  FAILED: {e}")
                n_failed += 1
            finally:
                pyro.enable_validation(True)
    finally:
        stability_logger.setLevel(prev_stability_level)

    n_success = len(a_rmse_bilinear_list)
    if n_success < max(1, config.n_datasets // 2):
        return {
            "status": "insufficient_data",
            "n_success": n_success,
            "n_failed": n_failed,
            "n_datasets": config.n_datasets,
        }

    return {
        "a_rmse_bilinear_list": a_rmse_bilinear_list,
        "a_rmse_linear_list": a_rmse_linear_list,
        "time_bilinear_list": time_bilinear_list,
        "time_linear_list": time_linear_list,
        "posterior_list": posterior_list,
        "b_true_list": b_true_list,
        "a_true_list": a_true_list,
        "a_inferred_bilinear_list": a_inferred_bilinear_list,
        "a_inferred_linear_list": a_inferred_linear_list,
        "init_scale_used_bilinear_list": init_scale_used_bilinear_list,
        "mean_a_rmse_bilinear": float(np.mean(a_rmse_bilinear_list)),
        "mean_a_rmse_linear": float(np.mean(a_rmse_linear_list)),
        "mean_time_bilinear": float(np.mean(time_bilinear_list)),
        "mean_time_linear": float(np.mean(time_linear_list)),
        "a_rmse_bilinear_stats": compute_summary_stats(a_rmse_bilinear_list),
        "a_rmse_linear_stats": compute_summary_stats(a_rmse_linear_list),
        "n_success": n_success,
        "n_failed": n_failed,
        "metadata": {
            "variant": "task_bilinear",
            "method": "svi",
            "n_regions": N,
            "J": 1,
            "duration": _DURATION,
            "TR": _TR,
            "SNR": _SNR,
            "num_steps": num_steps,
            "guide_type": "auto_normal",
            "init_scale_bilinear": _BILINEAR_INIT_SCALE,
            "init_scale_bilinear_retry": _BILINEAR_INIT_SCALE_RETRY,
            "init_scale_bilinear_n_retries": int(
                sum(
                    1 for v in init_scale_used_bilinear_list
                    if v == _BILINEAR_INIT_SCALE_RETRY
                )
            ),
            "init_scale_linear": 0.01,
            "quick": config.quick,
            "b_true_magnitudes": {"B[1,0]": _B_10, "B[2,1]": _B_21},
            "c_magnitude": _C_00,
            "epoch_schedule": {
                "event_times": _EPOCH_TIMES,
                "event_durations": _EPOCH_DURATIONS,
            },
            "stimulus_mod_factory": (
                "custom" if stimulus_mod_factory is not None
                else "default_epochs"
            ),
        },
    }
