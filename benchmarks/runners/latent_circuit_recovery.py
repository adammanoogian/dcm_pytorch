"""Latent-circuit DCM SVI recovery benchmark runner (v0.6.0 Phase 20).

Implements the simulate -> fit -> evaluate loop for the latent-circuit
DCM synthetic validation (SYNTH-01, SYNTH-02 in
.planning/phases/20-latent-circuit-forward-model/20-CONTEXT.md).

For each seed:
1. Generate N=4 bilinear latent-circuit ground truth via
   ``simulate_latent_circuit``.
2. Split 80/20 into training and held-out test trajectories.
3. Fit ``latent_circuit_dcm_model`` with multi-start SVI.
4. Extract posterior via ``extract_posterior_params``.
5. Compute per-seed metrics: A-RMSE, B-RMSE, sign recovery,
   95% CI coverage, shrinkage, trajectory R-squared (held-out).

Supports ``BenchmarkConfig.quick`` mode (3 seeds, 50 SVI steps) and
full mode (10+ seeds, 1000 SVI steps). The ``lc_a_prior_var`` and
``lc_b_prior_var`` keyword arguments temporarily monkey-patch the
``LC_A_PRIOR_VARIANCE`` and ``LC_B_PRIOR_VARIANCE`` module constants
for Plan 20-05 calibration sweep experiments.

References
----------
.planning/phases/20-latent-circuit-forward-model/20-CONTEXT.md
    SYNTH-01 (A/B recovery), SYNTH-02 (trajectory R-squared gate).
.planning/phases/20-latent-circuit-forward-model/20-04-PLAN.md
    Runner contract and fixture generation specification.
.planning/STATE.md
    Decision [20-03-D2]: AutoIAFNormal hidden_dim > latent_dim.
    Decision [20-02]: n_restarts=1 backward-compat path.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from functools import partial
from statistics import median
from typing import Any, Generator

import numpy as np
import pyro
import torch

from benchmarks.config import BenchmarkConfig
from benchmarks.latent_circuit_metrics import (
    compute_trajectory_r_squared,
)
from benchmarks.metrics import compute_rmse
from pyro_dcm.forward_models.neural_state import parameterize_A, parameterize_B
from pyro_dcm.models import (
    create_guide,
    extract_posterior_params,
    run_svi,
)
from pyro_dcm.models.latent_circuit_dcm_model import (
    latent_circuit_dcm_model,
    LC_A_PRIOR_VARIANCE as _LC_A_PRIOR_VARIANCE_DEFAULT,
    LC_B_PRIOR_VARIANCE as _LC_B_PRIOR_VARIANCE_DEFAULT,
)
# Import the actual submodule (not the __init__ re-export of the function)
# for monkey-patching the LC_*_PRIOR_VARIANCE constants during calibration.
import importlib as _importlib
_lc_model_submodule = _importlib.import_module(
    "pyro_dcm.models.latent_circuit_dcm_model"
)
from pyro_dcm.simulators.latent_circuit_simulator import (
    make_stable_latent_circuit_A,
    simulate_latent_circuit,
)
from pyro_dcm.simulators.task_simulator import make_block_stimulus, make_epoch_stimulus
from pyro_dcm.utils.ode_integrator import (
    PiecewiseConstantInput,
    integrate_ode,
    merge_piecewise_inputs,
)
from pyro_dcm.forward_models.coupled_system import CoupledDCMSystem

_log = logging.getLogger("pyro_dcm.latent_circuit_recovery")

# ---------------------------------------------------------------------------
# Ground-truth constants (Plan 20-04 specification)
# ---------------------------------------------------------------------------

_N_REGIONS: int = 4
"""Number of latent dimensions for the recovery benchmark."""

_DURATION: float = 100.0
"""Simulation duration in seconds."""

_DT: float = 0.01
"""ODE time step (seconds)."""

_SNR: float = 10.0
"""Signal-to-noise ratio for synthetic trajectories."""

_DRIVING_N_BLOCKS: int = 5
_DRIVING_BLOCK_DURATION: float = 10.0
_DRIVING_REST_DURATION: float = 10.0

# Modulator epochs are placed as FRACTIONS of the simulation duration so that
# all three windows fall inside the training split for any duration. Phase
# 20-05 originally hardcoded absolute times [10, 40, 70]s designed for a 100s
# run; when the acceptance run was reworked to 50s with an 80/20 split, the
# t=40 epoch fell in the held-out test segment and t=70 was cut entirely,
# leaving a SINGLE 8s modulator window (t=10-18) in training. With B entering
# the dynamics only while u_mod is ON, that starved B of identifying signal
# and collapsed the B posterior toward zero (B-RMSE ~0.31). Fractions
# [0.10, 0.35, 0.60] keep all three epochs within the first 60% of the
# trajectory -> always inside the 80% training window, well separated.
# See .planning/phases/20-latent-circuit-forward-model/20-05-SUMMARY.md (root cause 1).
_MOD_EVENT_FRACTIONS: list[float] = [0.10, 0.35, 0.60]
_MOD_EVENT_DURATIONS: list[float] = [8.0, 8.0, 8.0]
_MOD_EVENT_AMPLITUDES: list[float] = [1.0, 1.0, 1.0]

_C_00: float = 1.0  # driving input amplitude to region 0

# Bilinear B-matrix values: descending chain 0->1->2->3.
_B_10: float = 0.4
_B_21: float = 0.3
_B_32: float = 0.2

# Train/test split fraction.
_TRAIN_FRACTION: float = 0.80

# Seed-pool: skip corrupt (NaN BOLD) seeds up to 2x requested count.
_MAX_POOL_MULTIPLIER: int = 2


# ---------------------------------------------------------------------------
# Prior monkey-patch context manager for calibration sweep (Plan 20-05)
# ---------------------------------------------------------------------------

@contextmanager
def _patch_lc_priors(
    lc_a_prior_var: float | None,
    lc_b_prior_var: float | None,
) -> Generator[None, None, None]:
    """Temporarily patch LC_A_PRIOR_VARIANCE and LC_B_PRIOR_VARIANCE.

    Used by the Plan 20-05 calibration sweep to test different prior
    scales without code changes. The original values are restored even
    if the body raises an exception.

    Parameters
    ----------
    lc_a_prior_var : float or None
        Override for ``LC_A_PRIOR_VARIANCE``. None means no change.
    lc_b_prior_var : float or None
        Override for ``LC_B_PRIOR_VARIANCE``. None means no change.

    Yields
    ------
    None
    """
    orig_a = _lc_model_submodule.LC_A_PRIOR_VARIANCE
    orig_b = _lc_model_submodule.LC_B_PRIOR_VARIANCE
    try:
        if lc_a_prior_var is not None:
            _lc_model_submodule.LC_A_PRIOR_VARIANCE = lc_a_prior_var
        if lc_b_prior_var is not None:
            _lc_model_submodule.LC_B_PRIOR_VARIANCE = lc_b_prior_var
        yield
    finally:
        _lc_model_submodule.LC_A_PRIOR_VARIANCE = orig_a
        _lc_model_submodule.LC_B_PRIOR_VARIANCE = orig_b


# ---------------------------------------------------------------------------
# Ground-truth builder
# ---------------------------------------------------------------------------

def _build_ground_truth(
    seed: int | None = None,
    duration: float = _DURATION,
) -> dict[str, Any]:
    """Construct fixed N=4 bilinear latent-circuit ground truth.

    The topology is a directed descending chain A[i, i+1] > 0 with
    B modulation on the same chain. The base A matrix uses
    self_inhibition=0.5 Hz (slow RNN timescale) plus explicit off-diagonal
    entries; stability is verified with a Gershgorin-style check and B is
    scaled if needed.

    Parameters
    ----------
    seed : int or None, optional
        Random seed for ``make_stable_latent_circuit_A``. Default None.
    duration : float, optional
        Simulation duration in seconds. Default ``_DURATION`` (100s).
        The modulator stimulus grid AND its three event times are derived
        from this value (events at ``_MOD_EVENT_FRACTIONS * duration``) so
        that all modulator windows land inside the training split regardless
        of duration. MUST match the ``duration`` passed to
        ``simulate_latent_circuit`` downstream, otherwise the modulator grid
        and the simulated trajectory grid disagree (the original 20-05 bug).

    Returns
    -------
    dict
        Keys: ``A_true``, ``B_true`` (shape (1, N, N)), ``C``,
        ``b_mask_0``, ``stim``, ``stim_mod``, ``a_mask``, ``c_mask``.
    """
    N = _N_REGIONS

    # Base A: stable self-inhibition + directed chain off-diagonals.
    A_base = make_stable_latent_circuit_A(
        N,
        density=0.0,  # start with only diagonal
        self_inhibition=0.5,
        seed=seed,
    )
    # Add directed off-diagonals: A[i, i+1] = 0.15 (feedforward chain).
    A_true = A_base.clone()
    for i in range(N - 1):
        A_true[i + 1, i] = 0.15  # region i -> region i+1

    # Verify stability: max Re(eig(A + sum(B_j) * 1)) < -0.05.
    B_chain_sum = torch.zeros(N, N, dtype=torch.float64)
    B_chain_sum[1, 0] = _B_10
    B_chain_sum[2, 1] = _B_21
    B_chain_sum[3, 2] = _B_32
    scale = 1.0
    while True:
        A_eff = A_true + B_chain_sum * scale
        max_re = torch.linalg.eigvals(A_eff).real.max().item()
        if max_re < -0.05:
            break
        scale *= 0.5
        _log.warning(
            "A + B_true unstable (max Re eig=%.4f); scaling B by 0.5 to %.4f",
            max_re, scale,
        )

    # Final B_true scaled by `scale`.
    B_true = torch.zeros(1, N, N, dtype=torch.float64)
    B_true[0, 1, 0] = _B_10 * scale
    B_true[0, 2, 1] = _B_21 * scale
    B_true[0, 3, 2] = _B_32 * scale

    # B mask: 1 where B is non-zero.
    b_mask_0 = torch.zeros(N, N, dtype=torch.float64)
    b_mask_0[1, 0] = 1.0
    b_mask_0[2, 1] = 1.0
    b_mask_0[3, 2] = 1.0

    # C: driving input to region 0.
    C = torch.zeros(N, 1, dtype=torch.float64)
    C[0, 0] = _C_00

    # a_mask: all-ones (estimate all connections).
    a_mask = torch.ones(N, N, dtype=torch.float64)

    # c_mask: only region 0, input 0.
    c_mask = torch.zeros(N, 1, dtype=torch.float64)
    c_mask[0, 0] = 1.0

    # Driving stimulus: block design.
    stim = make_block_stimulus(
        n_blocks=_DRIVING_N_BLOCKS,
        block_duration=_DRIVING_BLOCK_DURATION,
        rest_duration=_DRIVING_REST_DURATION,
        n_inputs=1,
    )

    # Modulator: 3 epochs at fractions of the (effective) duration so they
    # all fall inside the training split. event_times and the stimulus grid
    # are both derived from `duration` (must match simulate_latent_circuit).
    event_times = [frac * duration for frac in _MOD_EVENT_FRACTIONS]
    stim_mod_dict = make_epoch_stimulus(
        event_times=event_times,
        event_durations=_MOD_EVENT_DURATIONS,
        event_amplitudes=_MOD_EVENT_AMPLITUDES,
        duration=duration,
        dt=_DT,
        n_inputs=1,
    )
    stim_mod = PiecewiseConstantInput(
        stim_mod_dict["times"], stim_mod_dict["values"],
    )

    return {
        "A_true": A_true,
        "B_true": B_true,
        "C": C,
        "b_mask_0": b_mask_0,
        "stim": stim,
        "stim_mod": stim_mod,
        "a_mask": a_mask,
        "c_mask": c_mask,
    }


# ---------------------------------------------------------------------------
# Trajectory prediction for held-out R-squared
# ---------------------------------------------------------------------------

def _predict_trajectories(
    A_free_mean: torch.Tensor,
    C_mean: torch.Tensor,
    B_free_mean: torch.Tensor,
    b_mask_0: torch.Tensor,
    stimulus: PiecewiseConstantInput,
    stim_mod: PiecewiseConstantInput,
    t_eval_full: torch.Tensor,
    dt: float,
    T_train: int,
) -> torch.Tensor:
    """Run the forward ODE from t=0 and return the held-out test segment.

    Integrates the full time grid starting from y0=zeros, then returns
    only the test portion (indices T_train:). This is necessary because
    the neural state at the train/test boundary is not zero — it has
    evolved under stimulus input for the entire training segment.

    Parameters
    ----------
    A_free_mean : torch.Tensor
        Posterior mean A_free, shape ``(N, N)``.
    C_mean : torch.Tensor
        Posterior mean C, shape ``(N, M)``.
    B_free_mean : torch.Tensor
        Posterior mean B_free_0, shape ``(N, N)`` (raw free params).
    b_mask_0 : torch.Tensor
        B structural mask, shape ``(N, N)``.
    stimulus : PiecewiseConstantInput
        Driving input function.
    stim_mod : PiecewiseConstantInput
        Modulator input function.
    t_eval_full : torch.Tensor
        Full time grid (train + test), shape ``(T_total,)``.
    dt : float
        ODE step size.
    T_train : int
        Number of training time points. The returned tensor starts
        at index ``T_train``.

    Returns
    -------
    torch.Tensor
        Predicted trajectories for the test segment, shape
        ``(T_total - T_train, N)``.
    """
    N = A_free_mean.shape[0]
    A = parameterize_A(A_free_mean.to(torch.float64))

    b_mask_stacked = b_mask_0.unsqueeze(0)  # (1, N, N)
    b_free_stacked = B_free_mean.unsqueeze(0)  # (1, N, N)
    B_stacked = parameterize_B(b_free_stacked, b_mask_stacked)  # (1, N, N)

    merged = merge_piecewise_inputs(stimulus, stim_mod)

    system = CoupledDCMSystem(
        A,
        C_mean.to(torch.float64),
        merged,
        hemodynamic=False,
        B=B_stacked,
        n_driving_inputs=C_mean.shape[1],
    )
    y0 = torch.zeros(N, dtype=torch.float64)

    with torch.no_grad():
        solution = integrate_ode(
            system,
            y0,
            t_eval_full.to(torch.float64),
            method="rk4",
            step_size=dt,
        )
    return solution[T_train:]  # (T_test, N)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_latent_circuit_recovery(
    config: BenchmarkConfig,
    *,
    n_regions: int = 4,
    n_modulators: int = 1,
    n_restarts: int = 10,
    init_scale: float = 0.1,
    lc_a_prior_var: float | None = None,
    lc_b_prior_var: float | None = None,
    _duration_override: float | None = None,
) -> dict[str, Any]:
    """Run latent-circuit DCM SVI recovery benchmark.

    For each seed: generate ground truth, simulate trajectories, split
    into training (80%) and held-out test (20%) segments, fit
    ``latent_circuit_dcm_model`` with multi-start SVI, extract
    posteriors, compute per-seed metrics.

    Parameters
    ----------
    config : BenchmarkConfig
        Benchmark configuration. Uses ``n_datasets`` (number of seeds),
        ``n_svi_steps``, ``seed`` (base seed), ``quick``.
    n_regions : int, optional
        Number of latent dimensions (N). Default 4. Currently the
        ground-truth topology is fixed for N=4; other values may produce
        warnings.
    n_modulators : int, optional
        Number of bilinear modulators (J). Default 1.
    n_restarts : int, optional
        Number of independent SVI restarts. Default 10 (minimum per
        pitfall LC11). Set to 1 for quick smoke tests.
    init_scale : float, optional
        Guide initial scale for ``AutoNormal``. Default 0.1, calibrated
        for the LC_A_PRIOR_VARIANCE=1/16 prior. Use 0.01 or smaller
        if SVI diverges at step 0.
    lc_a_prior_var : float or None, optional
        Override for ``LC_A_PRIOR_VARIANCE`` during this run (Plan 20-05
        calibration sweep). None means use the module constant.
    lc_b_prior_var : float or None, optional
        Override for ``LC_B_PRIOR_VARIANCE`` during this run. None means
        use the module constant.
    _duration_override : float or None, optional
        Override for the simulation duration (seconds). Default ``None``
        uses the module constant ``_DURATION = 100.0``. Set to a small
        value (e.g., 2.0) for import/API smoke tests only -- recovery
        metrics are not meaningful at very short durations. Not part of
        the public API; prefixed with underscore per convention.

    Returns
    -------
    dict
        Keys:

        - ``per_seed_results`` (list[dict]): per-seed metric dicts
          with keys ``a_rmse``, ``b_rmse``, ``sign_recovery``,
          ``ci_coverage_95``, ``trajectory_r_squared``,
          ``shrinkage_A``, ``shrinkage_B``, ``final_elbo``,
          ``elapsed_s``, ``seed``.
        - ``ground_truth`` (dict): ground-truth ``A_true``, ``B_true``,
          ``C`` tensors (common to all seeds).
        - ``aggregate`` (dict): medians of each metric across seeds.
        - ``config_summary`` (dict): runner configuration.
        - ``n_success`` (int): seeds that completed successfully.
        - ``n_failed`` (int): seeds that failed or were skipped.
        - ``seeds_used`` (list[int]): seeds that contributed results.
        - ``seeds_skipped`` (list[int]): seeds skipped (NaN data).

    Notes
    -----
    The ground truth is common to all seeds (same A, B, C topology).
    Per-seed variation comes from the noise realization in
    ``simulate_latent_circuit`` (different ``seed`` argument each run).

    **Compute routing.** Full runs (``quick=False``, ``n_datasets >= 10``,
    ``n_svi_steps >= 1000``) take substantially longer than 3 minutes on
    laptop CPU. Route full-scale runs to M3 via ``sbatch`` per project
    compute routing policy.

    References
    ----------
    .planning/phases/20-latent-circuit-forward-model/20-CONTEXT.md
        SYNTH-01, SYNTH-02, pitfall LC11 (multi-start).
    .planning/phases/20-latent-circuit-forward-model/20-04-PLAN.md
    """
    if n_regions != _N_REGIONS:
        _log.warning(
            "n_regions=%d requested but ground-truth topology is fixed for "
            "N=%d; A/B topology will be reused unchanged.",
            n_regions, _N_REGIONS,
        )

    # Effective duration: allow smoke-test override.
    duration = _duration_override if _duration_override is not None else _DURATION

    # Build shared ground truth (same for all seeds). Pass the effective
    # duration so the modulator epochs and grid match the simulated grid.
    gt = _build_ground_truth(seed=0, duration=duration)
    A_true = gt["A_true"]
    B_true = gt["B_true"]  # (1, N, N)
    C_true = gt["C"]
    b_mask_0 = gt["b_mask_0"]
    stim = gt["stim"]
    stim_mod = gt["stim_mod"]
    a_mask = gt["a_mask"]
    c_mask = gt["c_mask"]
    N = A_true.shape[0]

    # Stimulus as PiecewiseConstantInput.
    driving_stim = PiecewiseConstantInput(stim["times"], stim["values"])

    # Silence stability logger during SVI (expected occasional warnings).
    stability_logger = logging.getLogger("pyro_dcm.stability")
    prev_level = stability_logger.level
    stability_logger.setLevel(logging.ERROR)

    per_seed_results: list[dict[str, Any]] = []
    seeds_used: list[int] = []
    seeds_skipped: list[int] = []
    n_failed = 0

    num_steps = config.n_svi_steps
    base_seed = config.seed
    n_datasets = config.n_datasets
    max_pool = n_datasets * _MAX_POOL_MULTIPLIER

    try:
        pool_idx = 0
        while (
            len(seeds_used) < n_datasets
            and pool_idx < max_pool
        ):
            seed_i = base_seed + pool_idx
            pool_idx += 1
            slot = len(seeds_used) + 1
            print(
                f"  LC recovery: dataset {slot}/{n_datasets} "
                f"(seed {seed_i})..."
            )

            try:
                torch.manual_seed(seed_i)
                np.random.seed(seed_i)
                pyro.set_rng_seed(seed_i)
                pyro.enable_validation(False)

                # --- Simulate ---
                sim = simulate_latent_circuit(
                    A_true, C_true, stim,
                    duration=duration, dt=_DT, SNR=_SNR,
                    solver="rk4", seed=seed_i,
                    B_list=[B_true[0]],
                    stimulus_mod=stim_mod,
                )
                trajs = sim["trajectories"]  # (T, N) noisy

                # Pre-flight NaN check (matches task_bilinear pattern).
                if (
                    torch.isnan(trajs).any().item()
                    or torch.isinf(trajs).any().item()
                ):
                    seeds_skipped.append(seed_i)
                    print(
                        f"  SKIPPED seed {seed_i}: simulated trajectories "
                        f"contain NaN/Inf (ODE divergence)."
                    )
                    continue

                T_total = trajs.shape[0]
                T_train = int(T_total * _TRAIN_FRACTION)

                # Train and test splits.
                trajs_train = trajs[:T_train]    # (T_train, N)
                trajs_test = trajs[T_train:]     # (T_test, N)

                # Time grids.
                t_all = sim["times"]             # (T_total,)
                t_eval_train = t_all[:T_train]   # (T_train,)

                # --- Fit (with optional prior monkey-patch) ---
                model_args = (
                    trajs_train,
                    driving_stim,
                    a_mask,
                    c_mask,
                    t_eval_train,
                    _DT,
                )
                model_kwargs: dict[str, Any] = {
                    "b_masks": [b_mask_0],
                    "stim_mod": stim_mod,
                }

                guide_factory = partial(
                    create_guide,
                    latent_circuit_dcm_model,
                    guide_type="auto_normal",
                    init_scale=init_scale,
                )

                t0 = time.time()
                with _patch_lc_priors(lc_a_prior_var, lc_b_prior_var):
                    svi_result = run_svi(
                        latent_circuit_dcm_model,
                        guide_factory(),
                        model_args,
                        num_steps=num_steps,
                        lr=0.005,
                        clip_norm=10.0,
                        lr_decay_factor=0.01,
                        elbo_type="trace_elbo",
                        guide_type="auto_normal",
                        model_kwargs=model_kwargs,
                        n_restarts=n_restarts,
                        guide_factory=guide_factory,
                    )

                    # Build fresh guide from restored param store.
                    best_guide = guide_factory()
                    posterior = extract_posterior_params(
                        best_guide,
                        model_args,
                        model=partial(
                            latent_circuit_dcm_model,
                            **model_kwargs,
                        ),
                        num_samples=200,
                    )
                elapsed = time.time() - t0

                # --- Extract posterior means ---
                A_free_mean = posterior["A_free"]["mean"].to(torch.float64)
                C_mean = (
                    posterior["C"]["mean"].to(torch.float64)
                    * c_mask
                )
                B_free_mean = (
                    posterior["B_free_0"]["mean"].to(torch.float64)
                )
                A_inferred = parameterize_A(A_free_mean)

                # --- A-RMSE ---
                a_rmse = compute_rmse(A_true.to(torch.float64), A_inferred)

                # --- B-RMSE (magnitude-masked, shape (1, N, N)) ---
                B_inferred_unsqueeze = (
                    parameterize_B(
                        B_free_mean.unsqueeze(0),
                        b_mask_0.unsqueeze(0),
                    )
                )  # (1, N, N)
                b_rmse = float(
                    ((B_true.to(torch.float64) - B_inferred_unsqueeze) ** 2)
                    .mul(
                        (B_true.to(torch.float64).abs() > 0.1).float()
                    )
                    .sum()
                    / (B_true.to(torch.float64).abs() > 0.1).float().sum()
                    .clamp(min=1.0)
                )
                b_rmse = float(b_rmse ** 0.5)

                # --- Sign recovery (pooled, on B > 0.1) ---
                B_true_list = [B_true.to(torch.float64)]
                B_inferred_list = [B_inferred_unsqueeze.to(torch.float64)]
                sign_recovery = float(
                    _compute_sign_recovery_pooled(
                        B_true_list, B_inferred_list,
                    )
                )

                # --- CI coverage at 95% (null B elements) ---
                B_samples = (
                    posterior["B_free_0"]["samples"]
                    .to(torch.float64)
                    .unsqueeze(1)
                )  # (S, 1, N, N)
                ci_coverage_95 = float(
                    _compute_coverage_of_zero_single(
                        B_true.to(torch.float64),
                        B_samples,
                    )
                )

                # --- Shrinkage ---
                a_prior_var = (
                    lc_a_prior_var if lc_a_prior_var is not None
                    else _LC_A_PRIOR_VARIANCE_DEFAULT
                )
                b_prior_var = (
                    lc_b_prior_var if lc_b_prior_var is not None
                    else _LC_B_PRIOR_VARIANCE_DEFAULT
                )
                A_std_mean = (
                    posterior["A_free"]["std"].float().mean().item()
                )
                B_std_mean = (
                    posterior["B_free_0"]["std"].float().mean().item()
                )
                shrinkage_A = float(A_std_mean / (a_prior_var ** 0.5))
                shrinkage_B = float(B_std_mean / (b_prior_var ** 0.5))

                # --- Trajectory R-squared (held-out test segment) ---
                predicted_test = _predict_trajectories(
                    A_free_mean,
                    C_mean,
                    B_free_mean,
                    b_mask_0,
                    driving_stim,
                    stim_mod,
                    t_all,
                    _DT,
                    T_train,
                )
                traj_r2 = compute_trajectory_r_squared(
                    predicted_test, trajs_test.to(torch.float64),
                )

                # --- ELBO ---
                final_elbo = float(svi_result["final_loss"])

                result_dict: dict[str, Any] = {
                    "seed": seed_i,
                    "a_rmse": float(a_rmse),
                    "b_rmse": float(b_rmse),
                    "sign_recovery": float(sign_recovery),
                    "ci_coverage_95": float(ci_coverage_95),
                    "trajectory_r_squared": float(traj_r2),
                    "shrinkage_A": float(shrinkage_A),
                    "shrinkage_B": float(shrinkage_B),
                    "final_elbo": final_elbo,
                    "elapsed_s": float(elapsed),
                }
                per_seed_results.append(result_dict)
                seeds_used.append(seed_i)

                print(
                    f"  a_rmse={a_rmse:.4f}, b_rmse={b_rmse:.4f}, "
                    f"sign={sign_recovery:.3f}, cov95={ci_coverage_95:.3f}, "
                    f"traj_r2={traj_r2:.4f}, t={elapsed:.1f}s"
                )

            except (RuntimeError, ValueError, AssertionError) as exc:
                _log.error("Seed %d failed: %s", seed_i, exc)
                print(f"  FAILED seed {seed_i}: {exc}")
                n_failed += 1
            finally:
                pyro.enable_validation(True)

    finally:
        stability_logger.setLevel(prev_level)

    n_success = len(per_seed_results)

    # Aggregate medians.
    aggregate: dict[str, float] = {}
    if per_seed_results:
        for key in (
            "a_rmse", "b_rmse", "sign_recovery", "ci_coverage_95",
            "trajectory_r_squared", "shrinkage_A", "shrinkage_B",
        ):
            aggregate[key] = median(
                float(r[key]) for r in per_seed_results
            )

    return {
        "per_seed_results": per_seed_results,
        "ground_truth": {
            "A_true": A_true,
            "B_true": B_true,
            "C": C_true,
        },
        "aggregate": aggregate,
        "config_summary": {
            "n_regions": N,
            "n_modulators": n_modulators,
            "n_restarts": n_restarts,
            "init_scale": init_scale,
            "n_svi_steps": num_steps,
            "duration": duration,
            "dt": _DT,
            "SNR": _SNR,
            "train_fraction": _TRAIN_FRACTION,
            "lc_a_prior_var": lc_a_prior_var,
            "lc_b_prior_var": lc_b_prior_var,
        },
        "n_success": n_success,
        "n_failed": n_failed,
        "seeds_used": seeds_used,
        "seeds_skipped": seeds_skipped,
    }


# ---------------------------------------------------------------------------
# Private metric helpers (avoid circular import with bilinear_metrics)
# ---------------------------------------------------------------------------

def _compute_sign_recovery_pooled(
    B_true_list: list[torch.Tensor],
    B_inferred_list: list[torch.Tensor],
    magnitude_threshold: float = 0.1,
) -> float:
    """Pooled sign recovery for B elements with |B_true| > threshold.

    Parameters
    ----------
    B_true_list : list of torch.Tensor
        Per-seed ground-truth B tensors, shape ``(J, N, N)`` each.
    B_inferred_list : list of torch.Tensor
        Per-seed posterior-mean B tensors, same shape.
    magnitude_threshold : float, optional
        Default 0.1.

    Returns
    -------
    float
        Pooled sign recovery fraction, or 0.0 if no eligible elements.
    """
    total_matches = 0
    total_eligible = 0
    for B_true, B_inferred in zip(B_true_list, B_inferred_list, strict=True):
        mask = torch.abs(B_true) > magnitude_threshold
        if not mask.any():
            continue
        match = (torch.sign(B_inferred) == torch.sign(B_true))[mask]
        total_matches += int(match.sum().item())
        total_eligible += int(mask.sum().item())
    if total_eligible == 0:
        return 0.0
    return total_matches / total_eligible


def _compute_coverage_of_zero_single(
    B_true: torch.Tensor,
    B_samples: torch.Tensor,
    null_threshold: float = 0.1,
    ci_level: float = 0.95,
) -> float:
    """95% CI coverage-of-zero for null B elements (|B_true| < threshold).

    Parameters
    ----------
    B_true : torch.Tensor
        Ground-truth B tensor, shape ``(J, N, N)``.
    B_samples : torch.Tensor
        Posterior samples, shape ``(S, J, N, N)``.
    null_threshold : float, optional
        Default 0.1.
    ci_level : float, optional
        Default 0.95.

    Returns
    -------
    float
        Coverage fraction, or 0.0 if no null elements.
    """
    alpha = (1.0 - ci_level) / 2.0
    mask = torch.abs(B_true) < null_threshold
    if not mask.any():
        return 0.0
    lo = torch.quantile(B_samples.float(), alpha, dim=0)
    hi = torch.quantile(B_samples.float(), 1.0 - alpha, dim=0)
    contains_zero = (lo <= 0) & (0 <= hi)
    total = int(mask.sum().item())
    matched = int(contains_zero[mask].sum().item())
    return matched / total if total > 0 else 0.0
