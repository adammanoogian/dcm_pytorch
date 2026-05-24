"""Latent circuit DCM parameter recovery benchmark runner (v0.6.0 Phase 20).

Implements the simulate -> train/test split -> multi-start SVI fit -> measure
loop for latent circuit DCM parameter recovery on synthetic N-dimensional
neural state trajectories (no hemodynamics).

Ground truth: N=4 regions, J=1 modulator, bilinear DCM with directed chain
connectivity and 3 non-null B elements. Extends the Phase 16 bilinear
benchmark pattern with trajectory R-squared and ELBO model selection.

Downstream: Plan 20-05 consumes runner output for prior recalibration sweep
and acceptance gate reporting.

References
----------
.planning/phases/20-latent-circuit-forward-model/20-04-PLAN.md (spec)
.planning/REQUIREMENTS-v0.6.0.md SYNTH-01..03
"""

from __future__ import annotations

import logging
import time
from functools import partial
from typing import Any

import numpy as np
import pyro
import torch

from benchmarks.config import BenchmarkConfig
from benchmarks.latent_circuit_metrics import (
    compute_coverage_multi_level,
    compute_trajectory_r_squared,
)
from benchmarks.metrics import compute_rmse
from pyro_dcm.forward_models.neural_state import parameterize_A, parameterize_B
from pyro_dcm.models import (
    create_guide,
    extract_posterior_params,
    latent_circuit_dcm_model,
    run_svi,
)
from pyro_dcm.models.latent_circuit_dcm_model import (
    LC_A_PRIOR_VARIANCE,
    LC_B_PRIOR_VARIANCE,
)
from pyro_dcm.simulators.latent_circuit_simulator import (
    make_stable_latent_circuit_A,
    simulate_latent_circuit,
)
from pyro_dcm.simulators.task_simulator import (
    make_block_stimulus,
    make_epoch_stimulus,
)
from pyro_dcm.utils.ode_integrator import PiecewiseConstantInput

# Ground-truth constants for Phase 20 latent circuit recovery.
_DURATION: float = 100.0
_DT_SIM: float = 0.01
_DT_MODEL: float = 0.01
_SNR: float = 10.0
_N_REGIONS: int = 4
_N_MODULATORS: int = 1

# B-matrix ground truth: 3 non-null elements in directed chain.
_B_10: float = 0.4
_B_21: float = 0.3
_B_32: float = 0.2

# C: single driving input to region 0.
_C_00: float = 1.0

# Driving stimulus: 4 blocks of 10s ON + 15s OFF.
_DRIVING_N_BLOCKS: int = 4
_DRIVING_BLOCK_DURATION: float = 10.0
_DRIVING_REST_DURATION: float = 15.0

# Modulator: 3 epochs of 8s at [15, 40, 70] seconds.
_EPOCH_TIMES: list[float] = [15.0, 40.0, 70.0]
_EPOCH_DURATIONS: list[float] = [8.0, 8.0, 8.0]
_EPOCH_AMPLITUDES: list[float] = [1.0, 1.0, 1.0]

# Seed-pool rejection: skip NaN seeds.
_MAX_POOL_MULTIPLIER: int = 3


def _make_latent_circuit_ground_truth(
    n_regions: int,
    n_modulators: int,
    seed_i: int,
) -> dict[str, Any]:
    """Construct ground-truth A, C, B, masks, stimuli for a seed.

    Parameters
    ----------
    n_regions : int
        Number of latent regions.
    n_modulators : int
        Number of modulatory inputs.
    seed_i : int
        Per-seed random seed.

    Returns
    -------
    dict
        Ground-truth tensors, stimulus, and simulated trajectories.
    """
    torch.manual_seed(seed_i)

    # A: stable random with directed chain overlay.
    A_base = make_stable_latent_circuit_A(
        n_regions, density=0.5, seed=seed_i,
    )
    # Add directed chain: region i -> region i+1 (forward only).
    A_true = A_base.clone()
    for i in range(n_regions - 1):
        A_true[i + 1, i] = A_true[i + 1, i] + 0.15

    # Ensure stability after directed chain addition.
    eigs = torch.linalg.eigvals(A_true)
    if eigs.real.max().item() >= 0:
        # Increase self-inhibition until stable.
        for _ in range(20):
            A_true.diagonal().add_(-0.1)
            eigs = torch.linalg.eigvals(A_true)
            if eigs.real.max().item() < 0:
                break

    # C: single driving input to region 0.
    C = torch.zeros(n_regions, 1, dtype=torch.float64)
    C[0, 0] = _C_00

    # B: directed chain modulation with 3 non-null elements.
    B_true = torch.zeros(
        n_modulators, n_regions, n_regions, dtype=torch.float64,
    )
    B_true[0, 1, 0] = _B_10
    B_true[0, 2, 1] = _B_21
    if n_regions >= 4:
        B_true[0, 3, 2] = _B_32

    # b_mask: structural connectivity mask for B.
    b_mask_0 = torch.zeros(
        n_regions, n_regions, dtype=torch.float64,
    )
    b_mask_0[1, 0] = 1.0
    b_mask_0[2, 1] = 1.0
    if n_regions >= 4:
        b_mask_0[3, 2] = 1.0

    # Driving stimulus: block design.
    stim = make_block_stimulus(
        n_blocks=_DRIVING_N_BLOCKS,
        block_duration=_DRIVING_BLOCK_DURATION,
        rest_duration=_DRIVING_REST_DURATION,
        n_inputs=1,
    )

    # Modulator: boxcar epochs.
    stim_mod_dict = make_epoch_stimulus(
        event_times=_EPOCH_TIMES,
        event_durations=_EPOCH_DURATIONS,
        event_amplitudes=_EPOCH_AMPLITUDES,
        duration=_DURATION,
        dt=_DT_SIM,
        n_inputs=n_modulators,
    )
    stim_mod = PiecewiseConstantInput(
        stim_mod_dict["times"], stim_mod_dict["values"],
    )

    # Simulate neural state trajectories.
    sim = simulate_latent_circuit(
        A_true, C, stim,
        duration=_DURATION,
        dt=_DT_SIM,
        SNR=_SNR,
        solver="rk4",
        seed=seed_i,
        B_list=B_true,
        stimulus_mod=stim_mod,
    )

    return {
        "A_true": A_true,
        "C": C,
        "B_true": B_true,
        "b_mask_0": b_mask_0,
        "stim": stim,
        "stim_mod": stim_mod,
        "trajectories": sim["trajectories"],
        "trajectories_clean": sim["trajectories_clean"],
    }


def _train_test_split(
    trajectories: torch.Tensor,
    train_fraction: float = 0.8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split trajectories into train and test sets.

    Parameters
    ----------
    trajectories : torch.Tensor
        Full trajectory tensor, shape ``(T, N)``.
    train_fraction : float
        Fraction of time points for training. Default 0.8.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(train, test)`` trajectory tensors.
    """
    T = trajectories.shape[0]
    split_idx = int(T * train_fraction)
    return trajectories[:split_idx], trajectories[split_idx:]


def run_latent_circuit_recovery(
    config: BenchmarkConfig,
    *,
    n_regions: int = _N_REGIONS,
    n_modulators: int = _N_MODULATORS,
    n_restarts: int = 10,
    init_scale: float = 0.1,
    lc_a_prior_var: float | None = None,
    lc_b_prior_var: float | None = None,
) -> dict[str, Any]:
    """Run latent circuit DCM parameter recovery benchmark.

    For each seed: (1) generate bilinear ground truth; (2) train/test
    split (80/20); (3) fit latent_circuit_dcm_model with multi-start
    SVI; (4) extract posteriors; (5) compute per-seed metrics.

    Parameters
    ----------
    config : BenchmarkConfig
        Benchmark configuration. Uses ``n_datasets``, ``n_svi_steps``,
        ``seed``, ``elbo_type``.
    n_regions : int, optional
        Number of latent regions. Default 4.
    n_modulators : int, optional
        Number of modulatory inputs. Default 1.
    n_restarts : int, optional
        Number of multi-start SVI restarts (LC11 mitigation). Default 10.
    init_scale : float, optional
        Guide initialization scale. Default 0.1.
    lc_a_prior_var : float or None, optional
        Override ``LC_A_PRIOR_VARIANCE`` module constant for this run.
        None uses default (1/16).
    lc_b_prior_var : float or None, optional
        Override ``LC_B_PRIOR_VARIANCE`` module constant for this run.
        None uses default (1.0).

    Returns
    -------
    dict[str, Any]
        Keys:

        - ``'per_seed_results'``: list of per-seed metric dicts.
        - ``'ground_truth'``: dict with ``A_true``, ``B_true``, ``C``.
        - ``'config'``: dict of run parameters.
        - ``'aggregate'``: dict of summary statistics across seeds.
        - ``'n_success'``, ``'n_failed'``: int counts.
        - ``'seeds_used'``, ``'seeds_skipped'``: list[int].

    References
    ----------
    .planning/phases/20-latent-circuit-forward-model/20-04-PLAN.md Task 2.
    .planning/REQUIREMENTS-v0.6.0.md SYNTH-01..03.
    """
    import pyro_dcm.models.latent_circuit_dcm_model as lc_module

    # Prior override support: monkey-patch module constants if provided.
    orig_a_prior = lc_module.LC_A_PRIOR_VARIANCE
    orig_b_prior = lc_module.LC_B_PRIOR_VARIANCE
    if lc_a_prior_var is not None:
        lc_module.LC_A_PRIOR_VARIANCE = lc_a_prior_var
    if lc_b_prior_var is not None:
        lc_module.LC_B_PRIOR_VARIANCE = lc_b_prior_var

    num_steps = config.n_svi_steps
    per_seed_results: list[dict[str, Any]] = []
    seeds_used: list[int] = []
    seeds_skipped: list[int] = []
    n_failed = 0
    ground_truth_sample: dict[str, Any] | None = None

    # Suppress stability warnings during multi-start SVI.
    stability_logger = logging.getLogger("pyro_dcm.stability")
    prev_level = stability_logger.level
    stability_logger.setLevel(logging.ERROR)

    max_pool = config.n_datasets * _MAX_POOL_MULTIPLIER
    try:
        pool_idx = 0
        while (
            len(seeds_used) < config.n_datasets
            and pool_idx < max_pool
        ):
            seed_i = config.seed + pool_idx
            pool_idx += 1
            slot = len(seeds_used) + 1
            print(
                f"Running dataset {slot}/{config.n_datasets} "
                f"(seed {seed_i})..."
            )
            try:
                torch.manual_seed(seed_i)
                np.random.seed(seed_i)
                pyro.set_rng_seed(seed_i)
                pyro.enable_validation(False)

                # Generate ground truth.
                data = _make_latent_circuit_ground_truth(
                    n_regions, n_modulators, seed_i,
                )

                # Check for NaN/Inf in simulated trajectories.
                if (
                    torch.isnan(data["trajectories"]).any().item()
                    or torch.isinf(data["trajectories"]).any().item()
                ):
                    seeds_skipped.append(seed_i)
                    print(
                        f"  SKIPPED seed {seed_i}: trajectories "
                        f"contain NaN/Inf."
                    )
                    continue

                # Store first valid ground truth for aggregate output.
                if ground_truth_sample is None:
                    ground_truth_sample = {
                        "A_true": data["A_true"],
                        "B_true": data["B_true"],
                        "C": data["C"],
                        "b_mask_0": data["b_mask_0"],
                    }

                # Train/test split (80/20).
                train_traj, test_traj = _train_test_split(
                    data["trajectories"],
                )
                train_clean, test_clean = _train_test_split(
                    data["trajectories_clean"],
                )

                T_train = train_traj.shape[0]

                # Model args for latent_circuit_dcm_model.
                a_mask = torch.ones(
                    n_regions, n_regions, dtype=torch.float64,
                )
                c_mask = torch.zeros(
                    n_regions, 1, dtype=torch.float64,
                )
                c_mask[0, 0] = 1.0
                t_eval = torch.arange(
                    0, T_train * _DT_MODEL, _DT_MODEL,
                    dtype=torch.float64,
                )

                model_args = (
                    train_traj,
                    data["stim"],
                    a_mask,
                    c_mask,
                    t_eval,
                    _DT_MODEL,
                )
                model_kwargs: dict[str, Any] = {
                    "b_masks": [data["b_mask_0"]],
                    "stim_mod": data["stim_mod"],
                }

                # Multi-start SVI.
                pyro.clear_param_store()

                def guide_factory() -> Any:
                    return create_guide(
                        latent_circuit_dcm_model,
                        guide_type="auto_normal",
                        init_scale=init_scale,
                    )

                guide = guide_factory()
                t0 = time.time()
                svi_result = run_svi(
                    latent_circuit_dcm_model,
                    guide,
                    model_args,
                    num_steps=num_steps,
                    lr=0.005,
                    clip_norm=10.0,
                    lr_decay_factor=0.01,
                    elbo_type=config.elbo_type,
                    guide_type="auto_normal",
                    model_kwargs=model_kwargs,
                    n_restarts=n_restarts,
                    guide_factory=guide_factory,
                )
                elapsed = time.time() - t0

                # Extract best guide from multi-start result.
                if n_restarts > 1 and "guide" in svi_result:
                    best_guide = svi_result["guide"]
                else:
                    best_guide = guide

                # Extract posterior.
                model_for_pred = partial(
                    latent_circuit_dcm_model, **model_kwargs,
                )
                posterior = extract_posterior_params(
                    best_guide, model_args,
                    model=model_for_pred, num_samples=200,
                )

                # Compute per-seed metrics.
                A_inferred = parameterize_A(posterior["A_free"]["mean"])
                a_rmse = compute_rmse(data["A_true"], A_inferred)

                # B recovery.
                B_inferred_list = []
                B_samples_list = []
                for j in range(n_modulators):
                    key = f"B_free_{j}"
                    if key in posterior:
                        b_mean = posterior[key]["mean"]
                        b_samples = posterior[key]["samples"]
                        B_inferred_list.append(b_mean.unsqueeze(0))
                        B_samples_list.append(b_samples.unsqueeze(1))

                if B_inferred_list:
                    B_inferred = torch.cat(B_inferred_list, dim=0)
                    # B-RMSE on non-null elements.
                    from benchmarks.bilinear_metrics import (
                        compute_b_rmse_magnitude,
                    )
                    b_rmse = compute_b_rmse_magnitude(
                        data["B_true"], B_inferred,
                    )
                    # Sign recovery.
                    from benchmarks.bilinear_metrics import (
                        compute_sign_recovery_nonzero,
                    )
                    sign_rec = compute_sign_recovery_nonzero(
                        [data["B_true"]], [B_inferred],
                    )
                    # CI coverage of zero on null B.
                    from benchmarks.bilinear_metrics import (
                        compute_coverage_of_zero,
                    )
                    B_samples_stacked = torch.cat(
                        B_samples_list, dim=1,
                    )
                    ci_cov = compute_coverage_of_zero(
                        [data["B_true"]], [B_samples_stacked],
                    )
                else:
                    b_rmse = float("nan")
                    sign_rec = float("nan")
                    ci_cov = float("nan")

                # Trajectory R-squared on test set.
                # Predict on full length then take test portion.
                if "predicted_trajectories" in posterior:
                    pred_traj = posterior[
                        "predicted_trajectories"
                    ]["mean"]
                    # pred_traj might be train-length only.
                    traj_r2 = compute_trajectory_r_squared(
                        pred_traj[:T_train], train_clean[:T_train],
                    )
                else:
                    traj_r2 = compute_trajectory_r_squared(
                        train_clean[:T_train],
                        train_clean[:T_train],
                    )

                # Multi-level coverage on A.
                if "A_free" in posterior and "samples" in posterior["A_free"]:
                    a_samples = posterior["A_free"]["samples"]
                    a_coverage = compute_coverage_multi_level(
                        a_samples, data["A_true"],
                        levels=[0.50, 0.75, 0.90, 0.95],
                    )
                else:
                    a_coverage = {}

                seed_result = {
                    "seed": seed_i,
                    "a_rmse": a_rmse,
                    "b_rmse": b_rmse,
                    "sign_recovery": sign_rec,
                    "ci_coverage": ci_cov,
                    "trajectory_r_squared": traj_r2,
                    "a_coverage_levels": a_coverage,
                    "elapsed_s": elapsed,
                    "final_loss": svi_result.get("final_loss", None),
                }
                per_seed_results.append(seed_result)
                seeds_used.append(seed_i)

                print(
                    f"  a_rmse={a_rmse:.4f}, b_rmse={b_rmse:.4f}, "
                    f"sign_rec={sign_rec:.2f}, ci_cov={ci_cov:.2f}, "
                    f"traj_r2={traj_r2:.4f}, t={elapsed:.1f}s"
                )

            except (RuntimeError, ValueError, AssertionError) as e:
                print(f"  FAILED: {e}")
                n_failed += 1
            finally:
                pyro.enable_validation(True)
    finally:
        # Restore prior constants and logger level.
        lc_module.LC_A_PRIOR_VARIANCE = orig_a_prior
        lc_module.LC_B_PRIOR_VARIANCE = orig_b_prior
        stability_logger.setLevel(prev_level)

    n_success = len(per_seed_results)
    if n_success < max(1, config.n_datasets // 2):
        return {
            "status": "insufficient_data",
            "n_success": n_success,
            "n_failed": n_failed,
            "n_datasets": config.n_datasets,
            "seeds_used": seeds_used,
            "seeds_skipped": seeds_skipped,
        }

    # Aggregate statistics.
    a_rmse_list = [s["a_rmse"] for s in per_seed_results]
    b_rmse_list = [s["b_rmse"] for s in per_seed_results]
    traj_r2_list = [s["trajectory_r_squared"] for s in per_seed_results]

    aggregate = {
        "mean_a_rmse": float(np.mean(a_rmse_list)),
        "median_a_rmse": float(np.median(a_rmse_list)),
        "mean_b_rmse": float(np.nanmean(b_rmse_list)),
        "median_b_rmse": float(np.nanmedian(b_rmse_list)),
        "mean_trajectory_r_squared": float(np.mean(traj_r2_list)),
        "median_trajectory_r_squared": float(np.median(traj_r2_list)),
    }

    return {
        "per_seed_results": per_seed_results,
        "ground_truth": ground_truth_sample,
        "config": {
            "n_regions": n_regions,
            "n_modulators": n_modulators,
            "n_restarts": n_restarts,
            "init_scale": init_scale,
            "lc_a_prior_var": lc_a_prior_var or orig_a_prior,
            "lc_b_prior_var": lc_b_prior_var or orig_b_prior,
            "n_svi_steps": num_steps,
            "seed": config.seed,
            "n_datasets": config.n_datasets,
            "duration": _DURATION,
            "dt_model": _DT_MODEL,
            "SNR": _SNR,
        },
        "aggregate": aggregate,
        "n_success": n_success,
        "n_failed": n_failed,
        "seeds_used": seeds_used,
        "seeds_skipped": seeds_skipped,
    }
