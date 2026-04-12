"""Task DCM SVI benchmark runner.

Implements the simulate -> infer -> measure loop for task-based DCM
with per-subject SVI inference. Generates synthetic BOLD data with
known connectivity, runs Pyro SVI, and computes recovery metrics
using the consolidated benchmarks.metrics functions.

Reuses patterns from tests/test_task_dcm_recovery.py.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pyro
import torch

from benchmarks.config import BenchmarkConfig
from benchmarks.fixtures import load_fixture
from benchmarks.metrics import (
    compute_coverage_from_ci,
    compute_coverage_multi_level,
    compute_rmse,
    compute_summary_stats,
    pearson_corr,
)
from pyro_dcm.forward_models.neural_state import parameterize_A
from pyro_dcm.models import (
    create_guide,
    extract_posterior_params,
    run_svi,
    task_dcm_model,
)
from pyro_dcm.simulators.task_simulator import (
    make_block_stimulus,
    make_random_stable_A,
    simulate_task_dcm,
)
from pyro_dcm.utils.ode_integrator import PiecewiseConstantInput


def _build_A_ci(
    A_free_lo: torch.Tensor,
    A_free_hi: torch.Tensor,
    N: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert A_free quantile bounds to parameterized A bounds.

    For off-diagonal: identity transform, bounds pass through.
    For diagonal: a_ii = -exp(A_free_ii)/2 is monotone decreasing,
    so lower/upper swap.

    Parameters
    ----------
    A_free_lo : torch.Tensor
        Lower bound (2.5th percentile) of A_free, shape ``(N, N)``.
    A_free_hi : torch.Tensor
        Upper bound (97.5th percentile) of A_free, shape ``(N, N)``.
    N : int
        Number of regions.

    Returns
    -------
    tuple of torch.Tensor
        (A_lo, A_hi) each shape ``(N, N)`` in parameterized A space.
    """
    diag_mask = torch.eye(N, dtype=torch.bool)
    A_lo = A_free_lo.clone()
    A_hi = A_free_hi.clone()
    A_lo[diag_mask] = -torch.exp(A_free_hi[diag_mask]) / 2.0
    A_hi[diag_mask] = -torch.exp(A_free_lo[diag_mask]) / 2.0
    return A_lo, A_hi


def run_task_svi(config: BenchmarkConfig) -> dict[str, Any]:
    """Run task DCM SVI benchmark.

    For each dataset: generate synthetic BOLD with known A, run SVI
    inference, extract posterior, compute RMSE/coverage/correlation.

    Parameters
    ----------
    config : BenchmarkConfig
        Benchmark configuration with n_datasets, n_svi_steps, seed,
        n_regions, and quick flag.

    Returns
    -------
    dict
        Results dict with keys: rmse_list, coverage_list,
        correlation_list, elbo_list, time_list, n_steps_list,
        summary (mean/std of each metric), metadata.
    """
    N = config.n_regions
    M = 1  # single input
    n_blocks = 3 if config.quick else 5
    duration = 30.0 if config.quick else 90.0
    num_steps = config.n_svi_steps

    rmse_list: list[float] = []
    coverage_list: list[float] = []
    correlation_list: list[float] = []
    elbo_list: list[float] = []
    time_list: list[float] = []
    n_steps_list: list[int] = []
    a_true_list: list[list[float]] = []
    a_inferred_list: list[list[float]] = []
    coverage_multi: dict[float, list[float]] = {
        lv: [] for lv in [0.50, 0.75, 0.90, 0.95]
    }
    coverage_diag_multi: dict[float, list[float]] = {
        lv: [] for lv in [0.50, 0.75, 0.90, 0.95]
    }
    coverage_offdiag_multi: dict[float, list[float]] = {
        lv: [] for lv in [0.50, 0.75, 0.90, 0.95]
    }
    n_failed = 0

    for i in range(config.n_datasets):
        seed_i = config.seed + i
        print(f"Running dataset {i + 1}/{config.n_datasets}...")

        try:
            # Fix all seeds
            torch.manual_seed(seed_i)
            np.random.seed(seed_i)
            pyro.set_rng_seed(seed_i)
            pyro.enable_validation(False)
            pyro.clear_param_store()

            if config.fixtures_dir is not None:
                data = load_fixture(
                    "task", N, i, config.fixtures_dir,
                )
                A_true = data["A_true"]
                C = data["C"]
                bold = data["bold"]
                stim = PiecewiseConstantInput(
                    data["stimulus_times"],
                    data["stimulus_values"],
                )
                sim = {"bold": bold, "stimulus": stim}
                # Override duration from fixture metadata
                duration = float(data["duration"])
            else:
                # Generate ground truth
                A_true = make_random_stable_A(
                    N, density=0.5, seed=seed_i,
                )
                C = torch.zeros(N, M, dtype=torch.float64)
                C[0, 0] = 1.0

                stim = make_block_stimulus(
                    n_blocks=n_blocks,
                    block_duration=15.0,
                    rest_duration=15.0,
                    n_inputs=M,
                )

                # Simulate BOLD
                sim = simulate_task_dcm(
                    A_true, C, stim,
                    duration=duration, dt=0.01, TR=2.0,
                    SNR=5.0, seed=seed_i,
                )

            # Model args
            a_mask = torch.ones(N, N, dtype=torch.float64)
            c_mask = torch.zeros(N, M, dtype=torch.float64)
            c_mask[0, 0] = 1.0
            dt_model = 0.5
            t_eval = torch.arange(
                0, duration, dt_model, dtype=torch.float64,
            )
            TR = 2.0
            model_args = (
                sim["bold"], sim["stimulus"],
                a_mask, c_mask, t_eval, TR, dt_model,
            )

            # SVI
            guide = create_guide(
                task_dcm_model,
                init_scale=0.01,
                guide_type=config.guide_type,
                n_regions=N,
            )
            t0 = time.time()
            svi_result = run_svi(
                task_dcm_model, guide, model_args,
                num_steps=num_steps, lr=0.005,
                clip_norm=10.0, lr_decay_factor=0.01,
                elbo_type=config.elbo_type,
                guide_type=config.guide_type,
            )
            elapsed = time.time() - t0

            # Use post-Laplace guide if available
            extract_guide = svi_result.get("guide", guide)

            # Posterior via Predictive sampling
            posterior = extract_posterior_params(
                extract_guide, model_args,
            )
            A_free_mean = posterior["A_free"]["mean"]
            A_inferred = parameterize_A(A_free_mean)

            # 95% CI via sample-based quantiles
            A_free_samples = posterior["A_free"]["samples"]
            A_free_lo = torch.quantile(
                A_free_samples.float(), 0.025, dim=0,
            )
            A_free_hi = torch.quantile(
                A_free_samples.float(), 0.975, dim=0,
            )
            A_lo, A_hi = _build_A_ci(A_free_lo, A_free_hi, N)

            # Metrics
            rmse = compute_rmse(A_true, A_inferred)
            coverage = compute_coverage_from_ci(A_true, A_lo, A_hi)
            corr = pearson_corr(
                A_true.flatten(), A_inferred.flatten(),
            )

            # Multi-level coverage via parameterized samples
            A_param_samples = torch.stack(
                [parameterize_A(s) for s in A_free_samples],
            )
            ml_all = compute_coverage_multi_level(
                A_true.flatten(),
                A_param_samples.reshape(
                    A_param_samples.shape[0], -1,
                ),
            )
            diag_mask_ml = torch.eye(N, dtype=torch.bool)
            offdiag_mask_ml = ~diag_mask_ml
            ml_diag = compute_coverage_multi_level(
                A_true[diag_mask_ml],
                A_param_samples[:, diag_mask_ml],
            )
            ml_offdiag = compute_coverage_multi_level(
                A_true[offdiag_mask_ml],
                A_param_samples[:, offdiag_mask_ml],
            )
            for lv in coverage_multi:
                coverage_multi[lv].append(ml_all[lv])
                coverage_diag_multi[lv].append(ml_diag[lv])
                coverage_offdiag_multi[lv].append(
                    ml_offdiag[lv],
                )

            rmse_list.append(rmse)
            coverage_list.append(coverage)
            correlation_list.append(corr)
            elbo_list.append(svi_result["final_loss"])
            time_list.append(elapsed)
            n_steps_list.append(num_steps)
            a_true_list.append(A_true.flatten().tolist())
            a_inferred_list.append(
                A_inferred.flatten().tolist(),
            )

            print(
                f"  RMSE={rmse:.4f}, coverage={coverage:.3f}, "
                f"corr={corr:.3f}, time={elapsed:.1f}s"
            )

        except (RuntimeError, ValueError, AssertionError) as e:
            print(f"  FAILED: {e}")
            n_failed += 1
        finally:
            pyro.enable_validation(True)

    # Require >= 50% success
    n_success = len(rmse_list)
    if n_success < max(1, config.n_datasets // 2):
        return {
            "status": "insufficient_data",
            "n_success": n_success,
            "n_failed": n_failed,
            "n_datasets": config.n_datasets,
        }

    # Summary statistics (backward-compatible + median/IQR)
    summary: dict[str, Any] = {
        "mean_rmse": float(np.mean(rmse_list)),
        "std_rmse": float(np.std(rmse_list)),
        "mean_coverage": float(np.mean(coverage_list)),
        "std_coverage": float(np.std(coverage_list)),
        "mean_correlation": float(np.mean(correlation_list)),
        "std_correlation": float(np.std(correlation_list)),
        "mean_time": float(np.mean(time_list)),
        "mean_elbo": float(np.mean(elbo_list)),
        "rmse_stats": compute_summary_stats(rmse_list),
        "coverage_stats": compute_summary_stats(coverage_list),
        "correlation_stats": compute_summary_stats(
            correlation_list,
        ),
        "time_stats": compute_summary_stats(time_list),
    }

    return {
        "rmse_list": rmse_list,
        "coverage_list": coverage_list,
        "correlation_list": correlation_list,
        "elbo_list": elbo_list,
        "time_list": time_list,
        "n_steps_list": n_steps_list,
        "a_true_list": a_true_list,
        "a_inferred_list": a_inferred_list,
        "coverage_multi": {
            str(k): v for k, v in coverage_multi.items()
        },
        "coverage_diag_multi": {
            str(k): v for k, v in coverage_diag_multi.items()
        },
        "coverage_offdiag_multi": {
            str(k): v
            for k, v in coverage_offdiag_multi.items()
        },
        "n_success": n_success,
        "n_failed": n_failed,
        **summary,
        "metadata": {
            "variant": "task",
            "method": "svi",
            "n_regions": N,
            "n_blocks": n_blocks,
            "duration": duration,
            "num_steps": num_steps,
            "quick": config.quick,
        },
    }
