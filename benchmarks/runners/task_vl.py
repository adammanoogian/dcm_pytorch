"""Task DCM Variational Laplace benchmark runner (Plan 29-04, VLINFRA-02).

Implements the simulate -> infer -> measure loop for task-based DCM with
per-dataset Variational Laplace inference. Generates synthetic BOLD data with
known connectivity, fits via ``run_variational_laplace_generic`` using
``TaskDCMForward``, and computes recovery metrics.

This is the VL counterpart of ``benchmarks.runners.task_svi.run_task_svi``. It
follows the same ``(BenchmarkConfig) -> dict`` runner contract and reuses the
v0.2.0 metrics suite (``benchmarks.metrics``) without modification.

Critical configuration: the VL forward observation precision is a dense
``(T*N, T*N)`` matrix inverted in the ReML M-step, so this runner uses a coarse
model grid (``dt = 0.1`` integration step, BOLD sampled at ``TR``) and a short
duration so ``T*N`` stays well under the ``TaskDCMForward.build_precision`` cap
of 5000 (VLROBUST-02, the [29-03-D1] precision guard never trips).

References
----------
.planning/phases/29-vl-validation-infra-bmr-rank/29-04-PLAN.md
    Runner contract (VLINFRA-02), dt>=0.1 floor.
benchmarks/runners/task_svi.py
    Inline ground-truth generation (``make_block_stimulus`` /
    ``simulate_task_dcm`` current call signatures) mirrored here.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from benchmarks.config import BenchmarkConfig
from benchmarks.metrics import (
    compute_coverage_from_samples,
    compute_rmse,
    compute_summary_stats,
    pearson_corr,
)
from pyro_dcm.forward_models.neural_state import parameterize_A
from pyro_dcm.inference import (
    TaskDCMForward,
    extract_vl_posterior_generic,
    run_variational_laplace_generic,
)
from pyro_dcm.simulators.task_simulator import (
    make_block_stimulus,
    make_random_stable_A,
    simulate_task_dcm,
)

# Coarse VL model grid: keeps the dense (T*N, T*N) precision tractable.
_TR: float = 2.0
_DT_MODEL: float = 0.1


def run_task_vl(config: BenchmarkConfig) -> dict[str, Any]:
    """Run task DCM Variational Laplace benchmark.

    For each dataset: generate synthetic BOLD with known ``A`` and ``C``, fit via
    VL with ``TaskDCMForward`` at ``dt = 0.1`` (BOLD sampled at ``TR``), extract
    the Laplace posterior, and compute per-dataset A-RMSE, coverage, correlation,
    and convergence.

    Parameters
    ----------
    config : BenchmarkConfig
        Benchmark configuration. Uses ``n_datasets``, ``n_regions``, ``seed``,
        ``quick``, and ``max_iter`` (VL Gauss-Newton cap; defaults to the engine
        value of 128 when ``None``). ``quick`` shortens the duration so the
        precision matrix stays small.

    Returns
    -------
    dict
        Results dict with keys ``variant`` (``"task"``), ``method`` (``"vl"``),
        ``n_regions``, per-dataset lists (``rmse_list``, ``coverage_list``,
        ``correlation_list``, ``converged_list``, ``n_iterations_list``,
        ``time_list``, ``a_true_list``, ``a_inferred_list``), ``n_success``,
        ``n_failed``, flattened summary stats, a nested ``summary`` block, and a
        ``metadata`` block. Dataset failures append an ``"error"`` record and are
        skipped (mirrors ``run_task_svi`` robustness).
    """
    N = config.n_regions
    M = 1  # single driving input
    max_iter = config.max_iter if config.max_iter is not None else 128
    # Short duration keeps T*N = (duration / TR) * N well under the 5000 cap
    # AND keeps the finite-difference Jacobian (one ODE integration per param
    # per Gauss-Newton iteration) laptop-fast for the smoke test.
    duration = 40.0 if config.quick else 120.0
    n_blocks = 2 if config.quick else 3

    rmse_list: list[float] = []
    coverage_list: list[float] = []
    correlation_list: list[float] = []
    converged_list: list[bool] = []
    n_iterations_list: list[int] = []
    time_list: list[float] = []
    a_true_list: list[list[float]] = []
    a_inferred_list: list[list[float]] = []
    errors: list[dict[str, Any]] = []
    n_failed = 0

    for i in range(config.n_datasets):
        seed_i = config.seed + i
        print(f"Running dataset {i + 1}/{config.n_datasets} (seed {seed_i})...")

        try:
            torch.manual_seed(seed_i)
            np.random.seed(seed_i)

            # Ground truth (mirror run_task_svi inline branch signatures).
            A_true = make_random_stable_A(N, density=0.5, seed=seed_i)
            C_true = torch.zeros(N, M, dtype=torch.float64)
            C_true[0, 0] = 1.0
            stim = make_block_stimulus(
                n_blocks=n_blocks,
                block_duration=15.0,
                rest_duration=15.0,
                n_inputs=M,
            )
            sim = simulate_task_dcm(
                A_true, C_true, stim,
                duration=duration, dt=0.01, TR=_TR, SNR=5.0, seed=seed_i,
            )
            bold = sim["bold"].to(torch.float64)  # (T_TR, N)

            a_mask = torch.ones(N, N, dtype=torch.float64)
            c_mask = torch.zeros(N, M, dtype=torch.float64)
            c_mask[0, 0] = 1.0

            # t_eval at BOLD (TR) resolution -> predicted output rows match the
            # observed BOLD rows; dt=0.1 is the internal RK4 integration step.
            t_eval = torch.arange(
                0.0, bold.shape[0] * _TR, _TR, dtype=torch.float64,
            )[: bold.shape[0]]

            forward = TaskDCMForward(
                stimulus_fn=sim["stimulus"],
                c_mask=c_mask,
                t_eval=t_eval,
                dt=_DT_MODEL,
            )
            t0 = time.time()
            result = run_variational_laplace_generic(
                forward,
                observed=bold,
                a_mask=a_mask,
                n_regions=N,
                max_iter=max_iter,
                prior_variance=1.0 / 64.0,
                context={"a_mask": a_mask},
            )
            elapsed = time.time() - t0

            posterior = extract_vl_posterior_generic(result, forward, N)
            A_free_mean = posterior["A_free"]["mean"].to(torch.float64)
            A_inferred = parameterize_A(A_free_mean * a_mask)

            A_free_samples = posterior["A_free"]["samples"].to(torch.float64)
            A_param_samples = torch.stack(
                [parameterize_A(s * a_mask) for s in A_free_samples],
            )
            rmse = compute_rmse(A_true.to(torch.float64), A_inferred)
            coverage = compute_coverage_from_samples(
                A_true.to(torch.float64), A_param_samples, ci_level=0.95,
            )
            corr = pearson_corr(
                A_true.to(torch.float64).flatten(), A_inferred.flatten(),
            )

            rmse_list.append(rmse)
            coverage_list.append(coverage)
            correlation_list.append(corr)
            converged_list.append(bool(result.converged))
            n_iterations_list.append(int(result.n_iterations))
            time_list.append(elapsed)
            a_true_list.append(A_true.to(torch.float64).flatten().tolist())
            a_inferred_list.append(A_inferred.flatten().tolist())

            print(
                f"  RMSE={rmse:.4f}, coverage={coverage:.3f}, corr={corr:.3f}, "
                f"iters={result.n_iterations}, converged={result.converged}, "
                f"time={elapsed:.1f}s"
            )

        except (RuntimeError, ValueError) as e:
            print(f"  FAILED: {e}")
            n_failed += 1
            errors.append({"seed": seed_i, "error": str(e)})

    n_success = len(rmse_list)
    summary: dict[str, Any] = {}
    if rmse_list:
        summary = {
            "rmse_stats": compute_summary_stats(rmse_list),
            "coverage_stats": compute_summary_stats(coverage_list),
            "correlation_stats": compute_summary_stats(correlation_list),
            "convergence_rate": float(np.mean(converged_list)),
        }

    return {
        "variant": "task",
        "method": "vl",
        "n_regions": N,
        "rmse_list": rmse_list,
        "coverage_list": coverage_list,
        "correlation_list": correlation_list,
        "converged_list": converged_list,
        "n_iterations_list": n_iterations_list,
        "time_list": time_list,
        "a_true_list": a_true_list,
        "a_inferred_list": a_inferred_list,
        "n_success": n_success,
        "n_failed": n_failed,
        "errors": errors,
        "summary": summary,
        "metadata": {
            "variant": "task",
            "method": "vl",
            "n_regions": N,
            "n_blocks": n_blocks,
            "duration": duration,
            "dt_model": _DT_MODEL,
            "TR": _TR,
            "max_iter": max_iter,
            "quick": config.quick,
        },
    }
