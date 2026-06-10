"""Spectral DCM Variational Laplace benchmark runner (Plan 29-04, VLINFRA-02).

Implements the simulate -> infer -> measure loop for spectral DCM with
per-dataset Variational Laplace inference. Generates synthetic CSD data with
known connectivity, fits via ``run_variational_laplace_generic`` using
``SpectralDCMForward``, and computes recovery metrics.

This is the VL counterpart of ``benchmarks.runners.spectral_svi.run_spectral_svi``.
It follows the same ``(BenchmarkConfig) -> dict`` runner contract and reuses the
v0.2.0 metrics suite (``benchmarks.metrics``) without modification so the figure
pipeline and ``recovery_validation.py`` consume it unchanged.

References
----------
.planning/phases/29-vl-validation-infra-bmr-rank/29-04-PLAN.md
    Runner contract (VLINFRA-02), VL fit pattern.
cluster/scripts/lc_vl_acceptance_run.py
    The proven ``run_variational_laplace_generic`` + ``extract_vl_posterior_generic``
    usage mirrored here.
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
    SpectralDCMForward,
    extract_vl_posterior_generic,
    run_variational_laplace_generic,
)
from pyro_dcm.simulators.spectral_simulator import (
    make_stable_A_spectral,
    simulate_spectral_dcm,
)


def run_spectral_vl(config: BenchmarkConfig) -> dict[str, Any]:
    """Run spectral DCM Variational Laplace benchmark.

    For each dataset: generate synthetic CSD with known ``A``, fit via VL with
    ``SpectralDCMForward``, extract the Laplace posterior, and compute per-dataset
    A-RMSE, coverage, correlation, and convergence.

    Parameters
    ----------
    config : BenchmarkConfig
        Benchmark configuration. Uses ``n_datasets``, ``n_regions``, ``seed``,
        ``quick``, and ``max_iter`` (VL Gauss-Newton cap; defaults to the engine
        value of 128 when ``None``).

    Returns
    -------
    dict
        Results dict with keys ``variant`` (``"spectral"``), ``method`` (``"vl"``),
        ``n_regions``, per-dataset lists (``rmse_list``, ``coverage_list``,
        ``correlation_list``, ``converged_list``, ``n_iterations_list``,
        ``time_list``, ``a_true_list``, ``a_inferred_list``), ``n_success``,
        ``n_failed``, flattened summary stats, a nested ``summary`` block, and a
        ``metadata`` block. Dataset failures append an ``"error"`` record and are
        skipped (mirrors ``run_spectral_svi`` robustness).
    """
    N = config.n_regions
    max_iter = config.max_iter if config.max_iter is not None else 128

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

            # Ground truth + synthetic CSD (mirror spectral_svi inline branch).
            A_true = make_stable_A_spectral(N, seed=seed_i)
            sim = simulate_spectral_dcm(A_true, TR=2.0, n_freqs=32, seed=seed_i)
            csd_obs = sim["csd"].to(torch.complex128)
            freqs = sim["freqs"].to(torch.float64)
            a_mask = torch.ones(N, N, dtype=torch.float64)

            # Variational Laplace fit.
            forward = SpectralDCMForward()
            t0 = time.time()
            result = run_variational_laplace_generic(
                forward,
                observed=csd_obs,
                a_mask=a_mask,
                n_regions=N,
                max_iter=max_iter,
                prior_variance=1.0 / 64.0,
                context={"freqs": freqs},
            )
            elapsed = time.time() - t0

            posterior = extract_vl_posterior_generic(result, forward, N)
            A_free_mean = posterior["A_free"]["mean"].to(torch.float64)
            A_inferred = parameterize_A(A_free_mean * a_mask)

            # Coverage on parameterized A samples (off-diag identity transform).
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
        "variant": "spectral",
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
            "variant": "spectral",
            "method": "vl",
            "n_regions": N,
            "n_freqs": 32,
            "max_iter": max_iter,
            "quick": config.quick,
        },
    }
