"""Spectral DCM SVI benchmark runner.

Implements the simulate -> infer -> measure loop for spectral DCM
with per-subject SVI inference. Generates synthetic CSD data with
known connectivity, adds noise at specified SNR, runs Pyro SVI,
and computes recovery metrics.

Reuses patterns from tests/test_spectral_dcm_recovery.py.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pyro
import torch

from benchmarks.config import BenchmarkConfig
from benchmarks.metrics import (
    compute_coverage_from_ci,
    compute_rmse,
    pearson_corr,
)
from pyro_dcm.forward_models.neural_state import parameterize_A
from pyro_dcm.models import (
    create_guide,
    extract_posterior_params,
    run_svi,
    spectral_dcm_model,
)
from pyro_dcm.models.spectral_dcm_model import decompose_csd_for_likelihood
from pyro_dcm.simulators.spectral_simulator import (
    make_stable_A_spectral,
    simulate_spectral_dcm,
)


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


def run_spectral_svi(config: BenchmarkConfig) -> dict[str, Any]:
    """Run spectral DCM SVI benchmark.

    For each dataset: generate synthetic CSD with known A, add SNR=10
    noise, run SVI inference, extract posterior, compute metrics.

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
    # Spectral DCM converges fast; use 500 steps regardless of config
    num_steps = 500
    snr = 10.0

    rmse_list: list[float] = []
    coverage_list: list[float] = []
    correlation_list: list[float] = []
    elbo_list: list[float] = []
    time_list: list[float] = []
    n_steps_list: list[int] = []
    n_failed = 0

    for i in range(config.n_datasets):
        seed_i = config.seed + i
        print(f"Running dataset {i + 1}/{config.n_datasets}...")

        try:
            # Fix all seeds
            torch.manual_seed(seed_i)
            np.random.seed(seed_i)
            pyro.set_rng_seed(seed_i)
            pyro.clear_param_store()

            # Generate ground truth A
            A_true = make_stable_A_spectral(N, seed=seed_i)

            # Simulate CSD
            sim = simulate_spectral_dcm(
                A_true, TR=2.0, n_freqs=32, seed=seed_i,
            )

            # Add noise in decomposed real/imag space
            obs_real = decompose_csd_for_likelihood(sim["csd"])
            signal_power = obs_real.pow(2).mean().sqrt()
            noise_std = signal_power / snr
            torch.manual_seed(seed_i + 1000)
            noisy_obs = obs_real + noise_std * torch.randn_like(
                obs_real,
            )

            # Reconstruct noisy complex CSD
            F, n, _ = sim["csd"].shape
            half = F * n * n
            noisy_real = noisy_obs[:half].reshape(F, n, n)
            noisy_imag = noisy_obs[half:].reshape(F, n, n)
            noisy_csd = torch.complex(noisy_real, noisy_imag)

            # Model args
            a_mask = torch.ones(N, N, dtype=torch.float64)
            model_args = (noisy_csd, sim["freqs"], a_mask, N)

            # SVI
            guide = create_guide(
                spectral_dcm_model, init_scale=0.01,
            )
            t0 = time.time()
            svi_result = run_svi(
                spectral_dcm_model, guide, model_args,
                num_steps=num_steps, lr=0.01,
                clip_norm=10.0, lr_decay_factor=0.1,
            )
            elapsed = time.time() - t0

            # Posterior
            posterior = extract_posterior_params(guide, model_args)
            A_free_median = posterior["median"]["A_free"]
            A_inferred = parameterize_A(A_free_median)

            # 95% CI via quantiles
            quantiles = guide.quantiles(
                [0.025, 0.975], *model_args,
            )
            A_free_lo = quantiles["A_free"][0]
            A_free_hi = quantiles["A_free"][1]
            A_lo, A_hi = _build_A_ci(A_free_lo, A_free_hi, N)

            # Metrics
            rmse = compute_rmse(A_true, A_inferred)
            coverage = compute_coverage_from_ci(A_true, A_lo, A_hi)
            corr = pearson_corr(
                A_true.flatten(), A_inferred.flatten(),
            )

            rmse_list.append(rmse)
            coverage_list.append(coverage)
            correlation_list.append(corr)
            elbo_list.append(svi_result["final_loss"])
            time_list.append(elapsed)
            n_steps_list.append(num_steps)

            print(
                f"  RMSE={rmse:.4f}, coverage={coverage:.3f}, "
                f"corr={corr:.3f}, time={elapsed:.1f}s"
            )

        except (RuntimeError, ValueError) as e:
            print(f"  FAILED: {e}")
            n_failed += 1

    # Require >= 50% success
    n_success = len(rmse_list)
    if n_success < max(1, config.n_datasets // 2):
        return {
            "status": "insufficient_data",
            "n_success": n_success,
            "n_failed": n_failed,
            "n_datasets": config.n_datasets,
        }

    # Summary statistics
    summary: dict[str, Any] = {
        "mean_rmse": float(np.mean(rmse_list)),
        "std_rmse": float(np.std(rmse_list)),
        "mean_coverage": float(np.mean(coverage_list)),
        "std_coverage": float(np.std(coverage_list)),
        "mean_correlation": float(np.mean(correlation_list)),
        "std_correlation": float(np.std(correlation_list)),
        "mean_time": float(np.mean(time_list)),
        "mean_elbo": float(np.mean(elbo_list)),
    }

    return {
        "rmse_list": rmse_list,
        "coverage_list": coverage_list,
        "correlation_list": correlation_list,
        "elbo_list": elbo_list,
        "time_list": time_list,
        "n_steps_list": n_steps_list,
        "n_success": n_success,
        "n_failed": n_failed,
        **summary,
        "metadata": {
            "variant": "spectral",
            "method": "svi",
            "n_regions": N,
            "n_freqs": 32,
            "snr": snr,
            "num_steps": num_steps,
            "quick": config.quick,
        },
    }
