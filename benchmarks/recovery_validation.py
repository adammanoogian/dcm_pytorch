"""Comprehensive parameter recovery validation for spectral DCM.

Tests both VL and SVI across controlled conditions:
- Multiple region counts (N = 3, 5, 10)
- Multiple SNR levels (5, 10, 20)
- Multiple ground-truth connectivity patterns (sparse, dense, hub)
- Reports RMSE, correlation, coverage, convergence, wall time

This is the "known everything going in" validation — ground truth A
matrices are generated with controlled structure, CSD is simulated
from the exact generative model, and noise is added at known levels.

Usage:
    python benchmarks/recovery_validation.py
    python benchmarks/recovery_validation.py --quick  # 3 seeds, N<=5

Output:
    benchmarks/results/recovery_validation.csv
    benchmarks/results/recovery_summary.txt
"""

from __future__ import annotations

import argparse
import csv
import time
import traceback
from pathlib import Path

import torch
import numpy as np

from pyro_dcm.forward_models.neural_state import parameterize_A
from pyro_dcm.models.spectral_dcm_model import decompose_csd_for_likelihood
from pyro_dcm.simulators.spectral_simulator import (
    make_stable_A_spectral,
    simulate_spectral_dcm,
)
from pyro_dcm.inference.variational_laplace import (
    run_variational_laplace,
    extract_vl_posterior,
)


def _pearson_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    xd = x - x.mean()
    yd = y - y.mean()
    num = (xd * yd).sum()
    denom = (xd.pow(2).sum() * yd.pow(2).sum()).sqrt()
    if denom < 1e-15:
        return 0.0
    return (num / denom).item()


def make_structured_A(
    N: int,
    pattern: str = "random",
    density: float = 0.3,
    seed: int = 0,
) -> torch.Tensor:
    """Generate a stable A matrix with controlled structure.

    Parameters
    ----------
    N : int
        Number of regions.
    pattern : str
        One of 'random', 'sparse', 'hub', 'chain'.
    density : float
        Fraction of non-zero off-diagonal connections.
    seed : int
        Random seed.

    Returns
    -------
    torch.Tensor
        Stable (N, N) A matrix with negative diagonal.
    """
    rng = np.random.RandomState(seed)

    if pattern == "random":
        return make_stable_A_spectral(N, seed=seed)

    A_free = torch.zeros(N, N, dtype=torch.float64)

    if pattern == "sparse":
        n_off = N * (N - 1)
        n_active = max(1, int(density * n_off))
        indices = rng.choice(n_off, size=n_active, replace=False)
        off_diag_mask = ~torch.eye(N, dtype=torch.bool)
        flat_off = A_free[off_diag_mask]
        values = torch.tensor(
            rng.uniform(-0.3, 0.3, size=n_active), dtype=torch.float64
        )
        flat_off[indices] = values
        A_free[off_diag_mask] = flat_off

    elif pattern == "hub":
        hub = 0
        for j in range(1, N):
            A_free[j, hub] = rng.uniform(0.05, 0.3)
            A_free[hub, j] = rng.uniform(-0.1, 0.1)

    elif pattern == "chain":
        for i in range(N - 1):
            A_free[i + 1, i] = rng.uniform(0.05, 0.3)
            A_free[i, i + 1] = rng.uniform(-0.05, 0.1)

    return parameterize_A(A_free)


def make_noisy_csd(
    sim: dict,
    snr: float,
    seed: int,
) -> torch.Tensor:
    """Add Gaussian noise to simulated CSD."""
    obs_real = decompose_csd_for_likelihood(sim["csd"])
    signal_power = obs_real.pow(2).mean().sqrt()
    noise_std = signal_power / snr
    torch.manual_seed(seed + 1000)
    noisy_obs = obs_real + noise_std * torch.randn_like(obs_real)
    F, n, _ = sim["csd"].shape
    half = F * n * n
    return torch.complex(
        noisy_obs[:half].reshape(F, n, n),
        noisy_obs[half:].reshape(F, n, n),
    )


def run_recovery_trial(
    N: int,
    pattern: str,
    snr: float,
    seed: int,
    method: str = "VL",
    max_iter: int = 64,
) -> dict:
    """Run one recovery trial and return metrics."""
    A_true = make_structured_A(N, pattern=pattern, seed=seed)
    sim = simulate_spectral_dcm(A_true, TR=2.0, n_freqs=32, seed=seed)
    noisy_csd = make_noisy_csd(sim, snr=snr, seed=seed)
    a_mask = torch.ones(N, N, dtype=torch.float64)

    t0 = time.perf_counter()

    if method == "VL":
        result = run_variational_laplace(
            noisy_csd, sim["freqs"], a_mask,
            N=N, max_iter=max_iter, tolerance=0.1,
        )
        A_post = result.theta_post["A"]
        converged = result.converged
        n_iters = result.n_iterations

        posterior = extract_vl_posterior(result, N, num_samples=500)
        A_free_samples = posterior["A_free"]["samples"]
        A_free_lo = A_free_samples.quantile(0.025, dim=0)
        A_free_hi = A_free_samples.quantile(0.975, dim=0)

        diag_mask = torch.eye(N, dtype=torch.bool)
        A_lo = A_free_lo.clone()
        A_hi = A_free_hi.clone()
        A_lo[diag_mask] = -torch.exp(A_free_hi[diag_mask]) / 2.0
        A_hi[diag_mask] = -torch.exp(A_free_lo[diag_mask]) / 2.0

        covered = ((A_true >= A_lo) & (A_true <= A_hi)).float().mean().item()
    else:
        raise ValueError(f"Unknown method: {method}")

    wall_time = time.perf_counter() - t0

    rmse = torch.sqrt(torch.mean((A_true - A_post) ** 2)).item()
    corr = _pearson_corr(A_true.reshape(-1), A_post.reshape(-1))
    max_err = (A_true - A_post).abs().max().item()

    off_mask = ~torch.eye(N, dtype=torch.bool)
    rmse_off = torch.sqrt(
        torch.mean((A_true[off_mask] - A_post[off_mask]) ** 2)
    ).item()

    return {
        "n_regions": N,
        "pattern": pattern,
        "snr": snr,
        "seed": seed,
        "method": method,
        "wall_time": round(wall_time, 2),
        "rmse": round(rmse, 4),
        "rmse_offdiag": round(rmse_off, 4),
        "max_error": round(max_err, 4),
        "correlation": round(corr, 4),
        "coverage_95": round(covered, 3),
        "converged": converged,
        "n_iterations": n_iters,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Parameter recovery validation"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: fewer seeds, smaller N",
    )
    args = parser.parse_args()

    if args.quick:
        region_counts = [3, 5]
        snr_levels = [10.0]
        patterns = ["random", "sparse"]
        seeds = [100, 101, 102]
    else:
        region_counts = [3, 5, 10]
        snr_levels = [5.0, 10.0, 20.0]
        patterns = ["random", "sparse", "hub", "chain"]
        seeds = [100, 101, 102, 103, 104]

    results_dir = Path("benchmarks/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "recovery_validation.csv"

    fieldnames = [
        "n_regions", "pattern", "snr", "seed", "method",
        "wall_time", "rmse", "rmse_offdiag", "max_error",
        "correlation", "coverage_95", "converged", "n_iterations",
    ]

    total = len(region_counts) * len(snr_levels) * len(patterns) * len(seeds)
    done = 0

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for N in region_counts:
            for snr in snr_levels:
                for pattern in patterns:
                    for seed in seeds:
                        done += 1
                        label = f"[{done}/{total}] N={N} {pattern} SNR={snr} s={seed}"
                        print(f"{label}...", end=" ", flush=True)
                        try:
                            row = run_recovery_trial(
                                N, pattern, snr, seed, method="VL",
                            )
                            writer.writerow(row)
                            f.flush()
                            print(
                                f"RMSE={row['rmse']} r={row['correlation']} "
                                f"cov={row['coverage_95']} t={row['wall_time']}s"
                            )
                        except Exception as e:
                            print(f"FAILED: {e}")
                            traceback.print_exc()

    print(f"\nResults saved to {csv_path}")

    import pandas as pd

    df = pd.read_csv(csv_path)
    summary = df.groupby(["n_regions", "pattern", "snr"]).agg(
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),
        corr_mean=("correlation", "mean"),
        coverage_mean=("coverage_95", "mean"),
        wall_time_mean=("wall_time", "mean"),
        n_converged=("converged", "sum"),
        n_trials=("converged", "count"),
    ).round(3)

    summary_path = results_dir / "recovery_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Spectral DCM Parameter Recovery Validation\n")
        f.write("=" * 50 + "\n\n")
        f.write(summary.to_string())
        f.write("\n\n--- Aggregate by N ---\n")
        agg_n = df.groupby("n_regions").agg(
            rmse_mean=("rmse", "mean"),
            corr_mean=("correlation", "mean"),
            wall_time_mean=("wall_time", "mean"),
        ).round(3)
        f.write(agg_n.to_string())
    print(f"Summary saved to {summary_path}")
    print("\n" + summary.to_string())


if __name__ == "__main__":
    main()
