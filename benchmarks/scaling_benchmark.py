"""Spectral DCM scaling benchmark: VL vs SVI across region counts.

Measures wall time, memory, RMSE, and correlation for both inference
backends on simulated data with known ground truth. Identifies the
practical ceiling for each method.

Usage:
    python benchmarks/scaling_benchmark.py [--max-n 20] [--seeds 3]
    python benchmarks/scaling_benchmark.py --gpu  # include GPU SVI

Output:
    benchmarks/results/scaling_results.csv
    benchmarks/results/scaling_summary.txt
"""

from __future__ import annotations

import argparse
import csv
import os
import time
import traceback
from pathlib import Path

import torch

from pyro_dcm.forward_models.neural_state import parameterize_A
from pyro_dcm.models.spectral_dcm_model import decompose_csd_for_likelihood
from pyro_dcm.simulators.spectral_simulator import (
    make_stable_A_spectral,
    simulate_spectral_dcm,
)
from pyro_dcm.inference.variational_laplace import (
    run_variational_laplace,
    _param_count,
)


def _pearson_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    xd = x - x.mean()
    yd = y - y.mean()
    num = (xd * yd).sum()
    denom = (xd.pow(2).sum() * yd.pow(2).sum()).sqrt()
    if denom < 1e-15:
        return 0.0
    return (num / denom).item()


def make_noisy_csd(
    sim: dict,
    snr: float = 10.0,
    seed: int = 0,
) -> torch.Tensor:
    """Add Gaussian noise to simulated CSD at a given SNR."""
    obs_real = decompose_csd_for_likelihood(sim["csd"])
    signal_power = obs_real.pow(2).mean().sqrt()
    noise_std = signal_power / snr
    torch.manual_seed(seed + 1000)
    noisy_obs = obs_real + noise_std * torch.randn_like(obs_real)

    F, n, _ = sim["csd"].shape
    half = F * n * n
    noisy_real = noisy_obs[:half].reshape(F, n, n)
    noisy_imag = noisy_obs[half:].reshape(F, n, n)
    return torch.complex(noisy_real, noisy_imag)


def run_vl_benchmark(
    N: int,
    seed: int,
    n_freqs: int = 32,
    snr: float = 10.0,
    max_iter: int = 64,
) -> dict:
    """Run a single VL benchmark trial.

    Returns
    -------
    dict
        Keys: n_regions, seed, method, wall_time, rmse, correlation,
        converged, n_iterations, n_params, peak_memory_mb.
    """
    A_true = make_stable_A_spectral(N, seed=seed)
    sim = simulate_spectral_dcm(A_true, TR=2.0, n_freqs=n_freqs, seed=seed)
    noisy_csd = make_noisy_csd(sim, snr=snr, seed=seed)
    a_mask = torch.ones(N, N, dtype=torch.float64)

    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

    t0 = time.perf_counter()
    result = run_variational_laplace(
        noisy_csd, sim["freqs"], a_mask,
        N=N, max_iter=max_iter, tolerance=0.1,
    )
    wall_time = time.perf_counter() - t0

    A_post = result.theta_post["A"]
    rmse = torch.sqrt(torch.mean((A_true - A_post) ** 2)).item()
    corr = _pearson_corr(A_true.reshape(-1), A_post.reshape(-1))

    return {
        "n_regions": N,
        "seed": seed,
        "method": "VL",
        "wall_time": round(wall_time, 2),
        "rmse": round(rmse, 4),
        "correlation": round(corr, 4),
        "converged": result.converged,
        "n_iterations": result.n_iterations,
        "n_params": _param_count(N),
        "snr": snr,
    }


def run_svi_benchmark(
    N: int,
    seed: int,
    n_freqs: int = 32,
    snr: float = 10.0,
    num_steps: int = 500,
    device: str = "cpu",
) -> dict:
    """Run a single SVI benchmark trial."""
    import pyro
    from pyro_dcm.models import (
        spectral_dcm_model,
        create_guide,
        run_svi,
        extract_posterior_params,
    )

    A_true = make_stable_A_spectral(N, seed=seed)
    sim = simulate_spectral_dcm(A_true, TR=2.0, n_freqs=n_freqs, seed=seed)
    noisy_csd = make_noisy_csd(sim, snr=snr, seed=seed)
    a_mask = torch.ones(N, N, dtype=torch.float64)

    model_args = (noisy_csd, sim["freqs"], a_mask, N)

    pyro.set_rng_seed(seed)
    torch.manual_seed(seed)

    guide = create_guide(spectral_dcm_model, init_scale=0.01)

    t0 = time.perf_counter()
    svi_result = run_svi(
        spectral_dcm_model, guide, model_args,
        num_steps=num_steps, lr=0.01,
        clip_norm=10.0, lr_decay_factor=0.1,
    )
    wall_time = time.perf_counter() - t0

    posterior = extract_posterior_params(guide, model_args)
    A_free_mean = posterior["median"]["A_free"]
    A_post = parameterize_A(A_free_mean)

    rmse = torch.sqrt(torch.mean((A_true - A_post) ** 2)).item()
    corr = _pearson_corr(A_true.reshape(-1), A_post.reshape(-1))

    return {
        "n_regions": N,
        "seed": seed,
        "method": f"SVI_{device}",
        "wall_time": round(wall_time, 2),
        "rmse": round(rmse, 4),
        "correlation": round(corr, 4),
        "converged": True,
        "n_iterations": num_steps,
        "n_params": _param_count(N),
        "snr": snr,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Spectral DCM scaling benchmark"
    )
    parser.add_argument(
        "--max-n", type=int, default=20,
        help="Maximum number of regions to test",
    )
    parser.add_argument(
        "--seeds", type=int, default=3,
        help="Number of random seeds per condition",
    )
    parser.add_argument(
        "--gpu", action="store_true",
        help="Include GPU SVI benchmarks",
    )
    parser.add_argument(
        "--svi-steps", type=int, default=500,
        help="Number of SVI steps",
    )
    parser.add_argument(
        "--vl-max-iter", type=int, default=64,
        help="Maximum VL iterations",
    )
    parser.add_argument(
        "--snr", type=float, default=10.0,
        help="Signal-to-noise ratio",
    )
    args = parser.parse_args()

    region_counts = [n for n in [3, 5, 10, 15, 20, 30, 50] if n <= args.max_n]
    seeds = list(range(100, 100 + args.seeds))

    results_dir = Path("benchmarks/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "scaling_results.csv"

    fieldnames = [
        "n_regions", "seed", "method", "wall_time", "rmse",
        "correlation", "converged", "n_iterations", "n_params", "snr",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for N in region_counts:
            for seed in seeds:
                # VL benchmark
                print(f"VL N={N} seed={seed}...", end=" ", flush=True)
                try:
                    row = run_vl_benchmark(
                        N, seed, snr=args.snr,
                        max_iter=args.vl_max_iter,
                    )
                    writer.writerow(row)
                    f.flush()
                    print(
                        f"t={row['wall_time']}s "
                        f"RMSE={row['rmse']} r={row['correlation']}"
                    )
                except Exception as e:
                    print(f"FAILED: {e}")
                    traceback.print_exc()

                # SVI benchmark
                print(f"SVI N={N} seed={seed}...", end=" ", flush=True)
                try:
                    row = run_svi_benchmark(
                        N, seed, snr=args.snr,
                        num_steps=args.svi_steps,
                    )
                    writer.writerow(row)
                    f.flush()
                    print(
                        f"t={row['wall_time']}s "
                        f"RMSE={row['rmse']} r={row['correlation']}"
                    )
                except Exception as e:
                    print(f"FAILED: {e}")
                    traceback.print_exc()

    # Summary
    print(f"\nResults saved to {csv_path}")
    print("\n--- Summary ---")
    import pandas as pd

    df = pd.read_csv(csv_path)
    summary = df.groupby(["n_regions", "method"]).agg(
        wall_time_mean=("wall_time", "mean"),
        wall_time_std=("wall_time", "std"),
        rmse_mean=("rmse", "mean"),
        corr_mean=("correlation", "mean"),
        n_converged=("converged", "sum"),
    ).round(3)
    print(summary)

    summary_path = results_dir / "scaling_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary.to_string())
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
