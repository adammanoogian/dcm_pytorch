"""Latent circuit prior calibration sweep on M3 cluster.

Sweeps joint (LC_A_PRIOR_VARIANCE, LC_B_PRIOR_VARIANCE, init_scale) grid
on single seed (42) with 1000 SVI steps, 3 restarts. Identifies optimal
combination that minimizes B-RMSE (Phase 16.1 lesson: this metric is most
sensitive to prior/init interaction).

Selection criteria:
  Primary: lowest B-RMSE
  Secondary: lowest A-RMSE
  Tertiary: trajectory R-squared > 0.90
  Disqualifying: shrinkage_B > 0.95 (posterior collapsed to prior)
"""
from __future__ import annotations

import json
import os
import sys
import time
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.config import BenchmarkConfig
from benchmarks.runners.latent_circuit_recovery import run_latent_circuit_recovery

A_PRIOR_VARS = [1 / 64, 1 / 16, 1 / 4, 1.0]
B_PRIOR_VARS = [0.25, 1.0, 4.0]
INIT_SCALES = [0.01, 0.05, 0.1, 0.5]

N_SVI_STEPS = 1000
N_RESTARTS = 3
N_DATASETS = 1
SEED = 42


def run_sweep() -> list[dict]:
    """Run calibration sweep over all grid combinations."""
    grid = list(product(A_PRIOR_VARS, B_PRIOR_VARS, INIT_SCALES))
    print(f"Calibration sweep: {len(grid)} combinations")
    print(f"  A_prior_vars: {A_PRIOR_VARS}")
    print(f"  B_prior_vars: {B_PRIOR_VARS}")
    print(f"  init_scales: {INIT_SCALES}")
    print(f"  steps={N_SVI_STEPS}, restarts={N_RESTARTS}, seed={SEED}")
    print()

    results = []
    config = BenchmarkConfig(n_datasets=N_DATASETS, n_svi_steps=N_SVI_STEPS)

    for i, (a_var, b_var, init_scale) in enumerate(grid):
        t0 = time.time()
        print(
            f"[{i + 1}/{len(grid)}] a_var={a_var:.4f}, "
            f"b_var={b_var:.2f}, init={init_scale:.3f}",
            end=" ... ",
            flush=True,
        )

        try:
            result = run_latent_circuit_recovery(
                config,
                n_regions=4,
                n_modulators=1,
                n_restarts=N_RESTARTS,
                init_scale=init_scale,
                lc_a_prior_var=a_var,
                lc_b_prior_var=b_var,
            )
            seed_result = result["per_seed_results"][0]
            entry = {
                "a_prior_var": a_var,
                "b_prior_var": b_var,
                "init_scale": init_scale,
                "a_rmse": seed_result["a_rmse"],
                "b_rmse": seed_result["b_rmse"],
                "sign_recovery": seed_result["sign_recovery"],
                "ci_coverage_95": seed_result.get("ci_coverage_95", None),
                "trajectory_r_squared": seed_result["trajectory_r_squared"],
                "shrinkage_A": seed_result.get("shrinkage_A", None),
                "shrinkage_B": seed_result.get("shrinkage_B", None),
                "final_elbo": seed_result.get("final_elbo", None),
                "status": "ok",
            }
        except Exception as e:
            entry = {
                "a_prior_var": a_var,
                "b_prior_var": b_var,
                "init_scale": init_scale,
                "status": "error",
                "error": str(e),
            }

        elapsed = time.time() - t0
        entry["elapsed_s"] = round(elapsed, 1)
        results.append(entry)

        if entry["status"] == "ok":
            print(
                f"B-RMSE={entry['b_rmse']:.4f} "
                f"A-RMSE={entry['a_rmse']:.4f} "
                f"R2={entry['trajectory_r_squared']:.3f} "
                f"({elapsed:.0f}s)"
            )
        else:
            print(f"ERROR: {entry.get('error', 'unknown')} ({elapsed:.0f}s)")

    return results


def print_summary(results: list[dict]) -> None:
    """Print top-10 table sorted by B-RMSE."""
    ok_results = [r for r in results if r["status"] == "ok"]
    ok_results.sort(key=lambda r: r["b_rmse"])

    print("\n" + "=" * 80)
    print("  CALIBRATION SWEEP RESULTS (top 10 by B-RMSE)")
    print("=" * 80)
    print(
        f"{'Rank':<5} {'A_var':<8} {'B_var':<7} {'Init':<6} "
        f"{'B-RMSE':<9} {'A-RMSE':<9} {'R2':<7} {'Shrink_B':<10} {'Sign':<6}"
    )
    print("-" * 80)

    for i, r in enumerate(ok_results[:10]):
        shrink_b = r.get("shrinkage_B")
        shrink_str = f"{shrink_b:.3f}" if shrink_b is not None else "N/A"
        sign_str = (
            f"{r['sign_recovery']:.2f}" if r.get("sign_recovery") else "N/A"
        )
        print(
            f"{i + 1:<5} {r['a_prior_var']:<8.4f} {r['b_prior_var']:<7.2f} "
            f"{r['init_scale']:<6.3f} {r['b_rmse']:<9.4f} "
            f"{r['a_rmse']:<9.4f} {r['trajectory_r_squared']:<7.3f} "
            f"{shrink_str:<10} {sign_str:<6}"
        )

    print("-" * 80)

    # Select winner
    valid = [
        r
        for r in ok_results
        if r.get("shrinkage_B") is None or r["shrinkage_B"] < 0.95
    ]
    if valid:
        winner = valid[0]
        print(f"\n  WINNER: a_var={winner['a_prior_var']:.4f}, "
              f"b_var={winner['b_prior_var']:.2f}, "
              f"init_scale={winner['init_scale']:.3f}")
        print(f"  B-RMSE={winner['b_rmse']:.4f}, "
              f"A-RMSE={winner['a_rmse']:.4f}, "
              f"R2={winner['trajectory_r_squared']:.3f}")
    else:
        print("\n  WARNING: No valid winner (all shrinkage_B > 0.95)")

    print("=" * 80)


def main() -> None:
    """Run calibration sweep and save results."""
    results = run_sweep()
    print_summary(results)

    # Save results
    output_dir = Path("cluster/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    job_id = os.environ.get("SLURM_JOB_ID", "local")
    output_path = output_dir / f"lc_calibration_sweep_{job_id}.json"

    with open(output_path, "w") as f:
        json.dump(
            {
                "grid": {
                    "a_prior_vars": A_PRIOR_VARS,
                    "b_prior_vars": B_PRIOR_VARS,
                    "init_scales": INIT_SCALES,
                },
                "config": {
                    "n_svi_steps": N_SVI_STEPS,
                    "n_restarts": N_RESTARTS,
                    "seed": SEED,
                },
                "results": results,
                "n_total": len(results),
                "n_ok": sum(1 for r in results if r["status"] == "ok"),
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
