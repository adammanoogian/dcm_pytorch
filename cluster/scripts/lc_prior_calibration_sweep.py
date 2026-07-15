"""Latent circuit prior calibration sweep on M3 cluster (array job).

Sweeps joint (LC_A_PRIOR_VARIANCE, LC_B_PRIOR_VARIANCE, init_scale) grid
on single seed (42). Each SLURM array task runs one combo.

Uses _duration_override=10.0 (10s instead of 100s) for calibration --
relative ranking is preserved at shorter durations while making ODE
integration ~10x faster per SVI step.

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
import traceback
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import CLUSTER_RESULTS_DIR  # noqa: E402

from benchmarks.config import BenchmarkConfig
from benchmarks.runners.latent_circuit_recovery import run_latent_circuit_recovery

A_PRIOR_VARS = [1 / 64, 1 / 16, 1 / 4, 1.0]
B_PRIOR_VARS = [0.25, 1.0, 4.0]
INIT_SCALES = [0.01, 0.05, 0.1, 0.5]

GRID = list(product(A_PRIOR_VARS, B_PRIOR_VARS, INIT_SCALES))

N_SVI_STEPS = 300
N_RESTARTS = 2
N_DATASETS = 1
SEED = 42
DURATION_OVERRIDE = 30.0


def run_single_combo(task_id: int) -> dict:
    """Run one grid combination.

    Parameters
    ----------
    task_id : int
        0-based index into GRID.

    Returns
    -------
    dict
        Result entry with metrics or error status.
    """
    a_var, b_var, init_scale = GRID[task_id]
    print(
        f"[{task_id + 1}/{len(GRID)}] a_var={a_var:.4f}, "
        f"b_var={b_var:.2f}, init={init_scale:.3f}",
        flush=True,
    )

    config = BenchmarkConfig(
        variant="latent_circuit",
        method="svi",
        n_datasets=N_DATASETS,
        n_svi_steps=N_SVI_STEPS,
        n_regions=4,
        seed=SEED,
    )

    t0 = time.time()
    try:
        result = run_latent_circuit_recovery(
            config,
            n_regions=4,
            n_modulators=1,
            n_restarts=N_RESTARTS,
            init_scale=init_scale,
            lc_a_prior_var=a_var,
            lc_b_prior_var=b_var,
            _duration_override=DURATION_OVERRIDE,
        )
        seed_result = result["per_seed_results"][0]
        entry = {
            "task_id": task_id,
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
            "task_id": task_id,
            "a_prior_var": a_var,
            "b_prior_var": b_var,
            "init_scale": init_scale,
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }

    elapsed = time.time() - t0
    entry["elapsed_s"] = round(elapsed, 1)

    if entry["status"] == "ok":
        print(
            f"  B-RMSE={entry['b_rmse']:.4f} "
            f"A-RMSE={entry['a_rmse']:.4f} "
            f"R2={entry['trajectory_r_squared']:.3f} "
            f"({elapsed:.0f}s)"
        )
    else:
        print(f"  ERROR: {entry.get('error', 'unknown')} ({elapsed:.0f}s)")
        print(f"  Traceback:\n{entry.get('traceback', 'N/A')}")

    return entry


def main() -> None:
    """Run single combo from SLURM_ARRAY_TASK_ID, or all combos locally."""
    task_id_str = os.environ.get("SLURM_ARRAY_TASK_ID")
    job_id = os.environ.get("SLURM_JOB_ID", "local")

    output_dir = CLUSTER_RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if task_id_str is not None:
        task_id = int(task_id_str)
        print(f"SLURM array task {task_id} of {len(GRID)}")
        print(
            f"Config: steps={N_SVI_STEPS}, restarts={N_RESTARTS}, "
            f"seed={SEED}, duration={DURATION_OVERRIDE}s"
        )
        entry = run_single_combo(task_id)
        output_path = output_dir / f"lc_calib_{job_id}_{task_id}.json"
        with open(output_path, "w") as f:
            json.dump(entry, f, indent=2)
        print(f"\nResult saved to: {output_path}")
    else:
        print(f"Running all {len(GRID)} combinations locally")
        print(
            f"Config: steps={N_SVI_STEPS}, restarts={N_RESTARTS}, "
            f"seed={SEED}, duration={DURATION_OVERRIDE}s"
        )
        results = []
        for i in range(len(GRID)):
            entry = run_single_combo(i)
            results.append(entry)
        output_path = output_dir / "lc_calibration_sweep_local.json"
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
                        "duration_override": DURATION_OVERRIDE,
                    },
                    "results": results,
                    "n_total": len(results),
                    "n_ok": sum(1 for r in results if r["status"] == "ok"),
                },
                f,
                indent=2,
            )
        _print_summary(results)
        print(f"\nResults saved to: {output_path}")


def _print_summary(results: list[dict]) -> None:
    """Print top-10 table sorted by B-RMSE.

    Parameters
    ----------
    results : list[dict]
        Sweep results.
    """
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

    valid = [
        r
        for r in ok_results
        if r.get("shrinkage_B") is None or r["shrinkage_B"] < 0.95
    ]
    if valid:
        winner = valid[0]
        print(
            f"\n  WINNER: a_var={winner['a_prior_var']:.4f}, "
            f"b_var={winner['b_prior_var']:.2f}, "
            f"init_scale={winner['init_scale']:.3f}"
        )
        print(
            f"  B-RMSE={winner['b_rmse']:.4f}, "
            f"A-RMSE={winner['a_rmse']:.4f}, "
            f"R2={winner['trajectory_r_squared']:.3f}"
        )
    else:
        print("\n  WARNING: No valid winner (all shrinkage_B > 0.95)")

    print("=" * 80)


if __name__ == "__main__":
    main()
