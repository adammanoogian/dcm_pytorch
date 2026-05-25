"""Latent-circuit DCM 10-seed acceptance run on M3 cluster.

Runs 10-seed recovery with calibrated priors and full acceptance gates.
Each SLURM array task runs one seed.

References
----------
.planning/phases/20-latent-circuit-forward-model/20-05-PLAN.md -- Task 3
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.config import BenchmarkConfig
from benchmarks.runners.latent_circuit_recovery import run_latent_circuit_recovery

N_SVI_STEPS = 1000
N_RESTARTS = 10
INIT_SCALE = 0.1
BASE_SEED = 42


def main() -> None:
    """Run single-seed acceptance test from SLURM_ARRAY_TASK_ID."""
    task_id_str = os.environ.get("SLURM_ARRAY_TASK_ID")
    job_id = os.environ.get("SLURM_JOB_ID", "local")

    if task_id_str is not None:
        seed_offset = int(task_id_str)
    else:
        seed_offset = 0
        print("WARNING: SLURM_ARRAY_TASK_ID not set, using seed_offset=0")

    seed = BASE_SEED + seed_offset
    print(f"Acceptance run: seed={seed} (offset={seed_offset})")
    print(
        f"Config: steps={N_SVI_STEPS}, restarts={N_RESTARTS}, "
        f"init_scale={INIT_SCALE}"
    )

    config = BenchmarkConfig(
        variant="latent_circuit",
        method="svi",
        n_datasets=1,
        n_svi_steps=N_SVI_STEPS,
        n_regions=4,
        seed=seed,
    )

    output_dir = Path("cluster/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    try:
        result = run_latent_circuit_recovery(
            config,
            n_regions=4,
            n_modulators=1,
            n_restarts=N_RESTARTS,
            init_scale=INIT_SCALE,
        )
        seed_result = result["per_seed_results"][0]
        entry = {
            "seed": seed,
            "seed_offset": seed_offset,
            "a_rmse": seed_result["a_rmse"],
            "b_rmse": seed_result["b_rmse"],
            "sign_recovery": seed_result["sign_recovery"],
            "ci_coverage_95": seed_result.get("ci_coverage_95"),
            "trajectory_r_squared": seed_result["trajectory_r_squared"],
            "shrinkage_A": seed_result.get("shrinkage_A"),
            "shrinkage_B": seed_result.get("shrinkage_B"),
            "final_elbo": seed_result.get("final_elbo"),
            "status": "ok",
        }
        print(
            f"  A-RMSE={entry['a_rmse']:.4f} "
            f"B-RMSE={entry['b_rmse']:.4f} "
            f"sign={entry['sign_recovery']:.3f} "
            f"R2={entry['trajectory_r_squared']:.4f}"
        )
    except Exception as e:
        entry = {
            "seed": seed,
            "seed_offset": seed_offset,
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
        print(f"  ERROR: {e}")

    elapsed = time.time() - t0
    entry["elapsed_s"] = round(elapsed, 1)

    output_path = output_dir / f"lc_acceptance_{job_id}_{seed_offset}.json"
    with open(output_path, "w") as f:
        json.dump(entry, f, indent=2)
    print(f"\nResult saved to: {output_path} ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
