"""Env-driven single-cell entrypoint for the Phase 30 recovery-matrix sweep.

Each SLURM array task runs ONE ``(variant, N, SNR)`` cell of the validation grid
over ``GRID_SEEDS`` seeds and writes one per-cell JSON. Mirrors the structure of
``cluster/scripts/lc_vl_acceptance_run.py``: ``sys.path`` insertion for the repo
root + ``src``, ``SLURM_ARRAY_TASK_ID`` -> ``cell_for_index``, env-overridable
knobs, a try/except that records any failure as a ``status="error"`` entry so a
single bad cell never aborts the array, and a per-cell JSON written to
``cluster/results/``.

The actual fit logic lives in ``benchmarks.recovery_matrix_grid.run_one_cell``
(which reuses the Phase 29 VL runners); this script is only the SLURM glue.

Environment variables
---------------------
SLURM_ARRAY_TASK_ID : int, default 0
    Cell index into ``enumerate_cells()`` (local default 0).
SLURM_JOB_ID : str, default "local"
    Job id used in the output filename.
RECOVERY_MAX_ITER : int, default 64
    VL Gauss-Newton iteration cap.
RECOVERY_BASE_SEED : int, default 42
    Base seed (seed ``i`` = base + i).
RECOVERY_QUICK : {"0", "1"}, default "0"
    "1" shortens per-variant duration for the local faithfulness pre-check.

References
----------
benchmarks.recovery_matrix_grid.run_one_cell
    The per-cell driver (reuses the Phase 29 VL fit logic).
cluster/scripts/lc_vl_acceptance_run.py
    The mirrored per-task SLURM-glue structure.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from config import CLUSTER_RESULTS_DIR  # noqa: E402

from benchmarks.recovery_matrix_grid import (  # noqa: E402
    cell_for_index,
    run_one_cell,
)


def main() -> None:
    """Run a single recovery-matrix cell selected by SLURM_ARRAY_TASK_ID."""
    task_id_str = os.environ.get("SLURM_ARRAY_TASK_ID")
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    task_id = int(task_id_str) if task_id_str is not None else 0

    max_iter = int(os.environ.get("RECOVERY_MAX_ITER", "64"))
    base_seed = int(os.environ.get("RECOVERY_BASE_SEED", "42"))
    quick = os.environ.get("RECOVERY_QUICK", "0") == "1"

    cell = cell_for_index(task_id)
    print(
        f"Recovery-matrix cell {task_id}: variant={cell['variant']} "
        f"N={cell['n_regions']} SNR={cell['snr']} "
        f"(max_iter={max_iter}, base_seed={base_seed}, quick={quick})"
    )

    output_dir = CLUSTER_RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    entry: dict[str, Any]
    try:
        result = run_one_cell(
            cell, base_seed=base_seed, max_iter=max_iter, quick=quick,
        )
        entry = {"status": "ok", "task_id": task_id, "job_id": job_id, **result}
        metrics = result["metrics"]
        rmse_a = metrics.get("rmse_a")
        rmse_median = rmse_a["median"] if isinstance(rmse_a, dict) else None
        n_success = metrics.get("n_success")
        sign = metrics.get("sign_recovery_masked")
        rmse_str = f"{rmse_median:.4f}" if rmse_median is not None else "n/a"
        sign_str = f"{sign:.3f}" if isinstance(sign, float) else str(sign)
        print(
            f"  OK variant={cell['variant']} N={cell['n_regions']} "
            f"SNR={cell['snr']} rmse_a_median={rmse_str} "
            f"sign_masked={sign_str} converged={n_success}/{result['n_seeds']}"
        )
    except Exception as e:  # noqa: BLE001 -- record any failure for triage
        entry = {
            "status": "error",
            "task_id": task_id,
            "job_id": job_id,
            **cell,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
        print(f"  ERROR cell {task_id}: {e}")

    elapsed = time.time() - t0
    entry["elapsed_s"] = round(elapsed, 1)
    out_path = output_dir / f"recovery_matrix_{job_id}_{task_id}.json"
    with open(out_path, "w") as f:
        json.dump(entry, f, indent=2)
    print(f"\nResult saved to: {out_path} ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
