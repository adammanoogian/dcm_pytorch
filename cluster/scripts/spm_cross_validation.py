"""VLSPM-03 VL-vs-SPM12 cross-validation on M3 (the licensed-MATLAB run).

Single-task M3 entrypoint (NOT an array). The local MATLAB license server is
unreachable (FlexLM -15), so the actual ``spm_nlsi_GN`` cross-validation runs
here on M3 where MATLAB R2022a + SPM12 are licensed
(``/usr/local/matlab/r2022a/bin/matlab`` on the comp partition; SPM12 at
``$DCM_CLUSTER_ROOT/external/spm12``). It calls
:func:`validation.run_vl_validation.run_vl_spectral_dcm_validation` (which fits
the Phase 28 VL engine and injects the IDENTICAL CSD into SPM via the Plan 32-01
bridge), prints a human summary, and writes the JSON-safe result to
``cluster/results/spm_cross_validation_<jobid>.json``.

Record-don't-crash (mirrors 31-03-D3): a gate MISS -- including a real 5%
matched-F miss -- is a scientific result to RECORD in the JSON + stdout, not a
crash. The script exits 0 on a recorded gate miss and exits non-zero ONLY on an
unexpected exception (e.g. the MATLAB subprocess failing entirely). The strict
5% matched-F check is reported here (``matched_f_comparison.relative_error``,
the headline number); the HARD pass/fail enforcement lives in the laptop-gated
``tests/test_vl_spm_cross_validation.py``.

Environment variables
---------------------
SLURM_JOB_ID : str, default "local"
    Job id used in the output filename.
MATLAB_PATH : str
    MATLAB binary (the sbatch exports ``/usr/local/matlab/r2022a/bin/matlab``).
SPM12_PATH : str
    SPM12 location passed to the MATLAB child (the sbatch exports
    ``$DCM_CLUSTER_ROOT/external/spm12``).
SPM_XVAL_SEED : int, default 42
    Seed for the spectral simulation.
SPM_XVAL_MAX_ITER : int, default 64
    VL Gauss-Newton iteration cap.

References
----------
cluster/scripts/bmr_tempering_calibration.py
    The mirrored SLURM-glue structure (sys.path, env knobs, record-don't-crash).
validation.run_vl_validation.run_vl_spectral_dcm_validation
    The cross-validation orchestrator invoked here.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from config import CLUSTER_RESULTS_DIR  # noqa: E402

from validation.run_vl_validation import (  # noqa: E402
    run_vl_spectral_dcm_validation,
)


def _json_safe(obj: Any) -> Any:
    """Recursively cast numpy arrays / scalars to JSON-serializable values.

    Parameters
    ----------
    obj : Any
        Arbitrary nested structure (dict / list / tuple / numpy / python).

    Returns
    -------
    Any
        The same structure with numpy arrays cast via ``.tolist()`` and numpy
        scalars cast to native ``float`` / ``int``.
    """
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def main() -> int:
    """Run the VL-vs-SPM12 cross-validation and record the result.

    Returns
    -------
    int
        ``0`` on success OR on a recorded gate miss (record-don't-crash);
        ``1`` only on an unexpected exception.
    """
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    seed = int(os.environ.get("SPM_XVAL_SEED", "42"))
    max_iter = int(os.environ.get("SPM_XVAL_MAX_ITER", "64"))

    output_dir = CLUSTER_RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"spm_cross_validation_{job_id}.json"

    note = (
        "VLSPM-03 VL-vs-SPM12 cross-validation on the IDENTICAL injected CSD. "
        "A gate miss (incl. the strict 5% matched-F) is RECORDED, not crashed "
        "(record-don't-crash, mirroring 31-03-D3); exit non-zero only on an "
        "unexpected exception."
    )
    t0 = time.time()

    entry: dict[str, Any]
    exit_code = 0
    try:
        print(
            f"MATLAB_PATH={os.environ.get('MATLAB_PATH', '<unset>')} "
            f"SPM12_PATH={os.environ.get('SPM12_PATH', '<unset>')}"
        )
        print(
            f"Running VL-vs-SPM12 cross-validation "
            f"(seed={seed}, n_regions=2, max_iter={max_iter})..."
        )
        result = run_vl_spectral_dcm_validation(
            seed=seed, n_regions=2, max_iter=max_iter,
        )

        ep = result["ep_comparison"]
        f_cmp = result["matched_f_comparison"]
        ranking = result["ranking"]
        a01, a10 = result["ep_asymmetry"]

        # --- Human summary (the headline numbers) --------------------------
        print("\n=== VL vs SPM12 cross-validation summary ===")
        print(
            f"  Ep free-space within_tolerance (10%): "
            f"{ep['within_tolerance']} "
            f"(max={ep['max_relative_error']:.4f}, "
            f"mean={ep['mean_relative_error']:.4f})"
        )
        print(
            f"  Matched-F relative_error (HEADLINE): "
            f"{f_cmp['relative_error']:.4f} "
            f"within_tolerance (5%): {f_cmp['within_tolerance']} "
            f"[VL F={f_cmp['vl_free_energy']:.4f}, SPM F={f_cmp['spm_F']:.4f}]"
        )
        print(
            f"  Ranking agreement_rate (>=0.80): "
            f"{ranking['agreement_rate']:.4f} "
            f"({ranking['agreements']}/{ranking['total_pairs']} pairs)"
        )
        print(f"  S4 ep_asymmetry: Ep.A[0,1]={a01:.5f} Ep.A[1,0]={a10:.5f}")

        all_gates_pass = bool(
            ep["within_tolerance"]
            and f_cmp["within_tolerance"]
            and ranking["agreement_rate"] >= 0.80
            and a01 != a10
        )
        if all_gates_pass:
            print("  ALL GATES PASS.")
        else:
            # A gate miss is a RECORDED scientific result, not a crash.
            print("  GATE MISS RECORDED (see JSON; exit 0 by design).")

        entry = {
            "status": "ok",
            "job_id": job_id,
            "note": note,
            "seed": seed,
            "max_iter": max_iter,
            "all_gates_pass": all_gates_pass,
            "ep_comparison": _json_safe(ep),
            "matched_f_comparison": _json_safe(f_cmp),
            "matched_f_relative_error": float(f_cmp["relative_error"]),
            "matched_f_within_tolerance": bool(f_cmp["within_tolerance"]),
            "ranking": _json_safe(ranking),
            "ranking_agreement_rate": float(ranking["agreement_rate"]),
            "ep_asymmetry": [float(a01), float(a10)],
            "vl_A_free": _json_safe(result["vl_A_free"]),
            "spm_Ep_A": _json_safe(result["spm_Ep_A"]),
            "vl_F": float(result["vl_F"]),
            "spm_F": float(result["spm_F"]),
            "A_true": _json_safe(result["A_true"]),
            "n_regions": int(result["n_regions"]),
        }
    except Exception as e:  # noqa: BLE001 -- record unexpected failures, exit 1
        entry = {
            "status": "error",
            "job_id": job_id,
            "note": note,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
        exit_code = 1
        print(f"  ERROR (unexpected): {e}")

    elapsed = time.time() - t0
    entry["elapsed_s"] = round(elapsed, 1)
    with open(out_path, "w") as f:
        json.dump(entry, f, indent=2)
    print(f"\nResult saved to: {out_path} ({elapsed:.0f}s)")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
