"""Generate the single-source CMC-ERP SPM12 fixtures on M3 (Phase 33-02, CMC-06).

Single-task M3 entrypoint (NOT an array). The local MATLAB license server is
unreachable (FlexLM -15), so the actual ``spm_int_L`` / ``spm_fx_cmc`` fixture
run happens here on M3 where MATLAB R2022a + SPM12 are licensed
(``/usr/local/matlab/r2022a/bin/matlab`` on the comp partition; SPM12 at
``/home/aman0087/fc37/Carrick/spm12``). It

1. writes the frozen single-source DCM input ``.mat`` via
   :func:`validation.export_to_mat.export_erp_dcm`,
2. invokes ``run_spm_erp_dcm.m`` through ``matlab -batch`` to produce the 5 frozen
   fixture arrays (``f_field``, ``J0``, ``dtJ``, ``Eexp``, ``Q_update``,
   ``y_states``) + a provenance ``meta`` (D == 1, x0 == 0, SPM ``$Id``, dt, ns,
   ons, dur, x_test, u_test) into ``validation/data/erp_single_source_fixtures.mat``,
3. round-trips the fixtures, checks the shapes / ``meta.D`` / ``meta.x0``, and
4. writes a JSON provenance record to ``cluster/results/``.

Record-don't-crash (mirrors 31-03-D3 / the Phase-32 harness): a soft fixture-shape
or meta mismatch is RECORDED in the JSON + stdout (``checks_pass = False``) and the
script exits 0. The script exits non-zero ONLY on an unexpected exception -- the
MATLAB subprocess failing entirely or the fixture file not being produced.

Environment variables
---------------------
SLURM_JOB_ID : str, default "local"
    Job id used in the output filename.
MATLAB_PATH : str
    MATLAB binary (the sbatch exports ``/usr/local/matlab/r2022a/bin/matlab``).
SPM12_PATH : str
    SPM12 location passed to the MATLAB child (the sbatch exports
    ``/home/aman0087/fc37/Carrick/spm12``); ``run_spm_erp_dcm.m`` reads it via
    ``getenv('SPM12_PATH')`` with a local-default fallback.

References
----------
cluster/scripts/spm_cross_validation.py
    The mirrored SLURM-glue structure (sys.path, env knobs, record-don't-crash).
validation/matlab_scripts/run_spm_erp_dcm.m
    The MATLAB fixture generator invoked here.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from config import MATLAB_PATH  # noqa: E402
from validation.export_to_mat import export_erp_dcm  # noqa: E402

_MATLAB_SCRIPTS_DIR = str(_REPO_ROOT / "validation" / "matlab_scripts").replace(
    "\\", "/"
)
_DATA_DIR = _REPO_ROOT / "validation" / "data"
_INPUT_PATH = (_DATA_DIR / "erp_single_source_input.mat").as_posix()
_FIXTURE_PATH = (_DATA_DIR / "erp_single_source_fixtures.mat").as_posix()

# Expected frozen fixture shapes (the Wave-3 parity ladder asserts against these).
_EXPECTED_SHAPES = {
    "f_field": (8,),
    "J0": (8, 8),
    "dtJ": (8, 8),
    "Eexp": (8, 8),
    "Q_update": (8, 8),
    "y_states": (128, 8),
}


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


def _check_fixtures(path: str) -> dict[str, Any]:
    """Round-trip the produced fixtures and validate shapes + provenance meta.

    Parameters
    ----------
    path : str
        Path to ``erp_single_source_fixtures.mat``.

    Returns
    -------
    dict
        ``{checks_pass, shapes, shape_ok, meta_D, x0_is_zero, meta}`` -- a recorded
        diagnostic, never raised (record-don't-crash).
    """
    mat = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    shapes = {}
    shape_ok = True
    for name, expected in _EXPECTED_SHAPES.items():
        arr = np.atleast_1d(np.asarray(mat[name]))
        shapes[name] = list(arr.shape)
        if tuple(arr.shape) != expected:
            shape_ok = False

    meta = mat["meta"]
    meta_d = float(np.asarray(meta.D).reshape(-1)[0])
    x0 = np.asarray(meta.x0).reshape(-1)
    x0_is_zero = bool(np.all(x0 == 0.0)) and x0.size == 8
    nargout_mf = int(np.asarray(meta.nargout_Mf).reshape(-1)[0])

    meta_record = {
        "spm_ver": str(np.asarray(meta.spm_ver).reshape(-1)[0]),
        "id_spm_int_L": str(np.asarray(meta.id_spm_int_L).reshape(-1)[0]),
        "id_spm_fx_cmc": str(np.asarray(meta.id_spm_fx_cmc).reshape(-1)[0]),
        "D": meta_d,
        "nargout_Mf": nargout_mf,
        "dt": float(np.asarray(meta.dt).reshape(-1)[0]),
        "ns": int(np.asarray(meta.ns).reshape(-1)[0]),
        "ons": float(np.asarray(meta.ons).reshape(-1)[0]),
        "dur": float(np.asarray(meta.dur).reshape(-1)[0]),
        "u_test": float(np.asarray(meta.u_test).reshape(-1)[0]),
        "x_test": _json_safe(np.asarray(meta.x_test)),
    }

    checks_pass = bool(
        shape_ok
        and meta_d == 1.0
        and nargout_mf == 2
        and x0_is_zero
        and meta_record["dt"] == 0.004
        and meta_record["ns"] == 128
        and meta_record["dur"] == 16.0
    )
    return {
        "checks_pass": checks_pass,
        "shapes": shapes,
        "shape_ok": shape_ok,
        "meta_D": meta_d,
        "x0_is_zero": x0_is_zero,
        "meta": meta_record,
    }


def main() -> int:
    """Generate + validate the CMC-ERP fixtures and record the result.

    Returns
    -------
    int
        ``0`` on success OR on a recorded soft check miss (record-don't-crash);
        ``1`` only on an unexpected exception (MATLAB failed / no fixture file).
    """
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    output_dir = Path("cluster/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"erp_cross_validation_{job_id}.json"
    _DATA_DIR.mkdir(parents=True, exist_ok=True)

    note = (
        "Phase 33-02 single-source CMC-ERP fixture generation (D=1, x0==0). The "
        "fixtures are produced by run_spm_erp_dcm.m under MATLAB R2022a + SPM12 on "
        "M3. A soft shape/meta mismatch is RECORDED (checks_pass=False, exit 0); "
        "exit non-zero only on an unexpected MATLAB/IO failure (record-don't-crash)."
    )
    t0 = time.time()

    entry: dict[str, Any]
    exit_code = 0
    try:
        print(
            f"MATLAB_PATH={os.environ.get('MATLAB_PATH', str(MATLAB_PATH))} "
            f"SPM12_PATH={os.environ.get('SPM12_PATH', '<unset>')}"
        )

        # 1. Write the frozen single-source DCM input .mat.
        export_meta = export_erp_dcm(output_path=_INPUT_PATH)
        print(f"Wrote DCM input: {_INPUT_PATH}")
        print(
            f"  frozen grid: dt={export_meta['dt']} ns={export_meta['ns']} "
            f"ons={export_meta['ons']} dur={export_meta['dur']} "
            f"u_test={export_meta['u_test']}"
        )

        # 2. Invoke run_spm_erp_dcm.m via matlab -batch.
        matlab_cmd = (
            f"cd('{_MATLAB_SCRIPTS_DIR}'); "
            f"setenv('DCM_INPUT_PATH', '{_INPUT_PATH}'); "
            f"setenv('DCM_OUTPUT_PATH', '{_FIXTURE_PATH}'); "
            f"run_spm_erp_dcm"
        )
        # Pass SPM12_PATH through to the MATLAB child; the .m falls back to its
        # local default when the variable is absent (laptop + M3 share one code).
        child_env = dict(os.environ)
        matlab_bin = os.environ.get("MATLAB_PATH", str(MATLAB_PATH))
        print(f"Invoking MATLAB: {matlab_bin} -batch run_spm_erp_dcm")
        proc = subprocess.run(
            [matlab_bin, "-batch", matlab_cmd],
            capture_output=True,
            text=True,
            timeout=1800,
            env=child_env,
        )
        print("---- MATLAB stdout (tail) ----")
        print(proc.stdout[-2000:])
        if proc.returncode != 0:
            raise RuntimeError(
                f"MATLAB/SPM12 fixture generation failed (rc={proc.returncode}).\n"
                f"stderr: {proc.stderr[-1000:]}"
            )
        if not Path(_FIXTURE_PATH).exists():
            raise RuntimeError(
                f"MATLAB returned 0 but {_FIXTURE_PATH} was not produced."
            )

        # 3. Round-trip + validate the fixtures.
        checks = _check_fixtures(_FIXTURE_PATH)
        print("\n=== ERP fixture provenance ===")
        print(f"  shapes: {checks['shapes']}")
        print(
            f"  meta.D={checks['meta_D']} x0_is_zero={checks['x0_is_zero']} "
            f"spm_ver={checks['meta']['spm_ver']} "
            f"id_spm_int_L={checks['meta']['id_spm_int_L']}"
        )
        if checks["checks_pass"]:
            print("  ALL FIXTURE CHECKS PASS.")
        else:
            print("  FIXTURE CHECK MISS RECORDED (see JSON; exit 0 by design).")

        entry = {
            "status": "ok",
            "job_id": job_id,
            "note": note,
            "fixture_path": _FIXTURE_PATH,
            "input_path": _INPUT_PATH,
            "checks_pass": checks["checks_pass"],
            "shapes": checks["shapes"],
            "meta": checks["meta"],
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
