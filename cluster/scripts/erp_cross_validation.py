"""Generate the single-source CMC-ERP SPM12 fixtures on M3 (Phase 33-02, CMC-06).

Single-task M3 entrypoint (NOT an array). The local MATLAB license server is
unreachable (FlexLM -15), so the actual ``spm_int_L`` / ``spm_fx_cmc`` fixture
run happens here on M3 where MATLAB R2022a + SPM12 are licensed
(``/usr/local/matlab/r2022a/bin/matlab`` on the comp partition; SPM12 at
``$DCM_CLUSTER_ROOT/external/spm12``). It

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
    ``$DCM_CLUSTER_ROOT/external/spm12``); ``run_spm_erp_dcm.m`` reads it via
    ``getenv('SPM12_PATH')`` with a local-default fallback.

References
----------
cluster/scripts/spm_cross_validation.py
    The mirrored SLURM-glue structure (sys.path, env knobs, record-don't-crash).
validation/matlab_scripts/run_spm_erp_dcm.m
    The MATLAB fixture generator invoked here.
"""

from __future__ import annotations

import argparse
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
from config import CLUSTER_RESULTS_DIR  # noqa: E402

from config import MATLAB_PATH  # noqa: E402
from validation.export_to_mat import (  # noqa: E402
    export_erp_dcm,
    export_erp_dcm_leadfield,
    export_erp_dcm_multisource,
)

_MATLAB_SCRIPTS_DIR = str(_REPO_ROOT / "validation" / "matlab_scripts").replace(
    "\\", "/"
)
_DATA_DIR = _REPO_ROOT / "validation" / "data"
_INPUT_PATH = (_DATA_DIR / "erp_single_source_input.mat").as_posix()
_FIXTURE_PATH = (_DATA_DIR / "erp_single_source_fixtures.mat").as_posix()

# Multi-source (Phase 34-02) paths + the MATLAB generator entrypoint.
_MS_INPUT_PATH = (_DATA_DIR / "erp_multisource_input.mat").as_posix()
_MS_FIXTURE_PATH = (_DATA_DIR / "erp_multisource_fixtures.mat").as_posix()
_MS_N = 5
_MS_CND = 2
_MS_NS = 128
_MS_DIM = 8 * _MS_N  # 40 -- the network state dimension.

# Lead-field (Phase 35-02) paths + the MATLAB generator entrypoint.
_LF_INPUT_PATH = (_DATA_DIR / "erp_leadfield_input.mat").as_posix()
_LF_FIXTURE_PATH = (_DATA_DIR / "erp_leadfield_fixtures.mat").as_posix()
_LF_NC = 5            # LFP channels == sources.

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


def _check_multisource_fixtures(path: str) -> dict[str, Any]:
    """Round-trip the 5-source fixtures and validate shapes + provenance meta.

    Parameters
    ----------
    path : str
        Path to ``erp_multisource_fixtures.mat``.

    Returns
    -------
    dict
        ``{checks_pass, shapes, shape_ok, meta_D, nargout_Mf, x0_is_zero, N,
        meta}`` -- a recorded diagnostic, never raised (record-don't-crash). The
        cell-array fixtures (``QA``, ``QG``, ``J0``, ``Qupd``, ``y``) are checked
        per condition: ``QA{c}`` a length-4 cell of ``(5,5)``, ``QG{c}`` a
        ``(5,)`` column, ``J0{c}``/``Qupd{c}`` ``(40,40)``, ``y{c}``
        ``(128,40)``.
    """
    mat = scipy.io.loadmat(path, squeeze_me=False, struct_as_record=False)
    shapes: dict[str, Any] = {}
    shape_ok = True

    qa = mat["QA"]  # (1, Cnd) cell, each a (1,4) cell of (5,5)
    qg = mat["QG"]
    j0 = mat["J0"]
    qupd = mat["Qupd"]
    yy = mat["y"]
    cnd = qa.shape[1]
    shapes["Cnd"] = cnd
    shape_ok = shape_ok and cnd == _MS_CND

    qa_blocks = []
    for c in range(cnd):
        a_cell = np.asarray(qa[0, c])
        n_blocks = int(a_cell.size)
        block0 = np.asarray(a_cell.ravel()[0])
        qa_blocks.append([n_blocks, list(block0.shape)])
        shape_ok = shape_ok and n_blocks == 4 and block0.shape == (_MS_N, _MS_N)

        qg_c = np.atleast_1d(np.asarray(qg[0, c]).ravel())
        shape_ok = shape_ok and qg_c.size == _MS_N

        j0_c = np.asarray(j0[0, c])
        qupd_c = np.asarray(qupd[0, c])
        y_c = np.asarray(yy[0, c])
        shape_ok = shape_ok and j0_c.shape == (_MS_DIM, _MS_DIM)
        shape_ok = shape_ok and qupd_c.shape == (_MS_DIM, _MS_DIM)
        shape_ok = shape_ok and y_c.shape == (_MS_NS, _MS_DIM)
    shapes["QA"] = qa_blocks
    shapes["J0_0"] = list(np.asarray(j0[0, 0]).shape)
    shapes["Qupd_0"] = list(np.asarray(qupd[0, 0]).shape)
    shapes["y_0"] = list(np.asarray(yy[0, 0]).shape)

    meta = mat["meta"][0, 0]
    meta_d = float(np.asarray(meta.D).reshape(-1)[0])
    nargout_mf = int(np.asarray(meta.nargout_Mf).reshape(-1)[0])
    n_src = int(np.asarray(meta.N).reshape(-1)[0])
    x0 = np.asarray(meta.x0).reshape(-1)
    x0_is_zero = bool(np.all(x0 == 0.0)) and x0.size == _MS_DIM

    def _id(name: str) -> str:
        try:
            return str(np.asarray(getattr(meta, name)).reshape(-1)[0])
        except Exception:  # noqa: BLE001
            return ""

    meta_record = {
        "spm_ver": _id("spm_ver"),
        "id_spm_int_L": _id("id_spm_int_L"),
        "id_spm_fx_cmc": _id("id_spm_fx_cmc"),
        "id_spm_gen_Q": _id("id_spm_gen_Q"),
        "id_spm_gen_erp": _id("id_spm_gen_erp"),
        "D": meta_d,
        "nargout_Mf": nargout_mf,
        "N": n_src,
        "dt": float(np.asarray(meta.dt).reshape(-1)[0]),
        "ns": int(np.asarray(meta.ns).reshape(-1)[0]),
        "ons": float(np.asarray(meta.ons).reshape(-1)[0]),
        "dur": float(np.asarray(meta.dur).reshape(-1)[0]),
        "X": _json_safe(np.asarray(meta.X)),
    }

    checks_pass = bool(
        shape_ok
        and meta_d == 1.0
        and nargout_mf == 2
        and n_src == _MS_N
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
        "nargout_Mf": nargout_mf,
        "x0_is_zero": x0_is_zero,
        "N": n_src,
        "meta": meta_record,
    }


def main_multisource() -> int:
    """Generate + validate the 5-source CMC-ERP fixtures and record the result.

    Mirrors :func:`main` (export -> ``matlab -batch`` -> round-trip check -> JSON
    record, record-don't-crash) but for the multi-source MMN reference: it writes
    the DCM input via :func:`export_erp_dcm_multisource`, invokes
    ``run_spm_erp_dcm_multisource.m``, and validates the per-condition cell-array
    fixtures (``QA``, ``QG``, ``J0``, ``Qupd``, ``y``).

    Returns
    -------
    int
        ``0`` on success OR on a recorded soft check miss (record-don't-crash);
        ``1`` only on an unexpected exception (MATLAB failed / no fixture file).
    """
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    output_dir = CLUSTER_RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"erp_cross_validation_multisource_{job_id}.json"
    _DATA_DIR.mkdir(parents=True, exist_ok=True)

    note = (
        "Phase 34-02 multi-source (5-source MMN) CMC-ERP fixture generation "
        "(D=1, x0==0, N=5). The fixtures (QA, QG, J0, Qupd, y) are produced by "
        "run_spm_erp_dcm_multisource.m under MATLAB R2022a + SPM12 on M3. A soft "
        "shape/meta mismatch is RECORDED (checks_pass=False, exit 0); exit "
        "non-zero only on an unexpected MATLAB/IO failure (record-don't-crash)."
    )
    t0 = time.time()

    entry: dict[str, Any]
    exit_code = 0
    try:
        print(
            f"MATLAB_PATH={os.environ.get('MATLAB_PATH', str(MATLAB_PATH))} "
            f"SPM12_PATH={os.environ.get('SPM12_PATH', '<unset>')}"
        )

        # 1. Write the frozen 5-source DCM input .mat.
        export_meta = export_erp_dcm_multisource(output_path=_MS_INPUT_PATH)
        print(f"Wrote multi-source DCM input: {_MS_INPUT_PATH}")
        print(
            f"  N={export_meta['N']} dt={export_meta['dt']} "
            f"ns={export_meta['ns']} ons={export_meta['ons']} "
            f"dur={export_meta['dur']} n_effects={export_meta['n_effects']}"
        )
        print(f"  source_names: {export_meta['source_names']}")

        # 2. Invoke run_spm_erp_dcm_multisource.m via matlab -batch.
        matlab_cmd = (
            f"cd('{_MATLAB_SCRIPTS_DIR}'); "
            f"setenv('DCM_INPUT_PATH', '{_MS_INPUT_PATH}'); "
            f"setenv('DCM_OUTPUT_PATH', '{_MS_FIXTURE_PATH}'); "
            f"run_spm_erp_dcm_multisource"
        )
        child_env = dict(os.environ)
        matlab_bin = os.environ.get("MATLAB_PATH", str(MATLAB_PATH))
        print(f"Invoking MATLAB: {matlab_bin} -batch run_spm_erp_dcm_multisource")
        proc = subprocess.run(
            [matlab_bin, "-batch", matlab_cmd],
            capture_output=True,
            text=True,
            timeout=1800,
            env=child_env,
        )
        print("---- MATLAB stdout (tail) ----")
        print(proc.stdout[-2500:])
        if proc.returncode != 0:
            raise RuntimeError(
                f"MATLAB/SPM12 fixture generation failed (rc={proc.returncode}).\n"
                f"stderr: {proc.stderr[-1000:]}"
            )
        if not Path(_MS_FIXTURE_PATH).exists():
            raise RuntimeError(
                f"MATLAB returned 0 but {_MS_FIXTURE_PATH} was not produced."
            )

        # 3. Round-trip + validate the fixtures.
        checks = _check_multisource_fixtures(_MS_FIXTURE_PATH)
        print("\n=== Multi-source ERP fixture provenance ===")
        print(f"  shapes: {checks['shapes']}")
        print(
            f"  meta.D={checks['meta_D']} nargout_Mf={checks['nargout_Mf']} "
            f"N={checks['N']} x0_is_zero={checks['x0_is_zero']} "
            f"spm_ver={checks['meta']['spm_ver']} "
            f"id_spm_gen_Q={checks['meta']['id_spm_gen_Q']}"
        )
        if checks["checks_pass"]:
            print("  ALL MULTI-SOURCE FIXTURE CHECKS PASS.")
        else:
            print("  FIXTURE CHECK MISS RECORDED (see JSON; exit 0 by design).")

        entry = {
            "status": "ok",
            "mode": "multisource",
            "job_id": job_id,
            "note": note,
            "fixture_path": _MS_FIXTURE_PATH,
            "input_path": _MS_INPUT_PATH,
            "checks_pass": checks["checks_pass"],
            "shapes": checks["shapes"],
            "meta": checks["meta"],
        }
    except Exception as e:  # noqa: BLE001 -- record unexpected failures, exit 1
        entry = {
            "status": "error",
            "mode": "multisource",
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


def main() -> int:
    """Generate + validate the CMC-ERP fixtures and record the result.

    Returns
    -------
    int
        ``0`` on success OR on a recorded soft check miss (record-don't-crash);
        ``1`` only on an unexpected exception (MATLAB failed / no fixture file).
    """
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    output_dir = CLUSTER_RESULTS_DIR
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


def _check_leadfield_fixtures(path: str) -> dict[str, Any]:
    """Round-trip the LFP lead-field fixtures and validate shapes + meta.

    Parameters
    ----------
    path : str
        Path to ``erp_leadfield_fixtures.mat``.

    Returns
    -------
    dict
        ``{checks_pass, shapes, shape_ok, meta_D, nargout_Mf, N, Nc,
        dipfit_type, x0_is_zero, pj_index, meta}`` -- a recorded diagnostic,
        never raised (record-don't-crash). ``L_full`` is ``(Nc,8N)=(5,40)``,
        ``y_scalp`` a length-``Cnd`` cell of ``(ns,Nc)=(128,5)``, ``diff_wave``
        ``(128,5)``.
    """
    mat = scipy.io.loadmat(path, squeeze_me=False, struct_as_record=False)
    shapes: dict[str, Any] = {}
    shape_ok = True

    l_full = np.asarray(mat["L_full"])
    shapes["L_full"] = list(l_full.shape)
    shape_ok = shape_ok and l_full.shape == (_LF_NC, _MS_DIM)

    y_scalp = mat["y_scalp"]  # (1, Cnd) cell, each (ns, Nc)
    cnd = y_scalp.shape[1]
    shapes["Cnd"] = cnd
    shape_ok = shape_ok and cnd == _MS_CND
    y_shapes = []
    for c in range(cnd):
        y_c = np.asarray(y_scalp[0, c])
        y_shapes.append(list(y_c.shape))
        shape_ok = shape_ok and y_c.shape == (_MS_NS, _LF_NC)
    shapes["y_scalp"] = y_shapes

    diff_wave = np.asarray(mat["diff_wave"])
    shapes["diff_wave"] = list(diff_wave.shape)
    shape_ok = shape_ok and diff_wave.shape == (_MS_NS, _LF_NC)

    meta = mat["meta"][0, 0]
    meta_d = float(np.asarray(meta.D).reshape(-1)[0])
    nargout_mf = int(np.asarray(meta.nargout_Mf).reshape(-1)[0])
    n_src = int(np.asarray(meta.N).reshape(-1)[0])
    n_chan = int(np.asarray(meta.Nc).reshape(-1)[0])
    dipfit_type = str(np.asarray(meta.dipfit_type).reshape(-1)[0])
    x0 = np.asarray(meta.x0).reshape(-1)
    x0_is_zero = bool(np.all(x0 == 0.0)) and x0.size == _MS_DIM
    p_j = np.asarray(meta.P_J).reshape(-1)
    pj_index = int(np.argmax(p_j)) if p_j.size else -1

    def _id(name: str) -> str:
        try:
            return str(np.asarray(getattr(meta, name)).reshape(-1)[0])
        except Exception:  # noqa: BLE001
            return ""

    meta_record = {
        "spm_ver": _id("spm_ver"),
        "id_spm_lx_erp": _id("id_spm_lx_erp"),
        "id_spm_erp_L": _id("id_spm_erp_L"),
        "id_spm_L_priors": _id("id_spm_L_priors"),
        "id_spm_gen_Q": _id("id_spm_gen_Q"),
        "id_spm_int_L": _id("id_spm_int_L"),
        "D": meta_d,
        "nargout_Mf": nargout_mf,
        "N": n_src,
        "Nc": n_chan,
        "dipfit_type": dipfit_type,
        "pj_index": pj_index,
        "dt": float(np.asarray(meta.dt).reshape(-1)[0]),
        "ns": int(np.asarray(meta.ns).reshape(-1)[0]),
        "ons": float(np.asarray(meta.ons).reshape(-1)[0]),
        "dur": float(np.asarray(meta.dur).reshape(-1)[0]),
        "P_J": _json_safe(np.asarray(meta.P_J)),
        "P_L": _json_safe(np.asarray(meta.P_L)),
        "X": _json_safe(np.asarray(meta.X)),
    }

    checks_pass = bool(
        shape_ok
        and meta_d == 1.0
        and nargout_mf == 2
        and n_src == _MS_N
        and n_chan == _LF_NC
        and dipfit_type == "LFP"
        and pj_index == 2
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
        "nargout_Mf": nargout_mf,
        "N": n_src,
        "Nc": n_chan,
        "dipfit_type": dipfit_type,
        "x0_is_zero": x0_is_zero,
        "pj_index": pj_index,
        "meta": meta_record,
    }


def main_leadfield() -> int:
    """Generate + validate the LFP lead-field + scalp-ERP fixtures and record it.

    Mirrors :func:`main_multisource` (export -> ``matlab -batch`` -> round-trip
    check -> JSON record, record-don't-crash) but for the Phase-35 LFP lead
    field: it writes the DCM input via :func:`export_erp_dcm_leadfield`, invokes
    ``run_spm_erp_dcm_leadfield.m``, and validates ``L_full`` (5,40), the
    per-condition ``y_scalp`` cell (128,5), and ``diff_wave`` (128,5).

    Returns
    -------
    int
        ``0`` on success OR on a recorded soft check miss (record-don't-crash);
        ``1`` only on an unexpected exception (MATLAB failed / no fixture file).
    """
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    output_dir = CLUSTER_RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"erp_cross_validation_leadfield_{job_id}.json"
    _DATA_DIR.mkdir(parents=True, exist_ok=True)

    note = (
        "Phase 35-02 LFP lead-field + scalp-ERP fixture generation (D=1, x0==0, "
        "N=5, Nc=5, dipfit.type='LFP'). The fixtures (L_full, y_scalp, diff_wave) "
        "are produced by run_spm_erp_dcm_leadfield.m under MATLAB R2022a + SPM12 "
        "on M3. A soft shape/meta mismatch is RECORDED (checks_pass=False, exit "
        "0); exit non-zero only on an unexpected MATLAB/IO failure "
        "(record-don't-crash)."
    )
    t0 = time.time()

    entry: dict[str, Any]
    exit_code = 0
    try:
        print(
            f"MATLAB_PATH={os.environ.get('MATLAB_PATH', str(MATLAB_PATH))} "
            f"SPM12_PATH={os.environ.get('SPM12_PATH', '<unset>')}"
        )

        # 1. Write the frozen LFP lead-field DCM input .mat.
        export_meta = export_erp_dcm_leadfield(output_path=_LF_INPUT_PATH)
        print(f"Wrote lead-field DCM input: {_LF_INPUT_PATH}")
        print(
            f"  N={export_meta['N']} Nc={export_meta['Nc']} "
            f"dipfit={export_meta['dipfit_type']} dt={export_meta['dt']} "
            f"ns={export_meta['ns']} n_effects={export_meta['n_effects']}"
        )
        print(f"  source_names: {export_meta['source_names']}")

        # 2. Invoke run_spm_erp_dcm_leadfield.m via matlab -batch.
        matlab_cmd = (
            f"cd('{_MATLAB_SCRIPTS_DIR}'); "
            f"setenv('DCM_INPUT_PATH', '{_LF_INPUT_PATH}'); "
            f"setenv('DCM_OUTPUT_PATH', '{_LF_FIXTURE_PATH}'); "
            f"run_spm_erp_dcm_leadfield"
        )
        child_env = dict(os.environ)
        matlab_bin = os.environ.get("MATLAB_PATH", str(MATLAB_PATH))
        print(f"Invoking MATLAB: {matlab_bin} -batch run_spm_erp_dcm_leadfield")
        proc = subprocess.run(
            [matlab_bin, "-batch", matlab_cmd],
            capture_output=True,
            text=True,
            timeout=1800,
            env=child_env,
        )
        print("---- MATLAB stdout (tail) ----")
        print(proc.stdout[-2500:])
        if proc.returncode != 0:
            raise RuntimeError(
                f"MATLAB/SPM12 fixture generation failed (rc={proc.returncode}).\n"
                f"stderr: {proc.stderr[-1000:]}"
            )
        if not Path(_LF_FIXTURE_PATH).exists():
            raise RuntimeError(
                f"MATLAB returned 0 but {_LF_FIXTURE_PATH} was not produced."
            )

        # 3. Round-trip + validate the fixtures.
        checks = _check_leadfield_fixtures(_LF_FIXTURE_PATH)
        print("\n=== LFP lead-field ERP fixture provenance ===")
        print(f"  shapes: {checks['shapes']}")
        print(
            f"  meta.D={checks['meta_D']} nargout_Mf={checks['nargout_Mf']} "
            f"N={checks['N']} Nc={checks['Nc']} "
            f"dipfit_type={checks['dipfit_type']} pj_index={checks['pj_index']} "
            f"x0_is_zero={checks['x0_is_zero']} "
            f"spm_ver={checks['meta']['spm_ver']} "
            f"id_spm_lx_erp={checks['meta']['id_spm_lx_erp']}"
        )
        if checks["checks_pass"]:
            print("  ALL LEAD-FIELD FIXTURE CHECKS PASS.")
        else:
            print("  FIXTURE CHECK MISS RECORDED (see JSON; exit 0 by design).")

        entry = {
            "status": "ok",
            "mode": "leadfield",
            "job_id": job_id,
            "note": note,
            "fixture_path": _LF_FIXTURE_PATH,
            "input_path": _LF_INPUT_PATH,
            "checks_pass": checks["checks_pass"],
            "shapes": checks["shapes"],
            "meta": checks["meta"],
        }
    except Exception as e:  # noqa: BLE001 -- record unexpected failures, exit 1
        entry = {
            "status": "error",
            "mode": "leadfield",
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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the ``--mode {single,multisource}`` selector.

    Parameters
    ----------
    argv : list of str, optional
        Argument vector (defaults to ``sys.argv[1:]``).

    Returns
    -------
    argparse.Namespace
        ``mode`` -- ``"single"`` (default, the Phase-33 behaviour, unchanged) or
        ``"multisource"`` (the Phase-34 5-source MMN fixtures).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generate the CMC-ERP SPM12 fixtures on M3. --mode single (default) "
            "writes the Phase-33 single-source fixtures; --mode multisource "
            "writes the Phase-34 5-source MMN fixtures; --mode leadfield writes "
            "the Phase-35 LFP lead-field + scalp-ERP fixtures."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("single", "multisource", "leadfield"),
        default="single",
        help="Fixture set to generate (default: single).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    _args = _parse_args()
    if _args.mode == "leadfield":
        sys.exit(main_leadfield())
    if _args.mode == "multisource":
        sys.exit(main_multisource())
    sys.exit(main())
