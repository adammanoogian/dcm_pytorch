"""Harvest the Phase 30 per-cell JSONs into the recovery matrix + report.

Reads every per-cell JSON the M3 recovery-matrix array job wrote (one file per
``(variant, N, SNR)`` cell), classifies each cell against the documented
per-cell thresholds (``benchmarks.recovery_matrix_thresholds.classify_cell``),
and emits:

- ``recovery_matrix.csv`` and ``recovery_matrix.json`` -- the flat matrix over
  ``(variant, n_regions, snr)`` with the recovery metrics + the per-cell
  pass/identifiability-limit verdict.
- A human-readable markdown report -- a per-cell verdict table, the
  eig_clamp / stability-boundary regime characterization (VLROBUST-03), and an
  explicit "no silent failures" listing of every error-status cell with its
  traceback summary.

This is the "harvest + report" half of the two-phase async sweep: it runs once
SLURM reports the array finished, on the locally-synced results (Mutagen, NOT
git). It is runnable both on M3 (post-harvest) and locally.

VLREC-04 is enforced structurally: an error-status cell is NEVER silently
dropped -- it is loaded, listed in the report, and counted in the final summary.
A non-erroring cell that misses a documented threshold is labelled
``identifiability_limit`` WITH evidence, not failed/hidden. The script fails loud
(expected-vs-actual) ONLY when zero result files match, so the orchestrator knows
the harvest ran before the job actually finished.

Environment variables
---------------------
RECOVERY_RESULTS_GLOB : str, default "cluster/results/recovery_matrix_*.json"
    Glob for the per-cell JSON files to harvest.
RECOVERY_OUT_DIR : str, default "benchmarks/results"
    Directory the matrix CSV/JSON are written to.
RECOVERY_REPORT_PATH : str, default
    ".planning/phases/30-recovery-matrix-sweep/30-RECOVERY-MATRIX-REPORT.md"
    Path the human-readable markdown report is written to.

References
----------
benchmarks.recovery_matrix_thresholds.classify_cell
    Per-cell pass/identifiability-limit classifier (VLREC-04).
benchmarks.recovery_matrix_metrics.NEAR_BOUNDARY_LO / NEAR_BOUNDARY_HI
    The near-stability-boundary exclusion band characterized here (VLROBUST-03).
"""

from __future__ import annotations

import csv
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from benchmarks.recovery_matrix_metrics import (  # noqa: E402
    NEAR_BOUNDARY_HI,
    NEAR_BOUNDARY_LO,
)
from benchmarks.recovery_matrix_thresholds import (  # noqa: E402
    SHRINKAGE_SOFT_TARGET,
    classify_cell,
)

DEFAULT_GLOB = "cluster/results/recovery_matrix_*.json"
DEFAULT_OUT_DIR = "benchmarks/results"
DEFAULT_REPORT_PATH = (
    ".planning/phases/30-recovery-matrix-sweep/30-RECOVERY-MATRIX-REPORT.md"
)

# Files matching DEFAULT_GLOB that are NOT real sweep cells (local pre-checks).
_EXCLUDE_SUBSTRINGS = ("_local_",)

MATRIX_COLUMNS = [
    "cell_index",
    "variant",
    "n_regions",
    "snr",
    "rmse_a_median",
    "rmse_a_q25",
    "rmse_a_q75",
    "sign_recovery_masked",
    "coverage_95",
    "shrinkage_median",
    "r2_per_region",
    "convergence_rate",
    "n_success",
    "n_failed",
    "status",
    "reason",
]


def harvest_files(results_glob: str) -> list[str]:
    """Return the sorted list of per-cell JSON files to harvest.

    Local pre-check files (``recovery_matrix_local_*.json``) are excluded so a
    leftover laptop faithfulness check never pollutes the matrix.

    Parameters
    ----------
    results_glob : str
        Glob for candidate per-cell JSON files.

    Returns
    -------
    list of str
        Sorted matching file paths, with local pre-check files removed.
    """
    matches = sorted(glob.glob(results_glob))
    return [
        f
        for f in matches
        if not any(sub in os.path.basename(f) for sub in _EXCLUDE_SUBSTRINGS)
    ]


def _eig_band_summary(max_eig_list: list[float] | None) -> dict[str, Any]:
    """Summarize how close a cell's ground-truth eigenvalues sit to the band.

    The 30-02 driver excluded ground-truth ``A`` whose max real eigenvalue fell
    in the near-stability-boundary band ``[NEAR_BOUNDARY_LO, NEAR_BOUNDARY_HI]``
    (pitfall N2). This reports, per cell, the max real eigenvalue distribution and
    how many of the accepted draws landed inside the rejected band (should be 0 if
    the exclusion held) -- the eig_clamp / boundary-regime evidence (VLROBUST-03).

    Parameters
    ----------
    max_eig_list : list of float or None
        Per-seed max real eigenvalues of the accepted ground-truth ``A`` (from the
        cell's ``raw.max_real_eig_list``), or ``None`` when unavailable.

    Returns
    -------
    dict
        ``{"n_seeds", "max_eig_max", "max_eig_min", "in_band_count",
        "nearest_to_boundary"}``; counts/extrema are ``None`` when the list is
        absent.
    """
    if not max_eig_list:
        return {
            "n_seeds": 0,
            "max_eig_max": None,
            "max_eig_min": None,
            "in_band_count": None,
            "nearest_to_boundary": None,
        }
    vals = [float(v) for v in max_eig_list]
    in_band = [v for v in vals if NEAR_BOUNDARY_LO <= v <= NEAR_BOUNDARY_HI]
    # The eigenvalue closest to the boundary band (largest, i.e. least stable).
    nearest = max(vals)
    return {
        "n_seeds": len(vals),
        "max_eig_max": max(vals),
        "max_eig_min": min(vals),
        "in_band_count": len(in_band),
        "nearest_to_boundary": nearest,
    }


def _row_from_ok_cell(entry: dict[str, Any]) -> dict[str, Any]:
    """Build a matrix row + verdict for a successful (status=ok) cell.

    Parameters
    ----------
    entry : dict
        A loaded per-cell JSON with ``status == "ok"`` (carries ``metrics`` and
        ``raw``).

    Returns
    -------
    dict
        ``{"row", "verdict", "eig_band"}`` -- the flat matrix row, the
        ``classify_cell`` verdict, and the per-cell eigenvalue-band summary.
    """
    metrics = entry.get("metrics") or {}
    cell = {
        "variant": entry.get("variant"),
        "n_regions": entry.get("n_regions"),
        "snr": entry.get("snr"),
    }
    verdict = classify_cell(metrics, cell=cell)
    rmse_a = metrics.get("rmse_a")
    rmse_median = rmse_a.get("median") if isinstance(rmse_a, dict) else None
    rmse_q25 = rmse_a.get("q25") if isinstance(rmse_a, dict) else None
    rmse_q75 = rmse_a.get("q75") if isinstance(rmse_a, dict) else None
    row = {
        "cell_index": entry.get("cell_index"),
        "variant": entry.get("variant"),
        "n_regions": entry.get("n_regions"),
        "snr": entry.get("snr"),
        "rmse_a_median": rmse_median,
        "rmse_a_q25": rmse_q25,
        "rmse_a_q75": rmse_q75,
        "sign_recovery_masked": metrics.get("sign_recovery_masked"),
        "coverage_95": metrics.get("coverage_95"),
        "shrinkage_median": metrics.get("shrinkage"),
        "r2_per_region": metrics.get("r2_per_region"),
        "convergence_rate": metrics.get("convergence_rate"),
        "n_success": metrics.get("n_success"),
        "n_failed": metrics.get("n_failed"),
        "status": verdict["status"],
        "reason": verdict["reason"],
    }
    eig_band = _eig_band_summary((entry.get("raw") or {}).get("max_real_eig_list"))
    return {"row": row, "verdict": verdict, "eig_band": eig_band}


def _row_from_error_cell(entry: dict[str, Any]) -> dict[str, Any]:
    """Build a matrix row for an error-status cell (surfaced, never dropped).

    Parameters
    ----------
    entry : dict
        A loaded per-cell JSON with ``status == "error"``.

    Returns
    -------
    dict
        ``{"cell_index", "variant", "n_regions", "snr", "error", "traceback"}``
        for the report's no-silent-failures listing.
    """
    return {
        "cell_index": entry.get("cell_index"),
        "variant": entry.get("variant"),
        "n_regions": entry.get("n_regions"),
        "snr": entry.get("snr"),
        "error": entry.get("error", "<no error message>"),
        "traceback": entry.get("traceback", ""),
    }


def aggregate(
    *,
    results_glob: str | None = None,
    out_dir: str | Path | None = None,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Harvest per-cell JSONs into the recovery matrix CSV/JSON + report.

    Parameters
    ----------
    results_glob : str or None, optional
        Glob for the per-cell JSONs. Defaults to ``RECOVERY_RESULTS_GLOB`` env or
        ``cluster/results/recovery_matrix_*.json``.
    out_dir : str or Path or None, optional
        Directory for ``recovery_matrix.csv`` / ``recovery_matrix.json``. Defaults
        to ``RECOVERY_OUT_DIR`` env or ``benchmarks/results``.
    report_path : str or Path or None, optional
        Markdown report path. Defaults to ``RECOVERY_REPORT_PATH`` env or the
        Phase 30 report under ``.planning/``. Parameterized so tests write to a
        ``tmp_path`` instead of the real planning file.

    Returns
    -------
    dict
        ``{"n_pass", "n_limit", "n_error", "rows", "errored", "boundary_regime",
        "csv_path", "json_path", "report_path"}``.

    Raises
    ------
    FileNotFoundError
        If zero result files match ``results_glob`` (fail loud so the orchestrator
        knows the harvest ran before the job finished).
    """
    results_glob = results_glob or os.environ.get("RECOVERY_RESULTS_GLOB", DEFAULT_GLOB)
    out_dir = Path(out_dir or os.environ.get("RECOVERY_OUT_DIR", DEFAULT_OUT_DIR))
    report_path = Path(
        report_path
        or os.environ.get("RECOVERY_REPORT_PATH", DEFAULT_REPORT_PATH)
    )

    files = harvest_files(results_glob)
    if not files:
        raise FileNotFoundError(
            "No recovery-matrix result files found: expected >= 1 file matching "
            f"glob {results_glob!r}, got 0. Did the M3 array job finish and sync "
            "before this harvest ran?"
        )

    rows: list[dict[str, Any]] = []
    errored: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    n_pass = 0
    n_limit = 0

    for f in files:
        with open(f) as fh:
            entry = json.load(fh)
        if entry.get("status") == "error":
            err = _row_from_error_cell(entry)
            errored.append(err)
            # Still emit a matrix row so the cell is visible in the CSV.
            rows.append(
                {
                    "cell_index": entry.get("cell_index"),
                    "variant": entry.get("variant"),
                    "n_regions": entry.get("n_regions"),
                    "snr": entry.get("snr"),
                    "rmse_a_median": None,
                    "rmse_a_q25": None,
                    "rmse_a_q75": None,
                    "sign_recovery_masked": None,
                    "coverage_95": None,
                    "shrinkage_median": None,
                    "r2_per_region": None,
                    "convergence_rate": None,
                    "n_success": None,
                    "n_failed": None,
                    "status": "error",
                    "reason": f"errored: {err['error']}",
                }
            )
            continue

        built = _row_from_ok_cell(entry)
        rows.append(built["row"])
        if built["row"]["status"] == "pass":
            n_pass += 1
        else:
            n_limit += 1
        boundary_rows.append(
            {
                "cell_index": entry.get("cell_index"),
                "variant": entry.get("variant"),
                "n_regions": entry.get("n_regions"),
                "snr": entry.get("snr"),
                "shrinkage_median": (entry.get("metrics") or {}).get("shrinkage"),
                "coverage_95": (entry.get("metrics") or {}).get("coverage_95"),
                **built["eig_band"],
            }
        )

    rows.sort(key=lambda r: (r["cell_index"] is None, r["cell_index"]))
    boundary_rows.sort(key=lambda r: (r["cell_index"] is None, r["cell_index"]))
    errored.sort(key=lambda r: (r["cell_index"] is None, r["cell_index"]))

    boundary_regime = _build_boundary_regime(boundary_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "recovery_matrix.csv"
    json_path = out_dir / "recovery_matrix.json"
    _write_csv(csv_path, rows)
    matrix_payload = {
        "n_cells": len(rows),
        "n_pass": n_pass,
        "n_identifiability_limit": n_limit,
        "n_error": len(errored),
        "rows": rows,
        "errored": errored,
        "boundary_regime": boundary_regime,
    }
    with open(json_path, "w") as fh:
        json.dump(matrix_payload, fh, indent=2)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        _render_report(rows, errored, boundary_regime, n_pass, n_limit),
        encoding="utf-8",
    )

    return {
        "n_pass": n_pass,
        "n_limit": n_limit,
        "n_error": len(errored),
        "rows": rows,
        "errored": errored,
        "boundary_regime": boundary_regime,
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "report_path": str(report_path),
    }


def _build_boundary_regime(boundary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-cell eigenvalue/shrinkage evidence into a regime block.

    Parameters
    ----------
    boundary_rows : list of dict
        Per-cell eigenvalue-band + shrinkage/coverage summaries.

    Returns
    -------
    dict
        ``{"band", "total_in_band", "per_cell", "overconfident_cells"}`` -- the
        exclusion band, the total count of accepted draws that still fell in the
        rejected band (should be 0), and the cells flagged as the
        overconfident/non-injective regime (low shrinkage relative to the soft
        target).
    """
    total_in_band = sum(
        r["in_band_count"] for r in boundary_rows if r["in_band_count"]
    )
    overconfident = [
        {
            "cell_index": r["cell_index"],
            "variant": r["variant"],
            "n_regions": r["n_regions"],
            "snr": r["snr"],
            "shrinkage_median": r["shrinkage_median"],
        }
        for r in boundary_rows
        if r["shrinkage_median"] is not None
        and r["shrinkage_median"] < SHRINKAGE_SOFT_TARGET
    ]
    return {
        "band": [NEAR_BOUNDARY_LO, NEAR_BOUNDARY_HI],
        "shrinkage_soft_target": SHRINKAGE_SOFT_TARGET,
        "total_accepted_draws_in_band": total_in_band,
        "per_cell": boundary_rows,
        "overconfident_cells": overconfident,
    }


def _write_csv(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the flat matrix rows to ``csv_path`` with a stable column order.

    Parameters
    ----------
    csv_path : Path
        Destination CSV path.
    rows : list of dict
        Matrix rows keyed by ``MATRIX_COLUMNS``.
    """
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MATRIX_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col) for col in MATRIX_COLUMNS})


def _fmt(value: Any) -> str:
    """Format a metric value for the markdown table (floats to 4 sig figs)."""
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _render_report(
    rows: list[dict[str, Any]],
    errored: list[dict[str, Any]],
    boundary_regime: dict[str, Any],
    n_pass: int,
    n_limit: int,
) -> str:
    """Render the human-readable per-cell verdict + boundary-regime report.

    Parameters
    ----------
    rows : list of dict
        Matrix rows (one per cell, including errored cells).
    errored : list of dict
        Error-status cells, surfaced under "no silent failures".
    boundary_regime : dict
        The eig_clamp / stability-boundary regime characterization.
    n_pass : int
        Count of cells classified ``pass``.
    n_limit : int
        Count of cells classified ``identifiability_limit``.

    Returns
    -------
    str
        The full markdown report body.
    """
    lines: list[str] = []
    lines.append("# Phase 30 Recovery Matrix Report")
    lines.append("")
    lines.append(
        f"**Verdict summary:** {n_pass} PASS · {n_limit} "
        f"IDENTIFIABILITY-LIMIT-WITH-EVIDENCE · {len(errored)} ERRORED "
        f"(surfaced below, no silent failures). Total cells: {len(rows)}."
    )
    lines.append("")
    lines.append(
        "Every cell receives an explicit verdict (VLREC-04): a cell either meets "
        "the documented per-cell thresholds (PASS) or is documented as an "
        "identifiability limit WITH evidence; error-status cells are listed "
        "explicitly, never dropped."
    )
    lines.append("")

    lines.append("## Per-cell verdicts")
    lines.append("")
    lines.append(
        "| cell | variant | N | SNR | RMSE_A med | RMSE_A IQR | sign(masked) | "
        "cov95 | shrink | R2/region | conv | verdict |"
    )
    lines.append(
        "|---|---|---|---|---|---|---|---|---|---|---|---|"
    )
    for r in rows:
        iqr = f"{_fmt(r['rmse_a_q25'])}–{_fmt(r['rmse_a_q75'])}"
        verdict = (
            "PASS"
            if r["status"] == "pass"
            else ("ERROR" if r["status"] == "error" else "IDENT-LIMIT")
        )
        lines.append(
            f"| {_fmt(r['cell_index'])} | {r['variant']} | {_fmt(r['n_regions'])} "
            f"| {_fmt(r['snr'])} | {_fmt(r['rmse_a_median'])} | {iqr} "
            f"| {_fmt(r['sign_recovery_masked'])} | {_fmt(r['coverage_95'])} "
            f"| {_fmt(r['shrinkage_median'])} | {_fmt(r['r2_per_region'])} "
            f"| {_fmt(r['convergence_rate'])} | {verdict} |"
        )
    lines.append("")

    # Identifiability limits with their reasons.
    limits = [r for r in rows if r["status"] == "identifiability_limit"]
    lines.append("## Identifiability limits (with evidence)")
    lines.append("")
    if limits:
        for r in limits:
            lines.append(
                f"- **cell {r['cell_index']} ({r['variant']} N={r['n_regions']} "
                f"SNR={r['snr']})** — {r['reason']}"
            )
    else:
        lines.append(
            "None — every successfully-fit cell met its documented thresholds."
        )
    lines.append("")

    # No silent failures: errored cells.
    lines.append("## No silent failures — errored cells")
    lines.append("")
    if errored:
        lines.append(
            "These cells errored during the fit and are surfaced here (NOT "
            "dropped from the matrix); each appears in the CSV with status "
            "`error`:"
        )
        lines.append("")
        for e in errored:
            summary = (e["error"] or "").strip().splitlines()
            summary_str = summary[-1] if summary else "<no message>"
            lines.append(
                f"- **cell {e['cell_index']} ({e['variant']} N={e['n_regions']} "
                f"SNR={e['snr']})** — `{summary_str}`"
            )
    else:
        lines.append("None — every cell produced a fit result.")
    lines.append("")

    # Boundary regime characterization (VLROBUST-03).
    lines.append("## eig_clamp / stability-boundary regime (VLROBUST-03)")
    lines.append("")
    band = boundary_regime["band"]
    lines.append(
        f"Ground-truth `A` was drawn with max-real-eigenvalue EXCLUDED from the "
        f"near-stability-boundary band `[{band[0]}, {band[1]}]` (eig_clamp "
        f"non-injectivity, pitfall N2). Accepted draws still falling in that band "
        f"(should be 0): **{boundary_regime['total_accepted_draws_in_band']}**."
    )
    lines.append("")
    lines.append(
        "| cell | variant | N | SNR | max-eig (max) | max-eig (min) | "
        "in-band | shrink | cov95 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in boundary_regime["per_cell"]:
        lines.append(
            f"| {_fmt(r['cell_index'])} | {r['variant']} | {_fmt(r['n_regions'])} "
            f"| {_fmt(r['snr'])} | {_fmt(r['max_eig_max'])} "
            f"| {_fmt(r['max_eig_min'])} | {_fmt(r['in_band_count'])} "
            f"| {_fmt(r['shrinkage_median'])} | {_fmt(r['coverage_95'])} |"
        )
    lines.append("")
    soft = boundary_regime["shrinkage_soft_target"]
    over = boundary_regime["overconfident_cells"]
    lines.append(
        f"**Overconfident / non-injective regime.** Shrinkage soft target is "
        f"{soft} (RECOV-07); very-low shrinkage is the EXPECTED Laplace "
        f"overconfidence signal (job 55772525), documented here as evidence, NOT "
        f"flagged as a bug. Cells with shrinkage below the soft target:"
    )
    lines.append("")
    if over:
        for o in over:
            lines.append(
                f"- cell {o['cell_index']} ({o['variant']} N={o['n_regions']} "
                f"SNR={o['snr']}) — shrinkage {_fmt(o['shrinkage_median'])} "
                f"(< {soft})"
            )
    else:
        lines.append("- None — all cells at or above the soft target.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    """Run the harvest+aggregate and print the pass/limit/error summary."""
    result = aggregate()
    print(
        f"Recovery matrix harvested: {result['n_pass']} pass / "
        f"{result['n_limit']} identifiability-limit / {result['n_error']} errored."
    )
    print(f"  matrix CSV : {result['csv_path']}")
    print(f"  matrix JSON: {result['json_path']}")
    print(f"  report     : {result['report_path']}")
    if result["errored"]:
        print("  errored cells (surfaced, not dropped):")
        for e in result["errored"]:
            print(
                f"    cell {e['cell_index']} {e['variant']} "
                f"N={e['n_regions']} SNR={e['snr']}: {e['error']}"
            )


if __name__ == "__main__":
    main()
