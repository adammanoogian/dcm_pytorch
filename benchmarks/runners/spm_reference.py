"""SPM12 reference results loader.

Loads Phase 6 cross-validation results from VALIDATION_REPORT.md
rather than running MATLAB. Provides SPM12 comparison baselines
for the benchmark dashboard.

This runner does NOT execute MATLAB. It parses existing results
from the validation report generated in Phase 6.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from benchmarks.config import BenchmarkConfig


_REPORT_PATH = Path("validation/VALIDATION_REPORT.md")


def _parse_val01_table(content: str) -> list[dict[str, str]]:
    """Parse Task DCM vs SPM12 results table (VAL-01).

    Parameters
    ----------
    content : str
        Full VALIDATION_REPORT.md content.

    Returns
    -------
    list of dict
        Parsed rows with keys: Seed, Max_Rel_Error,
        Mean_Rel_Error, Sign_Agreement, SPM_F, Pyro_ELBO.
    """
    rows = []
    # Find the table under "## 1. Task DCM"
    section = re.search(
        r"## 1\. Task DCM.*?\n\|.*?Seed.*?\n\|[-|]+\n(.*?)(?:\n\n|\n##)",
        content,
        re.DOTALL,
    )
    if not section:
        return rows

    for line in section.group(1).strip().split("\n"):
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if len(cells) >= 5:
            rows.append({
                "Seed": cells[0],
                "Max_Rel_Error": cells[1],
                "Mean_Rel_Error": cells[2],
                "Sign_Agreement": cells[3],
                "SPM_F": cells[4],
            })

    return rows


def _parse_val04_table(content: str) -> list[dict[str, str]]:
    """Parse model ranking results table (VAL-04).

    Parameters
    ----------
    content : str
        Full VALIDATION_REPORT.md content.

    Returns
    -------
    list of dict
        Parsed rows with keys: Seed, F_correct, F_missing,
        F_diag, Correct_gt_Missing, Correct_gt_Diag.
    """
    rows = []
    section = re.search(
        r"## 4\. Model Ranking.*?"
        r"\| Seed \| F_correct.*?\n\|[-|]+\n(.*?)(?:\n\n|\n\*\*)",
        content,
        re.DOTALL,
    )
    if not section:
        return rows

    for line in section.group(1).strip().split("\n"):
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if len(cells) >= 5:
            rows.append({
                "Seed": cells[0],
                "F_correct": cells[1],
                "F_missing": cells[2],
                "F_diag": cells[3],
                "Correct_gt_Missing": cells[4],
                "Correct_gt_Diag": cells[5]
                if len(cells) > 5 else "",
            })

    return rows


def _has_pending_results(rows: list[dict[str, str]]) -> bool:
    """Check if table rows contain pending (--) values.

    Parameters
    ----------
    rows : list of dict
        Parsed table rows.

    Returns
    -------
    bool
        True if any cell value is "--" (pending).
    """
    for row in rows:
        for v in row.values():
            if v.strip() == "--":
                return True
    return False


def run_spm_reference(config: BenchmarkConfig) -> dict[str, Any]:
    """Load SPM12 results from VALIDATION_REPORT.md.

    Parses the validation report tables to extract cross-validation
    results from Phase 6. Does NOT run MATLAB.

    Parameters
    ----------
    config : BenchmarkConfig
        Benchmark configuration (mostly unused; seed for metadata).

    Returns
    -------
    dict
        Results dict with parsed validation tables or "pending"
        status if MATLAB results are not yet available.
    """
    if not _REPORT_PATH.exists():
        return {
            "status": "not_found",
            "reason": (
                f"Validation report not found at {_REPORT_PATH}. "
                f"Run Phase 6 validation first."
            ),
        }

    content = _REPORT_PATH.read_text(encoding="utf-8")

    # Parse tables
    val01_rows = _parse_val01_table(content)
    val04_rows = _parse_val04_table(content)

    # Check for pending results
    if _has_pending_results(val01_rows):
        return {
            "status": "pending",
            "reason": (
                "MATLAB results not yet available. "
                "Run plan 06-02 MATLAB batch scripts first."
            ),
            "val01_parsed": val01_rows,
            "val04_parsed": val04_rows,
        }

    # Map to standard metrics format
    result: dict[str, Any] = {
        "val01_task_dcm": val01_rows,
        "val04_model_ranking": val04_rows,
        "metadata": {
            "variant": "spm",
            "method": "reference",
            "source": "Phase 6 validation (not re-run)",
            "report_path": str(_REPORT_PATH),
        },
    }

    # Extract summary metrics from val04 (model ranking)
    if val04_rows:
        n_correct = sum(
            1 for r in val04_rows
            if r.get("Correct_gt_Missing", "").strip() == "YES"
        )
        result["ranking_agreement"] = (
            n_correct / len(val04_rows)
            if val04_rows else 0.0
        )

    return result
