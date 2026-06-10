"""Tests for the Phase 30 recovery-matrix harvest/aggregate + classifier.

Laptop-fast, hermetic, NO cluster and NO real fits: synthetic per-cell JSON
fixtures matching the ``run_one_cell`` output schema are written into ``tmp_path``,
and the aggregator + classifier are exercised against them. The two VLREC-04
guarantees are concretely enforced:

- ``test_no_silent_failures`` asserts an error-status fixture cell appears in the
  report (its variant / N / SNR listed), proving failures are surfaced not
  dropped.
- ``test_empty_results_fails_loud`` asserts pointing the glob at an empty dir
  raises an expected-vs-actual ``FileNotFoundError``.

All tests are marked ``@pytest.mark.vl`` (registered in ``pyproject.toml`` since
Phase 29) and parameterize the report path to ``tmp_path`` so the real
``.planning/`` report is never touched.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.recovery_matrix_thresholds import classify_cell

# Import the aggregator module by path-independent name; it lives under
# cluster/scripts which is on sys.path via the repo root in test runs.
_AGG_PATH = (
    Path(__file__).resolve().parents[1]
    / "cluster"
    / "scripts"
    / "recovery_matrix_aggregate.py"
)
_spec = importlib.util.spec_from_file_location("recovery_matrix_aggregate", _AGG_PATH)
assert _spec is not None and _spec.loader is not None
recovery_matrix_aggregate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(recovery_matrix_aggregate)

pytestmark = pytest.mark.vl


def _passing_spectral_cell() -> dict[str, Any]:
    """Build a successful spectral cell that should classify ``pass``."""
    return {
        "status": "ok",
        "task_id": 0,
        "job_id": "TESTJOB",
        "cell_index": 0,
        "variant": "spectral",
        "n_regions": 2,
        "snr": 1.0,
        "metrics": {
            "variant": "spectral",
            "method": "vl",
            "n_regions": 2,
            "n_success": 10,
            "n_failed": 0,
            "convergence_rate": 1.0,
            "rmse_a": {"median": 0.0014, "q25": 0.0002, "q75": 0.0025},
            "coverage_95": 1.0,
            "sign_recovery_masked": 1.0,
            "r2_per_region": None,
            "shrinkage": 0.21,
        },
        "raw": {"max_real_eig_list": [-0.5, -0.4, -0.5]},
        "n_seeds": 10,
        "elapsed_s": 13.7,
    }


def _failing_task_cell() -> dict[str, Any]:
    """Build a high-RMSE task cell that should be an ``identifiability_limit``."""
    return {
        "status": "ok",
        "task_id": 1,
        "job_id": "TESTJOB",
        "cell_index": 1,
        "variant": "task",
        "n_regions": 4,
        "snr": 1.0,
        "metrics": {
            "variant": "task",
            "method": "vl",
            "n_regions": 4,
            "n_success": 8,
            "n_failed": 2,
            "convergence_rate": 0.8,
            "rmse_a": {"median": 0.42, "q25": 0.30, "q75": 0.55},
            "coverage_95": 0.40,
            "sign_recovery_masked": 0.30,
            "r2_per_region": None,
            "shrinkage": 0.05,
        },
        "raw": {"max_real_eig_list": [-0.3, -0.25]},
        "n_seeds": 10,
        "elapsed_s": 42.0,
    }


def _error_latent_cell() -> dict[str, Any]:
    """Build an error-status latent_circuit cell (no metrics)."""
    return {
        "status": "error",
        "task_id": 2,
        "job_id": "TESTJOB",
        "cell_index": 2,
        "variant": "latent_circuit",
        "n_regions": 4,
        "snr": 3.0,
        "error": "underflow in dt 0.0",
        "traceback": "Traceback (most recent call last):\n...\nunderflow in dt 0.0\n",
        "elapsed_s": 3.1,
    }


def _write_fixtures(fixture_dir: Path) -> None:
    """Write the three synthetic per-cell JSON fixtures into ``fixture_dir``."""
    fixture_dir.mkdir(parents=True, exist_ok=True)
    cells = [
        ("recovery_matrix_TESTJOB_0.json", _passing_spectral_cell()),
        ("recovery_matrix_TESTJOB_1.json", _failing_task_cell()),
        ("recovery_matrix_TESTJOB_2.json", _error_latent_cell()),
    ]
    for name, obj in cells:
        (fixture_dir / name).write_text(json.dumps(obj), encoding="utf-8")


def test_classify_pass_and_limit() -> None:
    """Passing cell -> ``pass``; high-RMSE cell -> ``identifiability_limit``."""
    good = _passing_spectral_cell()
    verdict_good = classify_cell(
        good["metrics"],
        cell={"variant": "spectral", "n_regions": 2, "snr": 1.0},
    )
    assert verdict_good["status"] == "pass"

    bad = _failing_task_cell()
    verdict_bad = classify_cell(
        bad["metrics"],
        cell={"variant": "task", "n_regions": 4, "snr": 1.0},
    )
    assert verdict_bad["status"] == "identifiability_limit"
    assert verdict_bad["reason"]
    assert verdict_bad["evidence"]["rmse_a_median"] == pytest.approx(0.42)
    # Failing cell is a documented limit, never an exception.
    assert verdict_bad["checks"]["rmse_a"]["pass"] is False


def test_aggregate_writes_outputs(tmp_path: Path) -> None:
    """Aggregator writes CSV, JSON and the report with one row per fixture."""
    fixture_dir = tmp_path / "results"
    _write_fixtures(fixture_dir)
    out_dir = tmp_path / "out"
    report_path = tmp_path / "report.md"

    result = recovery_matrix_aggregate.aggregate(
        results_glob=str(fixture_dir / "*.json"),
        out_dir=out_dir,
        report_path=report_path,
    )

    csv_path = out_dir / "recovery_matrix.csv"
    json_path = out_dir / "recovery_matrix.json"
    assert csv_path.exists()
    assert json_path.exists()
    assert report_path.exists()

    # One header + one row per fixture cell (3).
    data_rows = csv_path.read_text(encoding="utf-8").strip().splitlines()[1:]
    assert len(data_rows) == 3
    assert result["n_pass"] == 1
    assert result["n_limit"] == 1
    assert result["n_error"] == 1


def test_no_silent_failures(tmp_path: Path) -> None:
    """The error-status fixture is surfaced in the report, not dropped."""
    fixture_dir = tmp_path / "results"
    _write_fixtures(fixture_dir)
    out_dir = tmp_path / "out"
    report_path = tmp_path / "report.md"

    recovery_matrix_aggregate.aggregate(
        results_glob=str(fixture_dir / "*.json"),
        out_dir=out_dir,
        report_path=report_path,
    )

    report = report_path.read_text(encoding="utf-8")
    # The errored cell's identity must appear under the no-silent-failures section.
    assert "latent_circuit" in report
    assert "underflow in dt 0.0" in report
    # And it must be in the CSV with status=error.
    csv_text = (out_dir / "recovery_matrix.csv").read_text(encoding="utf-8")
    assert "error" in csv_text


def test_empty_results_fails_loud(tmp_path: Path) -> None:
    """Pointing the glob at an empty dir fails loud (expected-vs-actual)."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="expected >= 1 file"):
        recovery_matrix_aggregate.aggregate(
            results_glob=str(empty_dir / "*.json"),
            out_dir=tmp_path / "out",
            report_path=tmp_path / "report.md",
        )
