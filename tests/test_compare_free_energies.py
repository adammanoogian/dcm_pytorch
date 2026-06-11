"""Unit tests for the strict-5% matched free-energy comparator.

Covers ``compare_free_energies`` (VLSPM-02): the relative-tolerance gate on
the SAME matched problem, treated as a HARD pass/fail criterion. Also pins the
S3 boundary by asserting that cross-model agreement is computed via the
existing ``compare_model_ranking`` (relative ranking), never via absolute F.

All tests are fast and laptop-only (no MATLAB).
"""

from __future__ import annotations

import math

import pytest

from validation.compare_results import (
    compare_free_energies,
    compare_model_ranking,
)

pytestmark = pytest.mark.vl


def test_within_tolerance_passes() -> None:
    """VL F within 5% of SPM F on the matched problem passes the gate."""
    result = compare_free_energies(-1000.0, -1010.0)
    assert result["within_tolerance"] is True
    assert result["relative_error"] == pytest.approx(10.0 / 1010.0)


def test_outside_tolerance_fails() -> None:
    """VL F more than 5% from SPM F fails the strict gate."""
    result = compare_free_energies(-1000.0, -1200.0)
    assert result["within_tolerance"] is False
    assert result["relative_error"] == pytest.approx(200.0 / 1200.0)


def test_custom_tolerance() -> None:
    """A looser ``rel_tolerance`` admits the previously-failing case."""
    result = compare_free_energies(-1000.0, -1200.0, rel_tolerance=0.20)
    assert result["within_tolerance"] is True
    assert result["rel_tolerance"] == pytest.approx(0.20)


def test_zero_spm_F_no_div_by_zero() -> None:
    """A zero SPM F does not raise and yields a finite relative error."""
    result = compare_free_energies(0.5, 0.0)
    assert math.isfinite(result["relative_error"])
    assert result["relative_error"] > 0.0


def test_cross_model_ranking_is_separate_path() -> None:
    """Cross-model agreement uses ``compare_model_ranking``, not absolute F.

    Pitfall S3: comparing absolute free energies across DIFFERENT models is
    forbidden. ``compare_free_energies`` is single-problem-only; the cross-model
    criterion is the RELATIVE ranking computed here. This test documents that
    boundary in code.
    """
    scenarios = [
        {"spm_F": -10.0, "pyro_elbo": -10.0},
        {"spm_F": -20.0, "pyro_elbo": -25.0},
    ]
    result = compare_model_ranking(scenarios)
    assert result["agreement_rate"] == pytest.approx(1.0)
