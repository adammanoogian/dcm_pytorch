"""Unit tests for benchmarks.metrics consolidated metric functions.

Validates compute_rmse, pearson_corr, compute_coverage_from_ci,
compute_coverage_from_samples, and compute_amortization_gap with
known-value tests.
"""

from __future__ import annotations

import math

import pytest
import torch

from benchmarks.metrics import (
    compute_amortization_gap,
    compute_coverage_from_ci,
    compute_coverage_from_samples,
    compute_rmse,
    pearson_corr,
)


class TestComputeRMSE:
    """Tests for compute_rmse."""

    def test_compute_rmse_identical(self) -> None:
        """RMSE of identical matrices is 0.0."""
        A = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64,
        )
        assert compute_rmse(A, A) == pytest.approx(0.0, abs=1e-10)

    def test_compute_rmse_known_value(self) -> None:
        """RMSE of identity vs zeros is sqrt(2/4) = sqrt(0.5)."""
        A_true = torch.eye(2, dtype=torch.float64)
        A_inferred = torch.zeros(2, 2, dtype=torch.float64)
        expected = math.sqrt(0.5)  # sqrt(2/4)
        assert compute_rmse(A_true, A_inferred) == pytest.approx(
            expected, rel=1e-6,
        )


class TestPearsonCorr:
    """Tests for pearson_corr."""

    def test_pearson_corr_perfect(self) -> None:
        """Perfectly correlated vectors have correlation 1.0."""
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        y = torch.tensor([2.0, 4.0, 6.0], dtype=torch.float64)
        assert pearson_corr(x, y) == pytest.approx(1.0, abs=1e-10)

    def test_pearson_corr_anticorrelated(self) -> None:
        """Perfectly anti-correlated vectors have correlation -1.0."""
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        y = torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64)
        assert pearson_corr(x, y) == pytest.approx(-1.0, abs=1e-10)

    def test_pearson_corr_zero(self) -> None:
        """Constant vector has zero correlation with any vector."""
        x = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        y = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        assert pearson_corr(x, y) == pytest.approx(0.0, abs=1e-10)


class TestCoverageFromCI:
    """Tests for compute_coverage_from_ci."""

    def test_coverage_from_ci_all_covered(self) -> None:
        """All true values inside bounds yields coverage 1.0."""
        A_true = torch.tensor(
            [[0.5, 0.5], [0.5, 0.5]], dtype=torch.float64,
        )
        A_lo = torch.zeros(2, 2, dtype=torch.float64)
        A_hi = torch.ones(2, 2, dtype=torch.float64)
        assert compute_coverage_from_ci(A_true, A_lo, A_hi) == pytest.approx(
            1.0, abs=1e-10,
        )

    def test_coverage_from_ci_none_covered(self) -> None:
        """All true values outside bounds yields coverage 0.0."""
        A_true = torch.tensor(
            [[2.0, 2.0], [2.0, 2.0]], dtype=torch.float64,
        )
        A_lo = torch.zeros(2, 2, dtype=torch.float64)
        A_hi = torch.ones(2, 2, dtype=torch.float64)
        assert compute_coverage_from_ci(A_true, A_lo, A_hi) == pytest.approx(
            0.0, abs=1e-10,
        )


class TestCoverageFromSamples:
    """Tests for compute_coverage_from_samples."""

    def test_coverage_from_samples_known(self) -> None:
        """Samples from N(0,1) with true value 0 -> ~90% coverage at 90% CI.

        Uses large sample size (10000) for stable estimate. Coverage
        should be close to the CI level (0.90) within tolerance.
        """
        torch.manual_seed(42)
        n_samples = 10000
        n_params = 20

        true_vals = torch.zeros(n_params, dtype=torch.float64)
        samples = torch.randn(
            n_samples, n_params, dtype=torch.float64,
        )

        coverage = compute_coverage_from_samples(
            true_vals, samples, ci_level=0.90,
        )
        # 90% CI coverage should be close to 0.90
        assert coverage == pytest.approx(0.90, abs=0.10)


class TestAmortizationGap:
    """Tests for compute_amortization_gap."""

    def test_amortization_gap(self) -> None:
        """Absolute and relative gap computed correctly."""
        elbo_svi = -100.0
        elbo_amortized = -120.0

        result = compute_amortization_gap(elbo_svi, elbo_amortized)

        expected_abs = -120.0 - (-100.0)  # = -20.0
        expected_rel = -20.0 / abs(-100.0)  # = -0.2

        assert result["absolute_gap"] == pytest.approx(
            expected_abs, abs=1e-10,
        )
        assert result["relative_gap"] == pytest.approx(
            expected_rel, abs=1e-10,
        )
