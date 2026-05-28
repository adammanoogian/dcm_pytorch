"""Unit tests for cross-modal A-matrix comparison metrics."""

from __future__ import annotations

import numpy as np
import pytest

from pyro_dcm.foundation.comparison import (
    compute_credible_interval_overlap,
    compute_pearson_correlation,
    compute_sign_kappa,
    normalize_a_matrix,
)


class TestNormalizeAMatrix:
    """Tests for Frobenius-norm normalization."""

    def test_normalize_a_matrix_unit_norm(self) -> None:
        """Normalized matrix has Frobenius norm 1.0."""
        rng = np.random.default_rng(42)
        a = rng.standard_normal((4, 4))
        a_norm = normalize_a_matrix(a)
        assert np.linalg.norm(a_norm, "fro") == pytest.approx(
            1.0, abs=1e-10
        )

    def test_normalize_preserves_sign_pattern(self) -> None:
        """Normalization preserves the sign of every element."""
        rng = np.random.default_rng(123)
        a = rng.standard_normal((4, 4))
        a_norm = normalize_a_matrix(a)
        np.testing.assert_array_equal(np.sign(a), np.sign(a_norm))

    def test_normalize_zero_matrix_raises(self) -> None:
        """Zero matrix raises ValueError."""
        a = np.zeros((3, 3))
        with pytest.raises(ValueError, match="zero Frobenius norm"):
            normalize_a_matrix(a)


class TestPearsonCorrelation:
    """Tests for Pearson r between A matrices."""

    def test_pearson_identical_matrices(self) -> None:
        """Identical matrices have r = 1.0."""
        rng = np.random.default_rng(42)
        a = rng.standard_normal((4, 4))
        r = compute_pearson_correlation(a, a)
        assert r == pytest.approx(1.0, abs=1e-10)

    def test_pearson_uncorrelated_matrices(self) -> None:
        """Orthogonal-structured matrices have low |r|."""
        # Upper triangular (excluding diagonal)
        a1 = np.array(
            [
                [0.0, 1.0, 2.0, 3.0],
                [0.0, 0.0, 4.0, 5.0],
                [0.0, 0.0, 0.0, 6.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        # Lower triangular (excluding diagonal) with different values
        a2 = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [7.0, 0.0, 0.0, 0.0],
                [8.0, 9.0, 0.0, 0.0],
                [10.0, 11.0, 12.0, 0.0],
            ]
        )
        r = compute_pearson_correlation(a1, a2)
        assert abs(r) < 0.5

    def test_pearson_shape_mismatch_raises(self) -> None:
        """Mismatched shapes raise ValueError."""
        a1 = np.zeros((3, 3))
        a2 = np.zeros((4, 4))
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_pearson_correlation(a1, a2)


class TestSignKappa:
    """Tests for sign-pattern Cohen's kappa."""

    def test_sign_kappa_perfect_agreement(self) -> None:
        """Matrices with same sign pattern but different magnitudes."""
        a1 = np.array(
            [[1.0, -2.0], [3.0, -4.0]]
        )
        a2 = np.array(
            [[5.0, -1.0], [0.5, -10.0]]
        )
        kappa = compute_sign_kappa(a1, a2)
        assert kappa == pytest.approx(1.0, abs=1e-10)

    def test_sign_kappa_random_disagreement(self) -> None:
        """Opposite sign patterns yield negative kappa."""
        a1 = np.array(
            [[1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, 1.0]]
        )
        a2 = -a1
        kappa = compute_sign_kappa(a1, a2)
        assert kappa < 0.0


class TestCredibleIntervalOverlap:
    """Tests for CI overlap fraction."""

    def test_ci_overlap_identical(self) -> None:
        """Identical posteriors have 100% overlap."""
        mean = np.array([[1.0, -0.5], [0.3, -0.2]])
        std = 0.1 * np.ones_like(mean)
        overlap = compute_credible_interval_overlap(
            mean, std, mean, std
        )
        assert overlap == pytest.approx(1.0, abs=1e-10)

    def test_ci_overlap_separated(self) -> None:
        """Well-separated posteriors have 0% overlap."""
        a1_mean = np.zeros((3, 3))
        a2_mean = 10.0 * np.ones((3, 3))
        std = 0.1 * np.ones((3, 3))
        overlap = compute_credible_interval_overlap(
            a1_mean, std, a2_mean, std
        )
        assert overlap == pytest.approx(0.0, abs=1e-10)

    def test_ci_overlap_shape_mismatch_raises(self) -> None:
        """Mismatched shapes raise ValueError."""
        with pytest.raises(ValueError, match="same shape"):
            compute_credible_interval_overlap(
                np.zeros((2, 2)),
                np.zeros((2, 2)),
                np.zeros((3, 3)),
                np.zeros((3, 3)),
            )
