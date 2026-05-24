"""Tests for latent extraction and PCA dimensionality reduction (DIM-01/02/03).

Tests PCA reduction, variance-explained diagnostic, output-R-squared gate,
and z-score normalization. Marked with @pytest.mark.latent where scikit-learn
is required.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.latent
class TestPcaReduce:
    """Tests for pca_reduce()."""

    def test_pca_reduce_shape(self) -> None:
        """pca_reduce returns projected array of shape (n_samples, n_components)."""
        from pyro_dcm.rnn import pca_reduce

        rng = np.random.default_rng(0)
        h = rng.standard_normal((500, 32)).astype(np.float32)
        pca, projected = pca_reduce(h, n_components=5)
        assert projected.shape == (500, 5), (
            f"Projected shape {projected.shape} != (500, 5)"
        )
        assert hasattr(pca, "explained_variance_ratio_"), (
            "PCA object missing explained_variance_ratio_"
        )
        assert len(pca.explained_variance_ratio_) == 5, (
            f"Expected 5 variance ratios, got {len(pca.explained_variance_ratio_)}"
        )

    def test_pca_reduce_variance_positive(self) -> None:
        """All explained variance ratios are non-negative and sum <= 1."""
        from pyro_dcm.rnn import pca_reduce

        rng = np.random.default_rng(1)
        h = rng.standard_normal((200, 16)).astype(np.float32)
        pca, _ = pca_reduce(h, n_components=8)
        ratios = pca.explained_variance_ratio_
        assert np.all(ratios >= 0), f"Negative variance ratios found: {ratios}"
        assert ratios.sum() <= 1.0 + 1e-6, (
            f"Sum of variance ratios {ratios.sum():.4f} exceeds 1"
        )

    def test_pca_reduce_components_attribute(self) -> None:
        """pca.components_ has shape (n_components, H)."""
        from pyro_dcm.rnn import pca_reduce

        rng = np.random.default_rng(2)
        h = rng.standard_normal((300, 20)).astype(np.float32)
        pca, _ = pca_reduce(h, n_components=4)
        assert pca.components_.shape == (4, 20), (
            f"components_ shape {pca.components_.shape} != (4, 20)"
        )


@pytest.mark.latent
class TestVarianceExplainedDiagnostic:
    """Tests for variance_explained_diagnostic()."""

    def test_diagnostic_monotone_cumulative(self) -> None:
        """Cumulative variance is monotonically non-decreasing."""
        from pyro_dcm.rnn import pca_reduce, variance_explained_diagnostic

        rng = np.random.default_rng(10)
        # Create 3-factor data embedded in 32-dim space
        factors = rng.standard_normal((500, 3))
        projection = rng.standard_normal((3, 32))
        h = (factors @ projection + 0.01 * rng.standard_normal((500, 32))).astype(
            np.float32
        )
        pca, _ = pca_reduce(h, n_components=10)
        diag = variance_explained_diagnostic(pca)

        cumulative = diag["cumulative"]
        assert np.all(np.diff(cumulative) >= -1e-10), (
            f"Cumulative variance not monotone: {cumulative}"
        )

    def test_diagnostic_marginal_nonneg(self) -> None:
        """Marginal variance ratios are non-negative."""
        from pyro_dcm.rnn import pca_reduce, variance_explained_diagnostic

        rng = np.random.default_rng(11)
        h = rng.standard_normal((300, 16)).astype(np.float32)
        pca, _ = pca_reduce(h, n_components=8)
        diag = variance_explained_diagnostic(pca)
        assert np.all(diag["marginal"] >= 0), (
            f"Marginal variance ratios contain negatives: {diag['marginal']}"
        )

    def test_diagnostic_recommended_n_positive_int(self) -> None:
        """recommended_n is a positive integer."""
        from pyro_dcm.rnn import pca_reduce, variance_explained_diagnostic

        rng = np.random.default_rng(12)
        h = rng.standard_normal((400, 24)).astype(np.float32)
        pca, _ = pca_reduce(h, n_components=12)
        diag = variance_explained_diagnostic(pca)
        assert isinstance(diag["recommended_n"], int), (
            f"recommended_n type {type(diag['recommended_n'])} != int"
        )
        assert diag["recommended_n"] >= 1, (
            f"recommended_n {diag['recommended_n']} < 1"
        )

    def test_diagnostic_recommended_n_low_rank_data(self) -> None:
        """recommended_n is small for strongly low-rank data.

        A 2-factor model should recommend N <= 5 (the 3rd+ components
        contribute very little variance in clean 2-factor data).
        """
        from pyro_dcm.rnn import pca_reduce, variance_explained_diagnostic

        rng = np.random.default_rng(13)
        factors = rng.standard_normal((1000, 2))
        projection = rng.standard_normal((2, 32))
        # Very low noise so top-2 PCs dominate
        h = (factors @ projection + 1e-4 * rng.standard_normal((1000, 32))).astype(
            np.float32
        )
        pca, _ = pca_reduce(h, n_components=10)
        diag = variance_explained_diagnostic(pca)
        # Components beyond index 2 should be < 5%, so recommended <= 3
        assert diag["recommended_n"] <= 3, (
            f"Expected recommended_n <= 3 for 2-factor data, "
            f"got {diag['recommended_n']}"
        )


@pytest.mark.latent
class TestOutputRSquaredGate:
    """Tests for output_r_squared_gate()."""

    def test_gate_passes_sufficient_components(self) -> None:
        """Gate passes (R2 >= 0.90) when PCA captures all task-relevant variance.

        Construct data from a 3-dim latent embedded in H=32 space.
        PCA with N=3 captures all variance; output readout should reconstruct
        perfectly, giving R2 close to 1.0.
        """
        from pyro_dcm.rnn import output_r_squared_gate, pca_reduce

        rng = np.random.default_rng(20)
        n_samples = 500
        H = 32
        N_latent = 3
        act_size = 2

        # Generate latent structure
        factors = rng.standard_normal((n_samples, N_latent)).astype(np.float32)
        embed = rng.standard_normal((N_latent, H)).astype(np.float32)
        h_all = factors @ embed  # (n_samples, H) -- exactly 3-dim

        # Output weights
        w_out = rng.standard_normal((act_size, H)).astype(np.float32)
        z_true = h_all @ w_out.T  # (n_samples, act_size)

        # PCA with enough components to capture the full variance
        pca, h_projected = pca_reduce(h_all, n_components=N_latent)

        result = output_r_squared_gate(h_projected, z_true, w_out, pca, threshold=0.90)
        assert result["passed"] is True, (
            f"Expected gate to pass with N=3 for 3-factor data. "
            f"R2={result['r_squared']:.4f}"
        )
        assert result["r_squared"] >= 0.90, (
            f"R2={result['r_squared']:.4f} < 0.90"
        )

    def test_gate_fails_insufficient_components(self) -> None:
        """Gate fails (R2 < 0.90) when PCA uses too few components.

        Constructs data from 3 independent orthogonal factors with equal
        variance, embedded in H=32 space. The output readout uses all 3
        factors equally. PCA with N=1 captures only ~33% of the signal,
        so R2 < 0.90.
        """
        from pyro_dcm.rnn import output_r_squared_gate, pca_reduce

        rng = np.random.default_rng(50)
        n_samples = 1000
        H = 32
        N_latent = 3
        act_size = 2

        # 3 truly independent, equal-variance latent factors
        factors = rng.standard_normal((n_samples, N_latent)).astype(np.float32)

        # 3 orthogonal embedding vectors of unit norm (ensures equal variance)
        # Use fixed orthogonal basis vectors projected into H-dim space
        basis = np.zeros((N_latent, H), dtype=np.float32)
        basis[0, :H // 3] = 1.0 / np.sqrt(H // 3)
        basis[1, H // 3 : 2 * H // 3] = 1.0 / np.sqrt(H // 3)
        basis[2, 2 * H // 3 :] = 1.0 / np.sqrt(H - 2 * H // 3)
        h_all = factors @ basis  # (n_samples, H) -- exactly 3-dim, equal variance

        # Output weights that depend equally on all 3 latent directions
        # w_out selects the 3 latent factors equally
        w_out_latent = np.ones((act_size, N_latent), dtype=np.float32)
        w_out_latent[0, :] = [1.0, 0.0, 0.0]  # output 0 depends on factor 0
        w_out_latent[1, :] = [0.0, 1.0, 1.0]  # output 1 depends on factors 1+2
        # Map back to H-dim: w_out = w_out_latent @ basis
        w_out = w_out_latent @ basis  # (act_size, H)
        z_true = h_all @ w_out.T  # (n_samples, act_size)

        # PCA with N=1: can only capture 1/3 of the variance
        # (all 3 components have equal variance so PC1 ~ 33%)
        pca, h_projected = pca_reduce(h_all, n_components=1)

        result = output_r_squared_gate(h_projected, z_true, w_out, pca, threshold=0.90)
        assert result["passed"] is False, (
            f"Expected gate to fail with N=1 for 3-factor equal-variance data. "
            f"R2={result['r_squared']:.4f} (PC1 variance: "
            f"{pca.explained_variance_ratio_[0]:.3f})"
        )

    def test_gate_result_keys(self) -> None:
        """output_r_squared_gate returns dict with required keys and types."""
        from pyro_dcm.rnn import output_r_squared_gate, pca_reduce

        rng = np.random.default_rng(22)
        h = rng.standard_normal((100, 8)).astype(np.float32)
        w_out = rng.standard_normal((2, 8)).astype(np.float32)
        z_true = h @ w_out.T
        pca, h_proj = pca_reduce(h, n_components=4)

        result = output_r_squared_gate(h_proj, z_true, w_out, pca)
        assert "r_squared" in result
        assert "passed" in result
        assert "threshold" in result
        assert isinstance(result["r_squared"], float)
        assert isinstance(result["passed"], bool)
        assert isinstance(result["threshold"], float)


@pytest.mark.latent
class TestZscoreTrajectories:
    """Tests for zscore_trajectories()."""

    def test_zscore_zero_mean_unit_std(self) -> None:
        """Z-scored output has mean ~0 and std ~1 per column."""
        from pyro_dcm.rnn import zscore_trajectories

        rng = np.random.default_rng(30)
        # Data with non-trivial mean and std
        h = rng.standard_normal((1000, 5)).astype(np.float32)
        h = h * np.array([2.0, 5.0, 0.5, 10.0, 1.0]) + np.array(
            [3.0, -1.0, 7.0, 0.0, -4.0]
        )

        z_scored, means, stds = zscore_trajectories(h)

        col_means = z_scored.mean(axis=0)
        col_stds = z_scored.std(axis=0)
        np.testing.assert_allclose(col_means, 0.0, atol=1e-5, err_msg="Mean != 0")
        np.testing.assert_allclose(col_stds, 1.0, atol=1e-5, err_msg="Std != 1")

    def test_zscore_returns_original_stats(self) -> None:
        """Returned means and stds match the original column statistics."""
        from pyro_dcm.rnn import zscore_trajectories

        rng = np.random.default_rng(31)
        h = rng.standard_normal((500, 4)).astype(np.float32) * 3.0 + 2.0

        _, means, stds = zscore_trajectories(h)
        expected_means = h.mean(axis=0)
        expected_stds = h.std(axis=0)
        np.testing.assert_allclose(means, expected_means, atol=1e-5)
        np.testing.assert_allclose(stds, expected_stds, atol=1e-5)

    def test_zscore_inverse_recovers_original(self) -> None:
        """Inverse z-score (z * stds + means) recovers the original data."""
        from pyro_dcm.rnn import zscore_trajectories

        rng = np.random.default_rng(32)
        h = rng.standard_normal((300, 6)).astype(np.float32) * 4.0 - 1.5

        z_scored, means, stds = zscore_trajectories(h)
        h_reconstructed = z_scored * stds + means
        np.testing.assert_allclose(
            h_reconstructed,
            h,
            atol=1e-5,
            err_msg="Inverse z-score does not recover original data",
        )

    def test_zscore_near_zero_std_handled(self) -> None:
        """Near-constant columns do not cause division by zero."""
        from pyro_dcm.rnn import zscore_trajectories

        rng = np.random.default_rng(33)
        # First column is constant (zero std)
        h = rng.standard_normal((200, 3)).astype(np.float32)
        h[:, 0] = 5.0  # constant column

        z_scored, means, stds = zscore_trajectories(h)
        # Should not raise; constant column z-scores to 0 everywhere
        assert np.all(np.isfinite(z_scored)), "z_scored contains NaN or Inf"
        assert stds[0] >= 1e-8, f"std clipped incorrectly: {stds[0]}"
