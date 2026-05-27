"""Tests for Bayesian Model Reduction (BMR).

Validates the core BMR function [REF-070] and helper utilities in
``pyro_dcm.model_selection.bmr``.
"""

from __future__ import annotations

import math

import pytest
import torch

from pyro_dcm.model_selection.bmr import (
    bayesian_model_reduction,
    make_reduced_prior_zero_connection,
)


# -- Helpers ---------------------------------------------------------


def _make_spd(dim: int, seed: int = 0) -> torch.Tensor:
    """Generate a random symmetric positive-definite matrix."""
    rng = torch.Generator().manual_seed(seed)
    L = torch.randn(dim, dim, generator=rng, dtype=torch.float64)
    return L @ L.T + torch.eye(dim, dtype=torch.float64)


# -- Tests -----------------------------------------------------------


class TestBayesianModelReduction:
    """Test suite for :func:`bayesian_model_reduction`."""

    def test_identical_priors_zero_delta(self) -> None:
        """Delta F should be zero when reduced prior == full prior."""
        dim = 4
        mu_f = torch.randn(dim, dtype=torch.float64)
        sigma_f = _make_spd(dim, seed=1)
        mu_0 = torch.randn(dim, dtype=torch.float64)
        sigma_0 = _make_spd(dim, seed=2)

        delta_f, mu_r, sigma_r = bayesian_model_reduction(
            posterior_mean=mu_f,
            posterior_cov=sigma_f,
            prior_mean=mu_0,
            prior_cov=sigma_0,
            reduced_prior_mean=mu_0,
            reduced_prior_cov=sigma_0,
        )

        assert delta_f == pytest.approx(0.0, abs=1e-10)
        torch.testing.assert_close(mu_r, mu_f, atol=1e-10, rtol=0)
        torch.testing.assert_close(sigma_r, sigma_f, atol=1e-10, rtol=0)

    def test_analytic_1d_case(self) -> None:
        """Compare against closed-form 1-D Gaussian Bayes factor.

        For 1-D Gaussians the log Bayes factor between the reduced and
        full models can be verified with explicit scalar algebra.
        """
        # Full prior: N(0, 1)
        mu_0 = torch.tensor([0.0], dtype=torch.float64)
        sigma_0 = torch.tensor([[1.0]], dtype=torch.float64)

        # Full posterior: N(0.5, 0.2)
        mu_f = torch.tensor([0.5], dtype=torch.float64)
        sigma_f = torch.tensor([[0.2]], dtype=torch.float64)

        # Reduced prior: N(0, 0.01)  -- very tight around zero
        mu_r0 = torch.tensor([0.0], dtype=torch.float64)
        sigma_r0 = torch.tensor([[0.01]], dtype=torch.float64)

        delta_f, mu_r, sigma_r = bayesian_model_reduction(
            mu_f,
            sigma_f,
            mu_0,
            sigma_0,
            mu_r0,
            sigma_r0,
        )

        # Manual 1-D computation
        pf = 1.0 / 0.2  # posterior precision
        p0 = 1.0 / 1.0  # prior precision
        pr0 = 1.0 / 0.01  # reduced prior precision

        pr_post = pf + pr0 - p0  # reduced posterior precision
        sr_post = 1.0 / pr_post  # reduced posterior variance

        mu_r_expected = sr_post * (pf * 0.5 + pr0 * 0.0 - p0 * 0.0)

        # delta_F = 0.5 * [log|Sigma_r| - log|Sigma_f|
        #                + log|Sigma_0| - log|Sigma_r0|
        #                - (mu_f - mu_r0)' P_r0 (mu_f - mu_r0)
        #                + (mu_f - mu_0)' P_0 (mu_f - mu_0)]
        logdet_term = math.log(sr_post) - math.log(0.2) + math.log(1.0) - math.log(0.01)
        quad_term = -pr0 * (0.5 - 0.0) ** 2 + p0 * (0.5 - 0.0) ** 2
        expected_delta = 0.5 * (logdet_term + quad_term)

        assert delta_f == pytest.approx(expected_delta, abs=1e-10)
        assert mu_r.item() == pytest.approx(mu_r_expected, abs=1e-10)
        assert sigma_r.item() == pytest.approx(sr_post, abs=1e-10)

    def test_tight_reduction_on_correct_zero(self) -> None:
        """Reducing a near-zero parameter with moderately tight prior helps.

        When the full posterior already has a near-zero mean for a
        parameter, shrinking it with a moderately tight reduced prior
        should yield delta F > 0 (Occam factor compensates for small
        quadratic penalty).
        """
        dim = 3
        # Posterior: parameter 0 is near zero
        mu_f = torch.tensor([0.01, 1.5, -0.8], dtype=torch.float64)
        sigma_f = 0.1 * torch.eye(dim, dtype=torch.float64)

        # Full prior: broad
        mu_0 = torch.zeros(dim, dtype=torch.float64)
        sigma_0 = torch.eye(dim, dtype=torch.float64)

        # Reduced prior: shrink parameter 0 with moderate variance
        mu_r0, sigma_r0 = make_reduced_prior_zero_connection(
            mu_0,
            sigma_0,
            indices=[0],
            shrinkage_variance=0.01,
        )

        delta_f, _, _ = bayesian_model_reduction(
            mu_f,
            sigma_f,
            mu_0,
            sigma_0,
            mu_r0,
            sigma_r0,
        )

        # Should be positive (reduced model preferred)
        assert delta_f > 0.0, (
            f"Expected positive delta F for near-zero parameter, got {delta_f:.6f}"
        )

    def test_tight_reduction_on_nonzero(self) -> None:
        """Reducing a clearly nonzero parameter should yield negative delta F.

        Shrinking a parameter whose posterior is far from zero should be
        strongly penalised, yielding a negative delta F.
        """
        dim = 3
        # Posterior: parameter 1 is clearly nonzero
        mu_f = torch.tensor([0.0, 3.0, 0.0], dtype=torch.float64)
        sigma_f = 0.1 * torch.eye(dim, dtype=torch.float64)

        mu_0 = torch.zeros(dim, dtype=torch.float64)
        sigma_0 = torch.eye(dim, dtype=torch.float64)

        # Reduced prior: shrink parameter 1
        mu_r0, sigma_r0 = make_reduced_prior_zero_connection(
            mu_0,
            sigma_0,
            indices=[1],
        )

        delta_f, _, _ = bayesian_model_reduction(
            mu_f,
            sigma_f,
            mu_0,
            sigma_0,
            mu_r0,
            sigma_r0,
        )

        # Should be strongly negative (full model preferred)
        assert delta_f < -1.0, (
            f"Expected strongly negative delta F for nonzero parameter, "
            f"got {delta_f:.6f}"
        )

    def test_reduced_posterior_shape(self) -> None:
        """Reduced posterior mean and covariance have correct shapes."""
        dim = 5
        mu_f = torch.randn(dim, dtype=torch.float64)
        sigma_f = 0.1 * torch.eye(dim, dtype=torch.float64)

        mu_0 = torch.zeros(dim, dtype=torch.float64)
        sigma_0 = torch.eye(dim, dtype=torch.float64)

        # Reduced prior is tighter than full prior (guarantees PD)
        mu_r0 = torch.zeros(dim, dtype=torch.float64)
        sigma_r0 = 0.5 * torch.eye(dim, dtype=torch.float64)

        _, mu_r, sigma_r = bayesian_model_reduction(
            mu_f,
            sigma_f,
            mu_0,
            sigma_0,
            mu_r0,
            sigma_r0,
        )

        assert mu_r.shape == (dim,)
        assert sigma_r.shape == (dim, dim)
        # Covariance should be symmetric
        torch.testing.assert_close(
            sigma_r,
            sigma_r.T,
            atol=1e-12,
            rtol=0,
        )

    def test_multidimensional_consistency(self) -> None:
        """BMR on independent blocks should equal sum of 1-D results.

        When priors and posterior are diagonal (independent parameters),
        the multi-dimensional delta F should equal the sum of per-
        parameter 1-D delta F values.
        """
        dim = 4
        # Diagonal (independent) case
        mu_f = torch.tensor([0.5, -1.0, 0.2, 0.8], dtype=torch.float64)
        diag_f = torch.tensor([0.1, 0.3, 0.2, 0.15], dtype=torch.float64)
        sigma_f = torch.diag(diag_f)

        mu_0 = torch.zeros(dim, dtype=torch.float64)
        diag_0 = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
        sigma_0 = torch.diag(diag_0)

        mu_r0 = torch.zeros(dim, dtype=torch.float64)
        diag_r0 = torch.tensor([0.01, 1.0, 0.01, 1.0], dtype=torch.float64)
        sigma_r0 = torch.diag(diag_r0)

        # Full multi-D BMR
        delta_f_multi, _, _ = bayesian_model_reduction(
            mu_f,
            sigma_f,
            mu_0,
            sigma_0,
            mu_r0,
            sigma_r0,
        )

        # Sum of 1-D BMRs
        delta_f_sum = 0.0
        for i in range(dim):
            df_i, _, _ = bayesian_model_reduction(
                mu_f[i : i + 1],
                sigma_f[i : i + 1, i : i + 1],
                mu_0[i : i + 1],
                sigma_0[i : i + 1, i : i + 1],
                mu_r0[i : i + 1],
                sigma_r0[i : i + 1, i : i + 1],
            )
            delta_f_sum += df_i

        assert delta_f_multi == pytest.approx(delta_f_sum, abs=1e-10)

    def test_symmetry_of_delta_f(self) -> None:
        """Delta F is antisymmetric when priors share the same covariance.

        When the full and reduced priors have the same covariance matrix
        (differing only in mean), the reduced posterior precision is the
        same in both directions. This makes the Laplace approximation
        exact with respect to antisymmetry:
        ``delta_F(A->B) = -delta_F(B->A)``.
        """
        dim = 3
        torch.manual_seed(42)
        mu_f = torch.randn(dim, dtype=torch.float64)
        sigma_f = 0.1 * torch.eye(dim, dtype=torch.float64)

        # Same covariance, different means
        shared_cov = 0.5 * torch.eye(dim, dtype=torch.float64)
        mu_a = torch.randn(dim, dtype=torch.float64)
        mu_b = torch.randn(dim, dtype=torch.float64)

        # A -> B
        delta_ab, _, _ = bayesian_model_reduction(
            mu_f,
            sigma_f,
            mu_a,
            shared_cov,
            mu_b,
            shared_cov,
        )

        # B -> A
        delta_ba, _, _ = bayesian_model_reduction(
            mu_f,
            sigma_f,
            mu_b,
            shared_cov,
            mu_a,
            shared_cov,
        )

        assert math.isfinite(delta_ab), f"delta_ab is not finite: {delta_ab}"
        assert math.isfinite(delta_ba), f"delta_ba is not finite: {delta_ba}"
        assert delta_ab == pytest.approx(-delta_ba, abs=1e-10)


class TestMakeReducedPriorZeroConnection:
    """Test suite for :func:`make_reduced_prior_zero_connection`."""

    def test_make_reduced_prior_zero_connection(self) -> None:
        """Reduced prior zeroes specified indices and shrinks variance."""
        dim = 4
        mu_0 = torch.tensor([0.5, -0.3, 0.8, 0.1], dtype=torch.float64)
        sigma_0 = _make_spd(dim, seed=30)

        indices = [1, 3]
        reduced_mean, reduced_cov = make_reduced_prior_zero_connection(
            mu_0,
            sigma_0,
            indices,
        )

        # Zeroed means
        assert reduced_mean[1].item() == 0.0
        assert reduced_mean[3].item() == 0.0
        # Preserved means
        assert reduced_mean[0].item() == mu_0[0].item()
        assert reduced_mean[2].item() == mu_0[2].item()

        # Shrunk variances
        assert reduced_cov[1, 1].item() == pytest.approx(1e-8, abs=1e-15)
        assert reduced_cov[3, 3].item() == pytest.approx(1e-8, abs=1e-15)

        # Cross-covariances zeroed
        for idx in indices:
            for j in range(dim):
                if j != idx:
                    assert reduced_cov[idx, j].item() == 0.0
                    assert reduced_cov[j, idx].item() == 0.0

        # Preserved block (indices 0, 2) unchanged
        for i in [0, 2]:
            for j in [0, 2]:
                assert reduced_cov[i, j].item() == pytest.approx(
                    sigma_0[i, j].item(),
                    abs=1e-15,
                )
