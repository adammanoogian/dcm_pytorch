"""Tests for relative-ranking BMR and VL posterior tempering.

Validates :func:`rank_connections` (relative single-prune ranking with a
separation gap) and :func:`temper_vl_posterior` (temperature scaling with a
Cholesky positive-definiteness guard) on a small, deterministic, analytically
constructed circuit. No VL fit is performed, so the tests are laptop-fast.
"""

from __future__ import annotations

import pytest
import torch

from pyro_dcm.model_selection import rank_connections, temper_vl_posterior

# -- Helpers ---------------------------------------------------------


def _known_circuit() -> dict:
    """Build a deterministic D=4 circuit with two present, two absent edges.

    Indices {0, 1} are truly present (posterior mean 2.0); indices {2, 3}
    are truly absent (posterior mean 0.0). The posterior covariance is a
    tight diagonal (variance 1e-3), trivially positive-definite. The prior
    is a unit-variance standard normal.
    """
    prior_mean = torch.zeros(4, dtype=torch.float64)
    prior_cov = torch.eye(4, dtype=torch.float64)
    posterior_mean = torch.tensor([2.0, 2.0, 0.0, 0.0], dtype=torch.float64)
    posterior_cov = torch.eye(4, dtype=torch.float64) * 1e-3
    return {
        "posterior_mean": posterior_mean,
        "posterior_cov": posterior_cov,
        "prior_mean": prior_mean,
        "prior_cov": prior_cov,
    }


# -- rank_connections ------------------------------------------------


@pytest.mark.vl
class TestRankConnections:
    """Test suite for :func:`rank_connections`."""

    def test_rank_connections_runs_K_calls_and_orders_present_first(
        self,
    ) -> None:
        """Present edges rank above absent edges over K=4 single-prune calls."""
        circuit = _known_circuit()
        result = rank_connections(**circuit, prunable_indices=[0, 1, 2, 3])

        assert len(result["ranked"]) == 4
        top_two = {result["ranked"][0]["index"], result["ranked"][1]["index"]}
        bottom_two = {
            result["ranked"][2]["index"],
            result["ranked"][3]["index"],
        }
        assert top_two == {0, 1}, (
            f"expected present edges {{0, 1}} in top two ranks; got {top_two}"
        )
        assert bottom_two == {2, 3}, (
            f"expected absent edges {{2, 3}} in bottom two ranks; "
            f"got {bottom_two}"
        )
        for entry in result["ranked"]:
            assert set(entry.keys()) >= {
                "index",
                "prune_delta_f",
                "rank",
                "gap_to_next",
            }

    def test_separation_gap_positive_and_cut_between_present_and_absent(
        self,
    ) -> None:
        """Separation gap is positive and cuts after the 2 essential edges."""
        circuit = _known_circuit()
        result = rank_connections(**circuit, prunable_indices=[0, 1, 2, 3])

        assert result["separation_gap"] > 0, (
            f"expected positive separation_gap; got {result['separation_gap']}"
        )
        assert result["separation_after_rank"] == 2, (
            f"expected cut after rank 2 (essential|inessential); "
            f"got {result['separation_after_rank']}"
        )

    def test_rank_connections_empty_indices_raises(self) -> None:
        """Empty prunable_indices raises ValueError."""
        circuit = _known_circuit()
        with pytest.raises(ValueError, match="requires >=1 prunable index"):
            rank_connections(**circuit, prunable_indices=[])


# -- temper_vl_posterior ---------------------------------------------


@pytest.mark.vl
class TestTemperVlPosterior:
    """Test suite for :func:`temper_vl_posterior`."""

    def test_temper_inflates_and_preserves_pd(self) -> None:
        """Tempering scales the diagonal and stays positive-definite."""
        sigma = torch.diag(
            torch.tensor([0.01, 0.02, 0.03], dtype=torch.float64)
        )
        tempered = temper_vl_posterior(sigma, tempering_factor=5.0)

        torch.testing.assert_close(
            tempered.diagonal(),
            5.0 * sigma.diagonal(),
            atol=1e-12,
            rtol=0,
        )
        # Cholesky succeeds (already asserted inside, re-check explicitly).
        torch.linalg.cholesky(tempered)

    def test_temper_non_pd_raises_loud(self) -> None:
        """A non-PD input raises ValueError naming shape and factor."""
        non_pd = torch.diag(
            torch.tensor([1.0, -2.0, 1.0], dtype=torch.float64)
        )
        with pytest.raises(ValueError) as exc_info:
            temper_vl_posterior(non_pd, tempering_factor=3.0)

        message = str(exc_info.value)
        assert "(3, 3)" in message, (
            f"expected matrix shape (3, 3) in message; got: {message}"
        )
        assert "tempering_factor=3.0" in message, (
            f"expected tempering_factor=3.0 in message; got: {message}"
        )
