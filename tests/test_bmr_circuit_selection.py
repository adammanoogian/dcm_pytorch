"""Tests for BMR circuit-size selection.

Validates :func:`enumerate_reduced_models` and
:func:`bmr_circuit_selection` from
``pyro_dcm.model_selection.bmr``.
"""

from __future__ import annotations

import pytest
import torch

from pyro_dcm.model_selection.bmr import (
    bmr_circuit_selection,
    enumerate_reduced_models,
)


class TestEnumerateReducedModels:
    """Test suite for :func:`enumerate_reduced_models`."""

    def test_enumerate_count(self) -> None:
        """k=3 prunable indices produces 2^3 - 1 = 7 candidates."""
        dim = 5
        prior_mean = torch.zeros(dim, dtype=torch.float64)
        prior_cov = torch.eye(dim, dtype=torch.float64)

        candidates = enumerate_reduced_models(
            prior_mean,
            prior_cov,
            prunable_indices=[0, 2, 4],
        )

        assert len(candidates) == 7

    def test_enumerate_labels(self) -> None:
        """Verify labels and pruned_indices tuples for k=2."""
        dim = 4
        prior_mean = torch.zeros(dim, dtype=torch.float64)
        prior_cov = torch.eye(dim, dtype=torch.float64)

        candidates = enumerate_reduced_models(
            prior_mean,
            prior_cov,
            prunable_indices=[1, 3],
        )

        # k=2 -> 3 candidates: prune(1), prune(3), prune(1,3)
        assert len(candidates) == 3

        labels = [c["label"] for c in candidates]
        pruned = [c["pruned_indices"] for c in candidates]

        assert "prune(1)" in labels
        assert "prune(3)" in labels
        assert "prune(1,3)" in labels

        assert (1,) in pruned
        assert (3,) in pruned
        assert (1, 3) in pruned

        # Check n_pruned values
        for c in candidates:
            assert c["n_pruned"] == len(c["pruned_indices"])

    def test_enumerate_too_many_raises(self) -> None:
        """k=21 raises ValueError due to exponential cost."""
        dim = 25
        prior_mean = torch.zeros(dim, dtype=torch.float64)
        prior_cov = torch.eye(dim, dtype=torch.float64)

        with pytest.raises(ValueError, match="21 prunable indices"):
            enumerate_reduced_models(
                prior_mean,
                prior_cov,
                prunable_indices=list(range(21)),
            )


class TestBMRCircuitSelection:
    """Test suite for :func:`bmr_circuit_selection`."""

    def test_circuit_selection_identifies_sparse_truth(self) -> None:
        """Best model prunes the two zero-valued parameters.

        Ground truth: params = [0.5, 0.0, 0.3, 0.0, 0.8].
        The posterior is centred on these values with tight covariance.
        The best reduced model should prune indices {1, 3}.
        """
        dim = 5
        true_params = torch.tensor(
            [0.5, 0.0, 0.3, 0.0, 0.8], dtype=torch.float64
        )

        # Simulate tight posterior around ground truth
        posterior_mean = true_params.clone()
        posterior_cov = 0.05 * torch.eye(dim, dtype=torch.float64)

        # Broad prior
        prior_mean = torch.zeros(dim, dtype=torch.float64)
        prior_cov = torch.eye(dim, dtype=torch.float64)

        result = bmr_circuit_selection(
            posterior_mean,
            posterior_cov,
            prior_mean,
            prior_cov,
            prunable_indices=[0, 1, 2, 3, 4],
        )

        best = result["best"]
        assert set(best["pruned_indices"]) == {1, 3}, (
            f"Expected best to prune {{1, 3}}, "
            f"got {set(best['pruned_indices'])}"
        )

    def test_circuit_selection_full_model_included(self) -> None:
        """Full model appears with delta_log_evidence = 0."""
        dim = 3
        posterior_mean = torch.randn(dim, dtype=torch.float64)
        posterior_cov = 0.1 * torch.eye(dim, dtype=torch.float64)
        prior_mean = torch.zeros(dim, dtype=torch.float64)
        prior_cov = torch.eye(dim, dtype=torch.float64)

        result = bmr_circuit_selection(
            posterior_mean,
            posterior_cov,
            prior_mean,
            prior_cov,
            prunable_indices=[0, 1],
        )

        # Find full model entry
        full_entries = [
            r
            for r in result["results"]
            if r["pruned_indices"] == ()
        ]
        assert len(full_entries) == 1
        assert full_entries[0]["delta_log_evidence"] == 0.0
        assert full_entries[0]["label"] == "full_model"

    def test_circuit_selection_result_structure(self) -> None:
        """Result dict contains all required keys and sub-keys."""
        dim = 4
        posterior_mean = torch.randn(dim, dtype=torch.float64)
        posterior_cov = 0.1 * torch.eye(dim, dtype=torch.float64)
        prior_mean = torch.zeros(dim, dtype=torch.float64)
        prior_cov = torch.eye(dim, dtype=torch.float64)

        result = bmr_circuit_selection(
            posterior_mean,
            posterior_cov,
            prior_mean,
            prior_cov,
            prunable_indices=[1, 2],
        )

        # Top-level keys
        assert "results" in result
        assert "best" in result
        assert "full_model_rank" in result
        assert "n_candidates" in result
        assert "prunable_indices" in result

        # n_candidates = 2^2 - 1 reduced + 1 full = 4
        assert result["n_candidates"] == 4

        # Each result entry has required keys
        required_keys = {
            "pruned_indices",
            "delta_log_evidence",
            "n_pruned",
            "label",
            "reduced_posterior_mean",
            "reduced_posterior_cov",
        }
        for entry in result["results"]:
            assert required_keys.issubset(entry.keys()), (
                f"Missing keys: {required_keys - entry.keys()}"
            )

        # Results sorted descending by delta_log_evidence
        evidences = [
            r["delta_log_evidence"] for r in result["results"]
        ]
        for i in range(len(evidences) - 1):
            assert evidences[i] >= evidences[i + 1], (
                f"Results not sorted: {evidences}"
            )

    def test_circuit_selection_single_prunable(self) -> None:
        """k=1 produces exactly 2 candidates (1 reduced + full)."""
        dim = 3
        posterior_mean = torch.randn(dim, dtype=torch.float64)
        posterior_cov = 0.1 * torch.eye(dim, dtype=torch.float64)
        prior_mean = torch.zeros(dim, dtype=torch.float64)
        prior_cov = torch.eye(dim, dtype=torch.float64)

        result = bmr_circuit_selection(
            posterior_mean,
            posterior_cov,
            prior_mean,
            prior_cov,
            prunable_indices=[2],
        )

        assert result["n_candidates"] == 2
        labels = {r["label"] for r in result["results"]}
        assert labels == {"prune(2)", "full_model"}
