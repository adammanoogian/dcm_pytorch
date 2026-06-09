"""Unit tests for latent-circuit metric guards and benchmark ground truth.

Covers the Phase 20-05 methodology fixes (Tier A):

- ``compute_elbo_model_selection`` refuses cross-dimensional comparisons when
  given ``observed_element_counts`` (decision 20-05-D2).
- ``_build_ground_truth`` places all modulator epochs inside the training
  split (root cause 1 of the 20-05 acceptance failure).

These tests are fast (no SVI / no ODE fitting) and run on laptop.
"""

from __future__ import annotations

import pytest
import torch

from benchmarks.latent_circuit_metrics import compute_elbo_model_selection
from benchmarks.runners.latent_circuit_recovery import (
    _TRAIN_FRACTION,
    _build_ground_truth,
)


class TestElboSelectionGuard:
    """Guard against invalid cross-dimensional ELBO comparison (20-05-D2)."""

    def test_selects_min_loss_on_valid_input(self) -> None:
        """With no element counts, returns the lowest-loss candidate."""
        result = compute_elbo_model_selection(
            {2: 100.0, 3: 80.0, 4: 50.0, 5: 90.0}, true_n=4,
        )
        assert result["selected_n"] == 4
        assert result["correct"] is True
        assert result["best_loss"] == 50.0

    def test_equal_element_counts_pass(self) -> None:
        """Identical observed-element counts are a valid comparison."""
        result = compute_elbo_model_selection(
            {2: 100.0, 4: 50.0},
            true_n=4,
            observed_element_counts={2: 800, 4: 800},
        )
        assert result["selected_n"] == 4

    def test_differing_element_counts_raise(self) -> None:
        """Different observed dimensionality must fail loud, not pick N=2."""
        with pytest.raises(ValueError, match="identical observed data"):
            compute_elbo_model_selection(
                {2: 10.0, 4: 50.0, 6: 90.0},
                observed_element_counts={2: 400, 4: 800, 6: 1200},
            )

    def test_mismatched_keys_raise(self) -> None:
        """observed_element_counts keys must match elbo_dict keys."""
        with pytest.raises(ValueError, match="keys must match"):
            compute_elbo_model_selection(
                {2: 10.0, 4: 50.0},
                observed_element_counts={2: 800},
            )

    def test_empty_dict_raises(self) -> None:
        """Empty elbo_dict is an error."""
        with pytest.raises(ValueError, match="non-empty"):
            compute_elbo_model_selection({})


class TestModulatorInTrainingSplit:
    """All modulator epochs must fall inside the training split (root cause 1)."""

    @pytest.mark.parametrize("duration", [50.0, 100.0])
    def test_modulator_absent_from_test_segment(self, duration: float) -> None:
        """The held-out test segment [train_frac*D, D] sees no modulation.

        Regression for the 20-05 bug where epochs hardcoded for 100s leaked
        into (or were cut from) a 50s run, starving B of identifying signal.
        """
        gt = _build_ground_truth(seed=0, duration=duration)
        stim_mod = gt["stim_mod"]
        times = stim_mod.times.reshape(-1)
        values = stim_mod.values.reshape(times.shape[0], -1)

        train_boundary = _TRAIN_FRACTION * duration

        on_mask = values.abs().sum(dim=1) > 0
        # No modulator activity at or beyond the train/test boundary.
        leaked = on_mask & (times >= train_boundary)
        assert not bool(leaked.any()), (
            f"Modulator active in held-out test segment at t="
            f"{times[leaked].tolist()} (boundary {train_boundary}s, "
            f"duration {duration}s)."
        )
        # ...and the modulator IS active somewhere in training (B has signal).
        in_train = on_mask & (times < train_boundary)
        assert bool(in_train.any()), "Modulator never active in training split."

    def test_three_distinct_modulator_epochs(self) -> None:
        """Exactly three ON epochs are present (rising edges), all in train."""
        gt = _build_ground_truth(seed=0, duration=50.0)
        stim_mod = gt["stim_mod"]
        values = stim_mod.values.reshape(stim_mod.times.shape[0], -1)
        on = (values.abs().sum(dim=1) > 0).to(torch.int64)
        # Count 0->1 transitions (rising edges) = number of epochs.
        rising = ((on[1:] - on[:-1]) == 1).sum().item() + int(on[0].item() == 1)
        assert rising == 3, f"Expected 3 modulator epochs, found {rising}."
