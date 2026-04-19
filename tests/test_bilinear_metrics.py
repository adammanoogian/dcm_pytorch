"""Unit tests for benchmarks/bilinear_metrics.py (Phase 16 RECOV-03..08).

Each metric helper has one unit test with hand-computed expected values.
``compute_acceptance_gates`` is tested with a synthetic runner_result.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from benchmarks.bilinear_metrics import (
    compute_a_rmse_relative,
    compute_acceptance_gates,
    compute_b_rmse_magnitude,
    compute_coverage_of_zero,
    compute_shrinkage,
    compute_sign_recovery_nonzero,
)


def _make_minimal_runner_result(
    n_seeds: int = 3,
    b_true: tuple[float, float] = (0.4, 0.3),
    b_inferred: tuple[float, float] = (0.4, 0.3),
    b_std_nonnull: float = 0.05,
    b_std_null: float = 0.3,
    a_rmse_bi: float = 0.1,
    a_rmse_lin: float = 0.09,
    t_bi: float = 200.0,
    t_lin: float = 100.0,
    n_samples: int = 50,
) -> dict:
    """Build a synthetic runner_result dict for compute_acceptance_gates tests."""
    n_regions = 3
    B_true = np.zeros((1, n_regions, n_regions), dtype=np.float64)
    B_true[0, 1, 0] = b_true[0]
    B_true[0, 2, 1] = b_true[1]
    B_inferred_mean = np.zeros((n_regions, n_regions), dtype=np.float64)
    B_inferred_mean[1, 0] = b_inferred[0]
    B_inferred_mean[2, 1] = b_inferred[1]
    # Samples: normal around inferred mean with std=b_std_nonnull for free
    # elements, b_std_null for null elements.
    rng = np.random.default_rng(seed=0)
    B_std = np.full((n_regions, n_regions), b_std_null, dtype=np.float64)
    B_std[1, 0] = b_std_nonnull
    B_std[2, 1] = b_std_nonnull
    B_samples = rng.normal(
        loc=B_inferred_mean, scale=B_std,
        size=(n_samples, n_regions, n_regions),
    )
    post = {
        "B_free_0": {
            "mean": B_inferred_mean.tolist(),
            "std": B_std.tolist(),
            "samples": B_samples.tolist(),
        },
        "B_true": B_true.tolist(),
    }
    return {
        "a_rmse_bilinear_list": [a_rmse_bi] * n_seeds,
        "a_rmse_linear_list": [a_rmse_lin] * n_seeds,
        "time_bilinear_list": [t_bi] * n_seeds,
        "time_linear_list": [t_lin] * n_seeds,
        "posterior_list": [post for _ in range(n_seeds)],
        "b_true_list": [B_true.flatten().tolist() for _ in range(n_seeds)],
    }


class TestMetricHelpers:
    """Unit tests for the 5 pure metric helpers."""

    def test_b_rmse_magnitude_on_known_ground_truth(self) -> None:
        """Perfect recovery -> 0.0; zero inference -> sqrt of squared true means."""
        B_true = torch.tensor([[[0.0, 0.0, 0.0],
                                [0.4, 0.0, 0.0],
                                [0.0, 0.3, 0.0]]], dtype=torch.float64)
        # Perfect recovery.
        assert compute_b_rmse_magnitude(B_true, B_true.clone()) == pytest.approx(0.0)
        # Zero inference: rmse = sqrt((0.4^2 + 0.3^2) / 2) = sqrt(0.125) ~ 0.3536.
        B_zero = torch.zeros_like(B_true)
        expected = ((0.4 ** 2 + 0.3 ** 2) / 2) ** 0.5
        got = compute_b_rmse_magnitude(B_true, B_zero)
        assert got == pytest.approx(expected, abs=1e-6)
        # Vacuous mask.
        B_small = torch.full_like(B_true, 0.05)
        assert compute_b_rmse_magnitude(B_small, B_small) == 0.0

    def test_sign_recovery_nonzero_pooled_vs_perseed_mean(self) -> None:
        """Pooled over (seed, element) pairs; per-seed-mean path rejected by L5."""
        B_true = torch.tensor([[[0.0, 0.0, 0.0],
                                [0.4, 0.0, 0.0],
                                [0.0, 0.3, 0.0]]], dtype=torch.float64)
        # Seed 0: 100% sign match (both positive).
        # Seed 1: B[2,1] flipped (negative) -> 1/2 match.
        # Seed 2: both flipped -> 0/2 match.
        # Pooled: 2+1+0 / 2+2+2 = 3/6 = 0.5.
        B_inf_0 = B_true.clone()  # both positive
        B_inf_1 = B_true.clone()
        B_inf_1[0, 2, 1] = -0.3
        B_inf_2 = -B_true.clone()
        sr = compute_sign_recovery_nonzero(
            [B_true, B_true, B_true], [B_inf_0, B_inf_1, B_inf_2],
        )
        assert sr == pytest.approx(0.5, abs=1e-9)

    def test_coverage_of_zero_matches_ci_containment(self) -> None:
        """Null elements with wide CI that contain 0 -> coverage == 1."""
        B_true = torch.zeros(1, 3, 3, dtype=torch.float64)  # all zero
        # Samples: normal(0, 1) around each element.
        rng = np.random.default_rng(seed=42)
        B_samples = torch.from_numpy(
            rng.normal(loc=0.0, scale=1.0, size=(200, 1, 3, 3)).astype(np.float64),
        )
        cov = compute_coverage_of_zero([B_true], [B_samples])
        # 9 null elements, all with wide CI centered at 0 -> coverage >= 0.95
        # typically.
        assert cov >= 0.8, f"coverage={cov} too low for samples centered at 0"

        # Samples far from zero -> coverage should drop.
        B_samples_far = B_samples + 5.0
        cov_far = compute_coverage_of_zero([B_true], [B_samples_far])
        assert cov_far <= 0.2, (
            f"coverage={cov_far} too high for samples shifted by 5"
        )

    def test_shrinkage_elementwise(self) -> None:
        """std_post / sigma_prior element-wise."""
        B_std = torch.tensor([[[0.5, 0.6], [0.7, 0.8]]], dtype=torch.float64)
        sh = compute_shrinkage(B_std, sigma_prior=1.0)
        assert torch.allclose(sh, B_std)  # sigma_prior=1 -> identity
        sh2 = compute_shrinkage(B_std, sigma_prior=2.0)
        assert torch.allclose(sh2, B_std / 2.0)

    def test_a_rmse_relative(self) -> None:
        """Ratio = mean_bi / mean_lin; pass iff ratio <= 1.25."""
        r = compute_a_rmse_relative([0.10, 0.12], [0.10, 0.10])
        # mean_bi=0.11; mean_lin=0.10 -> ratio=1.1 -> pass (<= 1.25).
        assert r["mean_bilinear"] == pytest.approx(0.11)
        assert r["mean_linear"] == pytest.approx(0.10)
        assert r["ratio"] == pytest.approx(1.1, abs=1e-6)
        assert r["pass"] is True
        # Fail case: ratio = 1.5.
        r2 = compute_a_rmse_relative([0.15, 0.15], [0.10, 0.10])
        assert r2["ratio"] == pytest.approx(1.5)
        assert r2["pass"] is False


class TestAcceptanceGates:
    """Unit tests for compute_acceptance_gates end-to-end."""

    def test_acceptance_gates_all_pass_on_perfect_recovery(self) -> None:
        """Perfect B recovery + matched A-RMSE + low shrinkage -> all 4 gates pass."""
        rr = _make_minimal_runner_result(
            n_seeds=5,
            b_inferred=(0.4, 0.3),          # perfect
            a_rmse_bi=0.10, a_rmse_lin=0.09,
            b_std_nonnull=0.05, b_std_null=0.3,
            t_bi=200.0, t_lin=100.0,
        )
        gates = compute_acceptance_gates(rr)
        assert gates["RECOV-03"]["pass"], f"RECOV-03 observed: {gates['RECOV-03']}"
        assert gates["RECOV-04"]["pass"], f"RECOV-04 observed: {gates['RECOV-04']}"
        assert gates["RECOV-05"]["pass"], f"RECOV-05 observed: {gates['RECOV-05']}"
        assert gates["RECOV-06"]["pass"], f"RECOV-06 observed: {gates['RECOV-06']}"
        assert gates["all_pass"] is True

    def test_acceptance_gates_fail_on_flipped_signs(self) -> None:
        """Sign-flipped B inference -> RECOV-05 fails."""
        rr = _make_minimal_runner_result(
            n_seeds=5, b_inferred=(-0.4, -0.3),  # both flipped
        )
        gates = compute_acceptance_gates(rr)
        assert gates["RECOV-05"]["pass"] is False
        assert gates["all_pass"] is False

    def test_all_pass_false_when_only_recov_03_fails(self) -> None:
        """Single-gate-failure exercise for the all_pass AND-combination logic.

        FIX 2 per orchestrator revision. Catches an accidental ``or`` in the
        ``all_pass`` computation: constructs a result where RECOV-04/05/06
        all pass (perfect B recovery, matched sign and null-element coverage)
        but RECOV-03 fails (bilinear A-RMSE >> 1.25 * linear-baseline),
        and asserts all_pass is False.
        """
        rr = _make_minimal_runner_result(
            n_seeds=5,
            b_inferred=(0.4, 0.3),              # perfect B recovery
            a_rmse_bi=0.50, a_rmse_lin=0.10,    # 5x ratio >> 1.25x threshold
            b_std_nonnull=0.05, b_std_null=0.3,
            t_bi=200.0, t_lin=100.0,
        )
        gates = compute_acceptance_gates(rr)
        assert gates["RECOV-03"]["pass"] is False, (
            f"Expected RECOV-03 failure (a_rmse_ratio=5.0 > 1.25); "
            f"got {gates['RECOV-03']}"
        )
        assert gates["RECOV-04"]["pass"] is True
        assert gates["RECOV-05"]["pass"] is True
        assert gates["RECOV-06"]["pass"] is True
        assert gates["all_pass"] is False, (
            "all_pass must be False when any single RECOV gate fails "
            "(regression gate for AND-combination logic)"
        )

    def test_acceptance_gates_raises_on_insufficient_data(self) -> None:
        """status='insufficient_data' raises ValueError."""
        rr = {
            "status": "insufficient_data", "n_success": 0, "n_failed": 3,
            "n_datasets": 3,
        }
        with pytest.raises(ValueError, match="insufficient_data"):
            compute_acceptance_gates(rr)

    def test_recov_08_flag_triggers_above_10x(self) -> None:
        """Ratio > 10x -> RECOV-08 flag True."""
        rr = _make_minimal_runner_result(
            n_seeds=3, t_bi=1200.0, t_lin=100.0,  # 12x ratio
        )
        gates = compute_acceptance_gates(rr)
        assert gates["RECOV-08"]["ratio"] > 10.0
        assert gates["RECOV-08"]["flag_over_10x"] is True

    def test_recov_07_shrinkage_info_reported(self) -> None:
        """RECOV-07 shrinkage_nonnull has the two free-element means (no pass/fail)."""
        rr = _make_minimal_runner_result(n_seeds=3, b_std_nonnull=0.04)
        gates = compute_acceptance_gates(rr)
        r07 = gates["RECOV-07"]
        assert "shrinkage_nonnull" in r07
        assert len(r07["shrinkage_nonnull"]) == 2
        # 0.04 / 1.0 = 0.04 << 0.7
        for s in r07["shrinkage_nonnull"]:
            assert s == pytest.approx(0.04, abs=0.01)
        assert r07["all_below_soft_target"] is True
