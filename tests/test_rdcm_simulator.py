"""Integration tests for rDCM simulator and parameter recovery.

Tests cover:
- Simulator utilities (make_stable_A_rdcm, make_block_stimulus_rdcm)
- End-to-end rigid parameter recovery (correlation > 0.8)
- End-to-end sparse sparsity pattern recovery (F1 > 0.85)
- Cross-mode consistency (rigid vs sparse on same data)
- Package export verification for all Phase 3 functions

References
----------
[REF-020] Frassle et al. (2017), NeuroImage 145, 270-275.
"""

from __future__ import annotations

import pytest
import torch

from pyro_dcm.simulators.rdcm_simulator import (
    make_block_stimulus_rdcm,
    make_stable_A_rdcm,
    simulate_rdcm,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def known_A_3() -> torch.Tensor:
    """Known 3-region A matrix for rigid recovery tests."""
    return torch.tensor(
        [
            [-0.5, 0.2, 0.0],
            [0.1, -0.6, 0.15],
            [0.0, 0.1, -0.4],
        ],
        dtype=torch.float64,
    )


@pytest.fixture()
def known_C_3() -> torch.Tensor:
    """Known C matrix: 2 inputs to regions 0 and 1."""
    return torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ],
        dtype=torch.float64,
    )


# ---------------------------------------------------------------------------
# Test: make_stable_A_rdcm
# ---------------------------------------------------------------------------


class TestMakeStableARdcm:
    """Tests for stable A matrix generation."""

    def test_shape(self) -> None:
        """Output A and a_mask have shape (nr, nr)."""
        A, mask = make_stable_A_rdcm(5, seed=0)
        assert A.shape == (5, 5)
        assert mask.shape == (5, 5)

    def test_stability(self) -> None:
        """All eigenvalues of A have negative real parts."""
        A, _ = make_stable_A_rdcm(5, seed=0)
        eigvals = torch.linalg.eigvals(A)
        assert eigvals.real.max().item() < 0

    def test_density(self) -> None:
        """Approximately 30% of off-diagonal entries are non-zero."""
        A, mask = make_stable_A_rdcm(10, density=0.3, seed=42)
        nr = 10
        n_offdiag = nr * (nr - 1)
        # Count off-diagonal non-zeros in mask
        offdiag_present = (mask - torch.eye(nr, dtype=torch.float64)).sum().item()
        actual_density = offdiag_present / n_offdiag
        # Allow tolerance due to rounding
        assert abs(actual_density - 0.3) < 0.05

    def test_diagonal_mask(self) -> None:
        """a_mask diagonal is all ones."""
        _, mask = make_stable_A_rdcm(5, seed=0)
        assert torch.all(mask.diagonal() == 1.0)

    def test_reproducible(self) -> None:
        """Same seed produces identical A and a_mask."""
        A1, m1 = make_stable_A_rdcm(5, density=0.5, seed=123)
        A2, m2 = make_stable_A_rdcm(5, density=0.5, seed=123)
        assert torch.allclose(A1, A2)
        assert torch.allclose(m1, m2)


# ---------------------------------------------------------------------------
# Test: make_block_stimulus_rdcm
# ---------------------------------------------------------------------------


class TestMakeBlockStimulusRdcm:
    """Tests for block design stimulus generation."""

    def test_shape(self) -> None:
        """Output has shape (n_time, n_inputs)."""
        u = make_block_stimulus_rdcm(5000, 2, 0.01, seed=0)
        assert u.shape == (5000, 2)

    def test_binary(self) -> None:
        """All values are 0 or 1."""
        u = make_block_stimulus_rdcm(5000, 2, 0.01, seed=0)
        assert torch.all((u == 0.0) | (u == 1.0))

    def test_has_blocks(self) -> None:
        """There are transitions between 0 and 1 (not all-zero or all-one)."""
        u = make_block_stimulus_rdcm(5000, 2, 0.01, seed=0)
        for inp in range(2):
            col = u[:, inp]
            diff = torch.abs(col[1:] - col[:-1])
            n_transitions = (diff > 0).sum().item()
            assert n_transitions > 0, f"Input {inp} has no transitions"


# ---------------------------------------------------------------------------
# Test: End-to-end rigid recovery (key test)
# ---------------------------------------------------------------------------


class TestRigidRecovery:
    """Tests for rigid rDCM parameter recovery."""

    @pytest.mark.slow
    def test_rigid_recovery_3_region(
        self,
        known_A_3: torch.Tensor,
        known_C_3: torch.Tensor,
    ) -> None:
        """Rigid rDCM recovers 3-region A with correlation > 0.8.

        Uses a fully connected A, 2 inputs, block stimulus, SNR=3.
        This is the primary validation for the rigid rDCM pipeline.
        """
        u = make_block_stimulus_rdcm(5000, 2, 0.01, seed=7)
        a_mask = torch.ones(3, 3, dtype=torch.float64)
        c_mask = torch.ones(3, 2, dtype=torch.float64)

        result = simulate_rdcm(
            known_A_3,
            known_C_3,
            u,
            u_dt=0.01,
            y_dt=2.0,
            SNR=3.0,
            a_mask=a_mask,
            c_mask=c_mask,
            mode="rigid",
            seed=42,
        )

        # Pearson correlation between true and recovered A
        a_true = known_A_3.flatten()
        a_mu = result["A_mu"].flatten()
        corr = _pearson_corr(a_true, a_mu)

        assert corr > 0.8, (
            f"Rigid recovery correlation {corr:.3f} < 0.8"
        )

        # Diagonal of A_mu should be negative (self-inhibition)
        assert torch.all(result["A_mu"].diagonal() < 0), (
            "Posterior A diagonal should be negative"
        )

        # Free energy should be finite
        assert torch.isfinite(
            torch.as_tensor(result["F_total"])
        ), "F_total is not finite"

    @pytest.mark.slow
    def test_rigid_recovery_diagonal_dominance(self) -> None:
        """Strong diagonal is recovered with larger magnitude than off-diagonal."""
        A = torch.tensor(
            [
                [-0.5, 0.05, 0.0],
                [0.05, -0.5, 0.05],
                [0.0, 0.05, -0.5],
            ],
            dtype=torch.float64,
        )
        C = torch.tensor(
            [[1.0], [0.0], [0.0]],
            dtype=torch.float64,
        )
        u = make_block_stimulus_rdcm(5000, 1, 0.01, seed=10)
        a_mask = torch.ones(3, 3, dtype=torch.float64)
        c_mask = torch.ones(3, 1, dtype=torch.float64)

        result = simulate_rdcm(
            A, C, u,
            u_dt=0.01, y_dt=2.0, SNR=3.0,
            a_mask=a_mask, c_mask=c_mask,
            mode="rigid", seed=42,
        )

        A_mu = result["A_mu"]
        # Diagonal magnitude should exceed off-diagonal magnitude
        diag_mag = A_mu.diagonal().abs().mean().item()
        offdiag_mask = ~torch.eye(3, dtype=torch.bool)
        offdiag_mag = A_mu[offdiag_mask].abs().mean().item()
        assert diag_mag > offdiag_mag, (
            f"Diagonal magnitude {diag_mag:.4f} not > "
            f"off-diagonal {offdiag_mag:.4f}"
        )


# ---------------------------------------------------------------------------
# Test: End-to-end sparse recovery (key test)
# ---------------------------------------------------------------------------


class TestSparseRecovery:
    """Tests for sparse rDCM sparsity pattern recovery."""

    @pytest.mark.slow
    def test_sparse_recovery_3_region(self) -> None:
        """Sparse rDCM recovers sparsity pattern with F1 > 0.85.

        Uses a 3-region sparse network with known zero-pattern,
        15000 time points for sufficient frequency-domain data,
        and 20 reruns for robust ARD convergence.
        This validates the ARD binary indicator mechanism.
        """
        # 3-region chain: clear sparse structure with 6 true connections
        # out of 9 total entries (3 zeros to discover)
        A = torch.tensor(
            [
                [-0.5, 0.2, 0.0],
                [0.0, -0.5, 0.2],
                [0.15, 0.0, -0.5],
            ],
            dtype=torch.float64,
        )
        C = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
            dtype=torch.float64,
        )
        nr, nu = 3, 2

        u = make_block_stimulus_rdcm(15000, nu, 0.01, seed=7)

        result = simulate_rdcm(
            A, C, u,
            u_dt=0.01, y_dt=2.0, SNR=3.0,
            a_mask=torch.ones(nr, nr, dtype=torch.float64),
            c_mask=torch.ones(nr, nu, dtype=torch.float64),
            mode="sparse",
            sparse_kwargs={"n_reruns": 20, "p0": 0.5},
            seed=42,
        )

        # Compute F1 score for sparsity pattern
        true_pattern = (A != 0).float()
        z_matrix = _z_list_to_matrix(
            result["z_per_region"], nr, nr, nu, 1
        )
        inferred_pattern = (z_matrix > 0.5).float()

        f1 = _f1_score(true_pattern, inferred_pattern)
        assert f1 > 0.85, (
            f"Sparse recovery F1 {f1:.3f} < 0.85"
        )

        # Free energy should be finite
        assert torch.isfinite(
            torch.as_tensor(result["F_total"])
        ), "F_total is not finite"


# ---------------------------------------------------------------------------
# Test: Cross-mode consistency
# ---------------------------------------------------------------------------


class TestCrossModeConsistency:
    """Tests comparing rigid and sparse on same data."""

    @pytest.mark.slow
    def test_rigid_vs_sparse_fully_connected(
        self,
        known_A_3: torch.Tensor,
        known_C_3: torch.Tensor,
    ) -> None:
        """Rigid and sparse produce similar A_mu for fully connected network.

        For a fully connected A, sparse ARD should retain most connections
        and produce A_mu positively correlated with rigid A_mu.
        Threshold is 0.8 since sparse ARD naturally shrinks coefficients
        differently from rigid VB.
        """
        u = make_block_stimulus_rdcm(5000, 2, 0.01, seed=7)
        a_mask = torch.ones(3, 3, dtype=torch.float64)
        c_mask = torch.ones(3, 2, dtype=torch.float64)

        rigid_result = simulate_rdcm(
            known_A_3, known_C_3, u,
            u_dt=0.01, y_dt=2.0, SNR=3.0,
            a_mask=a_mask, c_mask=c_mask,
            mode="rigid", seed=42,
        )
        sparse_result = simulate_rdcm(
            known_A_3, known_C_3, u,
            u_dt=0.01, y_dt=2.0, SNR=3.0,
            a_mask=a_mask, c_mask=c_mask,
            mode="sparse",
            sparse_kwargs={"n_reruns": 10},
            seed=42,
        )

        # A_mu from both modes should be positively correlated
        rigid_flat = rigid_result["A_mu"].flatten()
        sparse_flat = sparse_result["A_mu"].flatten()
        corr = _pearson_corr(rigid_flat, sparse_flat)
        assert corr > 0.8, (
            f"Rigid vs sparse A_mu correlation {corr:.3f} < 0.8"
        )

        # Sparse z values should be mostly > 0.5 for true connections
        z_matrix = _z_list_to_matrix(
            sparse_result["z_per_region"], 3, 3, 2, 1
        )
        true_mask = (known_A_3 != 0)
        z_on_true = z_matrix[true_mask]
        frac_retained = (z_on_true > 0.5).float().mean().item()
        assert frac_retained > 0.7, (
            f"Only {frac_retained:.1%} of true connections retained"
        )


# ---------------------------------------------------------------------------
# Test: Package exports
# ---------------------------------------------------------------------------


class TestPackageExports:
    """Verify all Phase 3 functions are importable from top-level packages."""

    def test_forward_models_exports(self) -> None:
        """All Phase 3 forward model functions importable."""
        from pyro_dcm.forward_models import (  # noqa: F401
            compute_derivative_coefficients,
            compute_free_energy_rigid,
            compute_free_energy_sparse,
            compute_rdcm_likelihood,
            create_regressors,
            euler_integrate_dcm,
            generate_bold,
            get_hrf,
            get_priors_rigid,
            get_priors_sparse,
            rigid_inversion,
            sparse_inversion,
            split_real_imag,
        )

    def test_simulators_exports(self) -> None:
        """All Phase 3 simulator functions importable."""
        from pyro_dcm.simulators import (  # noqa: F401
            make_block_stimulus_rdcm,
            make_stable_A_rdcm,
            simulate_rdcm,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pearson_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute Pearson correlation between two 1D tensors.

    Manual computation to avoid numpy process abort on Windows.
    """
    x_mean = x.mean()
    y_mean = y.mean()
    xd = x - x_mean
    yd = y - y_mean
    num = (xd * yd).sum()
    denom = (xd.pow(2).sum() * yd.pow(2).sum()).sqrt()
    if denom < 1e-15:
        return 0.0
    return (num / denom).item()


def _z_list_to_matrix(
    z_per_region: list,
    nr: int,
    nr_cols: int,
    nu: int,
    nc: int,
) -> torch.Tensor:
    """Extract A-related z indicators from per-region z vectors.

    Each z vector has length (nr + nu + nc). The first nr entries
    correspond to A connections for that region. Returns an (nr, nr)
    matrix of z values for A.
    """
    z_A = torch.zeros(nr, nr_cols, dtype=torch.float64)
    for r in range(nr):
        z_r = z_per_region[r]
        if z_r is not None:
            z_A[r, :nr_cols] = z_r[:nr_cols]
    return z_A


def _f1_score(
    true_pattern: torch.Tensor,
    inferred_pattern: torch.Tensor,
) -> float:
    """Compute F1 score between two binary pattern matrices."""
    tp = ((true_pattern == 1) & (inferred_pattern == 1)).sum().float()
    fp = ((true_pattern == 0) & (inferred_pattern == 1)).sum().float()
    fn = ((true_pattern == 1) & (inferred_pattern == 0)).sum().float()
    precision = tp / (tp + fp) if (tp + fp) > 0 else torch.tensor(0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else torch.tensor(0.0)
    if (precision + recall) < 1e-10:
        return 0.0
    f1 = 2.0 * precision * recall / (precision + recall)
    return f1.item()
