"""Unit tests for bilinear utility functions.

Tests parameterize_B and compute_effective_A against known values,
verifying [REF-001] Eq. 1 modulator composition for the v0.3.0 bilinear
extension (BILIN-01, BILIN-02).
"""

from __future__ import annotations

import warnings

import pytest
import torch

from pyro_dcm.forward_models.neural_state import (
    compute_effective_A,
    parameterize_A,
    parameterize_B,
)


class TestParameterizeB:
    """Tests for parameterize_B factory (BILIN-01)."""

    def test_shape_roundtrip(self) -> None:
        """Output preserves the (J, N, N) input shape."""
        J, N = 2, 3
        B_free = torch.randn(J, N, N, dtype=torch.float64)
        b_mask = torch.ones(J, N, N, dtype=torch.float64)
        for j in range(J):
            b_mask[j].fill_diagonal_(0.0)

        B = parameterize_B(B_free, b_mask)

        expected_shape = torch.Size([J, N, N])
        assert B.shape == expected_shape, (
            f"parameterize_B shape mismatch: expected {expected_shape}, "
            f"got {B.shape}"
        )

    def test_mask_zeros_masked_entries(self) -> None:
        """Entries with mask=0 are exactly zero; mask=1 entries pass through."""
        B_free = torch.tensor(
            [
                [
                    [0.10, 0.20, 0.30],
                    [0.40, 0.50, 0.60],
                    [0.70, 0.80, 0.90],
                ]
            ],
            dtype=torch.float64,
        )
        b_mask = torch.tensor(
            [
                [
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                ]
            ],
            dtype=torch.float64,
        )

        B = parameterize_B(B_free, b_mask)

        # Masked-to-zero entries: (0,0), (0,2), (1,1), (2,0), (2,2)
        for idx in [(0, 0, 0), (0, 0, 2), (0, 1, 1), (0, 2, 0), (0, 2, 2)]:
            assert B[idx].item() == 0.0, (
                f"Mask=0 entry {idx}: expected 0.0, got {B[idx].item()}"
            )

        # Pass-through entries: (0,1), (1,0), (1,2), (2,1)
        for idx in [(0, 0, 1), (0, 1, 0), (0, 1, 2), (0, 2, 1)]:
            assert B[idx].item() == B_free[idx].item(), (
                f"Mask=1 entry {idx}: expected {B_free[idx].item()}, "
                f"got {B[idx].item()}"
            )

    def test_default_diagonal_is_zero(self) -> None:
        """Recommended default (b_mask.fill_diagonal_(0.0)) zeros B diagonal."""
        J, N = 3, 4
        B_free = torch.randn(J, N, N, dtype=torch.float64)
        b_mask = torch.ones(J, N, N, dtype=torch.float64)
        for j in range(J):
            b_mask[j].fill_diagonal_(0.0)

        B = parameterize_B(B_free, b_mask)

        diag = torch.diagonal(B, dim1=-2, dim2=-1)  # shape (J, N)
        assert torch.all(diag == 0.0), (
            "Default-mask diagonal must be exactly 0.0 across all J "
            f"modulators; got diagonals:\n{diag}"
        )

    def test_nonzero_diagonal_triggers_deprecation_warning(self) -> None:
        """Non-zero diagonal b_mask emits DeprecationWarning, still returns populated entry."""
        J, N = 2, 3
        B_free = torch.ones(J, N, N, dtype=torch.float64) * 0.5
        b_mask = torch.ones(J, N, N, dtype=torch.float64)
        for j in range(J):
            b_mask[j].fill_diagonal_(0.0)
        # Opt-in to a single self-modulation entry
        b_mask[0, 1, 1] = 1.0

        with pytest.warns(DeprecationWarning, match="non-zero diagonal"):
            B = parameterize_B(B_free, b_mask)

        # The warning is informational; the entry must still be populated.
        assert B[0, 1, 1].item() == 0.5, (
            "DeprecationWarning must not silently zero the opted-in entry; "
            f"expected B[0, 1, 1] == 0.5, got {B[0, 1, 1].item()}"
        )
        # Other diagonal entries still zero.
        assert B[0, 0, 0].item() == 0.0, (
            f"B[0, 0, 0]: expected 0.0, got {B[0, 0, 0].item()}"
        )
        assert B[1, 1, 1].item() == 0.0, (
            f"B[1, 1, 1]: expected 0.0, got {B[1, 1, 1].item()}"
        )

    def test_empty_J_roundtrip(self) -> None:
        """J=0 returns empty (0, N, N) tensor without emitting DeprecationWarning."""
        N = 3
        B_free = torch.zeros(0, N, N, dtype=torch.float64)
        b_mask = torch.zeros(0, N, N, dtype=torch.float64)

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always", DeprecationWarning)
            B = parameterize_B(B_free, b_mask)

        expected_shape = torch.Size([0, N, N])
        assert B.shape == expected_shape, (
            f"parameterize_B J=0 shape: expected {expected_shape}, got {B.shape}"
        )

        dep_records = [w for w in record if issubclass(w.category, DeprecationWarning)]
        assert len(dep_records) == 0, (
            f"Expected 0 DeprecationWarnings for J=0 case, got {len(dep_records)}: "
            f"{[str(w.message) for w in dep_records]}"
        )

    def test_shape_mismatch_raises_valueerror(self) -> None:
        """Mismatched B_free and b_mask shapes raise ValueError with context."""
        B_free = torch.randn(2, 3, 3, dtype=torch.float64)
        b_mask = torch.ones(2, 4, 4, dtype=torch.float64)

        with pytest.raises(ValueError, match="shape mismatch|B_free.shape"):
            parameterize_B(B_free, b_mask)


class TestComputeEffectiveA:
    """Tests for compute_effective_A einsum composition (BILIN-02)."""

    def test_linear_case_B_zero_returns_A(self) -> None:
        """B all-zero with any u_mod returns A bit-exactly via einsum path."""
        N = 3
        A = parameterize_A(torch.zeros(N, N, dtype=torch.float64))
        B = torch.zeros(1, N, N, dtype=torch.float64)
        u_mod = torch.tensor([1.0], dtype=torch.float64)

        A_eff = compute_effective_A(A, B, u_mod)

        assert torch.equal(A_eff, A), (
            "compute_effective_A with B=0 must equal A bit-exactly; "
            f"expected equal, got max|diff|={(A_eff - A).abs().max().item()}"
        )

    def test_einsum_correctness(self) -> None:
        """Hand-computed A_eff matches einsum output across two modulators."""
        A = torch.tensor(
            [[-0.5, 0.0], [0.0, -0.5]],
            dtype=torch.float64,
        )
        B = torch.zeros(2, 2, 2, dtype=torch.float64)
        B[0, 0, 1] = 0.3
        B[0, 1, 0] = 0.2
        B[1, 0, 1] = -0.1
        u_mod = torch.tensor([1.0, 0.5], dtype=torch.float64)

        A_eff = compute_effective_A(A, B, u_mod)

        # Expected:
        # A_eff[0, 0] = -0.5 (unchanged; B[:, 0, 0] = 0)
        # A_eff[1, 1] = -0.5 (unchanged; B[:, 1, 1] = 0)
        # A_eff[0, 1] = 0.0 + 1.0 * 0.3 + 0.5 * (-0.1) = 0.25
        # A_eff[1, 0] = 0.0 + 1.0 * 0.2 + 0.5 * 0.0 = 0.2
        expected = torch.tensor(
            [[-0.5, 0.25], [0.2, -0.5]],
            dtype=torch.float64,
        )
        torch.testing.assert_close(A_eff, expected, atol=1e-12, rtol=1e-12)

    def test_empty_J_returns_A_unchanged(self) -> None:
        """J=0 short-circuit returns A bit-exactly without allocating."""
        N = 4
        A = torch.eye(N, dtype=torch.float64)
        B = torch.zeros(0, N, N, dtype=torch.float64)
        u_mod = torch.zeros(0, dtype=torch.float64)

        A_eff = compute_effective_A(A, B, u_mod)

        assert torch.equal(A_eff, A), (
            "compute_effective_A J=0 must return A bit-exactly; "
            f"expected equal, got max|diff|={(A_eff - A).abs().max().item()}"
        )
