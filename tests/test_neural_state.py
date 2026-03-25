"""Unit tests for neural state equation module.

Tests parameterize_A and NeuralStateEquation against known values,
verifying [REF-001] Eq. 1 implementation.
"""

from __future__ import annotations

import torch

from pyro_dcm.forward_models.neural_state import (
    NeuralStateEquation,
    parameterize_A,
)


class TestParameterizeA:
    """Tests for A matrix parameterization."""

    def test_parameterize_A_diagonal(self, dtype: torch.dtype) -> None:
        """Diagonal elements are -exp(free)/2.

        For A_free_diag = 0, result should be -exp(0)/2 = -0.5.
        """
        A_free = torch.zeros(3, 3, dtype=dtype)
        A = parameterize_A(A_free)
        diag = torch.diag(A)
        expected = torch.full((3,), -0.5, dtype=dtype)
        torch.testing.assert_close(diag, expected)

    def test_parameterize_A_offdiagonal(self, dtype: torch.dtype) -> None:
        """Off-diagonal elements pass through unchanged."""
        A_free = torch.tensor(
            [[0.0, 0.3, -0.1], [0.2, 0.0, 0.4], [-0.5, 0.1, 0.0]],
            dtype=dtype,
        )
        A = parameterize_A(A_free)

        # Off-diagonal should be unchanged
        assert A[0, 1].item() == 0.3
        assert A[0, 2].item() == -0.1
        assert A[1, 0].item() == 0.2
        assert A[1, 2].item() == 0.4
        assert A[2, 0].item() == -0.5
        assert A[2, 1].item() == 0.1

    def test_parameterize_A_always_negative_diagonal(
        self, dtype: torch.dtype
    ) -> None:
        """Diagonal is always negative for any free parameter value."""
        for val in [-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0]:
            A_free = torch.full((3, 3), val, dtype=dtype)
            A = parameterize_A(A_free)
            diag = torch.diag(A)
            assert (diag < 0).all(), (
                f"Diagonal not negative for free param = {val}: {diag}"
            )

    def test_parameterize_A_specific_values(self, dtype: torch.dtype) -> None:
        """Verify specific diagonal transform values."""
        A_free = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=dtype)
        A = parameterize_A(A_free)

        # diag[0] = -exp(1)/2
        expected_00 = -torch.exp(torch.tensor(1.0, dtype=dtype)) / 2.0
        torch.testing.assert_close(A[0, 0], expected_00)

        # diag[1] = -exp(-1)/2
        expected_11 = -torch.exp(torch.tensor(-1.0, dtype=dtype)) / 2.0
        torch.testing.assert_close(A[1, 1], expected_11)


class TestNeuralStateEquation:
    """Tests for NeuralStateEquation derivatives."""

    def test_neural_state_zero_input(
        self, test_A: torch.Tensor, test_C: torch.Tensor
    ) -> None:
        """With u=0, dx/dt = A @ x."""
        nse = NeuralStateEquation(test_A, test_C)
        x = torch.tensor([0.5, -0.3, 0.1], dtype=torch.float64)
        u = torch.tensor([0.0], dtype=torch.float64)

        dx = nse.derivatives(x, u)
        expected = test_A @ x
        torch.testing.assert_close(dx, expected)

    def test_neural_state_zero_activity(
        self, test_A: torch.Tensor, test_C: torch.Tensor
    ) -> None:
        """With x=0, dx/dt = C @ u."""
        nse = NeuralStateEquation(test_A, test_C)
        x = torch.zeros(3, dtype=torch.float64)
        u = torch.tensor([2.0], dtype=torch.float64)

        dx = nse.derivatives(x, u)
        expected = test_C @ u
        torch.testing.assert_close(dx, expected)

    def test_neural_state_stability(
        self, test_A: torch.Tensor, test_C: torch.Tensor
    ) -> None:
        """With stable A (negative eigenvalues), Euler steps do not diverge."""
        nse = NeuralStateEquation(test_A, test_C)
        x = torch.tensor([1.0, 0.5, -0.5], dtype=torch.float64)
        u = torch.zeros(1, dtype=torch.float64)

        dt = 0.01
        for _ in range(100):
            dx = nse.derivatives(x, u)
            x = x + dt * dx

        # x should not have diverged; norms should be bounded
        assert torch.all(torch.isfinite(x)), f"Non-finite values: {x}"
        assert x.norm() < 10.0, f"Diverged: norm = {x.norm()}"

    def test_neural_state_known_values(self, dtype: torch.dtype) -> None:
        """Hand-compute dx/dt for a 2-region example.

        A = [[-0.5, 0.1], [0.2, -0.5]]
        C = [[1.0], [0.0]]
        x = [0.1, 0.2], u = [1.0]

        dx/dt = A @ x + C @ u
              = [(-0.5*0.1 + 0.1*0.2) + 1.0, (0.2*0.1 + (-0.5)*0.2) + 0.0]
              = [-0.05 + 0.02 + 1.0, 0.02 - 0.10 + 0.0]
              = [0.97, -0.08]
        """
        A = torch.tensor([[-0.5, 0.1], [0.2, -0.5]], dtype=dtype)
        C = torch.tensor([[1.0], [0.0]], dtype=dtype)
        nse = NeuralStateEquation(A, C)

        x = torch.tensor([0.1, 0.2], dtype=dtype)
        u = torch.tensor([1.0], dtype=dtype)

        dx = nse.derivatives(x, u)
        expected = torch.tensor([0.97, -0.08], dtype=dtype)
        torch.testing.assert_close(dx, expected, atol=1e-12, rtol=1e-12)
