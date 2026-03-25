"""Unit tests for BOLD signal observation equation.

Tests bold_signal against known values, verifying [REF-002] Eq. 6
implementation.
"""

from __future__ import annotations

import torch

from pyro_dcm.forward_models.bold_signal import bold_signal


class TestBoldSignal:
    """Tests for bold_signal function."""

    def test_steady_state_zero_bold(self, dtype: torch.dtype) -> None:
        """At v=1, q=1 (steady state), BOLD signal should be zero."""
        v = torch.ones(3, dtype=dtype)
        q = torch.ones(3, dtype=dtype)
        y = bold_signal(v, q)
        torch.testing.assert_close(
            y, torch.zeros(3, dtype=dtype), atol=1e-15, rtol=0.0
        )

    def test_realistic_range(self, dtype: torch.dtype) -> None:
        """For near-steady-state hemodynamics, BOLD is in [-5%, +5%]."""
        v = torch.linspace(0.95, 1.05, 20, dtype=dtype)
        q = torch.linspace(0.90, 1.10, 20, dtype=dtype)
        y = bold_signal(v, q)

        assert torch.all(y > -0.05), f"BOLD too negative: min={y.min()}"
        assert torch.all(y < 0.05), f"BOLD too positive: max={y.max()}"

    def test_bold_increases_with_volume(self, dtype: torch.dtype) -> None:
        """Decreasing q (less deoxyhemoglobin) with v=1 increases BOLD.

        With V0=0.02, k1=2.8, k2=2.0, k3=0.6:
        y = V0 * (k1*(1-q) + k2*(1-q/v) + k3*(1-v))
        At v=1: y = V0 * (k1*(1-q) + k2*(1-q) + 0)
              = V0 * (k1+k2)*(1-q)
        So decreasing q from 1 gives positive BOLD.
        """
        v = torch.ones(1, dtype=dtype)
        q_low = torch.tensor([0.95], dtype=dtype)
        q_high = torch.tensor([1.0], dtype=dtype)

        y_low = bold_signal(v, q_low)
        y_high = bold_signal(v, q_high)

        assert y_low > y_high, (
            f"BOLD should increase when q decreases: "
            f"y(q=0.95)={y_low.item()}, y(q=1.0)={y_high.item()}"
        )

    def test_bold_constants(self) -> None:
        """Verify k1, k2, k3 with E0=0.40.

        k1 = 7 * 0.40 = 2.8
        k2 = 2.0
        k3 = 2 * 0.40 - 0.2 = 0.6
        """
        E0 = 0.40
        k1 = 7.0 * E0
        k2 = 2.0
        k3 = 2.0 * E0 - 0.2

        assert abs(k1 - 2.8) < 1e-15
        assert abs(k2 - 2.0) < 1e-15
        assert abs(k3 - 0.6) < 1e-15

    def test_bold_batch_shape(self, dtype: torch.dtype) -> None:
        """Works with batched inputs (T, N) and returns shape (T, N)."""
        T, N = 50, 4
        v = torch.ones(T, N, dtype=dtype) + 0.02 * torch.randn(
            T, N, dtype=dtype
        )
        q = torch.ones(T, N, dtype=dtype) + 0.02 * torch.randn(
            T, N, dtype=dtype
        )

        y = bold_signal(v, q)
        assert y.shape == (T, N), f"Expected ({T}, {N}), got {y.shape}"

    def test_bold_differentiable(self, dtype: torch.dtype) -> None:
        """torch.autograd.grad through bold_signal gives non-None gradients."""
        v = torch.ones(3, dtype=dtype, requires_grad=True)
        q = torch.ones(3, dtype=dtype, requires_grad=True) * 0.95

        y = bold_signal(v, q)
        loss = y.sum()

        grads = torch.autograd.grad(loss, [v, q])
        assert grads[0] is not None, "Gradient w.r.t. v is None"
        assert grads[1] is not None, "Gradient w.r.t. q is None"
        assert torch.all(torch.isfinite(grads[0])), "Non-finite grad for v"
        assert torch.all(torch.isfinite(grads[1])), "Non-finite grad for q"
