"""Tests for fixed-point analysis utilities (RNN-04).

Tests fixed-point finding, Jacobian computation, stability classification,
and deduplication for the CT-RNN latent circuit analysis pipeline.
No special pytest markers needed -- pure PyTorch, no optional dependencies.
"""

from __future__ import annotations

import torch

from pyro_dcm.rnn import (
    ContinuousTimeRNN,
    classify_stability,
    compute_jacobian_at_fp,
    find_fixed_points,
)
from pyro_dcm.rnn.fixed_point_analysis import _deduplicate_fixed_points


class TestFindFixedPoints:
    """Tests for find_fixed_points()."""

    def test_find_fixed_points_linear_system(self) -> None:
        """Fixed points exist for a tanh CT-RNN with H=8 and zero input.

        With zero input (u=0, b=0), h=0 is a trivially accessible fixed point
        for many random RNNs. The optimizer should converge to at least one.
        We use convergence_threshold=1.0 (loose) to ensure at least 1 candidate.
        """
        torch.manual_seed(42)
        rnn = ContinuousTimeRNN(n_input=2, n_hidden=8, n_output=2, activation="tanh")
        # Zero out bias to guarantee h=0 is a fixed point
        with torch.no_grad():
            rnn.b.zero_()

        u_context = torch.zeros(2)
        fps = find_fixed_points(
            rnn,
            u_context,
            n_inits=20,
            n_steps=2000,
            lr=1e-2,
            convergence_threshold=1e-4,
        )
        assert len(fps) >= 1, (
            f"Expected at least 1 fixed point, found {len(fps)}. "
            "Increase n_inits or relax convergence_threshold."
        )

        # Verify the fixed point satisfies ||dh/dt|| < tolerance
        h_star = fps[0]
        with torch.no_grad():
            net = rnn.W_rec @ h_star + rnn.W_in @ u_context + rnn.b
            dh = -h_star + rnn.f(net)
            residual = torch.linalg.norm(dh).item()
        assert residual < 1e-4, (
            f"Fixed point residual ||dh/dt|| = {residual:.2e}, expected < 1e-4. "
            "Fixed point does not satisfy the dynamics equation."
        )

    def test_find_fixed_points_returns_tensors(self) -> None:
        """find_fixed_points returns a list of (H,) detached tensors."""
        torch.manual_seed(0)
        rnn = ContinuousTimeRNN(n_input=2, n_hidden=4, n_output=2, activation="tanh")
        u = torch.zeros(2)
        fps = find_fixed_points(
            rnn, u, n_inits=5, n_steps=200, lr=1e-2, convergence_threshold=1.0
        )
        for fp in fps:
            assert isinstance(fp, torch.Tensor), (
                f"Expected torch.Tensor, got {type(fp)}"
            )
            assert fp.shape == (4,), f"Expected shape (4,), got {fp.shape}"
            assert not fp.requires_grad, "Returned fixed point should be detached"


class TestComputeJacobianAtFp:
    """Tests for compute_jacobian_at_fp()."""

    def test_jacobian_shape(self) -> None:
        """Jacobian at fixed point has shape (H, H)."""
        torch.manual_seed(7)
        H = 8
        rnn = ContinuousTimeRNN(n_input=3, n_hidden=H, n_output=2, activation="tanh")
        # h=0 is a fixed point when b=0 and W_in @ 0 = 0
        with torch.no_grad():
            rnn.b.zero_()
        h_star = torch.zeros(H)
        u = torch.zeros(3)
        J = compute_jacobian_at_fp(rnn, h_star, u)
        assert J.shape == (H, H), (
            f"Jacobian shape {J.shape} != expected ({H}, {H})"
        )

    def test_jacobian_at_origin_zero_bias(self) -> None:
        """At h=0, zero bias, zero input: J = -I + diag(f'(0)) @ W_rec.

        For tanh: f'(0) = 1, so J = -I + W_rec.
        """
        torch.manual_seed(3)
        H = 4
        rnn = ContinuousTimeRNN(n_input=2, n_hidden=H, n_output=2, activation="tanh")
        with torch.no_grad():
            rnn.b.zero_()
        h_star = torch.zeros(H)
        u = torch.zeros(2)
        J = compute_jacobian_at_fp(rnn, h_star, u)
        expected = -torch.eye(H) + rnn.W_rec.detach()
        assert torch.allclose(J, expected, atol=1e-5), (
            f"Jacobian deviates from analytical form -I + W_rec. "
            f"Max deviation: {(J - expected).abs().max().item():.2e}"
        )


class TestClassifyStability:
    """Tests for classify_stability()."""

    def test_classify_stability_stable(self) -> None:
        """Diagonal Jacobian with all negative entries is stable."""
        J = -torch.eye(4)
        result = classify_stability(J)
        assert result["stable"] is True, (
            f"Expected stable=True for -I, got {result['stable']}"
        )
        assert result["n_unstable"] == 0, (
            f"Expected n_unstable=0, got {result['n_unstable']}"
        )
        assert result["max_real_part"] < 0.0, (
            f"Expected max_real_part < 0, got {result['max_real_part']:.4f}"
        )
        assert result["eigenvalues"].shape == (4,), (
            f"Eigenvalues shape {result['eigenvalues'].shape} != (4,)"
        )

    def test_classify_stability_unstable(self) -> None:
        """Diagonal Jacobian with one positive entry is unstable."""
        J = torch.diag(torch.tensor([-1.0, -1.0, 0.5, -1.0]))
        result = classify_stability(J)
        assert result["stable"] is False, (
            f"Expected stable=False for diag([-1,-1,0.5,-1]), got {result['stable']}"
        )
        assert result["n_unstable"] >= 1, (
            f"Expected n_unstable >= 1, got {result['n_unstable']}"
        )
        assert result["max_real_part"] > 0.0, (
            f"Expected max_real_part > 0, got {result['max_real_part']:.4f}"
        )

    def test_classify_stability_marginally_stable(self) -> None:
        """Jacobian with eigenvalue at zero is classified as unstable."""
        J = torch.diag(torch.tensor([-1.0, 0.0, -2.0, -0.5]))
        result = classify_stability(J)
        # Eigenvalue at 0 has non-negative real part -> not stable
        assert result["stable"] is False, (
            "Zero eigenvalue should result in stable=False (n_unstable >= 1)"
        )
        assert result["n_unstable"] >= 1

    def test_classify_stability_result_types(self) -> None:
        """classify_stability returns correct Python types."""
        J = -2.0 * torch.eye(3)
        result = classify_stability(J)
        assert isinstance(result["stable"], bool)
        assert isinstance(result["n_unstable"], int)
        assert isinstance(result["max_real_part"], float)
        assert isinstance(result["eigenvalues"], torch.Tensor)
        assert result["eigenvalues"].is_complex()


class TestDeduplicateFixedPoints:
    """Tests for _deduplicate_fixed_points()."""

    def test_deduplicate_identical_points(self) -> None:
        """Identical fixed points are merged into one."""
        h = torch.ones(4)
        fps = [h.clone(), h.clone(), h.clone()]
        unique = _deduplicate_fixed_points(fps, dist_threshold=1e-3)
        assert len(unique) == 1, (
            f"Expected 1 unique point after deduplication, got {len(unique)}"
        )

    def test_deduplicate_distinct_points(self) -> None:
        """Well-separated fixed points are all retained."""
        fps = [
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            torch.tensor([0.0, 1.0, 0.0, 0.0]),
            torch.tensor([0.0, 0.0, 1.0, 0.0]),
        ]
        unique = _deduplicate_fixed_points(fps, dist_threshold=1e-3)
        assert len(unique) == 3, (
            f"Expected 3 distinct points, got {len(unique)}"
        )

    def test_deduplicate_empty_list(self) -> None:
        """Empty list returns empty list."""
        unique = _deduplicate_fixed_points([], dist_threshold=1e-3)
        assert unique == []

    def test_deduplicate_single_rnn_multiple_inits(self) -> None:
        """Multiple inits converging to same FP produce exactly one FP.

        Uses a small tanh RNN with zero bias so h=0 is the only fixed point.
        With n_inits=10 all starting near origin, deduplication should return 1.
        """
        torch.manual_seed(99)
        rnn = ContinuousTimeRNN(n_input=2, n_hidden=4, n_output=2, activation="tanh")
        with torch.no_grad():
            rnn.b.zero_()
            # Scale down W_rec so the origin is strongly attracting
            rnn.W_rec.mul_(0.1)

        u = torch.zeros(2)
        fps = find_fixed_points(
            rnn,
            u,
            n_inits=10,
            n_steps=1000,
            lr=1e-2,
            convergence_threshold=1e-4,
        )
        # All should converge to origin; deduplication gives 1
        assert len(fps) == 1, (
            f"Expected 1 fixed point after deduplication for contractive RNN, "
            f"got {len(fps)}. Fixed points: {[fp.tolist() for fp in fps]}"
        )
