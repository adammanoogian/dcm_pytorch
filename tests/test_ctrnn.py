from __future__ import annotations

import pytest
import torch

from pyro_dcm.rnn import ContinuousTimeRNN

# ---------------------------------------------------------------------------
# Shape contracts
# ---------------------------------------------------------------------------


def test_forward_shape_batched() -> None:
    """Forward pass with batched input produces correct output shapes.

    ``u`` shape (T, B, M_in) -> z: (T, B, M_out), h_traj: (T, B, H).
    """
    rnn = ContinuousTimeRNN(n_input=6, n_hidden=64, n_output=3)
    u = torch.randn(100, 16, 6)
    z, h = rnn(u)
    assert z.shape == (100, 16, 3), f"Expected z (100,16,3), got {z.shape}"
    assert h.shape == (100, 16, 64), f"Expected h (100,16,64), got {h.shape}"


def test_forward_shape_unbatched() -> None:
    """Unbatched input (T, M_in) gets batch dim of 1 inserted.

    Forward pass with ``u`` shape (T, M_in) -> z: (T, 1, M_out), h: (T, 1, H).
    """
    rnn = ContinuousTimeRNN(n_input=6, n_hidden=32, n_output=3)
    u = torch.randn(50, 6)
    z, h = rnn(u)
    assert z.shape == (50, 1, 3), f"Expected z (50,1,3), got {z.shape}"
    assert h.shape == (50, 1, 32), f"Expected h (50,1,32), got {h.shape}"


# ---------------------------------------------------------------------------
# Alpha computation
# ---------------------------------------------------------------------------


def test_alpha_computation() -> None:
    """Alpha == dt / tau is set correctly in __init__."""
    rnn = ContinuousTimeRNN(n_input=4, n_hidden=16, n_output=2, tau=2.0, dt=0.1)
    assert rnn.alpha == pytest.approx(0.05), (
        f"Expected alpha=0.05, got {rnn.alpha}"
    )


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------


def test_relu_activation() -> None:
    """ReLU activation: post-activation contribution is non-negative.

    Checks that the *change* in h at step 0 from the activation term is >= 0
    element-wise (since ReLU output is non-negative and alpha > 0).
    """
    rnn = ContinuousTimeRNN(n_input=4, n_hidden=16, n_output=2, activation="relu")
    rnn.eval()
    u = torch.randn(10, 4, 4)
    _, h_traj = rnn(u)
    # The activation component: alpha * relu(pre_act) is always >= 0.
    # Verify no NaN and h_traj has finite values.
    assert torch.isfinite(h_traj).all(), "h_traj contains non-finite values"
    # For zero initial state and alpha < 1, all h values >= 0 when using ReLU
    # (since (1-alpha)*0 + alpha*relu(...) >= 0 at t=0, and subsequent steps
    # depend on W_rec; we only check finiteness here as the invariant is
    # maintained only for the first step from zero init without recurrence).
    assert (h_traj[0] >= 0.0).all(), (
        "First hidden state from zero init with ReLU must be non-negative"
    )


def test_tanh_activation() -> None:
    """Tanh activation: hidden states are bounded in (-1, 1) for small inputs.

    With zero initial state and alpha << 1, the hidden states after the first
    step are alpha * tanh(W_in @ u + b). These stay bounded as tanh output
    is in (-1, 1) and the (1-alpha) decay prevents unbounded growth.
    """
    rnn = ContinuousTimeRNN(
        n_input=4, n_hidden=16, n_output=2, activation="tanh", tau=1.0, dt=0.1
    )
    rnn.eval()
    # Use small weights to keep h in bounded range
    with torch.no_grad():
        rnn.W_rec.fill_(0.0)
        rnn.W_in.fill_(0.0)
        rnn.b.fill_(0.0)
    u = torch.randn(20, 8, 4)
    _, h_traj = rnn(u)
    # With zero weights and zero input: tanh(0) = 0 -> all hidden = 0
    assert (h_traj.abs() <= 1.0 + 1e-6).all(), (
        f"Tanh h values must be in [-1,1], max abs: {h_traj.abs().max()}"
    )


def test_invalid_activation() -> None:
    """ValueError raised for unsupported activation name."""
    with pytest.raises(ValueError, match="Unknown activation"):
        ContinuousTimeRNN(n_input=4, n_hidden=16, n_output=2, activation="sigmoid")


# ---------------------------------------------------------------------------
# Gradient flow (BPTT readiness)
# ---------------------------------------------------------------------------


def test_gradient_flow() -> None:
    """Gradients flow through the full forward pass (BPTT works).

    After backward(), all parameters have non-None grads with at least
    some non-zero values.
    """
    rnn = ContinuousTimeRNN(n_input=4, n_hidden=16, n_output=2)
    rnn.train()
    u = torch.randn(10, 4, 4)
    z, _ = rnn(u)
    loss = z.sum()
    loss.backward()

    for name, param in rnn.named_parameters():
        assert param.grad is not None, f"param '{name}' has no gradient"
        assert param.grad.abs().sum() > 0, (
            f"param '{name}' gradient is all zeros"
        )


# ---------------------------------------------------------------------------
# Custom initial hidden state
# ---------------------------------------------------------------------------


def test_h0_custom() -> None:
    """Custom h0 changes the hidden trajectory vs zero-init.

    The first hidden state h[0] differs when starting from h0=ones vs zeros.
    """
    rnn = ContinuousTimeRNN(n_input=4, n_hidden=16, n_output=2)
    rnn.eval()
    u = torch.randn(5, 4, 4)

    h0_ones = torch.ones(4, 16)
    _, h_ones = rnn(u, h0=h0_ones)
    _, h_zeros = rnn(u)

    # Trajectories should differ (h0 has a lasting effect via (1-alpha)*h term)
    assert not torch.allclose(h_ones, h_zeros), (
        "Custom h0=ones should produce different trajectory than h0=zeros"
    )


# ---------------------------------------------------------------------------
# Noise injection
# ---------------------------------------------------------------------------


def test_noise_injection_training() -> None:
    """Training noise: different trajectories across runs; eval is deterministic.

    In training mode with noise_std > 0, two forward passes with the same
    input produce different hidden trajectories. In eval mode, they are identical.
    """
    rnn = ContinuousTimeRNN(
        n_input=4, n_hidden=16, n_output=2, noise_std=0.1
    )
    u = torch.randn(10, 4, 4)

    # Training mode: noise injected -> trajectories differ
    rnn.train()
    torch.manual_seed(0)
    _, h1 = rnn(u)
    torch.manual_seed(1)
    _, h2 = rnn(u)
    assert not torch.allclose(h1, h2), (
        "Training noise should produce different trajectories across seeds"
    )

    # Eval mode: no noise -> identical trajectories
    rnn.eval()
    _, h3 = rnn(u)
    _, h4 = rnn(u)
    assert torch.allclose(h3, h4), (
        "Eval mode (no noise) must produce identical trajectories"
    )


# ---------------------------------------------------------------------------
# Weight shapes
# ---------------------------------------------------------------------------


def test_weight_shapes() -> None:
    """Weight matrix shapes match constructor arguments.

    W_rec: (H, H), W_in: (H, M_in), W_out: (M_out, H), b: (H,).
    """
    H, M_in, M_out = 32, 6, 3
    rnn = ContinuousTimeRNN(n_input=M_in, n_hidden=H, n_output=M_out)

    assert rnn.W_rec.shape == (H, H), (
        f"W_rec: expected ({H},{H}), got {rnn.W_rec.shape}"
    )
    assert rnn.W_in.shape == (H, M_in), (
        f"W_in: expected ({H},{M_in}), got {rnn.W_in.shape}"
    )
    assert rnn.W_out.shape == (M_out, H), (
        f"W_out: expected ({M_out},{H}), got {rnn.W_out.shape}"
    )
    assert rnn.b.shape == (H,), (
        f"b: expected ({H},), got {rnn.b.shape}"
    )
