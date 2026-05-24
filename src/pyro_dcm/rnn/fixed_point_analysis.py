"""Fixed-point analysis utilities for CT-RNN latent circuit diagnostics.

Provides fixed-point finding, Jacobian computation, and eigenvalue-based
stability classification. Used by Phase 22 (PIPE-03) linearization quality
diagnostic.

References
----------
Langdon & Engel (2025) trainRNNbrain (interim cite; formal REF-ID in Phase 25).
Golub et al. (2018) Neuron -- fixed-point finding via ||dh/dt||^2 minimization.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.autograd.functional import jacobian

from pyro_dcm.rnn.continuous_time_rnn import ContinuousTimeRNN


def find_fixed_points(
    rnn: ContinuousTimeRNN,
    u_context: torch.Tensor,
    n_inits: int = 100,
    n_steps: int = 5000,
    lr: float = 1e-3,
    tol: float = 1e-12,
    convergence_threshold: float = 1e-6,
) -> list[torch.Tensor]:
    """Find fixed points of CT-RNN dynamics via Adam optimization.

    Minimizes ``||dh/dt||^2 = ||-h + f(W_rec @ h + W_in @ u + b)||^2``
    over hidden state ``h`` for a fixed input context ``u_context``.
    Multiple random initializations are used and results are deduplicated.

    Parameters
    ----------
    rnn : ContinuousTimeRNN
        Trained CT-RNN model whose fixed points are sought.
    u_context : torch.Tensor, shape (M_in,)
        Fixed input context at which fixed points are computed.
    n_inits : int, optional
        Number of random initializations. Default 100.
    n_steps : int, optional
        Maximum Adam optimization steps per initialization. Default 5000.
    lr : float, optional
        Adam learning rate. Default 1e-3.
    tol : float, optional
        Early-stopping tolerance: stop if ``loss < tol``. Default 1e-12.
    convergence_threshold : float, optional
        Keep a candidate if final ``loss < convergence_threshold``.
        Default 1e-6.

    Returns
    -------
    list of torch.Tensor
        Deduplicated list of fixed-point tensors, each of shape ``(H,)``.
        Empty list if no initializations converged.

    Notes
    -----
    Fixed points satisfy ``tau * dh/dt = -h + f(W_rec @ h + W_in @ u + b) = 0``,
    i.e., ``h* = f(W_rec @ h* + W_in @ u + b)``.
    The optimization objective is ``(1/H) * sum_i (dh_i/dt)^2``.
    Deduplication merges points within L2 distance ``< 1e-3``.
    """
    rnn.eval()
    fixed_points: list[torch.Tensor] = []

    for _ in range(n_inits):
        h = nn.Parameter(torch.randn(rnn.n_hidden, device=u_context.device) * 0.1)
        opt = torch.optim.Adam([h], lr=lr)

        final_loss = float("inf")
        for _ in range(n_steps):
            net = rnn.W_rec @ h + rnn.W_in @ u_context + rnn.b
            dh = -h + rnn.f(net)
            loss = (dh**2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            final_loss = loss.item()
            if final_loss < tol:
                break

        if final_loss < convergence_threshold:
            fixed_points.append(h.detach().clone())

    return _deduplicate_fixed_points(fixed_points)


def _deduplicate_fixed_points(
    fps: list[torch.Tensor],
    dist_threshold: float = 1e-3,
) -> list[torch.Tensor]:
    """Remove duplicate fixed points within a given L2 distance threshold.

    Parameters
    ----------
    fps : list of torch.Tensor
        Candidate fixed points, each of shape ``(H,)``.
    dist_threshold : float, optional
        L2 distance below which two fixed points are considered identical.
        Default 1e-3.

    Returns
    -------
    list of torch.Tensor
        Deduplicated fixed points.
    """
    if not fps:
        return []

    unique: list[torch.Tensor] = [fps[0]]
    for candidate in fps[1:]:
        is_dup = False
        with torch.no_grad():
            for existing in unique:
                dist = torch.linalg.norm(candidate - existing).item()
                if dist < dist_threshold:
                    is_dup = True
                    break
        if not is_dup:
            unique.append(candidate)
    return unique


def compute_jacobian_at_fp(
    rnn: ContinuousTimeRNN,
    h_star: torch.Tensor,
    u: torch.Tensor,
) -> torch.Tensor:
    """Compute the Jacobian of CT-RNN dynamics at a fixed point.

    Returns ``d(dh/dt)/dh`` evaluated at ``h_star`` for fixed input ``u``.
    Analytically this equals ``-I + diag(f'(net)) @ W_rec``, but is computed
    via automatic differentiation for generality.

    Parameters
    ----------
    rnn : ContinuousTimeRNN
        CT-RNN model. Weights are not modified.
    h_star : torch.Tensor, shape (H,)
        Fixed point at which the Jacobian is evaluated.
    u : torch.Tensor, shape (M_in,)
        Input context at the fixed point.

    Returns
    -------
    torch.Tensor, shape (H, H)
        Jacobian matrix ``J = d(dh/dt)/dh`` at ``h_star``.

    Notes
    -----
    Uses ``torch.autograd.functional.jacobian`` for exact automatic
    differentiation. The dynamics function is::

        dh/dt = -h + f(W_rec @ h + W_in @ u + b)

    The Jacobian captures the local linearization used in linearization
    quality diagnostics (Phase 22, PIPE-03).
    """

    def dynamics(h: torch.Tensor) -> torch.Tensor:
        return -h + rnn.f(rnn.W_rec @ h + rnn.W_in @ u + rnn.b)

    h_star_detached = h_star.detach().requires_grad_(True)
    J = jacobian(dynamics, h_star_detached)
    return J


def classify_stability(jacobian_matrix: torch.Tensor) -> dict:
    """Classify fixed-point stability from eigenvalues of the Jacobian.

    A fixed point is asymptotically stable if all eigenvalues have strictly
    negative real parts (i.e., all trajectories in the linearized system
    decay to zero).

    Parameters
    ----------
    jacobian_matrix : torch.Tensor, shape (H, H)
        Jacobian of the dynamics at a fixed point, as returned by
        ``compute_jacobian_at_fp``.

    Returns
    -------
    dict
        Keys:

        ``eigenvalues`` : torch.Tensor, shape (H,), complex
            Full complex eigenvalue spectrum.
        ``stable`` : bool
            ``True`` iff all eigenvalue real parts are strictly negative.
        ``n_unstable`` : int
            Count of eigenvalues with non-negative real part.
        ``max_real_part`` : float
            Maximum real part across all eigenvalues (negative => stable).

    Notes
    -----
    Uses ``torch.linalg.eig`` which returns complex eigenvalues for general
    (non-symmetric) matrices. The real part of each eigenvalue determines
    growth (positive) or decay (negative) along the corresponding mode.
    """
    eigenvalues = torch.linalg.eig(jacobian_matrix).eigenvalues  # (H,) complex
    real_parts = eigenvalues.real
    return {
        "eigenvalues": eigenvalues,
        "stable": bool((real_parts < 0).all().item()),
        "n_unstable": int((real_parts >= 0).sum().item()),
        "max_real_part": float(real_parts.max().item()),
    }
