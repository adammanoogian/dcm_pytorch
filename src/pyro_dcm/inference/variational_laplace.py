"""Variational Laplace (Gauss-Newton) inversion for DCM.

Implements the SPM-style Variational Laplace algorithm: second-order
Gauss-Newton optimization of the log-joint (MAP estimation) followed
by a Laplace approximation of the posterior covariance via the inverse
Hessian at the MAP.

This is the same algorithm SPM12 uses in ``spm_nlsi_GN.m`` for DCM
inversion, adapted to PyTorch with autograd-based Jacobians instead
of SPM's finite differences.

Key properties vs Pyro SVI with AutoNormal:

- **Second-order**: Newton step ``H⁻¹ g`` rescales weakly identified
  parameters (like B in bilinear DCM) by their inverse curvature,
  giving them larger steps. First-order methods (Adam) lack this.
- **Full covariance**: the Gauss-Newton Hessian ``J'ΛJ + Σ₀⁻¹`` is
  dense, capturing all pairwise parameter correlations (A↔B, A↔C).
  Mean-field SVI cannot do this.
- **Deterministic**: no stochastic ELBO gradients, no sampling noise.

References
----------
[REF-040] Friston et al. (2007). Variational free energy and the
    Laplace approximation. NeuroImage, 34(1), 220-234.
SPM12 source: ``spm_nlsi_GN.m``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

logger = logging.getLogger(__name__)


@dataclass
class VLResult:
    """Result of Variational Laplace inversion.

    Attributes
    ----------
    theta_map : torch.Tensor
        MAP estimate in unconstrained parameter space, shape ``(P,)``.
    posterior_cov : torch.Tensor
        Laplace posterior covariance, shape ``(P, P)``. Computed as the
        inverse Gauss-Newton Hessian at the MAP.
    free_energy : list[float]
        Free energy (negative variational free energy) at each iteration.
    n_iterations : int
        Number of Gauss-Newton iterations performed.
    converged : bool
        Whether the optimizer converged (step norm < ``tol``).
    noise_prec : float
        Estimated observation noise precision (1/σ²).
    """

    theta_map: torch.Tensor
    posterior_cov: torch.Tensor
    free_energy: list[float] = field(default_factory=list)
    n_iterations: int = 0
    converged: bool = False
    noise_prec: float = 1.0


def _jacobian_fd(
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    theta: torch.Tensor,
    y0: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Compute the Jacobian via forward finite differences.

    Parameters
    ----------
    forward_fn : callable
        ``theta -> predicted_bold.flatten()``, shape ``(T*N,)``.
    theta : torch.Tensor
        Current parameter vector, shape ``(P,)``.
    y0 : torch.Tensor
        ``forward_fn(theta)`` pre-computed, shape ``(T*N,)``.
    eps : float
        Finite-difference step size.

    Returns
    -------
    torch.Tensor
        Jacobian matrix, shape ``(T*N, P)``.
    """
    P = theta.numel()
    D = y0.numel()
    J = torch.zeros(D, P, dtype=theta.dtype, device=theta.device)
    for j in range(P):
        theta_plus = theta.clone()
        theta_plus[j] += eps
        y_plus = forward_fn(theta_plus)
        J[:, j] = (y_plus - y0) / eps
    return J


def _compute_free_energy(
    residuals: torch.Tensor,
    theta: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_prec: torch.Tensor,
    noise_prec: float,
    log_det_posterior_cov: float,
) -> float:
    """Compute the (negative) variational free energy.

    F = log p(y|θ) + log p(θ) + ½ log|Σ_post|

    Implements [REF-040] Eq. 4 (simplified for fixed noise precision).

    Parameters
    ----------
    residuals : torch.Tensor
        ``(y - ŷ).flatten()``, shape ``(D,)``.
    theta : torch.Tensor
        Current parameter vector, shape ``(P,)``.
    prior_mean : torch.Tensor
        Prior mean, shape ``(P,)``.
    prior_prec : torch.Tensor
        Prior precision matrix, shape ``(P, P)``.
    noise_prec : float
        Observation noise precision.
    log_det_posterior_cov : float
        Log-determinant of the current posterior covariance approximation.

    Returns
    -------
    float
        Variational free energy (higher is better).
    """
    D = residuals.numel()
    P = theta.numel()
    d_theta = theta - prior_mean
    log_lik = -0.5 * noise_prec * residuals @ residuals + 0.5 * D * torch.log(
        torch.tensor(noise_prec, dtype=theta.dtype)
    )
    log_prior = -0.5 * d_theta @ prior_prec @ d_theta
    two_pi = torch.tensor(2.0 * torch.pi)
    entropy = 0.5 * log_det_posterior_cov + 0.5 * P * (1.0 + torch.log(two_pi))
    return float((log_lik + log_prior + entropy).item())


def variational_laplace(
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    y_obs: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_cov: torch.Tensor,
    *,
    noise_prec_init: float = 1.0,
    max_iter: int = 128,
    tol: float = 1e-6,
    min_damping: float = 1e-4,
    max_damping: float = 1e8,
    fd_eps: float = 1e-4,
    update_noise_prec: bool = True,
) -> VLResult:
    """Gauss-Newton Variational Laplace inversion.

    Implements the core algorithm from ``spm_nlsi_GN.m`` [REF-040]:

    1. Start at prior mean.
    2. At each iteration, compute the Jacobian J = ∂ŷ/∂θ via finite
       differences (P+1 forward passes).
    3. Form the Gauss-Newton Hessian H = λ·J'J + Σ₀⁻¹.
    4. Compute gradient g = λ·J'r - Σ₀⁻¹(θ-μ₀).
    5. Solve for Newton step: Δθ = (H + αI)⁻¹ g.
    6. Accept step if free energy improves; increase damping otherwise.
    7. At convergence, Σ_post = H⁻¹.

    Parameters
    ----------
    forward_fn : callable
        Maps a flat parameter vector ``theta`` of shape ``(P,)`` to
        flattened predicted observations of shape ``(D,)`` where
        ``D = T * N``. Must be deterministic and differentiable
        (at least numerically).
    y_obs : torch.Tensor
        Observed data, shape ``(D,)`` (pre-flattened).
    prior_mean : torch.Tensor
        Prior mean in unconstrained parameter space, shape ``(P,)``.
    prior_cov : torch.Tensor
        Prior covariance, shape ``(P, P)``. Diagonal is typical.
    noise_prec_init : float
        Initial observation noise precision (1/σ²). Updated via
        ReML if ``update_noise_prec=True``.
    max_iter : int
        Maximum Gauss-Newton iterations.
    tol : float
        Convergence tolerance on step norm ``||Δθ||``.
    min_damping : float
        Minimum Levenberg-Marquardt damping factor.
    max_damping : float
        Maximum damping factor (triggers convergence failure).
    fd_eps : float
        Finite-difference step size for Jacobian computation.
    update_noise_prec : bool
        If True, update noise precision via ReML at each iteration.

    Returns
    -------
    VLResult
        MAP estimate, posterior covariance, free energy trace, and
        convergence diagnostics.
    """
    dtype = prior_mean.dtype
    device = prior_mean.device
    P = prior_mean.numel()
    D = y_obs.numel()

    prior_prec = torch.linalg.inv(prior_cov)
    theta = prior_mean.clone()
    noise_prec = noise_prec_init
    alpha = 1.0
    fe_trace: list[float] = []

    eye_P = torch.eye(P, dtype=dtype, device=device)

    for iteration in range(max_iter):
        y_pred = forward_fn(theta)
        residuals = y_obs - y_pred

        J = _jacobian_fd(forward_fn, theta, y_pred, eps=fd_eps)

        JtJ = J.T @ J
        H = noise_prec * JtJ + prior_prec
        d_theta = theta - prior_mean
        g = noise_prec * (J.T @ residuals) - prior_prec @ d_theta

        # Damped Newton step with line search on free energy
        step_accepted = False
        for _ls in range(16):
            H_damped = H + alpha * eye_P
            try:
                delta = torch.linalg.solve(H_damped, g)
            except torch.linalg.LinAlgError:
                alpha *= 10.0
                continue

            theta_new = theta + delta
            y_pred_new = forward_fn(theta_new)
            residuals_new = y_obs - y_pred_new

            try:
                log_det_cov = -torch.linalg.slogdet(H_damped)[1].item()
            except torch.linalg.LinAlgError:
                log_det_cov = 0.0

            fe_new = _compute_free_energy(
                residuals_new, theta_new, prior_mean, prior_prec,
                noise_prec, log_det_cov,
            )

            if iteration == 0 or fe_new >= fe_trace[-1] - 1e-4:
                step_accepted = True
                theta = theta_new
                residuals = residuals_new
                y_pred = y_pred_new
                fe_trace.append(fe_new)
                alpha = max(alpha / 2.0, min_damping)
                break
            else:
                alpha *= 10.0
                if alpha > max_damping:
                    break

        if not step_accepted:
            try:
                log_det_cov = -torch.linalg.slogdet(H)[1].item()
            except torch.linalg.LinAlgError:
                log_det_cov = 0.0
            fe_current = _compute_free_energy(
                residuals, theta, prior_mean, prior_prec,
                noise_prec, log_det_cov,
            )
            fe_trace.append(fe_current)
            logger.info(
                "VL iteration %d: step rejected (damping=%.1e), stopping",
                iteration, alpha,
            )
            break

        if update_noise_prec:
            sse = float((residuals @ residuals).item())
            noise_prec = max(D / sse, 1e-6) if sse > 0 else noise_prec

        step_norm = float(delta.norm().item())
        logger.debug(
            "VL iter %d: F=%.2f, ||Δθ||=%.2e, λ_noise=%.2f, α=%.2e",
            iteration, fe_trace[-1], step_norm, noise_prec, alpha,
        )

        if step_norm < tol:
            logger.info(
                "VL converged at iteration %d (||Δθ||=%.2e < tol=%.2e)",
                iteration, step_norm, tol,
            )
            break

    # Final posterior covariance from undamped Hessian
    y_pred_final = forward_fn(theta)
    J_final = _jacobian_fd(forward_fn, theta, y_pred_final, eps=fd_eps)
    H_final = noise_prec * (J_final.T @ J_final) + prior_prec
    try:
        posterior_cov = torch.linalg.inv(H_final)
    except torch.linalg.LinAlgError:
        logger.warning("Hessian singular at MAP; using damped inverse")
        posterior_cov = torch.linalg.inv(H_final + min_damping * eye_P)

    converged = iteration < max_iter - 1 and step_norm < tol

    return VLResult(
        theta_map=theta.detach(),
        posterior_cov=posterior_cov.detach(),
        free_energy=fe_trace,
        n_iterations=iteration + 1,
        converged=converged,
        noise_prec=noise_prec,
    )
