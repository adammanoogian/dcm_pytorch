"""Variational Laplace inference for spectral DCM.

Implements SPM12's ``spm_nlsi_GN`` Gauss-Newton optimization under the
Laplace approximation. Uses the Gauss-Newton Hessian approximation
(J^T @ precision @ J + prior_precision) to avoid second derivatives
through eigendecomposition, matching SPM12's approach.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from pyro_dcm.forward_models.neural_state import parameterize_A
from pyro_dcm.forward_models.spectral_transfer import spectral_dcm_forward


@dataclass
class VariationalLaplaceResult:
    """Result container for Variational Laplace inference.

    Attributes
    ----------
    theta_post : dict[str, torch.Tensor]
        Posterior means for each parameter group.
    sigma_post : torch.Tensor
        Posterior covariance matrix (Laplace approximation).
    free_energy : list[float]
        Free energy at each iteration.
    converged : bool
        Whether the optimizer converged.
    n_iterations : int
        Number of iterations performed.
    predicted_csd : torch.Tensor
        Predicted CSD at the posterior mode, shape ``(F, N, N)``.
    """

    theta_post: dict[str, torch.Tensor] = field(default_factory=dict)
    sigma_post: torch.Tensor | None = None
    free_energy: list[float] = field(default_factory=list)
    converged: bool = False
    n_iterations: int = 0
    predicted_csd: torch.Tensor | None = None


def _pack_params(
    A_free: torch.Tensor,
    noise_a: torch.Tensor,
    noise_b: torch.Tensor,
    noise_c: torch.Tensor,
) -> torch.Tensor:
    """Flatten parameter tensors into a single vector."""
    return torch.cat([
        A_free.reshape(-1),
        noise_a.reshape(-1),
        noise_b.reshape(-1),
        noise_c.reshape(-1),
    ])


def _unpack_params(
    theta: torch.Tensor,
    N: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reshape flat parameter vector back into named tensors.

    SPM12 fMRI noise layout:
      A_free: (N, N)   -- N*N params
      noise_a: (2, 1)  -- 2 params (shared neuronal)
      noise_b: (2, 1)  -- 2 params (global observation)
      noise_c: (1, N)  -- N params (regional amplitude only)
    """
    idx = 0
    A_free = theta[idx : idx + N * N].reshape(N, N)
    idx += N * N
    noise_a = theta[idx : idx + 2].reshape(2, 1)
    idx += 2
    noise_b = theta[idx : idx + 2].reshape(2, 1)
    idx += 2
    noise_c = theta[idx : idx + N].reshape(1, N)
    idx += N
    return A_free, noise_a, noise_b, noise_c


def _param_count(N: int) -> int:
    """Total number of free parameters for N regions.

    A_free(N*N) + noise_a(2) + noise_b(2) + noise_c(N) = N*N + N + 4.
    """
    return N * N + 2 + 2 + N


def _predicted_residual(
    theta: torch.Tensor,
    observed_csd: torch.Tensor,
    freqs: torch.Tensor,
    a_mask: torch.Tensor,
    N: int,
    eig_clamp: float | None = -1.0 / 32.0,
) -> torch.Tensor:
    """Compute residual vector (observed - predicted) as real vector."""
    A_free, noise_a, noise_b, noise_c = _unpack_params(theta, N)
    A = parameterize_A(A_free * a_mask.to(A_free.device))
    pred_csd = spectral_dcm_forward(
        A, freqs, noise_a, noise_b, noise_c, eig_clamp=eig_clamp
    )
    residual = observed_csd - pred_csd
    return torch.cat([residual.real.reshape(-1), residual.imag.reshape(-1)])


def _compute_jacobian(
    theta: torch.Tensor,
    observed_csd: torch.Tensor,
    freqs: torch.Tensor,
    a_mask: torch.Tensor,
    N: int,
    dx: float = 1e-6,
    eig_clamp: float | None = -1.0 / 32.0,
) -> torch.Tensor:
    """Compute Jacobian of residual w.r.t. theta via finite differences.

    Matches SPM12's ``spm_diff`` approach. Autograd through
    eigendecomposition is numerically unstable when eigenvalues are
    degenerate, so central finite differences are used instead.

    Returns J where J[i, j] = d(residual_i) / d(theta_j).
    Shape: (n_data, n_params).
    """
    n_params = theta.shape[0]
    res0 = _predicted_residual(
        theta, observed_csd, freqs, a_mask, N, eig_clamp=eig_clamp
    )
    n_data = res0.shape[0]
    J = torch.zeros(n_data, n_params, dtype=torch.float64, device=theta.device)

    for j in range(n_params):
        theta_plus = theta.clone()
        theta_plus[j] += dx
        res_plus = _predicted_residual(
            theta_plus, observed_csd, freqs, a_mask, N, eig_clamp=eig_clamp
        )
        J[:, j] = (res_plus - res0) / dx

    return J


def run_variational_laplace(
    observed_csd: torch.Tensor,
    freqs: torch.Tensor,
    a_mask: torch.Tensor,
    N: int | None = None,
    max_iter: int = 128,
    tolerance: float = 1e-2,
    prior_variance: float = 1.0 / 64.0,
    initial_lambda: float = 8.0,
    regularization: float = 1.0 / 128.0,
    eig_clamp: float | None = -1.0 / 32.0,
) -> VariationalLaplaceResult:
    """Run Variational Laplace (Gauss-Newton) inference for spectral DCM.

    Implements SPM12's ``spm_nlsi_GN`` optimization using the
    Gauss-Newton Hessian approximation: H ≈ J^T @ Λ @ J + Π_prior,
    where J is the Jacobian of the predicted CSD residual and Λ is
    the observation precision. This avoids second derivatives through
    eigendecomposition.

    Parameters
    ----------
    observed_csd : torch.Tensor
        Observed cross-spectral density, shape ``(F, N, N)``, complex128.
    freqs : torch.Tensor
        Frequency vector in Hz, shape ``(F,)``, float64.
    a_mask : torch.Tensor
        Binary structural mask for A connections, shape ``(N, N)``.
    N : int or None
        Number of regions. Inferred from ``a_mask`` if None.
    max_iter : int
        Maximum Gauss-Newton iterations.
    tolerance : float
        Convergence criterion on free energy change.
    prior_variance : float
        Prior variance for all parameters (SPM12 default: 1/64).
        Use 1/16 for MEG/latent-circuit models.
    initial_lambda : float
        Initial log-precision for observation noise.
    regularization : float
        Levenberg-Marquardt damping added to Hessian diagonal.
    eig_clamp : float or None
        Maximum real part of eigenvalues for A matrix clamping.
        Default ``-1/32`` preserves fMRI behavior; use ``-1.0``
        for MEG; ``None`` disables clamping entirely.

    Returns
    -------
    VariationalLaplaceResult
        Posterior parameters, covariance, free energy trace, and
        convergence flag.
    """
    if N is None:
        N = a_mask.shape[0]

    n_params = _param_count(N)
    device = observed_csd.device

    prior_mean = torch.zeros(n_params, dtype=torch.float64, device=device)
    prior_precision = (1.0 / prior_variance) * torch.eye(
        n_params, dtype=torch.float64, device=device
    )

    theta = prior_mean.clone()
    lambda_precision = torch.tensor(
        initial_lambda, dtype=torch.float64, device=device
    )

    result = VariationalLaplaceResult()
    prev_F = float("inf")
    convergence_count = 0

    for iteration in range(max_iter):
        if not torch.isfinite(theta).all():
            theta = prior_mean.clone()

        res = _predicted_residual(
            theta, observed_csd, freqs, a_mask, N, eig_clamp=eig_clamp
        )
        n_data = res.shape[0]
        precision = torch.exp(lambda_precision)

        # Negative log-likelihood
        sse = (res @ res).item()
        nll = 0.5 * precision.item() * sse - 0.5 * n_data * lambda_precision.item()

        # KL from prior
        diff = theta - prior_mean
        kl = 0.5 * (diff @ prior_precision @ diff).item()

        F_val = -(nll + kl)
        result.free_energy.append(F_val)

        dF = abs(prev_F - F_val)
        if iteration > 0 and dF < tolerance:
            convergence_count += 1
            if convergence_count >= 4:
                result.converged = True
                break
        else:
            convergence_count = 0
        prev_F = F_val

        # Gauss-Newton: J^T @ precision @ J + prior_precision
        J = _compute_jacobian(
            theta, observed_csd, freqs, a_mask, N, eig_clamp=eig_clamp
        )

        H_gn = precision * (J.T @ J) + prior_precision
        H_gn = H_gn + regularization * torch.eye(
            n_params, dtype=torch.float64, device=device
        )

        # Gradient: precision * J^T @ residual + prior_precision @ (theta - mu_0)
        grad = -precision * (J.T @ res) + prior_precision @ (theta - prior_mean)

        try:
            L = torch.linalg.cholesky(H_gn)
            dtheta = torch.cholesky_solve(
                -grad.unsqueeze(1), L
            ).squeeze(1)
        except torch.linalg.LinAlgError:
            dtheta = torch.linalg.solve(
                H_gn + 0.1 * torch.eye(
                    n_params, dtype=torch.float64, device=device
                ),
                -grad,
            )

        # Adaptive step size (SPM12-style): backtrack until F improves
        step = 1.0
        theta_new = theta + step * dtheta
        for _ in range(8):
            if not torch.isfinite(theta_new).all():
                step *= 0.5
                theta_new = theta + step * dtheta
                continue
            try:
                res_trial = _predicted_residual(
                    theta_new, observed_csd, freqs, a_mask, N,
                    eig_clamp=eig_clamp,
                )
                sse_trial = (res_trial @ res_trial).item()
                diff_trial = theta_new - prior_mean
                kl_trial = 0.5 * (diff_trial @ prior_precision @ diff_trial).item()
                nll_trial = (
                    0.5 * precision.item() * sse_trial
                    - 0.5 * n_data * lambda_precision.item()
                )
                F_trial = -(nll_trial + kl_trial)
                if F_trial > F_val or step < 1e-4:
                    break
            except RuntimeError:
                pass
            step *= 0.5
            theta_new = theta + step * dtheta

        theta = theta_new

        # Update hyperparameter (observation log-precision)
        with torch.no_grad():
            try:
                res_new = _predicted_residual(
                    theta, observed_csd, freqs, a_mask, N,
                    eig_clamp=eig_clamp,
                )
                sse_new = (res_new @ res_new).item()
            except RuntimeError:
                sse_new = sse
            lambda_precision = torch.tensor(
                max(-4.0, torch.log(torch.tensor(n_data / max(sse_new, 1e-16))).item()),
                dtype=torch.float64, device=device,
            )

    result.n_iterations = iteration + 1

    # Final Gauss-Newton Hessian for posterior covariance
    J_final = _compute_jacobian(
        theta, observed_csd, freqs, a_mask, N, eig_clamp=eig_clamp
    )
    precision_final = torch.exp(lambda_precision)
    H_final = precision_final * (J_final.T @ J_final) + prior_precision
    H_final = H_final + regularization * torch.eye(
        n_params, dtype=torch.float64, device=device
    )

    try:
        result.sigma_post = torch.inverse(H_final)
    except torch.linalg.LinAlgError:
        result.sigma_post = torch.linalg.pinv(H_final)

    with torch.no_grad():
        A_free_post, na_post, nb_post, nc_post = _unpack_params(theta, N)
        A_post = parameterize_A(A_free_post * a_mask.to(A_free_post.device))
        pred_final = spectral_dcm_forward(
            A_post, freqs, na_post, nb_post, nc_post, eig_clamp=eig_clamp
        )

    result.theta_post = {
        "A_free": A_free_post,
        "A": A_post,
        "noise_a": na_post,
        "noise_b": nb_post,
        "noise_c": nc_post,
    }
    result.predicted_csd = pred_final

    return result


def extract_vl_posterior(
    result: VariationalLaplaceResult,
    N: int,
    num_samples: int = 1000,
) -> dict[str, dict[str, torch.Tensor]]:
    """Extract posterior in the same format as ``extract_posterior_params``.

    Parameters
    ----------
    result : VariationalLaplaceResult
        Output from ``run_variational_laplace``.
    N : int
        Number of regions.
    num_samples : int
        Number of posterior samples to draw from Laplace approximation.

    Returns
    -------
    dict
        Same structure as ``guides.extract_posterior_params``:
        ``{param_name: {'mean': Tensor, 'std': Tensor, 'samples': Tensor}}``.
    """
    theta_mean = _pack_params(
        result.theta_post["A_free"],
        result.theta_post["noise_a"],
        result.theta_post["noise_b"],
        result.theta_post["noise_c"],
    )

    try:
        L = torch.linalg.cholesky(result.sigma_post)
        z = torch.randn(num_samples, theta_mean.shape[0], dtype=torch.float64)
        samples_flat = theta_mean.unsqueeze(0) + (z @ L.T)
    except torch.linalg.LinAlgError:
        diag_var = torch.clamp(result.sigma_post.diagonal(), min=1e-16)
        samples_flat = theta_mean.unsqueeze(0) + torch.randn(
            num_samples, theta_mean.shape[0], dtype=torch.float64
        ) * diag_var.sqrt().unsqueeze(0)

    posterior = {}
    std_vec = result.sigma_post.diagonal().clamp(min=0).sqrt()

    idx = 0
    n_a = N * N
    posterior["A_free"] = {
        "mean": result.theta_post["A_free"],
        "std": std_vec[idx : idx + n_a].reshape(N, N),
        "samples": samples_flat[:, idx : idx + n_a].reshape(num_samples, N, N),
    }
    idx += n_a

    n_na = 2  # noise_a is (2, 1) shared
    posterior["noise_a"] = {
        "mean": result.theta_post["noise_a"],
        "std": std_vec[idx : idx + n_na].reshape(2, 1),
        "samples": samples_flat[:, idx : idx + n_na].reshape(num_samples, 2, 1),
    }
    idx += n_na

    posterior["noise_b"] = {
        "mean": result.theta_post["noise_b"],
        "std": std_vec[idx : idx + 2].reshape(2, 1),
        "samples": samples_flat[:, idx : idx + 2].reshape(num_samples, 2, 1),
    }
    idx += 2

    n_nc = N  # noise_c is (1, N) amplitude only
    posterior["noise_c"] = {
        "mean": result.theta_post["noise_c"],
        "std": std_vec[idx : idx + n_nc].reshape(1, N),
        "samples": samples_flat[:, idx : idx + n_nc].reshape(num_samples, 1, N),
    }

    # Include parameterized A (from theta_post) for convenience
    posterior["A"] = {
        "mean": result.theta_post["A"],
        "std": std_vec[:n_a].reshape(N, N),
    }

    posterior["median"] = {
        k: v["mean"]
        for k, v in posterior.items()
        if k not in ("median", "A")
    }

    return posterior
