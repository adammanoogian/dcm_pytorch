"""Variational Laplace inference for spectral DCM.

Implements SPM12's ``spm_nlsi_GN`` Gauss-Newton optimization under the
Laplace approximation. Uses the Gauss-Newton Hessian approximation
(J^H @ iS @ J + prior_precision) with data-driven Wishart observation
precision Q from ``spm_dcm_csd_Q``, matching SPM12's approach.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from pyro_dcm.forward_models.neural_state import parameterize_A
from pyro_dcm.forward_models.spectral_transfer import spectral_dcm_forward
from pyro_dcm.inference.csd_precision import compute_csd_precision


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
    P_transit: torch.Tensor,
    P_decay: torch.Tensor,
    P_epsilon: torch.Tensor,
) -> torch.Tensor:
    """Flatten parameter tensors into a single vector.

    Layout: A_free(N*N) + noise_a(2) + noise_b(2) + noise_c(N) +
    P_transit(N) + P_decay(1) + P_epsilon(1) = N*N + 2N + 6.
    """
    return torch.cat([
        A_free.reshape(-1),
        noise_a.reshape(-1),
        noise_b.reshape(-1),
        noise_c.reshape(-1),
        P_transit.reshape(-1),
        P_decay.reshape(-1),
        P_epsilon.reshape(-1),
    ])


def _unpack_params(
    theta: torch.Tensor,
    N: int,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
]:
    """Reshape flat parameter vector back into named tensors.

    SPM12 fMRI parameter layout:
      A_free:    (N, N)  -- N*N params
      noise_a:   (2, 1)  -- 2 params (shared neuronal)
      noise_b:   (2, 1)  -- 2 params (global observation)
      noise_c:   (1, N)  -- N params (regional amplitude only)
      P_transit: (N,)    -- N params (per-region transit time)
      P_decay:   (1,)    -- 1 param  (shared signal decay)
      P_epsilon: (1,)    -- 1 param  (BOLD signal ratio)
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
    P_transit = theta[idx : idx + N]
    idx += N
    P_decay = theta[idx : idx + 1]
    idx += 1
    P_epsilon = theta[idx : idx + 1]
    idx += 1
    return A_free, noise_a, noise_b, noise_c, P_transit, P_decay, P_epsilon


def _param_count(N: int) -> int:
    """Total number of free parameters for N regions.

    A_free(N*N) + noise_a(2) + noise_b(2) + noise_c(N)
    + P_transit(N) + P_decay(1) + P_epsilon(1) = N*N + 2N + 6.
    """
    return N * N + 2 + 2 + N + N + 1 + 1


def _predicted_residual(
    theta: torch.Tensor,
    observed_csd: torch.Tensor,
    freqs: torch.Tensor,
    a_mask: torch.Tensor,
    N: int,
    eig_clamp: float | None = -1.0 / 32.0,
    mar_order: int = 7,
) -> torch.Tensor:
    """Compute residual vector (observed - predicted) as complex vector.

    Returns the complex residual matching SPM12's spm_vec(y) - spm_vec(f),
    which keeps complex entries as-is. Shape: (F*N*N,) complex128.
    """
    (A_free, noise_a, noise_b, noise_c,
     P_transit, P_decay, P_epsilon) = _unpack_params(theta, N)
    A = parameterize_A(A_free * a_mask.to(A_free.device))
    pred_csd = spectral_dcm_forward(
        A, freqs, noise_a, noise_b, noise_c, eig_clamp=eig_clamp,
        mar_order=mar_order,
        hemodynamic=True,
        P_transit=P_transit, P_decay=P_decay, P_epsilon=P_epsilon,
    )
    residual = observed_csd - pred_csd
    return residual.reshape(-1)


def _compute_jacobian(
    theta: torch.Tensor,
    observed_csd: torch.Tensor,
    freqs: torch.Tensor,
    a_mask: torch.Tensor,
    N: int,
    dx: float = 1e-6,
    eig_clamp: float | None = -1.0 / 32.0,
    mar_order: int = 7,
) -> torch.Tensor:
    """Compute Jacobian of residual w.r.t. theta via finite differences.

    Matches SPM12's ``spm_diff`` approach. Autograd through
    eigendecomposition is numerically unstable when eigenvalues are
    degenerate, so central finite differences are used instead.

    Returns complex-valued J where J[i, j] = d(residual_i) / d(theta_j).
    Shape: (n_data, n_params), dtype complex128.
    """
    n_params = theta.shape[0]
    res0 = _predicted_residual(
        theta, observed_csd, freqs, a_mask, N, eig_clamp=eig_clamp,
        mar_order=mar_order,
    )
    n_data = res0.shape[0]
    J = torch.zeros(n_data, n_params, dtype=torch.complex128, device=theta.device)

    for j in range(n_params):
        theta_plus = theta.clone()
        theta_plus[j] += dx
        res_plus = _predicted_residual(
            theta_plus, observed_csd, freqs, a_mask, N, eig_clamp=eig_clamp,
            mar_order=mar_order,
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
    regularization: float = 1.0 / 128.0,
    eig_clamp: float | None = -1.0 / 32.0,
    mar_order: int = 7,
) -> VariationalLaplaceResult:
    """Run Variational Laplace (Gauss-Newton) inference for spectral DCM.

    Implements SPM12's ``spm_nlsi_GN`` optimization using the
    Gauss-Newton Hessian approximation: H = real(J^H @ iS @ J) + ipC,
    where J is the complex Jacobian of the CSD residual and iS is the
    data-driven Wishart observation precision from ``spm_dcm_csd_Q``.

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
    regularization : float
        Levenberg-Marquardt damping added to Hessian diagonal.
    eig_clamp : float or None
        Maximum real part of eigenvalues for A matrix clamping.
        Default ``-1/32`` preserves fMRI behavior; use ``-1.0``
        for MEG; ``None`` disables clamping entirely.
    mar_order : int
        MAR model order for the CSD -> MAR -> CSD round-trip. Default
        7 matches SPM12's ``M.p - 1 = 8 - 1 = 7``. The round-trip is
        safe for VL (uses finite-difference Jacobians). Set to 0 to
        disable.

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

    # Build block-diagonal prior precision:
    # - A_free + noise params: prior_variance (default 1/64)
    # - Hemodynamic params (P_transit, P_decay, P_epsilon): 1/256
    #   (SPM12 spm_dcm_fmri_priors.m convention)
    hemo_var = 1.0 / 256.0
    n_connectivity_noise = N * N + 2 + 2 + N  # A_free + a + b + c
    n_hemo = N + 1 + 1  # P_transit + P_decay + P_epsilon
    prior_var_vec = torch.cat([
        torch.full((n_connectivity_noise,), prior_variance, dtype=torch.float64),
        torch.full((n_hemo,), hemo_var, dtype=torch.float64),
    ]).to(device)
    prior_precision = torch.diag(1.0 / prior_var_vec)

    # Compute data-driven observation precision Q (spm_dcm_csd_Q)
    Q_list, nq = compute_csd_precision(observed_csd)

    # Initialize hyperparameters h for observation precision components
    # SPM12 initializes h = hE = sparse(nh,1) - log(var(spm_vec(y))) + 4
    # For now, h = 0 (ReML M-step in Plan 02 will update h)
    nh = len(Q_list)
    h = torch.zeros(nh, dtype=torch.float64, device=device)

    # Compute iS = sum(Q{i} * exp(h(i))) -- observation precision
    # For fMRI, nh=1, so iS = Q[0] * exp(h[0])
    def _compute_iS() -> torch.Tensor:
        iS = torch.zeros_like(Q_list[0])
        for i_h in range(nh):
            iS = iS + Q_list[i_h] * (torch.exp(torch.tensor(-32.0)) + torch.exp(h[i_h]))
        return iS

    theta = prior_mean.clone()

    result = VariationalLaplaceResult()
    prev_F = float("inf")
    convergence_count = 0

    # SPM12 Levenberg-Marquardt regularization parameter (v in spm_nlsi_GN)
    v = 4.0  # initial value, decreased on success, increased on failure

    for iteration in range(max_iter):
        if not torch.isfinite(theta).all():
            theta = prior_mean.clone()

        # Compute complex residual: e = spm_vec(y) - spm_vec(f)
        e = _predicted_residual(
            theta, observed_csd, freqs, a_mask, N, eig_clamp=eig_clamp,
            mar_order=mar_order,
        )

        # Compute observation precision iS
        iS = _compute_iS()

        # Parameter deviation from prior
        p = theta - prior_mean

        # Free energy: F = L(1) + L(2) where
        # L(1) = -0.5 * real(e' @ iS @ e)   [observation fit]
        # L(2) = -0.5 * p' @ ipC @ p         [prior divergence]
        # (L(3) hyperprior term added in Plan 02 with ReML)
        ny = e.shape[0]
        L1 = -0.5 * (e.conj() @ iS @ e).real.item()
        L2 = -0.5 * (p @ prior_precision @ p).item()
        F_val = L1 + L2
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

        # Compute complex Jacobian: J[i,j] = d(e_i)/d(theta_j)
        J = _compute_jacobian(
            theta, observed_csd, freqs, a_mask, N, eig_clamp=eig_clamp,
            mar_order=mar_order,
        )

        # Gauss-Newton Hessian: Pp = real(J^H @ iS @ J)
        # SPM12 line 388: Pp = real(J'*iS*J)
        Pp = (J.conj().T @ iS @ J).real
        H_gn = Pp + prior_precision

        # Levenberg-Marquardt regularization (SPM12-style)
        # SPM12 uses spm_dx(dFdpp, dFdp, {v}) which applies (H + exp(v)*diag(H))
        H_gn = H_gn + regularization * torch.eye(
            n_params, dtype=torch.float64, device=device
        )

        # Gradient: dFdp = -real(J^H @ iS @ e) - ipC @ p
        # SPM12 line 468: dFdp = -real(J'*iS*e) - ipC*p
        grad = -(J.conj().T @ iS @ e).real - prior_precision @ p

        # Solve for update: dp = -H_gn^{-1} @ grad (Newton step on F)
        # Note: grad = dF/dp, and we want to maximize F, so dp = H^{-1} @ grad
        # But H_gn = -dF^2/dp^2 (negative Hessian), so dp = -H_gn^{-1} @ (-grad)
        # = H_gn^{-1} @ grad... Let me follow SPM12 exactly:
        # dFdpp = -Pp - ipC (line 469)
        # dp = spm_dx(dFdpp, dFdp, {v})
        # spm_dx with {v}: dp = inv(-dFdpp + exp(v)*diag(-dFdpp)) * dFdp
        # = inv(H_gn + exp(v)*diag(H_gn)) * grad
        lm_factor = torch.exp(torch.tensor(v, dtype=torch.float64))
        H_reg = H_gn + lm_factor * torch.diag(H_gn.diagonal())

        try:
            L_chol = torch.linalg.cholesky(H_reg)
            dtheta = torch.cholesky_solve(
                grad.unsqueeze(1), L_chol
            ).squeeze(1)
        except torch.linalg.LinAlgError:
            dtheta = torch.linalg.solve(
                H_reg + 0.1 * torch.eye(
                    n_params, dtype=torch.float64, device=device
                ),
                grad,
            )

        # Trial step: check if free energy improves
        theta_new = theta + dtheta
        if torch.isfinite(theta_new).all():
            try:
                e_trial = _predicted_residual(
                    theta_new, observed_csd, freqs, a_mask, N,
                    eig_clamp=eig_clamp, mar_order=mar_order,
                )
                p_trial = theta_new - prior_mean
                L1_trial = -0.5 * (e_trial.conj() @ iS @ e_trial).real.item()
                L2_trial = -0.5 * (p_trial @ prior_precision @ p_trial).item()
                F_trial = L1_trial + L2_trial
            except RuntimeError:
                F_trial = float("-inf")
        else:
            F_trial = float("-inf")

        # SPM12 accept/reject logic (lines 456-489)
        if F_trial > F_val or iteration < 3:
            # Accept: decrease regularization
            theta = theta_new
            v = min(v + 1.0 / 2.0, 4.0)
        else:
            # Reject: increase regularization, keep current theta
            v = min(v - 2.0, -4.0)

    result.n_iterations = iteration + 1

    # Final Gauss-Newton Hessian for posterior covariance
    iS_final = _compute_iS()
    J_final = _compute_jacobian(
        theta, observed_csd, freqs, a_mask, N, eig_clamp=eig_clamp,
        mar_order=mar_order,
    )
    Pp_final = (J_final.conj().T @ iS_final @ J_final).real
    H_final = Pp_final + prior_precision
    H_final = H_final + regularization * torch.eye(
        n_params, dtype=torch.float64, device=device
    )

    try:
        result.sigma_post = torch.inverse(H_final)
    except torch.linalg.LinAlgError:
        result.sigma_post = torch.linalg.pinv(H_final)

    with torch.no_grad():
        (A_free_post, na_post, nb_post, nc_post,
         pt_post, pd_post, pe_post) = _unpack_params(theta, N)
        A_post = parameterize_A(A_free_post * a_mask.to(A_free_post.device))
        pred_final = spectral_dcm_forward(
            A_post, freqs, na_post, nb_post, nc_post, eig_clamp=eig_clamp,
            mar_order=mar_order,
            hemodynamic=True,
            P_transit=pt_post, P_decay=pd_post, P_epsilon=pe_post,
        )

    result.theta_post = {
        "A_free": A_free_post,
        "A": A_post,
        "noise_a": na_post,
        "noise_b": nb_post,
        "noise_c": nc_post,
        "P_transit": pt_post,
        "P_decay": pd_post,
        "P_epsilon": pe_post,
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
        result.theta_post["P_transit"],
        result.theta_post["P_decay"],
        result.theta_post["P_epsilon"],
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
    idx += n_nc

    # Hemodynamic parameters
    posterior["P_transit"] = {
        "mean": result.theta_post["P_transit"],
        "std": std_vec[idx : idx + N],
        "samples": samples_flat[:, idx : idx + N],
    }
    idx += N

    posterior["P_decay"] = {
        "mean": result.theta_post["P_decay"],
        "std": std_vec[idx : idx + 1],
        "samples": samples_flat[:, idx : idx + 1],
    }
    idx += 1

    posterior["P_epsilon"] = {
        "mean": result.theta_post["P_epsilon"],
        "std": std_vec[idx : idx + 1],
        "samples": samples_flat[:, idx : idx + 1],
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
