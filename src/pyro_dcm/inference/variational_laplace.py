"""Variational Laplace inference for spectral DCM.

Implements SPM12's ``spm_nlsi_GN`` Gauss-Newton optimization under the
Laplace approximation. Uses the Gauss-Newton Hessian approximation
(J^H @ iS @ J + prior_precision) with data-driven Wishart observation
precision Q from ``spm_dcm_csd_Q``, and ReML M-step for hyperparameter
estimation, matching SPM12's approach.

The parameter space is projected onto the SVD subspace of the prior
covariance (``spm_svd(pC, 0)``), which removes zero-variance dimensions
(absent connections from ``a_mask``) and improves numerical stability.
All E-step/M-step operations run in the reduced space; the posterior
is mapped back to full space for output.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import torch

from pyro_dcm.forward_models.neural_state import parameterize_A
from pyro_dcm.forward_models.spectral_transfer import spectral_dcm_forward
from pyro_dcm.inference.csd_precision import compute_csd_precision

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SPM12 helper functions
# ---------------------------------------------------------------------------


def _spm_logdet(M: torch.Tensor) -> float:
    """Compute log-determinant robustly, matching ``spm_logdet.m``.

    Handles positive semi-definite matrices by:
    1. Removing zero-variance rows/cols (zero diagonal)
    2. Trying Cholesky (fast path for pos-def)
    3. Falling back to SVD, keeping only positive singular values

    Parameters
    ----------
    M : torch.Tensor
        Square matrix (real or complex). If complex, uses real part.

    Returns
    -------
    float
        log(det(M)), or sum of log of positive singular values.
    """
    if M.is_complex():
        M = M.real
    M = M.double()

    # Remove null variances (zero diagonal entries)
    diag_vals = M.diagonal()
    nonzero = diag_vals.abs() > 0
    if nonzero.sum() == 0:
        return 0.0
    idx = torch.where(nonzero)[0]
    M_sub = M[idx][:, idx]

    TOL = 1e-16

    # Try Cholesky first (fast for positive definite)
    try:
        L = torch.linalg.cholesky(M_sub)
        return 2.0 * L.diagonal().abs().log().sum().item()
    except torch.linalg.LinAlgError:
        pass

    # Fall back to SVD
    s = torch.linalg.svdvals(M_sub)
    s_pos = s[(s > TOL) & (s < 1.0 / TOL)]
    if s_pos.numel() == 0:
        return 0.0
    return s_pos.log().sum().item()


def _spm_trace(A: torch.Tensor, B: torch.Tensor) -> float:
    """Efficient trace(A @ B) matching ``spm_trace.m``.

    Uses the identity: trace(A @ B) = sum(A^T * B) for real matrices.
    SPM12: ``C = sum(sum(A'.*B))``.

    Parameters
    ----------
    A, B : torch.Tensor
        Square matrices of compatible dimensions.

    Returns
    -------
    float
        trace(A @ B)
    """
    val = (A.T * B).sum()
    return val.real.item() if val.is_complex() else val.item()


def _spm_dx(
    dfdx: torch.Tensor,
    f: torch.Tensor,
    t: float,
) -> torch.Tensor:
    """Regularized descent matching ``spm_dx.m``.

    Implements the augmented-system matrix exponential approach:

        dx = expm([0, 0; t*f, t*dfdx])[2:end, 1]

    where t = exp(t_input - spm_logdet(dfdx) / n) when t_input is a
    regularization parameter (cell-like usage in SPM12: ``spm_dx(H, g, {v})``).

    For large t (> exp(16)), falls back to the pseudo-inverse solution:
        dx = -pinv(dfdx) @ f

    Parameters
    ----------
    dfdx : torch.Tensor
        Jacobian/Hessian matrix, shape (n, n).
    f : torch.Tensor
        Gradient/force vector, shape (n,).
    t : float
        Regularization parameter. Corresponds to ``{t}`` in SPM12,
        which sets t_scalar = exp(t - spm_logdet(dfdx)/n).

    Returns
    -------
    torch.Tensor
        Parameter update dx, shape (n,).
    """
    n = f.shape[0]

    # Compute regularized time step: t_scalar = exp(t - logdet(dfdx)/n)
    logdet_val = _spm_logdet(dfdx)
    t_scalar = math.exp(t - logdet_val / n) if abs(logdet_val / n) < 500 else 0.0

    # If t_scalar is very large, use pseudo-inverse (Newton step)
    if t_scalar > math.exp(16):
        try:
            dx = -torch.linalg.pinv(dfdx) @ f
        except Exception:
            dx = torch.zeros_like(f)
        return dx.real if dx.is_complex() else dx

    # Augmented system: J_aug = [0, 0^T; t*f, t*dfdx]
    # Shape: (n+1, n+1)
    J_aug = torch.zeros(n + 1, n + 1, dtype=torch.float64, device=f.device)
    # J_aug[0, :] = 0  (already zero)
    J_aug[1:, 0] = t_scalar * f  # t*f column
    J_aug[1:, 1:] = t_scalar * dfdx  # t*dfdx block

    # Matrix exponential
    try:
        exp_J = torch.linalg.matrix_exp(J_aug)
        dx = exp_J[1:, 0]  # Extract first column, skip row 0
    except Exception:
        # Fallback to damped Newton
        try:
            dx = -torch.linalg.pinv(dfdx) @ f
        except Exception:
            dx = torch.zeros_like(f)

    return dx.real if dx.is_complex() else dx


def _spm_svd(
    pC: torch.Tensor,
    threshold: float = 0.0,
) -> torch.Tensor:
    """SVD-based dimension reduction matching ``spm_svd.m``.

    Computes the SVD of the prior covariance matrix and returns the
    left singular vectors corresponding to non-negligible singular
    values. With threshold=0, uses 64*eps as the effective threshold,
    which keeps all dimensions with non-zero prior variance and removes
    those with exactly zero variance (absent connections).

    Parameters
    ----------
    pC : torch.Tensor
        Prior covariance matrix, shape ``(np_full, np_full)``.
        Typically diagonal with per-parameter variances.
    threshold : float
        Threshold for normalized singular values. Values <= 0
        are replaced by 64*eps (matching SPM12 convention).

    Returns
    -------
    torch.Tensor
        Projection matrix V, shape ``(np_full, np_reduced)``,
        where np_reduced <= np_full.
    """
    # Match spm_svd.m threshold logic
    if threshold >= 1.0:
        threshold = threshold - 1e-6
    if threshold <= 0.0:
        threshold = 64.0 * torch.finfo(torch.float64).eps

    pC = pC.double()

    # For symmetric pC, eigendecomposition is equivalent to SVD
    # spm_svd does: [u, S, v] = svd(X, 0); s = diag(S).^2;
    # j = find(s*n/sum(s) > threshold); V = v(:, j)
    U, S, Vh = torch.linalg.svd(pC, full_matrices=False)
    s = S ** 2  # squared singular values (eigenvalues of pC^2)
    n = s.shape[0]
    s_sum = s.sum()

    if s_sum < 1e-32:
        # All zeros -- return identity (no reduction possible)
        return torch.eye(pC.shape[0], dtype=torch.float64, device=pC.device)

    # Keep dimensions where s_i * n / sum(s) > threshold
    keep = (s * n / s_sum) > threshold
    V = Vh.T[:, keep]

    if V.shape[1] == 0:
        # Fallback: keep at least one dimension
        V = Vh.T[:, :1]

    return V


@dataclass
class VariationalLaplaceResult:
    """Result container for Variational Laplace inference.

    Attributes
    ----------
    theta_post : dict[str, torch.Tensor]
        Posterior means for each parameter group.
    sigma_post : torch.Tensor
        Posterior covariance matrix (Laplace approximation),
        shape ``(np_full, np_full)`` in full parameter space.
    free_energy : list[float]
        Free energy at each iteration.
    converged : bool
        Whether the optimizer converged.
    n_iterations : int
        Number of iterations performed.
    predicted_csd : torch.Tensor
        Predicted CSD at the posterior mode, shape ``(F, N, N)``.
    n_reduced_params : int
        Number of parameters in the SVD-reduced space. Useful for
        callers that need to construct ``initial_p`` vectors for
        multi-restart optimization.
    """

    theta_post: dict[str, torch.Tensor] = field(default_factory=dict)
    sigma_post: torch.Tensor | None = None
    free_energy: list[float] = field(default_factory=list)
    converged: bool = False
    n_iterations: int = 0
    predicted_csd: torch.Tensor | None = None
    n_reduced_params: int = 0


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
    initial_p: torch.Tensor | None = None,
) -> VariationalLaplaceResult:
    """Run Variational Laplace (Gauss-Newton) inference for spectral DCM.

    Implements SPM12's ``spm_nlsi_GN`` optimization with:
    - SVD dimension reduction of parameter space (``spm_svd(pC, 0)``)
    - ReML M-step (8 inner Fisher-scoring iterations) for hyperparameters
    - 3-term free energy: L(1) data fit + L(2) parameter KL + L(3) hyperprior
    - Accept/reject with adaptive regularization (v parameter)
    - Convergence via predicted dF criterion (4 consecutive dF < 0.1)

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
        Convergence criterion (unused -- kept for API compat). SPM12
        uses predicted dF < 0.1 for 4 consecutive iterations.
    prior_variance : float
        Prior variance for connectivity/noise parameters (SPM12: 1/64).
    regularization : float
        Base Levenberg-Marquardt damping (unused -- kept for API compat).
        SPM12 uses adaptive v parameter with spm_dx.
    eig_clamp : float or None
        Maximum real part of eigenvalues for A matrix clamping.
    mar_order : int
        MAR model order for CSD round-trip (default 7 = SPM12 M.p-1).
    initial_p : torch.Tensor or None, optional
        Initial parameter vector in SVD-reduced space, shape
        ``(n_reduced_params,)``. If None, starts from zeros (SPM12
        default). Used for multi-restart optimization where each
        restart begins from a different starting point.

    Returns
    -------
    VariationalLaplaceResult
        Posterior parameters, covariance, free energy trace, and
        convergence flag.
    """
    if N is None:
        N = a_mask.shape[0]

    np_full = _param_count(N)
    device = observed_csd.device

    prior_mean = torch.zeros(np_full, dtype=torch.float64, device=device)

    # Build block-diagonal prior covariance (pC_full):
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

    # Zero out prior variance for A_free entries where a_mask is zero.
    # SPM12 spm_dcm_fmri_priors.m: absent connections get pC=0, so SVD
    # of pC removes those dimensions from the optimization.
    a_mask_flat = a_mask.reshape(-1).to(device)
    prior_var_vec[:N * N] = prior_var_vec[:N * N] * (a_mask_flat > 0).double()

    pC_full = torch.diag(prior_var_vec)

    # SVD dimension reduction (SPM12 spm_nlsi_GN.m lines 278-294)
    # V: (np_full, np_reduced) -- projects full space to reduced space
    # With threshold=0 -> 64*eps, this removes zero-variance dimensions
    # (e.g., A parameters where a_mask is zero)
    V = _spm_svd(pC_full, threshold=0.0)
    np_reduced = V.shape[1]
    log.info(
        "SVD dimension reduction: np_full=%d -> np_reduced=%d",
        np_full, np_reduced,
    )

    # Prior in reduced space (SPM12: pC = V'*pC*V; ipC = inv(pC))
    pC_reduced = V.T @ pC_full @ V
    ipC = torch.linalg.inv(pC_reduced)

    # Compute data-driven observation precision Q (spm_dcm_csd_Q)
    Q_list, nq = compute_csd_precision(observed_csd)
    nh = len(Q_list)

    # Hyperprior setup (SPM12 spm_nlsi_GN.m lines 248-267)
    # hE = sparse(nh,1) - log(var(spm_vec(y))) + 4
    y_vec = observed_csd.reshape(-1)
    # SPM12 uses var of the vectorized (complex) data; we use real parts
    # since var is not defined for complex in the same way
    var_y = torch.var(y_vec.real)
    if var_y < 1e-32:
        var_y = torch.tensor(1e-32, dtype=torch.float64)
    hE = torch.zeros(nh, dtype=torch.float64, device=device) - torch.log(var_y) + 4.0

    # ihC = eye(nh) * exp(4)  (hyperprior precision)
    ihC = torch.eye(nh, dtype=torch.float64, device=device) * math.exp(4)

    # Initialize hyperparameters at prior
    h = hE.clone()

    # Initialize reduced-space parameter deviation
    if initial_p is not None:
        if initial_p.shape[0] != np_reduced:
            msg = (
                f"initial_p has {initial_p.shape[0]} elements but "
                f"reduced space has {np_reduced} dimensions"
            )
            raise ValueError(msg)
        p = initial_p.to(dtype=torch.float64, device=device).clone()
        theta = prior_mean + V @ p
        log.info("Using provided initial_p (norm=%.4f)", p.norm().item())
    else:
        # p = V' @ (theta - prior_mean) = V' @ zeros = zeros
        p = torch.zeros(np_reduced, dtype=torch.float64, device=device)
        theta = prior_mean.clone()  # full-space parameters for forward model

    # SPM12 initialization (lines 299-304)
    criterion = [False, False, False, False]
    C_F = float("-inf")  # best free energy
    C_p = torch.zeros(np_reduced, dtype=torch.float64, device=device)
    C_h = h.clone()
    C_Cp = V.T @ pC_full @ V  # initial Cp = pC in reduced space
    v = -4.0  # log ascent rate (start heavily regularized)

    # Gradients (initialized, updated on accept) -- in reduced space
    dFdp = torch.zeros(np_reduced, dtype=torch.float64, device=device)
    dFdpp = torch.zeros(
        np_reduced, np_reduced, dtype=torch.float64, device=device,
    )

    result = VariationalLaplaceResult()
    result.n_reduced_params = np_reduced

    for iteration in range(max_iter):
        if not torch.isfinite(theta).all():
            theta = prior_mean.clone()
            p = torch.zeros(np_reduced, dtype=torch.float64, device=device)

        # E-Step prediction: residual and Jacobian
        # ================================================================
        e = _predicted_residual(
            theta, observed_csd, freqs, a_mask, N, eig_clamp=eig_clamp,
            mar_order=mar_order,
        )
        J_full = _compute_jacobian(
            theta, observed_csd, freqs, a_mask, N, eig_clamp=eig_clamp,
            mar_order=mar_order,
        )
        # Project Jacobian to reduced space: J = J_full @ V
        # SPM12: spm_diff(IS, Ep, M, U, 1, {V}) produces (ny, np_reduced)
        J = J_full @ V.to(dtype=torch.complex128)

        # M-step: Fisher scoring to find h = argmax F(p, h)
        # (SPM12 spm_nlsi_GN.m lines 378-430)
        # ================================================================
        for m_iter in range(8):
            # Precision from hyperparameters
            # iS = sum Q{i} * (exp(-32) + exp(h(i)))
            iS = torch.zeros_like(Q_list[0])
            for i_h in range(nh):
                iS = iS + Q_list[i_h] * (
                    math.exp(-32) + torch.exp(h[i_h]).item()
                )
            # S = inv(iS) -- needed for Fisher scoring derivatives
            try:
                S = torch.linalg.inv(iS)
            except torch.linalg.LinAlgError:
                S = torch.linalg.pinv(iS)

            # For nq=1, kron(eye(nq), iS) = iS, so skip Kronecker

            # Posterior covariance in reduced space
            # Pp = real(J' * iS * J); Cp = inv(Pp + ipC)
            Pp = (J.conj().T @ iS @ J).real
            Cp_m = torch.linalg.inv(Pp + ipC)

            # Precision operators for M-Step
            P_ops = []
            PS_ops = []
            JPJ_ops = []
            for i_h in range(nh):
                P_i = Q_list[i_h] * torch.exp(h[i_h]).item()
                PS_i = P_i @ S
                # For nq=1, kron is identity
                # JPJ is (np_reduced, np_reduced)
                JPJ_i = (J.conj().T @ P_i @ J).real
                P_ops.append(P_i)
                PS_ops.append(PS_i)
                JPJ_ops.append(JPJ_i)

            # Derivatives dL/dh
            dFdh_m = torch.zeros(nh, dtype=torch.float64, device=device)
            dFdhh_m = torch.zeros(nh, nh, dtype=torch.float64, device=device)

            for i_h in range(nh):
                # dFdh(i) = trace(PS{i})*nq/2 - real(e'*P{i}*e)/2
                #           - spm_trace(Cp, JPJ{i})/2
                trace_PS = PS_ops[i_h].trace().real.item() if PS_ops[i_h].is_complex() else PS_ops[i_h].trace().item()
                ePe = (e.conj() @ P_ops[i_h] @ e).real.item()
                tr_CpJPJ = _spm_trace(Cp_m, JPJ_ops[i_h])
                dFdh_m[i_h] = trace_PS * nq / 2.0 - ePe / 2.0 - tr_CpJPJ / 2.0

                for j_h in range(i_h, nh):
                    # dFdhh(i,j) = -spm_trace(PS{i}, PS{j})*nq/2
                    val = -_spm_trace(PS_ops[i_h], PS_ops[j_h]) * nq / 2.0
                    dFdhh_m[i_h, j_h] = val
                    dFdhh_m[j_h, i_h] = val

            # Add hyperpriors
            d_h = h - hE
            dFdh_m = dFdh_m - ihC @ d_h
            dFdhh_m = dFdhh_m - ihC
            # Ch = inv(-dFdhh) -- hyperparameter posterior covariance
            Ch = torch.linalg.inv(-dFdhh_m)

            # Update h via regularized descent: spm_dx(dFdhh, dFdh, {4})
            dh = _spm_dx(dFdhh_m, dFdh_m, 4.0)
            # Clamp step to [-1, 1]
            dh = torch.clamp(dh, -1.0, 1.0)
            h = h + dh

            # Check M-step convergence
            dF_m = (dFdh_m @ dh).item()
            if dF_m < 1e-2:
                break

        # After M-step, recompute final iS and Cp with converged h
        iS = torch.zeros_like(Q_list[0])
        for i_h in range(nh):
            iS = iS + Q_list[i_h] * (
                math.exp(-32) + torch.exp(h[i_h]).item()
            )
        Pp = (J.conj().T @ iS @ J).real
        Cp = torch.linalg.inv(Pp + ipC)

        # Free energy: F = L(1) + L(2) + L(3)
        # (SPM12 spm_nlsi_GN.m lines 438-441)
        # All in reduced space except L(1) which uses data-space e
        # ================================================================
        ny = e.shape[0]
        # L(1): data fit
        L_1 = (_spm_logdet(iS) * nq / 2.0
               - (e.conj() @ iS @ e).real.item() / 2.0
               - ny * math.log(2.0 * math.pi) / 2.0)
        # L(2): parameter KL (reduced space)
        L_2 = (_spm_logdet(ipC @ Cp) / 2.0
               - (p @ ipC @ p).item() / 2.0)
        # L(3): hyperparameter uncertainty
        d_hyper = h - hE
        L_3 = (_spm_logdet(ihC @ Ch) / 2.0
               - (d_hyper @ ihC @ d_hyper).item() / 2.0)

        F_val = L_1 + L_2 + L_3
        result.free_energy.append(F_val)

        # Accept/reject (SPM12 lines 456-489)
        # ================================================================
        if F_val > C_F or iteration < 3:
            # Accept current estimates (all in reduced space)
            C_p = p.clone()
            C_h = h.clone()
            C_F = F_val
            C_Cp = Cp.clone()

            # E-step gradients (reduced space)
            dFdp = -(J.conj().T @ iS @ e).real - ipC @ p
            dFdpp = -(J.conj().T @ iS @ J).real - ipC

            # Decrease regularization
            v = min(v + 0.5, 4.0)
        else:
            # Reject: revert to best cached state (reduced space)
            p = C_p.clone()
            h = C_h.clone()
            Cp = C_Cp.clone()

            # Increase regularization
            v = min(v - 2.0, -4.0)

        # E-step update (SPM12 line 493) -- in reduced space
        # ================================================================
        dp = _spm_dx(dFdpp, dFdp, v)
        p = p + dp
        # Map back to full space: Ep = pE + V * p(ip)
        theta = prior_mean + V @ p

        # Convergence (SPM12 lines 570-580)
        # ================================================================
        dF_pred = (dFdp @ dp).item()
        criterion = [dF_pred < 1e-1] + criterion[:3]
        if all(criterion):
            result.converged = True
            break

    result.n_iterations = iteration + 1

    # Final outputs use cached best state (SPM12 lines 590-594)
    # Map from reduced to full space:
    #   Ep = pE + V * C.p(ip)
    #   Cp = V * C.Cp(ip,ip) * V'
    # ================================================================
    theta_final = prior_mean + V @ C_p
    Cp_full = V @ C_Cp @ V.T  # (np_full, np_full)

    with torch.no_grad():
        (A_free_post, na_post, nb_post, nc_post,
         pt_post, pd_post, pe_post) = _unpack_params(theta_final, N)
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
    result.sigma_post = Cp_full
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
