"""Regression DCM analytic posterior inference.

Implements the rDCM variational Bayesian inversion for both rigid
(fixed architecture) and sparse (ARD with binary indicators) variants,
closed-form free energy computation, and prior specification.

This module provides:
- Prior specification for rigid and sparse rDCM (``get_priors_rigid``,
  ``get_priors_sparse``)
- Standalone log-likelihood computation (``compute_rdcm_likelihood``)
- Free energy with 5 components for rigid (``compute_free_energy_rigid``)
- Free energy with 7 components for sparse (``compute_free_energy_sparse``)
- Region-wise VB inversion for rigid rDCM (``rigid_inversion``)
- Region-wise VB inversion for sparse rDCM (``sparse_inversion``)

All operations use ``torch.float64`` for numerical precision.

References
----------
[REF-020] Frassle et al. (2017), NeuroImage 145, 270-275.
[REF-021] Frassle et al. (2018), NeuroImage 155, 406-421.
Julia source: RegressionDynamicCausalModeling.jl (rigid_inversion.jl,
sparse_inversion.jl, get_priors.jl).
"""

from __future__ import annotations

import math

import torch


def get_priors_rigid(
    a_mask: torch.Tensor,
    c_mask: torch.Tensor,
) -> dict[str, torch.Tensor | float]:
    """Compute rDCM priors for rigid inversion.

    Matching Julia ``get_priors.jl`` for ``RigidRdcm``.

    Prior mean [REF-020] Eq. 9:
        A: ``zeros(nr, nr) - 0.5 * eye(nr)`` (self-inhibition = -0.5).
        C: ``zeros(nr, nu)``.

    Prior covariance [REF-020] Eq. 10:
        A off-diagonal: ``8 / nr`` for present connections.
        A diagonal: ``1 / (8 * nr)``.
        C: ``1.0`` for present connections, ``0`` for absent.

    Prior precision is ``1 / covariance`` (inf for absent connections).

    Noise precision prior: Gamma(a0=2, b0=1).

    Parameters
    ----------
    a_mask : torch.Tensor
        Binary architecture mask for A, shape ``(nr, nr)``, float64.
    c_mask : torch.Tensor
        Binary mask for C, shape ``(nr, nu)``, float64.

    Returns
    -------
    dict
        Keys: ``'m0'`` (prior mean, ``(nr, nr+nu)``),
        ``'l0'`` (prior precision, ``(nr, nr+nu)``),
        ``'a0'`` (Gamma shape, float),
        ``'b0'`` (Gamma rate, float).

    References
    ----------
    [REF-020] Eq. 9-10. Julia ``get_priors.jl``.
    """
    nr = a_mask.shape[0]
    nu = c_mask.shape[1]
    dtype = torch.float64

    # Off-diagonal mask
    a_off = a_mask.clone().to(dtype)
    a_off.fill_diagonal_(0)

    # Prior mean [REF-020] Eq. 9
    pE_A = torch.zeros(nr, nr, dtype=dtype)
    pE_A -= 0.5 * torch.eye(nr, dtype=dtype)
    pE_C = torch.zeros(nr, nu, dtype=dtype)

    # Prior covariance [REF-020] Eq. 10
    pC_A = (a_off * 8.0) / nr + torch.eye(nr, dtype=dtype) / (8.0 * nr)
    pC_C = torch.zeros(nr, nu, dtype=dtype)
    pC_C[c_mask.bool()] = 1.0

    # Prior precision = 1 / covariance
    l0_A = 1.0 / pC_A
    l0_C = 1.0 / pC_C  # inf for absent connections

    # Concatenate A and C
    m0 = torch.cat([pE_A, pE_C], dim=1)  # (nr, nr+nu)
    l0 = torch.cat([l0_A, l0_C], dim=1)  # (nr, nr+nu)

    return {
        "m0": m0,
        "l0": l0,
        "a0": 2.0,
        "b0": 1.0,
    }


def get_priors_sparse(
    a_mask: torch.Tensor,
    c_mask: torch.Tensor,
) -> dict[str, torch.Tensor | float]:
    """Compute rDCM priors for sparse inversion.

    Matching Julia ``get_priors.jl`` for ``SparseRdcm``.

    Key difference from rigid: uses ``ones(nr, nr)`` and ``ones(nr, nu)``
    as architecture masks, since sparsity is learned via z indicators.
    The actual ``a_mask`` and ``c_mask`` are not used for masking; all
    connections are considered possible.

    Parameters
    ----------
    a_mask : torch.Tensor
        Binary architecture mask for A, shape ``(nr, nr)``, float64.
        Not used for masking (sparse learns sparsity via z), but needed
        to determine dimensions.
    c_mask : torch.Tensor
        Binary mask for C, shape ``(nr, nu)``, float64.
        Not used for masking.

    Returns
    -------
    dict
        Same keys as ``get_priors_rigid`` plus ``'p0'`` (Bernoulli
        prior probability, default 0.5).

    References
    ----------
    Julia ``get_priors.jl`` for ``SparseRdcm``.
    """
    nr = a_mask.shape[0]
    nu = c_mask.shape[1]
    dtype = torch.float64

    # Sparse uses full connectivity masks
    a_full = torch.ones(nr, nr, dtype=dtype)
    c_full = torch.ones(nr, nu, dtype=dtype)

    # Off-diagonal mask (all ones except diagonal)
    a_off = a_full.clone()
    a_off.fill_diagonal_(0)

    # Prior mean [REF-020] Eq. 9
    pE_A = torch.zeros(nr, nr, dtype=dtype)
    pE_A -= 0.5 * torch.eye(nr, dtype=dtype)
    pE_C = torch.zeros(nr, nu, dtype=dtype)

    # Prior covariance [REF-020] Eq. 10 with full connectivity
    pC_A = (a_off * 8.0) / nr + torch.eye(nr, dtype=dtype) / (8.0 * nr)
    pC_C = torch.zeros(nr, nu, dtype=dtype)
    pC_C[c_full.bool()] = 1.0

    # Prior precision
    l0_A = 1.0 / pC_A
    l0_C = 1.0 / pC_C

    # Concatenate
    m0 = torch.cat([pE_A, pE_C], dim=1)
    l0 = torch.cat([l0_A, l0_C], dim=1)

    return {
        "m0": m0,
        "l0": l0,
        "a0": 2.0,
        "b0": 1.0,
        "p0": 0.5,
    }


def compute_rdcm_likelihood(
    Y_r: torch.Tensor,
    X_r: torch.Tensor,
    mu_r: torch.Tensor,
    tau_r: float | torch.Tensor,
) -> torch.Tensor:
    """Standalone rDCM analytic log-likelihood for one region.

    Computes the frequency-domain Gaussian log-likelihood from
    [REF-020] Eq. 15 (log_lik component), exposed as a standalone
    function for testing and downstream use.

    Parameters
    ----------
    Y_r : torch.Tensor
        Data vector for region r (NaN-filtered), shape ``(N_eff_r,)``.
    X_r : torch.Tensor
        Design matrix for region r (NaN-filtered),
        shape ``(N_eff_r, D_r)``.
    mu_r : torch.Tensor
        Posterior mean weights, shape ``(D_r,)``.
    tau_r : float or torch.Tensor
        Posterior noise precision (``a_r / beta_r``).

    Returns
    -------
    torch.Tensor
        Scalar log-likelihood value.

    References
    ----------
    [REF-020] Eq. 15 (log-likelihood component).
    Julia ``rigid_inversion.jl``.
    """
    N_eff_r = Y_r.shape[0]
    residual = Y_r - X_r @ mu_r
    tau_t = torch.as_tensor(tau_r, dtype=Y_r.dtype)
    log_lik = (
        -0.5 * N_eff_r * math.log(2.0 * math.pi)
        + 0.5 * N_eff_r * torch.log(tau_t)
        - 0.5 * tau_t * (residual @ residual)
    )
    return log_lik


def compute_free_energy_rigid(
    N_eff: int,
    a_r: float | torch.Tensor,
    beta_r: float | torch.Tensor,
    QF: float | torch.Tensor,
    tau_r: float | torch.Tensor,
    l0_r: torch.Tensor,
    mu_r: torch.Tensor,
    mu0_r: torch.Tensor,
    Sigma_r: torch.Tensor,
    a0: float,
    beta0: float,
    D_r: int,
) -> torch.Tensor:
    """Compute negative free energy for one region (rigid rDCM).

    Five additive components from [REF-020] Eq. 15:
        ``F_r = log_lik + log_p_weight + log_p_prec
              + log_q_weight + log_q_prec``

    Parameters
    ----------
    N_eff : int
        Effective number of data points for this region.
    a_r : float or Tensor
        Posterior Gamma shape.
    beta_r : float or Tensor
        Posterior Gamma rate.
    QF : float or Tensor
        Quadratic form ``0.5 * (||Y - X mu||^2 + tr(W Sigma))``.
    tau_r : float or Tensor
        Posterior noise precision ``a_r / beta_r``.
    l0_r : torch.Tensor
        Prior precision matrix, shape ``(D_r, D_r)``.
    mu_r : torch.Tensor
        Posterior mean, shape ``(D_r,)``.
    mu0_r : torch.Tensor
        Prior mean, shape ``(D_r,)``.
    Sigma_r : torch.Tensor
        Posterior covariance, shape ``(D_r, D_r)``.
    a0 : float
        Prior Gamma shape.
    beta0 : float
        Prior Gamma rate.
    D_r : int
        Dimensionality of parameter vector for region r.

    Returns
    -------
    torch.Tensor
        Scalar negative free energy F_r.

    References
    ----------
    [REF-020] Eq. 15. Julia ``rigid_inversion.jl`` ``compute_F()``.
    """
    a_r = torch.as_tensor(a_r, dtype=mu_r.dtype)
    beta_r = torch.as_tensor(beta_r, dtype=mu_r.dtype)
    QF = torch.as_tensor(QF, dtype=mu_r.dtype)
    tau_r = torch.as_tensor(tau_r, dtype=mu_r.dtype)
    beta0_t = torch.as_tensor(beta0, dtype=mu_r.dtype)

    # Component 1: log-likelihood
    log_lik = (
        0.5
        * (
            N_eff * (torch.special.digamma(a_r) - torch.log(beta_r))
            - N_eff * math.log(2.0 * math.pi)
        )
        - QF * tau_r
    )

    # Component 2: log p(weights)
    sign_l0, logdet_l0 = torch.linalg.slogdet(l0_r)
    diff = mu_r - mu0_r
    log_p_weight = 0.5 * (
        logdet_l0
        - D_r * math.log(2.0 * math.pi)
        - diff @ l0_r @ diff
        - torch.trace(l0_r @ Sigma_r)
    )

    # Component 3: log p(precision)
    log_p_prec = (
        a0 * torch.log(beta0_t)
        - torch.lgamma(torch.tensor(a0, dtype=mu_r.dtype))
        + (a0 - 1.0)
        * (torch.special.digamma(a_r) - torch.log(beta_r))
        - beta0 * tau_r
    )

    # Component 4: -log q(weights) (entropy of posterior Gaussian)
    sign_S, logdet_S = torch.linalg.slogdet(Sigma_r)
    log_q_weight = 0.5 * (
        logdet_S + D_r * (1.0 + math.log(2.0 * math.pi))
    )

    # Component 5: -log q(precision) (neg-entropy of posterior Gamma)
    log_q_prec = (
        a_r
        - torch.log(beta_r)
        + torch.lgamma(a_r)
        + (1.0 - a_r) * torch.special.digamma(a_r)
    )

    F_r = log_lik + log_p_weight + log_p_prec + log_q_weight + log_q_prec
    return F_r


def rigid_inversion(
    X: torch.Tensor,
    Y: torch.Tensor,
    a_mask: torch.Tensor,
    c_mask: torch.Tensor,
    confound_cols: int = 1,
    max_iter: int = 500,
    tol: float = 1e-5,
) -> dict[str, object]:
    """Region-wise VB inversion for rigid rDCM.

    Matching Julia ``rigid_inversion.jl``.

    For each region r, selects columns from X based on architecture
    mask, filters NaN rows, and iterates VB update equations until
    convergence. Returns posterior parameters for all regions.

    Parameters
    ----------
    X : torch.Tensor
        Design matrix, shape ``(N_eff, nr+nu+nc)``, float64.
    Y : torch.Tensor
        Data matrix, shape ``(N_eff, nr)``, float64.
    a_mask : torch.Tensor
        Binary architecture mask for A, shape ``(nr, nr)``, float64.
    c_mask : torch.Tensor
        Binary mask for C, shape ``(nr, nu)``, float64.
    confound_cols : int, optional
        Number of confound columns. Default 1.
    max_iter : int, optional
        Maximum VB iterations. Default 500.
    tol : float, optional
        Convergence tolerance. Default 1e-5.

    Returns
    -------
    dict
        Keys: ``'A_mu'`` ``(nr, nr)``, ``'C_mu'`` ``(nr, nu)``,
        ``'mu_per_region'`` list, ``'Sigma_per_region'`` list,
        ``'a_per_region'`` tensor, ``'beta_per_region'`` tensor,
        ``'F_per_region'`` tensor, ``'F_total'`` scalar,
        ``'iterations_per_region'`` tensor.

    References
    ----------
    [REF-020] Eq. 11-15. Julia ``rigid_inversion.jl``.
    """
    nr = a_mask.shape[0]
    nu = c_mask.shape[1]
    nc = confound_cols
    dtype = torch.float64

    # Get priors
    priors = get_priors_rigid(a_mask, c_mask)
    m0 = priors["m0"]  # (nr, nr+nu)
    l0 = priors["l0"]  # (nr, nr+nu)
    a0 = priors["a0"]
    b0 = priors["b0"]

    # Storage
    mu_per_region: list[torch.Tensor] = []
    Sigma_per_region: list[torch.Tensor] = []
    a_per_region = torch.zeros(nr, dtype=dtype)
    beta_per_region = torch.zeros(nr, dtype=dtype)
    F_per_region = torch.zeros(nr, dtype=dtype)
    iter_per_region = torch.zeros(nr, dtype=torch.int64)

    for r in range(nr):
        # Build per-region index mask: [a_mask[r,:], c_mask[r,:], ones(nc)]
        idx_r = torch.cat([
            a_mask[r, :].to(dtype),
            c_mask[r, :].to(dtype),
            torch.ones(nc, dtype=dtype),
        ])
        col_select = idx_r.bool()

        # Select columns from X
        X_full = X[:, col_select]

        # Filter NaN rows for this region
        valid = ~torch.isnan(Y[:, r])
        X_r = X_full[valid]
        Y_r = Y[valid, r]

        N_eff_r = X_r.shape[0]
        D_r = X_r.shape[1]

        # Per-region priors: extract from concatenated priors
        # m0 has shape (nr, nr+nu), we need the selected indices
        # But idx_r only covers nr+nu, not confounds
        # For confounds, prior mean = 0, prior precision = small
        m0_full = torch.cat([
            m0[r, :],
            torch.zeros(nc, dtype=dtype),
        ])
        l0_full = torch.cat([
            l0[r, :],
            torch.ones(nc, dtype=dtype),  # weak prior on confounds
        ])
        mu0_r = m0_full[col_select]
        l0_diag_r = l0_full[col_select]

        # Handle inf precision (absent connections should be masked
        # out by col_select, but clamp for safety)
        l0_diag_r = torch.clamp(l0_diag_r, max=1e16)
        l0_r = torch.diag(l0_diag_r)

        # Precompute sufficient statistics
        W = X_r.T @ X_r  # (D_r, D_r)
        V = X_r.T @ Y_r  # (D_r,)

        # Initialize
        tau_r = a0 / b0
        a_r = a0 + N_eff_r * 0.5
        F_old = torch.tensor(float("-inf"), dtype=dtype)
        pr = tol**2

        mu_r = torch.zeros(D_r, dtype=dtype)
        Sigma_r = torch.zeros(D_r, D_r, dtype=dtype)
        F_r = torch.tensor(float("-inf"), dtype=dtype)
        n_iter = 0

        for iteration in range(max_iter):
            # Posterior covariance [REF-020] Eq. 12
            Sigma_r = torch.linalg.inv(tau_r * W + l0_r)
            Sigma_r = 0.5 * (Sigma_r + Sigma_r.T)

            # Posterior mean [REF-020] Eq. 11
            mu_r = Sigma_r @ (tau_r * V + l0_r @ mu0_r)

            # Quadratic form [REF-020] Eq. 14
            residual = Y_r - X_r @ mu_r
            QF = 0.5 * (
                residual @ residual + torch.trace(W @ Sigma_r)
            )

            # Posterior Gamma rate
            beta_r_val = b0 + QF
            tau_r = a_r / beta_r_val

            # Free energy [REF-020] Eq. 15
            F_r = compute_free_energy_rigid(
                N_eff_r,
                a_r,
                beta_r_val,
                QF,
                tau_r,
                l0_r,
                mu_r,
                mu0_r,
                Sigma_r,
                a0,
                b0,
                D_r,
            )

            # Convergence check
            n_iter = iteration + 1
            if (F_old - F_r) ** 2 < pr:
                break
            F_old = F_r

        # Store results
        mu_per_region.append(mu_r)
        Sigma_per_region.append(Sigma_r)
        a_per_region[r] = a_r
        beta_per_region[r] = beta_r_val
        F_per_region[r] = F_r
        iter_per_region[r] = n_iter

    # Assemble A_mu and C_mu from per-region posteriors
    A_mu = torch.zeros(nr, nr, dtype=dtype)
    C_mu = torch.zeros(nr, nu, dtype=dtype)

    for r in range(nr):
        idx_r = torch.cat([
            a_mask[r, :].to(dtype),
            c_mask[r, :].to(dtype),
            torch.ones(nc, dtype=dtype),
        ])
        col_select = idx_r.bool()

        mu_r = mu_per_region[r]
        # Map posterior mean back to A and C
        pos = 0
        for j in range(nr):
            if a_mask[r, j] > 0:
                A_mu[r, j] = mu_r[pos]
                pos += 1
        for j in range(nu):
            if c_mask[r, j] > 0:
                C_mu[r, j] = mu_r[pos]
                pos += 1
        # Remaining positions are confound weights (not stored)

    F_total = F_per_region.sum()

    return {
        "A_mu": A_mu,
        "C_mu": C_mu,
        "mu_per_region": mu_per_region,
        "Sigma_per_region": Sigma_per_region,
        "a_per_region": a_per_region,
        "beta_per_region": beta_per_region,
        "F_per_region": F_per_region,
        "F_total": F_total,
        "iterations_per_region": iter_per_region,
    }


def compute_free_energy_sparse(
    N_eff: int,
    a_r: float | torch.Tensor,
    beta_r: float | torch.Tensor,
    QF: float | torch.Tensor,
    tau_r: float | torch.Tensor,
    l0_r: torch.Tensor,
    mu_r: torch.Tensor,
    mu0_r: torch.Tensor,
    Sigma_r: torch.Tensor,
    a0: float,
    beta0: float,
    D_r: int,
    z_r: torch.Tensor,
    z_idx: torch.Tensor,
    p0: torch.Tensor,
) -> torch.Tensor:
    """Compute negative free energy for one region (sparse rDCM).

    Seven additive components: 5 from rigid + 2 for z indicators.

    Additional components beyond rigid:
        ``log_p_z``: prior on binary indicators.
        ``log_q_z``: entropy of z (Bernoulli posterior).

    Parameters
    ----------
    N_eff : int
        Effective number of data points.
    a_r : float or Tensor
        Posterior Gamma shape.
    beta_r : float or Tensor
        Posterior Gamma rate.
    QF : float or Tensor
        Quadratic form.
    tau_r : float or Tensor
        Posterior noise precision.
    l0_r : torch.Tensor
        Prior precision matrix, shape ``(D_r, D_r)``.
    mu_r : torch.Tensor
        Posterior mean, shape ``(D_r,)``.
    mu0_r : torch.Tensor
        Prior mean, shape ``(D_r,)``.
    Sigma_r : torch.Tensor
        Posterior covariance, shape ``(D_r, D_r)``.
    a0 : float
        Prior Gamma shape.
    beta0 : float
        Prior Gamma rate.
    D_r : int
        Dimensionality of parameter vector.
    z_r : torch.Tensor
        Binary indicator probabilities, shape ``(D_r,)``.
    z_idx : torch.Tensor
        Boolean mask of indices where ``tol^2 < z < 1``.
    p0 : torch.Tensor
        Bernoulli prior probabilities, shape ``(D_r,)``.

    Returns
    -------
    torch.Tensor
        Scalar negative free energy F_r.

    References
    ----------
    Julia ``sparse_inversion.jl`` ``compute_F_sparse()``.
    Frassle et al. (2018) [REF-021].
    """
    # Rigid components (5)
    F_rigid = compute_free_energy_rigid(
        N_eff, a_r, beta_r, QF, tau_r, l0_r, mu_r, mu0_r,
        Sigma_r, a0, beta0, D_r,
    )

    eps = 1e-16

    # Component 6: log p(z) -- prior on binary indicators
    if z_idx.any():
        p0_sel = p0[z_idx]
        z_sel = z_r[z_idx]
        log_p_z = torch.sum(
            torch.log(1.0 - p0_sel + eps)
            + z_sel * torch.log(p0_sel / (1.0 - p0_sel + eps) + eps)
        )
    else:
        log_p_z = torch.tensor(0.0, dtype=mu_r.dtype)

    # Component 7: -log q(z) -- entropy of Bernoulli posterior
    if z_idx.any():
        z_sel = z_r[z_idx]
        log_q_z = torch.sum(
            -(1.0 - z_sel) * torch.log(1.0 - z_sel + eps)
            - z_sel * torch.log(z_sel + eps)
        )
    else:
        log_q_z = torch.tensor(0.0, dtype=mu_r.dtype)

    return F_rigid + log_p_z + log_q_z


def _sparse_inversion_single_run(
    X: torch.Tensor,
    Y: torch.Tensor,
    nr: int,
    nu: int,
    nc: int,
    priors: dict,
    max_iter: int,
    tol: float,
    p0_val: float,
    restrict_inputs: bool,
) -> tuple[
    list[torch.Tensor],
    list[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[torch.Tensor],
]:
    """Single run of sparse VB inversion (internal helper).

    Parameters
    ----------
    X : torch.Tensor
        Full design matrix.
    Y : torch.Tensor
        Data matrix.
    nr, nu, nc : int
        Number of regions, inputs, confounds.
    priors : dict
        From ``get_priors_sparse``.
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance.
    p0_val : float
        Bernoulli prior probability.
    restrict_inputs : bool
        If True, keep z=1 for C indices.

    Returns
    -------
    tuple
        ``(mu_list, Sigma_list, a_arr, beta_arr, F_arr, iter_arr,
        z_list)``.
    """
    dtype = torch.float64
    m0 = priors["m0"]
    l0 = priors["l0"]
    a0 = priors["a0"]
    b0 = priors["b0"]

    mu_list: list[torch.Tensor] = []
    Sigma_list: list[torch.Tensor] = []
    a_arr = torch.zeros(nr, dtype=dtype)
    beta_arr = torch.zeros(nr, dtype=dtype)
    F_arr = torch.zeros(nr, dtype=dtype)
    iter_arr = torch.zeros(nr, dtype=torch.int64)
    z_list: list[torch.Tensor] = []

    for r in range(nr):
        # Sparse uses all A connections + all C + confounds
        # All connections are included (full connectivity)
        idx_a = torch.ones(nr, dtype=dtype)
        idx_c = torch.ones(nu, dtype=dtype)
        idx_conf = torch.ones(nc, dtype=dtype)
        idx_r = torch.cat([idx_a, idx_c, idx_conf])
        col_select = idx_r.bool()

        X_full = X[:, col_select]

        # Filter NaN rows
        valid = ~torch.isnan(Y[:, r])
        X_r = X_full[valid]
        Y_r = Y[valid, r]

        N_eff_r = X_r.shape[0]
        D_r = X_r.shape[1]

        # Per-region priors
        m0_full = torch.cat([m0[r, :], torch.zeros(nc, dtype=dtype)])
        l0_full = torch.cat([l0[r, :], torch.ones(nc, dtype=dtype)])
        mu0_r = m0_full[col_select]
        l0_diag_r = l0_full[col_select]
        l0_diag_r = torch.clamp(l0_diag_r, max=1e16)
        l0_r = torch.diag(l0_diag_r)

        # Precompute
        W = X_r.T @ X_r
        V = X_r.T @ Y_r

        # Initialize z indicators
        z_r = 0.5 * torch.ones(D_r, dtype=dtype)
        # C indices start at nr, end at nr+nu
        c_start = nr
        c_end = nr + nu
        if restrict_inputs:
            z_r[c_start:c_end] = 1.0
        # Confound indices always z=1
        z_r[nr + nu :] = 1.0

        # Bernoulli prior for all parameters
        p0_r = p0_val * torch.ones(D_r, dtype=dtype)

        # Initialize
        tau_r_val = a0 / b0
        a_r = a0 + N_eff_r * 0.5
        F_old = torch.tensor(float("-inf"), dtype=dtype)
        pr = tol

        mu_r = torch.zeros(D_r, dtype=dtype)
        Sigma_r = torch.zeros(D_r, D_r, dtype=dtype)
        F_r = torch.tensor(float("-inf"), dtype=dtype)
        n_iter = 0
        beta_r_val = torch.tensor(b0, dtype=dtype)

        for iteration in range(max_iter):
            # --- z update with random sweep ---
            # A_mat = W * (mu_r outer mu_r) + W * Sigma_r
            mu_outer = mu_r.unsqueeze(1) * mu_r.unsqueeze(0)
            A_mat = W * mu_outer + W * Sigma_r

            perm = torch.randperm(D_r)
            for i_idx in range(D_r):
                i = perm[i_idx].item()
                # Skip confound and (optionally) C indices
                if i >= nr + nu:
                    continue
                if restrict_inputs and c_start <= i < c_end:
                    continue

                g_i = (
                    math.log(p0_r[i] / (1.0 - p0_r[i]))
                    + tau_r_val * mu_r[i] * V[i]
                    + tau_r_val * A_mat[i, i] / 2.0
                )
                z_r[i] = 1.0  # temporarily set to 1
                g_i = g_i - tau_r_val * (z_r @ A_mat[:, i])
                z_r[i] = torch.sigmoid(
                    torch.as_tensor(g_i, dtype=dtype)
                ).item()

            # --- Z matrix and G matrix ---
            Z_diag = torch.diag(z_r)
            G = Z_diag @ W @ Z_diag
            # Correct diagonal: G[i,i] = z_r[i] * W[i,i]
            G_diag = z_r * W.diag()
            G.fill_diagonal_(0)
            G += torch.diag(G_diag)

            # Posterior covariance
            Sigma_r = torch.linalg.inv(tau_r_val * G + l0_r)
            Sigma_r = 0.5 * (Sigma_r + Sigma_r.T)

            # Posterior mean
            mu_r = Sigma_r @ (
                tau_r_val * Z_diag @ V + l0_r @ mu0_r
            )

            # QF (sparse version)
            YtY = Y_r @ Y_r
            QF = 0.5 * (
                YtY
                - 2.0 * mu_r @ Z_diag @ V
                + mu_r @ G @ mu_r
                + torch.trace(G @ Sigma_r)
            )

            # Posterior Gamma rate
            beta_r_val = b0 + QF
            tau_r_val = a_r / beta_r_val

            # Hard thresholding
            small = mu_r.abs() < 1e-5
            mu_r = mu_r.clone()
            mu_r[small] = 0.0
            z_r_new = z_r.clone()
            z_r_new[small] = 0.0
            # Restore confound z
            z_r_new[nr + nu :] = 1.0
            if restrict_inputs:
                z_r_new[c_start:c_end] = 1.0
            z_r = z_r_new

            # Free energy (sparse)
            tol_sq = tol**2
            z_idx = (z_r > tol_sq) & (z_r < 1.0)
            F_r = compute_free_energy_sparse(
                N_eff_r,
                a_r,
                beta_r_val,
                QF,
                tau_r_val,
                l0_r,
                mu_r,
                mu0_r,
                Sigma_r,
                a0,
                b0,
                D_r,
                z_r,
                z_idx,
                p0_r,
            )

            # Convergence
            n_iter = iteration + 1
            if (F_old - F_r) ** 2 < pr**2:
                break
            F_old = F_r

        mu_list.append(mu_r)
        Sigma_list.append(Sigma_r)
        a_arr[r] = a_r
        beta_arr[r] = beta_r_val
        F_arr[r] = F_r
        iter_arr[r] = n_iter
        z_list.append(z_r)

    return mu_list, Sigma_list, a_arr, beta_arr, F_arr, iter_arr, z_list


def sparse_inversion(
    X: torch.Tensor,
    Y: torch.Tensor,
    a_mask: torch.Tensor,
    c_mask: torch.Tensor,
    confound_cols: int = 1,
    max_iter: int = 500,
    tol: float = 1e-5,
    n_reruns: int = 100,
    p0: float = 0.5,
    restrict_inputs: bool = True,
) -> dict[str, object]:
    """Sparse VB inversion with binary indicators.

    Matching Julia ``sparse_inversion.jl``.

    Key differences from rigid:
    1. Uses ``get_priors_sparse`` (all connections possible).
    2. Initializes z_r = 0.5 for A connections, z_r = 1.0 for C
       (if ``restrict_inputs=True``).
    3. VB loop includes z update with random sweep ordering.
    4. Runs ``n_reruns`` times and selects best free energy.

    Parameters
    ----------
    X : torch.Tensor
        Design matrix, shape ``(N_eff, nr+nu+nc)``, float64.
    Y : torch.Tensor
        Data matrix, shape ``(N_eff, nr)``, float64.
    a_mask : torch.Tensor
        Binary architecture mask for A, shape ``(nr, nr)``, float64.
    c_mask : torch.Tensor
        Binary mask for C, shape ``(nr, nu)``, float64.
    confound_cols : int, optional
        Number of confound columns. Default 1.
    max_iter : int, optional
        Maximum VB iterations. Default 500.
    tol : float, optional
        Convergence tolerance. Default 1e-5.
    n_reruns : int, optional
        Number of random restarts. Default 100.
    p0 : float, optional
        Bernoulli prior probability. Default 0.5.
    restrict_inputs : bool, optional
        If True, keep z=1 for C columns. Default True.

    Returns
    -------
    dict
        Same keys as ``rigid_inversion`` plus ``'z_per_region'`` list.

    References
    ----------
    Julia ``sparse_inversion.jl``. Frassle et al. (2018) [REF-021].
    """
    nr = a_mask.shape[0]
    nu = c_mask.shape[1]
    nc = confound_cols
    dtype = torch.float64

    priors = get_priors_sparse(a_mask, c_mask)

    best_F_total = torch.tensor(float("-inf"), dtype=dtype)
    best_result = None

    for run in range(n_reruns):
        (
            mu_list,
            Sigma_list,
            a_arr,
            beta_arr,
            F_arr,
            iter_arr,
            z_list,
        ) = _sparse_inversion_single_run(
            X, Y, nr, nu, nc, priors, max_iter, tol, p0,
            restrict_inputs,
        )

        F_total = F_arr.sum()
        if F_total > best_F_total:
            best_F_total = F_total
            best_result = (
                mu_list,
                Sigma_list,
                a_arr,
                beta_arr,
                F_arr,
                iter_arr,
                z_list,
            )

    # Unpack best result
    (
        mu_list,
        Sigma_list,
        a_arr,
        beta_arr,
        F_arr,
        iter_arr,
        z_list,
    ) = best_result

    # Assemble A_mu and C_mu
    A_mu = torch.zeros(nr, nr, dtype=dtype)
    C_mu = torch.zeros(nr, nu, dtype=dtype)

    for r in range(nr):
        mu_r = mu_list[r]
        z_r = z_list[r]
        # A connections: indices 0..nr-1
        for j in range(nr):
            A_mu[r, j] = mu_r[j] * z_r[j]
        # C connections: indices nr..nr+nu-1
        for j in range(nu):
            C_mu[r, j] = mu_r[nr + j] * z_r[nr + j]

    return {
        "A_mu": A_mu,
        "C_mu": C_mu,
        "mu_per_region": mu_list,
        "Sigma_per_region": Sigma_list,
        "a_per_region": a_arr,
        "beta_per_region": beta_arr,
        "F_per_region": F_arr,
        "F_total": best_F_total,
        "iterations_per_region": iter_arr,
        "z_per_region": z_list,
    }
