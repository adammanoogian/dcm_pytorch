"""Spectral DCM transfer function and predicted cross-spectral density.

Implements the eigendecomposition-based transfer function from [REF-010]
Eq. 3 (Friston, Kahan, Biswal & Razi, 2014) and the predicted CSD
assembly from [REF-010] Eq. 4, matching SPM12 spm_dcm_mtf.m and
spm_csd_fmri_mtf.m conventions.

The transfer function maps neural dynamics to observed cross-spectral
density via modal decomposition of the effective connectivity matrix A.
Eigenvalue stabilization follows the SPM convention of clamping real
parts to max(-1/32) for fMRI frequency ranges. For MEG/EEG
electrophysiology (1-45 Hz), the clamp threshold can be relaxed via
the ``eig_clamp`` parameter.

When ``hemodynamic=True`` (default), the transfer function is computed
through the full 5N-dimensional hemodynamic system (neural +
vasodilation + flow + volume + deoxyHb), matching SPM12's
``spm_fx_fmri`` + ``spm_gx_fmri`` + ``spm_dcm_mtf`` pipeline. This
includes P.C/16 input scaling (Divergence 7 from RESEARCH.md).
"""

from __future__ import annotations

import torch

from pyro_dcm.forward_models.spectral_noise import (
    neuronal_noise_csd,
    observation_noise_csd,
)

# ---------------------------------------------------------------------------
# Hemodynamic constants (SPM12 spm_fx_fmri.m, spm_gx_fmri.m)
# ---------------------------------------------------------------------------

#: Signal decay rate (1/s). SPM12: H(1).
H_KAPPA: float = 0.64

#: Autoregulation gain (1/s). SPM12: H(2).
H_GAMMA: float = 0.32

#: Transit time (s). SPM12: H(3).
H_TAU: float = 2.00

#: Grubb's exponent (dimensionless). SPM12: H(4).
H_ALPHA: float = 0.32

#: Resting oxygen extraction fraction. SPM12: H(5) / E0.
H_E0: float = 0.4


def _hemodynamic_fx(
    x: torch.Tensor,
    A: torch.Tensor,
    C: torch.Tensor,
    P_decay: torch.Tensor,
    P_transit: torch.Tensor,
) -> torch.Tensor:
    """Hemodynamic state equation matching SPM12 ``spm_fx_fmri.m``.

    Computes dx/dt for the 5N-dimensional state vector:
    ``[x_neural(N), s(N), lnf(N), lnv(N), lnq(N)]``.

    Parameters
    ----------
    x : torch.Tensor
        State vector, shape ``(5*N,)``.
    A : torch.Tensor
        Neural effective connectivity, shape ``(N, N)``. Already
        parameterized (negative self-connections).
    C : torch.Tensor
        Input matrix, shape ``(N, N)``. Typically ``I/16`` matching
        SPM12 ``P.C = P.C/16`` (spm_fx_fmri.m line 49).
    P_decay : torch.Tensor
        Log signal decay deviation, shape ``(1,)``.
    P_transit : torch.Tensor
        Log transit time deviation per region, shape ``(N,)``.

    Returns
    -------
    torch.Tensor
        Time derivative dx/dt, shape ``(5*N,)``.
    """
    N = A.shape[0]

    # Unpack states
    x_neural = x[:N]
    s = x[N : 2 * N]
    lnf = x[2 * N : 3 * N]
    lnv = x[3 * N : 4 * N]
    lnq = x[4 * N : 5 * N]

    f = torch.exp(lnf)
    v = torch.exp(lnv)
    q = torch.exp(lnq)

    # Modulated hemodynamic parameters
    sd = H_KAPPA * torch.exp(P_decay)  # signal decay
    tt = H_TAU * torch.exp(P_transit)  # transit time per region

    # Outflow: fv = v^(1/alpha)
    fv = v ** (1.0 / H_ALPHA)

    # Oxygen extraction fraction: E(f) = (1 - (1-E0)^(1/f)) / E0
    ff = (1.0 - (1.0 - H_E0) ** (1.0 / f)) / H_E0

    # Neural dynamics: dx_neural/dt = A @ x_neural  (input u=0 for Jacobian)
    dx_neural = A @ x_neural

    # Vasodilation: ds/dt = x_neural - sd*s - gamma*(f - 1)
    ds = x_neural - sd * s - H_GAMMA * (f - 1.0)

    # Flow (in log space): dlnf/dt = s/f  (SPM12: f(:,3) = x(:,2)./x(:,3))
    # Note: x(:,3) in SPM12 is the exponentiated flow (f), and the state
    # equation gives d(lnf)/dt = s/f (chain rule through exp).
    dlnf = s / f

    # Volume (in log space): dlnv/dt = (f - fv)/(tt*v)
    dlnv = (f - fv) / (tt * v)

    # DeoxyHb (in log space): dlnq/dt = (ff*f - fv*q/v) / (tt*q)
    dlnq = (ff * f - fv * q / v) / (tt * q)

    return torch.cat([dx_neural, ds, dlnf, dlnv, dlnq])


def compute_hemodynamic_jacobian(
    A: torch.Tensor,
    P_decay: torch.Tensor,
    P_transit: torch.Tensor,
    P_epsilon: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute full 5N x 5N hemodynamic Jacobian at steady state.

    Uses **analytical Jacobian** matching SPM12's ``spm_fx_fmri``
    (lines 247-256) and ``spm_gx_fmri`` (lines 69-70). Evaluated
    at the steady state x=0 (log-space), where f=v=q=1 and s=0.

    Parameters
    ----------
    A : torch.Tensor
        Neural effective connectivity, shape ``(N, N)``, float64.
        Already parameterized (negative self-connections).
    P_decay : torch.Tensor
        Log signal decay deviation, shape ``(1,)``, float64.
    P_transit : torch.Tensor
        Log transit time deviation per region, shape ``(N,)``, float64.
    P_epsilon : torch.Tensor
        Log BOLD signal ratio deviation, shape ``(1,)``, float64.

    Returns
    -------
    dfdx : torch.Tensor
        Full system Jacobian, shape ``(5N, 5N)``, float64.
    dfdu : torch.Tensor
        Input projection, shape ``(5N, N)``, float64. Incorporates
        P.C/16 scaling (SPM12 spm_fx_fmri.m line 49).
    dgdx : torch.Tensor
        BOLD observation Jacobian, shape ``(N, 5N)``, float64.
        Matches SPM12 ``spm_gx_fmri.m`` (Stephan 2007 params).
    """
    N = A.shape[0]
    dtype = A.dtype
    device = A.device

    # Hemodynamic parameters at steady state
    sd = H_KAPPA * torch.exp(P_decay)       # signal decay (scalar)
    tt = H_TAU * torch.exp(P_transit)        # transit time (N,)

    # Steady-state values (x=0 in log-space → f=v=q=1, s=0)
    f = torch.ones(N, dtype=dtype, device=device)
    v = torch.ones(N, dtype=dtype, device=device)
    q = torch.ones(N, dtype=dtype, device=device)
    s = torch.zeros(N, dtype=dtype, device=device)

    # Derived steady-state quantities
    fv = v ** (1.0 / H_ALPHA)               # outflow = 1

    # --- Analytical dfdx (SPM12 spm_fx_fmri.m lines 218, 247-256) ---
    dfdx = torch.zeros(5 * N, 5 * N, dtype=dtype, device=device)

    # Block (1,1): neural Jacobian = A (EE matrix)
    dfdx[:N, :N] = A

    # Block (2,1): d(ds/dt)/d(x_neural) = I
    idx_n = torch.arange(N, device=device)
    dfdx[N + idx_n, idx_n] = 1.0

    # Block (2,2): d(ds/dt)/d(s) = -sd * I
    dfdx[N + idx_n, N + idx_n] = -sd.item()

    # Block (2,3): d(ds/dt)/d(lnf) = -H(2)*f  (SPM12: -H(2)*x(:,3))
    dfdx[N + idx_n, 2 * N + idx_n] = -H_GAMMA * f

    # Block (3,2): d(dlnf/dt)/d(s) = 1/f  (SPM12: 1./x(:,3))
    dfdx[2 * N + idx_n, N + idx_n] = 1.0 / f

    # Block (3,3): d(dlnf/dt)/d(lnf) = -s/f  (SPM12: -x(:,2)./x(:,3))
    dfdx[2 * N + idx_n, 2 * N + idx_n] = -s / f

    # Block (4,3): d(dlnv/dt)/d(lnf) = f/(tt*v)
    dfdx[3 * N + idx_n, 2 * N + idx_n] = f / (tt * v)

    # Block (4,4): SPM12 line 253
    # -v^(1/α-1)/(tt*α) - (1/v*(f - v^(1/α)))/tt
    dfdx[3 * N + idx_n, 3 * N + idx_n] = (
        -v ** (1.0 / H_ALPHA - 1.0) / (tt * H_ALPHA)
        - (1.0 / v * (f - fv)) / tt
    )

    # Block (5,3): SPM12 line 254
    # (f + log(1-E0)*(1-E0)^(1/f) - f*(1-E0)^(1/f)) / (tt*q*E0)
    one_minus_e0 = 1.0 - H_E0
    dfdx[4 * N + idx_n, 2 * N + idx_n] = (
        f + torch.log(torch.tensor(one_minus_e0, dtype=dtype)) * one_minus_e0 ** (1.0 / f)
        - f * one_minus_e0 ** (1.0 / f)
    ) / (tt * q * H_E0)

    # Block (5,4): SPM12 line 255
    # v^(1/α-1)*(α-1)/(tt*α)
    dfdx[4 * N + idx_n, 3 * N + idx_n] = (
        v ** (1.0 / H_ALPHA - 1.0) * (H_ALPHA - 1.0) / (tt * H_ALPHA)
    )

    # Block (5,5): SPM12 line 256
    # (f/q)*((1-E0)^(1/f) - 1)/(tt*E0)
    dfdx[4 * N + idx_n, 4 * N + idx_n] = (
        (f / q) * (one_minus_e0 ** (1.0 / f) - 1.0) / (tt * H_E0)
    )

    # --- dfdu: input projection (5N, N) ---
    # C = I/16 (SPM12 spm_fx_fmri.m line 49)
    dfdu = torch.zeros(5 * N, N, dtype=dtype, device=device)
    dfdu[:N, :] = torch.eye(N, dtype=dtype, device=device) / 16.0

    # --- dgdx: BOLD observation (SPM12 spm_gx_fmri.m lines 69-70) ---
    TE = 0.04
    V0 = 4.0
    nu0 = 40.3
    r0 = 25.0

    ep = torch.exp(P_epsilon)
    k1 = 4.3 * nu0 * H_E0 * TE
    k2 = ep * r0 * H_E0 * TE
    k3 = 1.0 - ep

    # SPM12: dgdx{1,4} = diag(-V0*(k3.*v - k2.*q./v))
    # SPM12: dgdx{1,5} = diag(-V0*(k1.*q + k2.*q./v))
    k2_s = k2.item() if k2.dim() == 0 else k2[0].item()
    k3_s = k3.item() if k3.dim() == 0 else k3[0].item()

    dgdx = torch.zeros(N, 5 * N, dtype=dtype, device=device)
    dgdx[idx_n, 3 * N + idx_n] = -V0 * (k3_s * v - k2_s * q / v)
    dgdx[idx_n, 4 * N + idx_n] = -V0 * (k1 * q + k2_s * q / v)

    return dfdx, dfdu, dgdx


def compute_transfer_function_hemodynamic(
    dfdx: torch.Tensor,
    dfdu: torch.Tensor,
    dgdx: torch.Tensor,
    freqs: torch.Tensor,
    *,
    eig_clamp: float | None = -1.0 / 32.0,
) -> torch.Tensor:
    """Transfer function through full hemodynamic system.

    Matches SPM12 ``spm_dcm_mtf.m`` eigendecomposition approach applied
    to the 5N x 5N hemodynamic Jacobian. Uses ``torch.linalg.pinv``
    for the eigenvector inverse, matching SPM12's ``pinv(v)`` (line 125
    of spm_dcm_mtf.m) for numerical stability.

    Parameters
    ----------
    dfdx : torch.Tensor
        Full system Jacobian, shape ``(5N, 5N)``, float64.
    dfdu : torch.Tensor
        Input projection, shape ``(5N, N)``, float64.
    dgdx : torch.Tensor
        BOLD observation Jacobian, shape ``(N, 5N)``, float64.
    freqs : torch.Tensor
        Frequencies in Hz, shape ``(F,)``, float64.
    eig_clamp : float or None
        Maximum value for real parts of eigenvalues. Default ``-1/32``
        matches SPM12 fMRI convention.

    Returns
    -------
    torch.Tensor
        Transfer function H, shape ``(F, N, N)``, complex128.
    """
    eigvals, eigvecs = torch.linalg.eig(dfdx.to(torch.complex128))

    # Stabilize eigenvalues (SPM12: s = 1j*imag(s) + min(real(s), -1/32))
    if eig_clamp is not None:
        eigvals = torch.complex(
            torch.clamp(eigvals.real, max=eig_clamp),
            eigvals.imag,
        )

    # SPM12 uses pinv(v) for numerical stability (spm_dcm_mtf.m line 125)
    dgdv = dgdx.to(torch.complex128) @ eigvecs
    dvdu = torch.linalg.pinv(eigvecs) @ dfdu.to(torch.complex128)

    w = freqs.to(torch.complex128)
    Sk = 1.0 / (1j * 2.0 * torch.pi * w[:, None] - eigvals[None, :])

    H = torch.einsum("ik,kj,fk->fij", dgdv, dvdu, Sk)
    return H


def default_frequency_grid(
    TR: float = 2.0,
    n_freqs: int = 32,
) -> torch.Tensor:
    """Generate default frequency grid matching SPM12 conventions.

    SPM12 uses linearly spaced frequencies from 1/128 Hz (lowest
    resolvable frequency for typical fMRI) to the Nyquist frequency
    1/(2*TR) Hz.

    Cite: SPM12 spm_dcm_fmri_csd_data.m.

    Parameters
    ----------
    TR : float
        Repetition time in seconds. Default 2.0.
    n_freqs : int
        Number of frequency bins. Default 32.

    Returns
    -------
    torch.Tensor
        Frequency vector in Hz, shape ``(n_freqs,)``, dtype float64.

    Examples
    --------
    >>> freqs = default_frequency_grid(TR=2.0, n_freqs=32)
    >>> freqs.shape  # (32,)
    >>> freqs[0]     # ~0.0078 Hz (1/128)
    """
    return torch.linspace(
        1.0 / 128.0,
        1.0 / (2.0 * TR),
        n_freqs,
        dtype=torch.float64,
    )


def default_frequency_grid_meg(
    sfreq: float = 250.0,
    n_freqs: int = 64,
) -> torch.Tensor:
    """Generate default frequency grid for MEG electrophysiology.

    Returns linearly spaced frequencies from 1 Hz to 45 Hz, matching
    the 1-45 Hz bandpass recommended for source-reconstructed MEG
    data in parcellation-based connectivity analyses.

    Parameters
    ----------
    sfreq : float
        Sampling frequency in Hz. Used only for Nyquist validation;
        the upper frequency bound (45 Hz) must be below ``sfreq / 2``.
        Default 250.0.
    n_freqs : int
        Number of frequency bins. Default 64.

    Returns
    -------
    torch.Tensor
        Frequency vector in Hz, shape ``(n_freqs,)``, dtype float64.

    Raises
    ------
    ValueError
        If the maximum frequency (45 Hz) exceeds the Nyquist frequency
        ``sfreq / 2``.

    Examples
    --------
    >>> freqs = default_frequency_grid_meg(sfreq=250.0, n_freqs=64)
    >>> freqs.shape  # (64,)
    >>> freqs[0]     # ~1.0 Hz
    >>> freqs[-1]    # ~45.0 Hz
    """
    fmax = 45.0
    nyquist = sfreq / 2.0
    if fmax > nyquist:
        msg = (
            f"Maximum frequency {fmax} Hz exceeds Nyquist frequency "
            f"{nyquist} Hz (sfreq={sfreq} Hz). Use sfreq >= {2 * fmax} Hz."
        )
        raise ValueError(msg)
    return torch.linspace(1.0, fmax, n_freqs, dtype=torch.float64)


def compute_transfer_function(
    A: torch.Tensor,
    C_in: torch.Tensor,
    C_out: torch.Tensor,
    freqs: torch.Tensor,
    *,
    eig_clamp: float | None = -1.0 / 32.0,
) -> torch.Tensor:
    """Compute spectral transfer function via eigendecomposition.

    Implements [REF-010] Eq. 3 (Friston et al. 2014):
        g(w) = C_out @ (iwI - A)^{-1} @ C_in

    using modal decomposition for numerical stability:
        H(w) = sum_k dgdv_k * dvdu_k / (i*2*pi*w - lambda_k)

    Eigenvalue stabilization clamps real parts to ``max(eig_clamp)``
    following the SPM12 convention. The default ``-1/32`` is appropriate
    for fMRI frequencies (0.008-0.25 Hz). For MEG electrophysiology
    (1-45 Hz), use ``eig_clamp=-1.0`` or ``eig_clamp=None`` (disables
    clamping entirely); see 22-RESEARCH.md Pitfall 4.

    Cite: [REF-010] Eq. 3 and SPM12 spm_dcm_mtf.m.

    Parameters
    ----------
    A : torch.Tensor
        Effective connectivity (Jacobian), shape ``(N, N)``, float64.
    C_in : torch.Tensor
        Input projection matrix, shape ``(N, nu)``, float64.
    C_out : torch.Tensor
        Output projection matrix, shape ``(nn, N)``, float64.
    freqs : torch.Tensor
        Frequencies in Hz, shape ``(F,)``, float64.
    eig_clamp : float or None
        Maximum value for real parts of eigenvalues of A. Default
        ``-1/32`` matches the SPM12 fMRI convention. Set to ``-1.0``
        for MEG, or ``None`` to disable clamping entirely (relies on
        ``parameterize_A`` upstream for stability).

    Returns
    -------
    torch.Tensor
        Transfer function H, shape ``(F, nn, nu)``, complex128.

    Examples
    --------
    >>> import torch
    >>> A = torch.diag(torch.tensor([-0.5, -0.5], dtype=torch.float64))
    >>> C_in = C_out = torch.eye(2, dtype=torch.float64)
    >>> freqs = default_frequency_grid(TR=2.0, n_freqs=16)
    >>> H = compute_transfer_function(A, C_in, C_out, freqs)
    >>> H.shape  # (16, 2, 2)
    """
    # Step 1: Eigendecompose A
    eigvals, eigvecs = torch.linalg.eig(A.to(torch.complex128))

    # Step 2: Stabilize eigenvalues
    # Default clamps real parts to max(-1/32) for fMRI (SPM convention).
    # For MEG, use eig_clamp=-1.0 or None to disable.
    if eig_clamp is not None:
        eigvals = torch.complex(
            torch.clamp(eigvals.real, max=eig_clamp),
            eigvals.imag,
        )

    # Step 3: Project through eigenvectors
    # dgdv: output projection through eigenvectors, shape (nn, N)
    dgdv = C_out.to(torch.complex128) @ eigvecs
    # dvdu: inverse eigenvectors applied to input, shape (N, nu)
    dvdu = torch.linalg.inv(eigvecs) @ C_in.to(torch.complex128)

    # Step 4: Modal transfer function
    # Sk(w) = 1 / (i*2*pi*w - lambda_k), shape (F, N)
    w = freqs.to(torch.complex128)
    Sk = 1.0 / (
        1j * 2.0 * torch.pi * w[:, None] - eigvals[None, :]
    )

    # Step 5: Assemble H(w) = sum_k dgdv(:,k) * dvdu(k,:) * Sk(w,k)
    # Using einsum: H[f, i, j] = sum_k dgdv[i, k] * dvdu[k, j] * Sk[f, k]
    H = torch.einsum("ik,kj,fk->fij", dgdv, dvdu, Sk)

    return H


def predicted_csd(
    H: torch.Tensor,
    Gu: torch.Tensor,
    Gn: torch.Tensor,
) -> torch.Tensor:
    """Compute predicted cross-spectral density.

    Implements [REF-010] Eq. 4 (Friston et al. 2014):
        S(w) = H(w) @ Gu(w) @ H(w)^H + Gn(w)

    where H is the transfer function, Gu is the neuronal noise CSD,
    Gn is the observation noise CSD, and ^H denotes conjugate transpose.

    Cite: [REF-010] Eq. 4 and SPM12 spm_csd_fmri_mtf.m.

    Parameters
    ----------
    H : torch.Tensor
        Transfer function, shape ``(F, nn, nu)``, complex128.
    Gu : torch.Tensor
        Neuronal noise CSD, shape ``(F, nu, nu)``, complex128.
    Gn : torch.Tensor
        Observation noise CSD, shape ``(F, nn, nn)``, complex128.

    Returns
    -------
    torch.Tensor
        Predicted CSD, shape ``(F, nn, nn)``, complex128.

    Examples
    --------
    >>> import torch
    >>> F, N = 16, 2
    >>> H = torch.randn(F, N, N, dtype=torch.complex128)
    >>> Gu = torch.eye(N, dtype=torch.complex128).unsqueeze(0).expand(F, -1, -1)
    >>> Gn = torch.zeros(F, N, N, dtype=torch.complex128)
    >>> S = predicted_csd(H, Gu, Gn)
    >>> S.shape  # (16, 2, 2)
    """
    # S(w) = H(w) @ Gu(w) @ H(w)^H + Gn(w)
    G = H @ Gu @ H.conj().transpose(-2, -1)
    return G + Gn


def csd_mar_roundtrip(
    csd: torch.Tensor,
    freqs: torch.Tensor,
    mar_order: int = 7,
) -> torch.Tensor:
    """Apply CSD -> MAR -> CSD round-trip matching SPM12.

    SPM12's ``spm_csd_fmri_mtf.m`` (line 157) applies this as the
    final step::

        y = spm_mar2csd(spm_csd2mar(y, M.Hz, M.p - 1), M.Hz)

    This constrains predicted CSD to lie in MAR(p) model space,
    acting as regularization/smoothing.

    .. warning::

        This function is **NOT differentiable** (uses numpy FFT +
        linear solve). Safe for VL inference (finite-difference
        Jacobians). For SVI, use ``mar_order=0`` to skip.

    Parameters
    ----------
    csd : torch.Tensor
        Predicted CSD, shape ``(F, N, N)``, complex128.
    freqs : torch.Tensor
        Frequency vector in Hz, shape ``(F,)``, float64.
    mar_order : int
        MAR model order for the round-trip. Default 7, matching
        SPM12's ``M.p - 1 = 8 - 1 = 7``.

    Returns
    -------
    torch.Tensor
        Smoothed CSD, shape ``(F, N, N)``, complex128. Same device
        as input.
    """
    import numpy as np

    from pyro_dcm.forward_models.mar_csd import csd2mar, mar2csd

    csd_np = csd.detach().cpu().numpy()
    freqs_np = freqs.detach().cpu().numpy()
    fs = 2.0 * freqs_np[-1]

    mar = csd2mar(csd_np, freqs_np, p=mar_order)
    csd_smooth = mar2csd(mar, freqs_np, fs)

    return torch.tensor(
        csd_smooth, dtype=torch.complex128, device=csd.device,
    )


def spectral_dcm_forward(
    A: torch.Tensor,
    freqs: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    *,
    eig_clamp: float | None = -1.0 / 32.0,
    mar_order: int = 7,
    hemodynamic: bool = True,
    P_transit: torch.Tensor | None = None,
    P_decay: torch.Tensor | None = None,
    P_epsilon: torch.Tensor | None = None,
) -> torch.Tensor:
    """Complete spectral DCM predicted CSD pipeline.

    Convenience function wrapping the full predicted CSD computation:
    transfer function (via eigendecomposition), neuronal and observation
    noise spectra, and CSD assembly.

    When ``hemodynamic=True`` (default, matching SPM12), the transfer
    function is computed through the full 5N-dimensional hemodynamic
    system using ``compute_hemodynamic_jacobian`` and
    ``compute_transfer_function_hemodynamic``. This includes P.C/16
    input scaling (Divergence 7 from RESEARCH.md) and the Stephan 2007
    BOLD observation function.

    When ``hemodynamic=False``, uses the neural-only N x N transfer
    function with C_in = C_out = identity (old behavior).

    When ``mar_order > 0``, applies a CSD -> MAR -> CSD round-trip as
    the final step, matching SPM12 ``spm_csd_fmri_mtf.m`` line 157.
    This constrains predicted CSD to lie in MAR(p) model space.

    Implements [REF-010] Eq. 3-7 (Friston et al. 2014), matching
    SPM12 spm_csd_fmri_mtf.m.

    Parameters
    ----------
    A : torch.Tensor
        Effective connectivity matrix, shape ``(N, N)``, float64.
    freqs : torch.Tensor
        Frequency vector in Hz, shape ``(F,)``, float64.
    a : torch.Tensor
        Neuronal noise parameters, float64.
        ``spm_fmri`` mode (default): shape ``(2, 1)`` -- shared.
        ``extended`` mode: shape ``(2, N)`` -- per-region.
    b : torch.Tensor
        Global observation noise parameters, shape ``(2, 1)``, float64.
        ``b[0, 0]`` = log amplitude, ``b[1, 0]`` = log exponent.
    c : torch.Tensor
        Regional observation noise parameters, float64.
        ``spm_fmri`` mode (default): shape ``(1, N)`` -- amplitude only.
        ``extended`` mode: shape ``(2, N)`` -- amplitude + exponent.
    eig_clamp : float or None
        Maximum value for real parts of eigenvalues of A. Default
        ``-1/32`` matches the SPM12 fMRI convention. Set to ``-1.0``
        for MEG, or ``None`` to disable clamping. Passed through to
        ``compute_transfer_function``.
    mar_order : int
        MAR model order for the CSD -> MAR -> CSD round-trip. Default
        7 matches SPM12's ``M.p - 1 = 8 - 1 = 7``. Set to 0 to
        disable the round-trip (required for SVI/autograd paths since
        the round-trip is not differentiable).
    hemodynamic : bool
        If True (default), compute transfer function through the full
        5N-dimensional hemodynamic system matching SPM12. If False,
        use neural-only N x N transfer function (old behavior).
    P_transit : torch.Tensor or None
        Log transit time deviation per region, shape ``(N,)``. If None,
        defaults to zeros (prior mean).
    P_decay : torch.Tensor or None
        Log signal decay deviation, shape ``(1,)``. If None, defaults
        to zeros (prior mean).
    P_epsilon : torch.Tensor or None
        Log BOLD signal ratio deviation, shape ``(1,)``. If None,
        defaults to zeros (prior mean).

    Returns
    -------
    torch.Tensor
        Predicted CSD, shape ``(F, N, N)``, complex128.

    Examples
    --------
    >>> import torch
    >>> A = torch.diag(torch.tensor([-0.5, -0.5], dtype=torch.float64))
    >>> freqs = default_frequency_grid(TR=2.0, n_freqs=16)
    >>> a = torch.zeros(2, 1, dtype=torch.float64)
    >>> b = torch.zeros(2, 1, dtype=torch.float64)
    >>> c = torch.zeros(1, 2, dtype=torch.float64)
    >>> csd = spectral_dcm_forward(A, freqs, a, b, c)
    >>> csd.shape  # (16, 2, 2)
    """
    N = A.shape[0]

    if hemodynamic:
        # Default hemodynamic params at prior mean (zeros)
        if P_transit is None:
            P_transit = torch.zeros(N, dtype=A.dtype, device=A.device)
        if P_decay is None:
            P_decay = torch.zeros(1, dtype=A.dtype, device=A.device)
        if P_epsilon is None:
            P_epsilon = torch.zeros(1, dtype=A.dtype, device=A.device)

        # Full 5N x 5N hemodynamic Jacobian at steady state
        dfdx, dfdu, dgdx = compute_hemodynamic_jacobian(
            A, P_decay, P_transit, P_epsilon,
        )

        # Transfer function through hemodynamic system
        H = compute_transfer_function_hemodynamic(
            dfdx, dfdu, dgdx, freqs, eig_clamp=eig_clamp,
        )
    else:
        # Neural-only transfer function (old behavior)
        C_in = torch.eye(N, dtype=torch.float64, device=A.device)
        C_out = torch.eye(N, dtype=torch.float64, device=A.device)
        H = compute_transfer_function(
            A, C_in, C_out, freqs, eig_clamp=eig_clamp,
        )

    # Compute noise spectra
    Gu = neuronal_noise_csd(freqs, a, n_regions=N)
    Gn = observation_noise_csd(freqs, b, c, N)

    # Assemble predicted CSD
    raw_csd = predicted_csd(H, Gu, Gn)

    # MAR round-trip (SPM12 spm_csd_fmri_mtf.m line 157)
    if mar_order > 0:
        return csd_mar_roundtrip(raw_csd, freqs, mar_order)
    return raw_csd
