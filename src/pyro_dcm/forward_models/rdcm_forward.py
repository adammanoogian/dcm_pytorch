"""Regression DCM forward pipeline: HRF, BOLD generation, and design matrix.

Implements the rDCM forward data-preparation pipeline from
[REF-020] Frässle et al. (2017), following the Julia
RegressionDynamicCausalModeling.jl implementation exactly.

This module provides:
- HRF generation via Euler integration of a minimal 1-region DCM
- Synthetic BOLD generation with zero-padded FFT convolution
- Frequency-domain design matrix construction with real/imaginary splitting
- Derivative coefficients for transforming DFT to temporal-derivative DFT

All real operations use ``torch.float64``, all frequency-domain operations
use ``torch.complex128``.

References
----------
[REF-020] Frässle et al. (2017), NeuroImage 145, 270-275.
Julia source: RegressionDynamicCausalModeling.jl (dcm_euler_integration.jl,
generate_BOLD.jl, create_regressors.jl).
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# Hemodynamic constants for rDCM Euler integration
# Source: Julia dcm_euler_integration.jl
# ---------------------------------------------------------------------------
_RDCM_HEMO_CONSTANTS: tuple[float, ...] = (0.64, 0.32, 2.00, 0.32, 0.32)
"""(kappa, gamma, tau, alpha, rho) -- matches Julia H vector."""

# BOLD signal constants from Julia dcm_euler_integration.jl
_RELAXATION_RATE_SLOPE: float = 25.0
_FREQUENCY_OFFSET: float = 40.3
_OXY_EXTRACTION_FRACTION: float = 0.4
_ECHO_TIME: float = 0.04
_RESTING_VENOUS_VOLUME: float = 4.0


def dcm_euler_step(
    x: torch.Tensor,
    s: torch.Tensor,
    f: torch.Tensor,
    v: torch.Tensor,
    q: torch.Tensor,
    A: torch.Tensor,
    C: torch.Tensor,
    u_t: torch.Tensor,
    dt: float,
    H: tuple[float, ...] = _RDCM_HEMO_CONSTANTS,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Single Euler step for a DCM with ``nr`` regions.

    Implements the Euler integration scheme from Julia
    ``dcm_euler_integration.jl``.

    Neural dynamics:
        ``x_new = x + dt * (A @ x + C @ u_t)``

    Hemodynamic dynamics (per region, sequential update matching Julia):
        ``s_new = s + dt * (x - H[0]*s - H[1]*(f - 1))``
        ``f_new = f + dt * s``
        ``v_new = v + dt * (f - v^(1/H[3])) / H[2]``
        ``q_new = q + dt * (f*(1-(1-H[4])^(1/f))/H[4]
                            - v^(1/H[3])*q/v) / H[2]``

    Parameters
    ----------
    x : torch.Tensor
        Neural activity, shape ``(nr,)``.
    s : torch.Tensor
        Vasodilatory signal, shape ``(nr,)``.
    f : torch.Tensor
        Blood flow (linear space), shape ``(nr,)``.
    v : torch.Tensor
        Venous volume (linear space), shape ``(nr,)``.
    q : torch.Tensor
        Deoxyhemoglobin (linear space), shape ``(nr,)``.
    A : torch.Tensor
        Connectivity matrix, shape ``(nr, nr)``.
    C : torch.Tensor
        Input weights, shape ``(nr, nu)``.
    u_t : torch.Tensor
        Input at current time step, shape ``(nu,)``.
    dt : float
        Euler step size (seconds).
    H : tuple of float
        Hemodynamic constants ``(kappa, gamma, tau, alpha, rho)``.

    Returns
    -------
    tuple of torch.Tensor
        ``(x_new, s_new, f_new, v_new, q_new)`` each shape ``(nr,)``.

    References
    ----------
    Julia ``dcm_euler_integration.jl``.
    """
    kappa, gamma, tau, alpha, rho = H

    # Neural dynamics
    x_new = x + dt * (A @ x + C @ u_t)

    # Hemodynamic dynamics
    s_new = s + dt * (x - kappa * s - gamma * (f - 1.0))
    f_new = f + dt * s

    # Clamp to prevent numerical issues
    f_safe = torch.clamp(f, min=1e-6)
    v_safe = torch.clamp(v, min=1e-6)
    q_safe = torch.clamp(q, min=1e-6)

    fv = v_safe.pow(1.0 / alpha)  # venous outflow
    E_f = (1.0 - (1.0 - rho) ** (1.0 / f_safe)) / rho

    v_new = v + dt * (f - fv) / tau
    q_new = q + dt * (f * E_f - fv * q_safe / v_safe) / tau

    # Clamp outputs
    f_new = torch.clamp(f_new, min=1e-6)
    v_new = torch.clamp(v_new, min=1e-6)
    q_new = torch.clamp(q_new, min=1e-6)

    return x_new, s_new, f_new, v_new, q_new


def euler_integrate_dcm(
    A: torch.Tensor,
    C: torch.Tensor,
    u: torch.Tensor,
    dt: float,
    H: tuple[float, ...] = _RDCM_HEMO_CONSTANTS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full Euler integration of a DCM over time.

    Implements the complete Euler integration loop from Julia
    ``dcm_euler_integration.jl``, producing both neural activity and
    BOLD signal at each time step.

    BOLD signal per step uses the Julia formula with constants:
        ``k1 = relaxationRateSlope * oxygenExtractionFraction * echoTime``
        ``k2 = frequencyOffset * oxygenExtractionFraction * echoTime``
        ``k3 = 1.0``
        ``bold = restingVenousVolume * (k1*(1-q) + k2*(1-q/v) + k3*(1-v))``

    Parameters
    ----------
    A : torch.Tensor
        Connectivity matrix, shape ``(nr, nr)``, float64.
    C : torch.Tensor
        Input weights, shape ``(nr, nu)``, float64.
    u : torch.Tensor
        Stimulus input, shape ``(N_t, nu)``, float64.
    dt : float
        Euler step size (seconds).
    H : tuple of float
        Hemodynamic constants ``(kappa, gamma, tau, alpha, rho)``.

    Returns
    -------
    tuple of torch.Tensor
        ``(x_all, bold_all)`` where ``x_all`` has shape ``(N_t, nr)``
        (neural activity) and ``bold_all`` has shape ``(N_t, nr)``
        (BOLD signal).

    References
    ----------
    Julia ``dcm_euler_integration.jl``, ``generate_BOLD.jl``.
    """
    N_t = u.shape[0]
    nr = A.shape[0]

    # BOLD signal constants (Julia convention)
    k1 = _RELAXATION_RATE_SLOPE * _OXY_EXTRACTION_FRACTION * _ECHO_TIME
    k2 = _FREQUENCY_OFFSET * _OXY_EXTRACTION_FRACTION * _ECHO_TIME
    k3 = 1.0

    # Initialize states at equilibrium
    x = torch.zeros(nr, dtype=A.dtype, device=A.device)
    s = torch.zeros(nr, dtype=A.dtype, device=A.device)
    f = torch.ones(nr, dtype=A.dtype, device=A.device)
    v = torch.ones(nr, dtype=A.dtype, device=A.device)
    q = torch.ones(nr, dtype=A.dtype, device=A.device)

    x_all = torch.zeros(N_t, nr, dtype=A.dtype, device=A.device)
    bold_all = torch.zeros(N_t, nr, dtype=A.dtype, device=A.device)

    for t in range(N_t):
        x, s, f, v, q = dcm_euler_step(x, s, f, v, q, A, C, u[t], dt, H)
        x_all[t] = x
        # BOLD signal (Julia convention)
        bold_all[t] = _RESTING_VENOUS_VOLUME * (
            k1 * (1.0 - q)
            + k2 * (1.0 - q / torch.clamp(v, min=1e-6))
            + k3 * (1.0 - v)
        )

    return x_all, bold_all


def get_hrf(
    N: int,
    u_dt: float,
    H: tuple[float, ...] = _RDCM_HEMO_CONSTANTS,
) -> torch.Tensor:
    """Generate HRF by Euler-integrating a minimal 1-region DCM.

    Creates ``A = [[-1.0]]``, ``C = [[16.0]]`` and a unit impulse
    input, then Euler-integrates the DCM to produce the BOLD response
    which serves as the HRF. This matches the Julia ``get_hrf()``
    function in ``generate_BOLD.jl``.

    Parameters
    ----------
    N : int
        Number of time steps.
    u_dt : float
        Time step (seconds).
    H : tuple of float
        Hemodynamic constants ``(kappa, gamma, tau, alpha, rho)``.

    Returns
    -------
    torch.Tensor
        HRF signal, shape ``(N,)``, float64.

    Notes
    -----
    The Julia code uses ``C = [[16.0]]`` for the minimal DCM.
    The neural dynamics are ``dx/dt = A*x + C*u = -x + 16*u``,
    so the impulse produces a strong initial neural response that
    decays exponentially.

    References
    ----------
    Julia ``generate_BOLD.jl`` ``get_hrf()``.

    Examples
    --------
    >>> hrf = get_hrf(N=1000, u_dt=0.5)
    >>> hrf.shape  # (1000,)
    >>> hrf.dtype   # torch.float64
    """
    A_hrf = torch.tensor([[-1.0]], dtype=torch.float64)
    C_hrf = torch.tensor([[16.0]], dtype=torch.float64)

    # Unit impulse input
    u_hrf = torch.zeros(N, 1, dtype=torch.float64)
    u_hrf[0, 0] = 1.0

    _, bold = euler_integrate_dcm(A_hrf, C_hrf, u_hrf, u_dt, H)
    return bold[:, 0]


def generate_bold(
    A: torch.Tensor,
    C: torch.Tensor,
    u: torch.Tensor,
    u_dt: float,
    y_dt: float,
    SNR: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Generate synthetic BOLD for rDCM simulation.

    Steps (matching Julia ``generate_BOLD.jl``):
    1. Compute HRF via ``get_hrf(N_u, u_dt)``.
    2. Euler-integrate DCM with A, C, u to get neural signal x.
    3. Zero-pad x and hrf to ``3 * N_u`` to avoid circular convolution.
    4. Convolve x with hrf in frequency domain per region.
    5. Truncate back to ``N_u``, subsample at ratio ``y_dt / u_dt``.
    6. Add Gaussian noise: ``noise = randn * std(y_col) / SNR``.

    Parameters
    ----------
    A : torch.Tensor
        Connectivity matrix, shape ``(nr, nr)``, float64.
    C : torch.Tensor
        Input weights, shape ``(nr, nu)``, float64.
    u : torch.Tensor
        Stimulus input, shape ``(N_u, nu)``, float64.
    u_dt : float
        Input sampling interval (seconds).
    y_dt : float
        BOLD sampling interval / TR (seconds).
    SNR : float, optional
        Signal-to-noise ratio. Default 1.0.

    Returns
    -------
    dict
        Keys: ``'y'`` (noisy BOLD, shape ``(N_y, nr)``),
        ``'y_clean'`` (clean BOLD, shape ``(N_y, nr)``),
        ``'x'`` (neural activity, shape ``(N_u, nr)``),
        ``'hrf'`` (HRF, shape ``(N_u,)``).

    References
    ----------
    [REF-020] Frässle et al. (2017). Julia ``generate_BOLD.jl``.
    """
    N_u = u.shape[0]
    nr = A.shape[0]

    # Step 1: compute HRF
    hrf = get_hrf(N_u, u_dt)

    # Step 2: Euler-integrate DCM to get neural signal
    x, _ = euler_integrate_dcm(A, C, u, u_dt)

    # Step 3: zero-pad to 3*N_u for non-circular convolution
    pad_len = 3 * N_u
    x_padded = torch.zeros(pad_len, nr, dtype=torch.float64)
    x_padded[:N_u] = x
    hrf_padded = torch.zeros(pad_len, dtype=torch.float64)
    hrf_padded[:N_u] = hrf

    # Step 4: convolve in frequency domain per region
    hrf_fft = torch.fft.fft(hrf_padded)  # (pad_len,)
    y_padded = torch.zeros(pad_len, nr, dtype=torch.float64)
    for r in range(nr):
        x_fft_r = torch.fft.fft(x_padded[:, r])
        conv_r = torch.fft.ifft(x_fft_r * hrf_fft)
        y_padded[:, r] = conv_r.real

    # Step 5: truncate and subsample
    y_full = y_padded[:N_u]  # truncate
    ratio = int(round(y_dt / u_dt))
    y_clean = y_full[::ratio]  # subsample

    # Step 6: add noise
    y_noisy = y_clean.clone()
    if SNR < 1e10:  # skip noise for very large SNR
        for r in range(nr):
            col_std = y_clean[:, r].std()
            if col_std > 0:
                noise = torch.randn_like(y_clean[:, r]) * col_std / SNR
                y_noisy[:, r] = y_clean[:, r] + noise

    return {"y": y_noisy, "y_clean": y_clean, "x": x, "hrf": hrf}


def compute_derivative_coefficients(N: int) -> torch.Tensor:
    """Compute DFT derivative coefficients.

    ``coef[k] = exp(2 * pi * i * k / N) - 1`` for ``k = 0, ..., N//2``.

    These coefficients transform the DFT of a signal into the DFT
    of its temporal derivative (discrete difference). Used in
    [REF-020] Eq. 6-7 to compute Y from the DFT of BOLD.

    Parameters
    ----------
    N : int
        Length of the original time-domain signal.

    Returns
    -------
    torch.Tensor
        Complex128 tensor, shape ``(N // 2 + 1,)``.

    References
    ----------
    [REF-020] Eq. 6-7. Julia ``create_regressors.jl``.
    """
    k = torch.arange(N // 2 + 1, dtype=torch.float64)
    coef = torch.exp(2.0 * torch.pi * 1j * k / N) - 1.0
    return coef.to(torch.complex128)


def split_real_imag(
    Z_complex: torch.Tensor,
    N_y: int,
) -> torch.Tensor:
    """Split complex frequency-domain data into stacked real/imag parts.

    For real-valued time-domain signals, the DFT has symmetry
    constraints: the imaginary part at DC (index 0) is always zero,
    and for even-length signals, the imaginary part at Nyquist
    (last rfft index) is also zero. These zero-valued components
    are excluded from the stacked output.

    Parameters
    ----------
    Z_complex : torch.Tensor
        Complex128 output of ``rfft``, shape ``(N_rfft, ...)``.
    N_y : int
        Length of the original time-domain signal (needed to
        determine even/odd for Nyquist handling).

    Returns
    -------
    torch.Tensor
        Float64 tensor with real and imaginary parts stacked
        along dim 0. DC imaginary is always excluded; Nyquist
        imaginary is excluded for even ``N_y``.

    References
    ----------
    Julia ``rigid_inversion.jl``, ``sparse_inversion.jl``.
    """
    if N_y % 2 == 0:
        # Even: exclude DC (idx 0) and Nyquist (idx -1) imaginary
        out = torch.cat(
            [Z_complex.real, Z_complex.imag[1:-1]], dim=0
        )
    else:
        # Odd: exclude DC imaginary only
        out = torch.cat(
            [Z_complex.real, Z_complex.imag[1:]], dim=0
        )
    return out.to(torch.float64)


def reduce_zeros(
    Y: torch.Tensor,
    X: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Balance zero vs non-zero frequency rows with NaN markers.

    For each region ``r``, count zero-valued rows in ``Y[:, r]``.
    If there are more zeros than non-zeros, randomly replace some
    zero rows with NaN to balance the two groups. This improves
    regression quality by preventing zero-dominated frequencies from
    biasing the fit.

    Parameters
    ----------
    Y : torch.Tensor
        Frequency-domain data, shape ``(N_eff, nr)``, float64.
    X : torch.Tensor
        Frequency-domain design matrix, shape ``(N_eff, D)``, float64.

    Returns
    -------
    tuple of torch.Tensor
        ``(Y_out, X_out)`` with NaN markers on excluded rows.
        Same shapes as inputs.

    References
    ----------
    Julia ``create_regressors.jl`` ``reduce_zeros!``.
    """
    Y_out = Y.clone()
    X_out = X.clone()
    nr = Y.shape[1]

    for r in range(nr):
        col = Y_out[:, r]
        zero_mask = col == 0.0
        n_zeros = int(zero_mask.sum().item())
        n_nonzeros = len(col) - n_zeros

        if n_zeros > n_nonzeros and n_nonzeros > 0:
            # Need to remove some zeros to balance
            n_to_remove = n_zeros - n_nonzeros
            zero_indices = torch.where(zero_mask)[0]
            perm = torch.randperm(len(zero_indices))[:n_to_remove]
            remove_indices = zero_indices[perm]
            Y_out[remove_indices, r] = float("nan")
            X_out[remove_indices] = float("nan")

    return Y_out, X_out


def create_regressors(
    hrf: torch.Tensor,
    y: torch.Tensor,
    u: torch.Tensor,
    u_dt: float,
    y_dt: float,
    X0: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Create frequency-domain design matrix and data for rDCM regression.

    Constructs the design matrix X and data vector Y in the frequency
    domain following [REF-020] Eq. 4-8 and Julia ``create_regressors.jl``.

    Steps:
    1. If ``X0`` is ``None``, create constant confound ``ones(N_u, 1)``.
    2. Compute FFT of HRF: ``hrf_fft = rfft(hrf)``.
    3. Compute FFT of stimulus: ``u_fft = rfft(u, dim=0)``.
    4. Convolve stimulus with HRF: ``uh = irfft(u_fft * hrf_fft)``.
    5. Subsample convolved inputs and confounds at ``y_dt / u_dt``.
    6. Compute derivative coefficients for ``N_y``.
    7. ``Y = coef * rfft(y) / y_dt`` (DFT of temporal derivative).
    8. ``X = cat([rfft(y), rfft(uh_sub) / y_dt, rfft(X0_sub)])``.
    9. Apply ``split_real_imag`` to both Y and X.
    10. Apply ``reduce_zeros``.

    Parameters
    ----------
    hrf : torch.Tensor
        HRF from ``get_hrf``, shape ``(N_u,)``, float64.
    y : torch.Tensor
        BOLD signal, shape ``(N_y, nr)``, float64.
    u : torch.Tensor
        Stimulus inputs, shape ``(N_u, nu)``, float64.
    u_dt : float
        Input sampling interval (seconds).
    y_dt : float
        BOLD sampling interval / TR (seconds).
    X0 : torch.Tensor or None, optional
        Confound regressors, shape ``(N_u, nc)``. If ``None``,
        a constant confound ``ones(N_u, 1)`` is used.

    Returns
    -------
    tuple
        ``(X, Y, N_eff)`` where ``X`` is float64 shape
        ``(N_eff, nr + nu + nc)``, ``Y`` is float64 shape
        ``(N_eff, nr)``, and ``N_eff`` is the number of effective
        frequency-domain data points.

    References
    ----------
    [REF-020] Eq. 4-8. Julia ``create_regressors.jl``.
    """
    N_u = u.shape[0]
    N_y = y.shape[0]
    nr = y.shape[1]

    # Step 1: default confound
    if X0 is None:
        X0 = torch.ones(N_u, 1, dtype=torch.float64)

    # Step 2: FFT of HRF
    hrf_fft = torch.fft.rfft(hrf)  # (N_u//2+1,)

    # Step 3: FFT of stimulus
    u_fft = torch.fft.rfft(u, dim=0)  # (N_u//2+1, nu)

    # Step 4: convolve stimulus with HRF in frequency domain
    uh_fft = u_fft * hrf_fft[:, None]
    uh = torch.fft.irfft(uh_fft, n=N_u, dim=0)  # (N_u, nu)

    # Step 5: subsample convolved inputs and confounds to match BOLD
    ratio = int(round(y_dt / u_dt))
    uh_sub = uh[::ratio]  # (N_y, nu)
    X0_sub = X0[::ratio]  # (N_y, nc)

    # Step 6: derivative coefficients for N_y
    coef = compute_derivative_coefficients(N_y)  # (N_y//2+1,)

    # Step 7: Y = coef * rfft(y) / y_dt
    y_fft = torch.fft.rfft(y, dim=0)  # (N_y//2+1, nr)
    Y_complex = coef[:, None] * y_fft / y_dt

    # Step 8: X = [rfft(y), rfft(uh_sub)/y_dt, rfft(X0_sub)]
    uh_fft_sub = torch.fft.rfft(uh_sub, dim=0)  # (N_y//2+1, nu)
    X0_fft_sub = torch.fft.rfft(X0_sub, dim=0)  # (N_y//2+1, nc)
    X_complex = torch.cat(
        [y_fft, uh_fft_sub / y_dt, X0_fft_sub],
        dim=1,
    )  # (N_y//2+1, nr+nu+nc)

    # Step 9: split real/imag
    Y_real = split_real_imag(Y_complex, N_y)
    X_real = split_real_imag(X_complex, N_y)

    # Step 10: reduce zeros
    Y_out, X_out = reduce_zeros(Y_real, X_real)

    # Compute effective data points (non-NaN rows)
    N_eff = Y_out.shape[0]

    return X_out, Y_out, N_eff
