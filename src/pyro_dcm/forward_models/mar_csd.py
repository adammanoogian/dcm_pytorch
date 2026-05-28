"""Multivariate Autoregressive CSD computation matching SPM12.

Implements the MAR-based cross-spectral density estimation used by
SPM12's spectral DCM internally. This allows dcm_pytorch to use
identical spectral input as SPM12 for a fair inference comparison.

The approach: fit a multivariate AR model (Yule-Walker), then compute
the transfer function H(f) = (I - sum_p A_p * exp(-2pi*i*f*p*dt))^{-1}
and CSD = H * Sigma * H^H, where Sigma is the residual covariance.

Functions for the full MAR pipeline:

- ``mar_ml``: Fit MAR model from time-domain data (Yule-Walker).
- ``mar2csd``: Convert MAR model to CSD (forward transform).
- ``csd2ccf``: Convert CSD to cross-covariance function (inverse FFT).
  Reimplements SPM12 ``spm_csd2ccf.m``.
- ``ccf2mar``: Convert cross-covariance to MAR coefficients
  (Yule-Walker on CCF). Reimplements SPM12 ``spm_ccf2mar.m``.
- ``csd2mar``: Convenience wrapper (CSD -> CCF -> MAR).

This matches SPM12's spm_csd_mtf.m / mar_ml / mar2csd logic and
Julia's SpectralDynamicCausalModeling.mar_ml + mar2csd.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import inv, solve
from scipy.linalg import toeplitz


def mar_ml(data: np.ndarray, order: int = 8) -> dict:
    """Fit multivariate AR model via Yule-Walker equations.

    Parameters
    ----------
    data : np.ndarray
        Time series, shape ``(T, n_channels)``.
    order : int
        AR model order (default 8, matching SPM12).

    Returns
    -------
    dict
        Keys: ``A`` (list of n_channels x n_channels AR coefficient
        matrices, length ``order``), ``noise_cov`` (residual covariance,
        n_channels x n_channels).
    """
    T, n = data.shape
    data = data - data.mean(axis=0)

    # Compute autocovariance matrices R(k) for k = 0..order
    R = []
    for k in range(order + 1):
        R_k = (data[k:].T @ data[:T - k]) / (T - k) if k > 0 else (data.T @ data) / T
        R.append(R_k)

    # Block Toeplitz system: [R(0) R(1)' ... R(p-1)'] [A1]   [R(1) ]
    #                        [R(1) R(0)  ... R(p-2)'] [A2] = [R(2) ]
    #                        [... ]                    [..]   [..   ]
    #                        [R(p-1) ...        R(0) ] [Ap]   [R(p) ]
    block_size = n
    rhs = np.vstack([R[k] for k in range(1, order + 1)])  # (order*n, n)

    # Build block Toeplitz matrix
    blocks = np.zeros((order * n, order * n))
    for i in range(order):
        for j in range(order):
            lag = abs(i - j)
            if i >= j:
                blocks[i * n:(i + 1) * n, j * n:(j + 1) * n] = R[lag]
            else:
                blocks[i * n:(i + 1) * n, j * n:(j + 1) * n] = R[lag].T

    # Solve for AR coefficients
    A_flat = solve(blocks, rhs)  # (order*n, n)
    A_list = [A_flat[k * n:(k + 1) * n].T for k in range(order)]

    # Residual covariance
    noise_cov = R[0].copy()
    for k in range(order):
        noise_cov -= A_list[k] @ R[k + 1]

    return {"A": A_list, "noise_cov": noise_cov}


def mar2csd(
    mar: dict,
    frequencies: np.ndarray,
    fs: float,
) -> np.ndarray:
    """Convert MAR model to cross-spectral density.

    Parameters
    ----------
    mar : dict
        Output of ``mar_ml``.
    frequencies : np.ndarray
        Frequency vector in Hz.
    fs : float
        Sampling rate (1/TR).

    Returns
    -------
    np.ndarray
        Complex CSD matrix, shape ``(n_freq, n_channels, n_channels)``.
    """
    A_list = mar["A"]
    noise_cov = mar["noise_cov"]
    order = len(A_list)
    n = A_list[0].shape[0]
    n_freq = len(frequencies)

    csd = np.zeros((n_freq, n, n), dtype=complex)
    I = np.eye(n)
    dt = 1.0 / fs

    for fi, f in enumerate(frequencies):
        # Transfer function: H(f) = (I - sum_p A_p * exp(-2pi*i*f*p*dt))^{-1}
        H_inv = I.copy().astype(complex)
        for p in range(order):
            H_inv -= A_list[p] * np.exp(-2j * np.pi * f * (p + 1) * dt)
        H = inv(H_inv)
        # CSD = H * Sigma * H^H
        csd[fi] = H @ noise_cov @ H.conj().T

    return csd


def csd2ccf(
    csd: np.ndarray,
    Hz: np.ndarray,
    dt: float | None = None,
) -> np.ndarray:
    """Convert CSD to cross-covariance function.

    Reimplements SPM12 ``spm_csd2ccf.m``: pads CSD onto a full FFT
    grid, performs inverse FFT, takes real part, and scales
    appropriately.

    Parameters
    ----------
    csd : np.ndarray
        Cross-spectral density, shape ``(n_freq, n_channels,
        n_channels)``.
    Hz : np.ndarray
        Frequency vector in Hz, shape ``(n_freq,)``.
    dt : float or None
        Sampling interval. If None, computed as ``1 / (2 * Hz[-1])``.

    Returns
    -------
    np.ndarray
        Cross-covariance function, shape ``(2*N+1, n_channels,
        n_channels)`` where ``N = ceil(ns / 2 / dw)``.
    """
    dw = Hz[1] - Hz[0]
    if dt is None:
        dt = 1.0 / (2.0 * Hz[-1])
    ns = 1.0 / dt
    N = int(np.ceil(ns / 2.0 / dw))

    n_channels = csd.shape[1]
    ccf = np.zeros((2 * N + 1, n_channels, n_channels))

    for i in range(n_channels):
        for j in range(n_channels):
            # Build zero-padded one-sided spectrum of length N
            g = np.zeros(N, dtype=complex)
            # Map CSD frequency bins into FFT grid indices
            gi = np.arange(len(Hz)) + int(np.ceil(Hz[0] / dw)) - 1
            valid = (gi >= 0) & (gi < N)
            g[gi[valid]] = csd[valid, i, j]
            # Build full two-sided spectrum and inverse FFT
            f = np.fft.ifft(
                np.concatenate([[0], g, np.flipud(np.conj(g))])
            )
            ccf[:, i, j] = np.real(np.fft.fftshift(f)) * N * dw

    return ccf


def ccf2mar(
    ccf: np.ndarray,
    p: int,
) -> dict:
    """Convert cross-covariance to MAR coefficients.

    Reimplements SPM12 ``spm_ccf2mar.m``: builds a block-Toeplitz
    system from the cross-covariance and solves via Yule-Walker.

    Parameters
    ----------
    ccf : np.ndarray
        Cross-covariance function, shape ``(2*n+1, m, m)`` where ``n``
        is the number of positive lags and ``m`` is number of channels.
    p : int
        MAR model order.

    Returns
    -------
    dict
        Keys: ``A`` (list of ``m x m`` AR coefficient matrices, length
        ``p``), ``noise_cov`` (residual covariance, ``m x m``).
        The AR coefficient sign convention matches ``mar2csd``:
        ``H_inv = I - sum_k A_k * exp(-2j*pi*f*k*dt)``.
    """
    N_total = ccf.shape[0]
    m = ccf.shape[1]
    n = (N_total - 1) // 2

    # Cap order at available lags
    p = min(p, n - 1)

    # Extract positive-lag cross-covariance: lags 0, 1, ..., n
    ccf_pos = ccf[n:, :, :]

    # Build RHS: stack ccf at lags 1..p -> shape (p*m, m)
    rhs = np.vstack([ccf_pos[k + 1] for k in range(p)])

    # Build block-Toeplitz matrix (p*m, p*m)
    B = np.zeros((p * m, p * m))
    for i in range(p):
        for j in range(p):
            lag = abs(i - j)
            if i >= j:
                B[i * m : (i + 1) * m, j * m : (j + 1) * m] = ccf_pos[lag]
            else:
                B[i * m : (i + 1) * m, j * m : (j + 1) * m] = ccf_pos[lag].T

    # Solve for coefficients: coeff shape (p*m, m)
    coeff = np.linalg.solve(B, rhs)

    # Extract per-lag AR coefficient matrices
    # Convention: A_k such that H_inv = I - sum_k A_k * exp(...)
    # This matches the existing mar2csd which subtracts A_list[p].
    A_list = [coeff[k * m : (k + 1) * m].T for k in range(p)]

    # Residual noise covariance
    noise_cov = ccf_pos[0].copy()
    for k in range(p):
        noise_cov = noise_cov - A_list[k] @ ccf_pos[k + 1]

    return {"A": A_list, "noise_cov": noise_cov}


def csd2mar(
    csd: np.ndarray,
    Hz: np.ndarray,
    p: int = 7,
    dt: float | None = None,
) -> dict:
    """Convert CSD to MAR model (CSD -> CCF -> MAR).

    Convenience wrapper matching SPM12's ``spm_csd2mar.m`` pipeline:
    first converts CSD to cross-covariance via ``csd2ccf``, then fits
    MAR coefficients via ``ccf2mar``.

    Parameters
    ----------
    csd : np.ndarray
        Cross-spectral density, shape ``(n_freq, n_channels,
        n_channels)``.
    Hz : np.ndarray
        Frequency vector in Hz, shape ``(n_freq,)``.
    p : int
        MAR model order. Default 7, matching SPM12's ``M.p - 1 = 7``.
    dt : float or None
        Sampling interval. If None, computed as ``1 / (2 * Hz[-1])``.

    Returns
    -------
    dict
        Keys: ``A`` (list of AR coefficient matrices), ``noise_cov``
        (residual covariance).
    """
    ccf = csd2ccf(csd, Hz, dt=dt)
    return ccf2mar(ccf, p)


def compute_spm_compatible_csd(
    bold: np.ndarray,
    tr: float,
    order: int = 8,
    n_freq: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute CSD matching SPM12's internal method.

    Parameters
    ----------
    bold : np.ndarray
        BOLD time series, shape ``(T, n_regions)``.
    tr : float
        Repetition time in seconds.
    order : int
        MAR model order (default 8).
    n_freq : int
        Number of frequency bins (default 32).

    Returns
    -------
    csd : np.ndarray
        Complex CSD, shape ``(n_freq, n_regions, n_regions)``.
    frequencies : np.ndarray
        Frequency vector in Hz, shape ``(n_freq,)``.
    """
    T = bold.shape[0]
    fs = 1.0 / tr

    f_min = 1.0 / min(128, T * tr)
    f_max = 1.0 / max(8, 2 * tr)
    frequencies = np.linspace(f_min, f_max, n_freq)

    mar = mar_ml(bold, order)
    csd = mar2csd(mar, frequencies, fs)

    return csd, frequencies
