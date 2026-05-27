"""Multivariate Autoregressive CSD computation matching SPM12.

Implements the MAR-based cross-spectral density estimation used by
SPM12's spectral DCM internally. This allows dcm_pytorch to use
identical spectral input as SPM12 for a fair inference comparison.

The approach: fit a multivariate AR model (Yule-Walker), then compute
the transfer function H(f) = (I - sum_p A_p * exp(-2pi*i*f*p*dt))^{-1}
and CSD = H * Sigma * H^H, where Sigma is the residual covariance.

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
