"""Empirical cross-spectral density computation from BOLD time series.

Computes cross-spectral density (CSD) matrices from multivariate BOLD
time series using the Welch periodogram method (scipy.signal.csd). This
is the data preparation module that converts raw BOLD time series into
the frequency-domain CSD format required by spectral DCM ([REF-010]).

The empirical CSD is compared against the predicted CSD from the spectral
DCM forward model during inference (Phase 4). This module implements the
empirical side of that comparison.

SPM12 uses MAR-based CSD estimation (spm_mar + spm_mar_spectra), but the
CONTEXT.md decision specifies Welch-based CSD as the standard signal
processing approach. The predicted CSD forward model (spectral_transfer.py)
matches SPM's formula exactly.

References
----------
[REF-010] Friston, Kahan, Biswal & Razi (2014). A DCM for resting state
    fMRI. NeuroImage, 94, 396-407. Eq. 3-4.
[REF-011] Razi et al. (2015). Construct validation of a DCM for resting
    state fMRI. NeuroImage, 106, 1-14.
"""

from __future__ import annotations

import numpy as np
import scipy.signal
import torch


def default_welch_params(T: int, fs: float) -> dict[str, int | str]:
    """Return sensible Welch parameters for fMRI data.

    Provides default segment length, overlap, and window for CSD
    estimation from fMRI time series. The segment length is clamped
    to ``min(256, T)`` to handle short time series gracefully.

    Parameters
    ----------
    T : int
        Number of time points in the BOLD time series.
    fs : float
        Sampling frequency in Hz (= 1/TR).

    Returns
    -------
    dict[str, int | str]
        Dictionary with keys:

        - ``nperseg`` : int -- segment length for Welch method.
        - ``noverlap`` : int -- overlap between segments (50%).
        - ``window`` : str -- window function name.
    """
    nperseg = min(256, T)
    noverlap = nperseg // 2
    return {"nperseg": nperseg, "noverlap": noverlap, "window": "hann"}


def compute_empirical_csd(
    bold: np.ndarray,
    fs: float,
    freqs: np.ndarray,
    nperseg: int | None = None,
) -> np.ndarray:
    """Compute cross-spectral density matrix from BOLD time series.

    Uses ``scipy.signal.csd`` (Welch periodogram) for each region pair,
    then interpolates the result onto the target frequency grid. Enforces
    Hermitian symmetry by computing only the upper triangle and
    conjugating for the lower triangle.

    Implements empirical CSD estimation for spectral DCM data preparation
    per [REF-010] (Friston et al. 2014) and [REF-011] (Razi et al. 2015).

    Parameters
    ----------
    bold : np.ndarray
        BOLD time series, shape ``(T, N)`` where T is time points and
        N is number of regions.
    fs : float
        Sampling frequency in Hz (= 1/TR).
    freqs : np.ndarray
        Target frequency grid in Hz, shape ``(F,)``. The raw Welch CSD
        is interpolated onto this grid.
    nperseg : int or None, optional
        Segment length for Welch method. If None, uses
        ``min(256, T)`` (see :func:`default_welch_params`).

    Returns
    -------
    np.ndarray
        Complex CSD matrix, shape ``(F, N, N)``, dtype complex128.
        Hermitian at each frequency: ``csd[f, i, j] == conj(csd[f, j, i])``.
        Diagonal (auto-spectra) are real and non-negative.

    Notes
    -----
    Scaling is ``'density'`` (power per Hz), matching one-sided PSD
    convention for real-valued signals. The interpolation uses
    ``np.interp`` on real and imaginary parts separately.

    References
    ----------
    [REF-010] Friston et al. (2014), Eq. 3-4 (empirical CSD input).
    [REF-011] Razi et al. (2015) — CSD estimation from BOLD.
    """
    T, N = bold.shape

    if nperseg is None:
        nperseg = min(256, T)

    F = len(freqs)
    csd = np.zeros((F, N, N), dtype=np.complex128)

    # Compute upper triangle + diagonal, enforce Hermitian symmetry
    for i in range(N):
        for j in range(i, N):
            f_raw, Pxy = scipy.signal.csd(
                bold[:, i],
                bold[:, j],
                fs=fs,
                nperseg=nperseg,
                scaling="density",
            )
            # Interpolate real and imaginary parts onto target grid
            real_interp = np.interp(freqs, f_raw, Pxy.real)
            imag_interp = np.interp(freqs, f_raw, Pxy.imag)
            csd[:, i, j] = real_interp + 1j * imag_interp

            if i != j:
                # Hermitian symmetry: S(f, j, i) = conj(S(f, i, j))
                csd[:, j, i] = np.conj(csd[:, i, j])

    return csd


def bold_to_csd_torch(
    bold: torch.Tensor,
    fs: float,
    freqs: torch.Tensor,
    nperseg: int | None = None,
) -> torch.Tensor:
    """Compute empirical CSD from BOLD time series (torch interface).

    Convenience wrapper that accepts ``torch.Tensor`` inputs, converts
    to numpy, calls :func:`compute_empirical_csd`, and returns a
    ``torch.complex128`` tensor. This is the interface that Phase 4's
    Pyro model uses to prepare observed data.

    Parameters
    ----------
    bold : torch.Tensor
        BOLD time series, shape ``(T, N)``.
    fs : float
        Sampling frequency in Hz (= 1/TR).
    freqs : torch.Tensor
        Target frequency grid in Hz, shape ``(F,)``.
    nperseg : int or None, optional
        Segment length for Welch method. If None, auto-selected.

    Returns
    -------
    torch.Tensor
        Complex CSD matrix, shape ``(F, N, N)``, dtype ``torch.complex128``.

    See Also
    --------
    compute_empirical_csd : NumPy implementation.
    """
    bold_np = bold.detach().cpu().numpy()
    freqs_np = freqs.detach().cpu().numpy()

    csd_np = compute_empirical_csd(bold_np, fs, freqs_np, nperseg=nperseg)

    return torch.as_tensor(csd_np, dtype=torch.complex128)
