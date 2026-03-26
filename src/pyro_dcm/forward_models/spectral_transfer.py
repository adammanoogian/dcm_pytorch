"""Spectral DCM transfer function and predicted cross-spectral density.

Implements the eigendecomposition-based transfer function from [REF-010]
Eq. 3 (Friston, Kahan, Biswal & Razi, 2014) and the predicted CSD
assembly from [REF-010] Eq. 4, matching SPM12 spm_dcm_mtf.m and
spm_csd_fmri_mtf.m conventions.

The transfer function maps neural dynamics to observed cross-spectral
density via modal decomposition of the effective connectivity matrix A.
Eigenvalue stabilization follows the SPM convention of clamping real
parts to max(-1/32) for fMRI frequency ranges.
"""

from __future__ import annotations

import torch

from pyro_dcm.forward_models.spectral_noise import (
    neuronal_noise_csd,
    observation_noise_csd,
)


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
    """
    return torch.linspace(
        1.0 / 128.0,
        1.0 / (2.0 * TR),
        n_freqs,
        dtype=torch.float64,
    )


def compute_transfer_function(
    A: torch.Tensor,
    C_in: torch.Tensor,
    C_out: torch.Tensor,
    freqs: torch.Tensor,
) -> torch.Tensor:
    """Compute spectral transfer function via eigendecomposition.

    Implements [REF-010] Eq. 3 (Friston et al. 2014):
        g(w) = C_out @ (iwI - A)^{-1} @ C_in

    using modal decomposition for numerical stability:
        H(w) = sum_k dgdv_k * dvdu_k / (i*2*pi*w - lambda_k)

    Eigenvalue stabilization clamps real parts to max(-1/32) following
    the SPM12 convention for fMRI frequencies.

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

    Returns
    -------
    torch.Tensor
        Transfer function H, shape ``(F, nn, nu)``, complex128.
    """
    # Step 1: Eigendecompose A
    eigvals, eigvecs = torch.linalg.eig(A.to(torch.complex128))

    # Step 2: Stabilize eigenvalues (SPM convention for fMRI)
    # Clamp real parts to max(-1/32) to prevent blow-up
    eigvals = torch.complex(
        torch.clamp(eigvals.real, max=-1.0 / 32.0),
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
    """
    # S(w) = H(w) @ Gu(w) @ H(w)^H + Gn(w)
    G = H @ Gu @ H.conj().transpose(-2, -1)
    return G + Gn


def spectral_dcm_forward(
    A: torch.Tensor,
    freqs: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
) -> torch.Tensor:
    """Complete spectral DCM predicted CSD pipeline.

    Convenience function wrapping the full predicted CSD computation:
    transfer function (via eigendecomposition), neuronal and observation
    noise spectra, and CSD assembly. Uses C_in = C_out = identity
    following the standard spDCM convention.

    Implements [REF-010] Eq. 3-7 (Friston et al. 2014), matching
    SPM12 spm_csd_fmri_mtf.m.

    Parameters
    ----------
    A : torch.Tensor
        Effective connectivity matrix, shape ``(N, N)``, float64.
    freqs : torch.Tensor
        Frequency vector in Hz, shape ``(F,)``, float64.
    a : torch.Tensor
        Neuronal noise parameters, shape ``(2, N)``, float64.
        ``a[0, :]`` = log amplitude, ``a[1, :]`` = log exponent.
    b : torch.Tensor
        Global observation noise parameters, shape ``(2, 1)``, float64.
        ``b[0, 0]`` = log amplitude, ``b[1, 0]`` = log exponent.
    c : torch.Tensor
        Regional observation noise parameters, shape ``(2, N)``, float64.
        ``c[0, :]`` = log amplitude, ``c[1, :]`` = log exponent.

    Returns
    -------
    torch.Tensor
        Predicted CSD, shape ``(F, N, N)``, complex128.
    """
    N = A.shape[0]

    # Standard spDCM convention: C_in = C_out = identity
    C_in = torch.eye(N, dtype=torch.float64, device=A.device)
    C_out = torch.eye(N, dtype=torch.float64, device=A.device)

    # Compute transfer function via eigendecomposition
    H = compute_transfer_function(A, C_in, C_out, freqs)

    # Compute noise spectra
    Gu = neuronal_noise_csd(freqs, a)
    Gn = observation_noise_csd(freqs, b, c, N)

    # Assemble predicted CSD
    return predicted_csd(H, Gu, Gn)
