"""Spectral noise models for spectral DCM.

Implements the neuronal fluctuation and observation noise cross-spectral
density models from [REF-010] Eq. 5-7 (Friston, Kahan, Biswal & Razi,
2014), matching SPM12 spm_csd_fmri_mtf.m parameterization.

The noise model uses three parameter groups in log-space:
  - P.a (2, N): neuronal fluctuation amplitude + exponent per region
  - P.b (2, 1): global observation noise amplitude + exponent
  - P.c (2, N): regional observation noise amplitude + exponent per region

Total parameters: 4N + 2, all in log-space with C = 1/256 scaling.
"""

from __future__ import annotations

import torch


# SPM scaling constant applied to all noise spectra
_C_SCALE: float = 1.0 / 256.0


def neuronal_noise_csd(
    freqs: torch.Tensor,
    a: torch.Tensor,
) -> torch.Tensor:
    """Compute neuronal fluctuation cross-spectral density (diagonal).

    Implements [REF-010] Eq. 5-6 (Friston et al. 2014):
        Gu_i(w) = C * exp(a[0,i]) * w^(-exp(a[1,i])) * 4.0

    where C = 1/256 is SPM's scaling constant. The neuronal noise is
    diagonal (independent fluctuations per region), producing a 1/f
    power-law spectrum.

    Cite: [REF-010] Eq. 5-6 and SPM12 spm_csd_fmri_mtf.m.

    Parameters
    ----------
    freqs : torch.Tensor
        Frequency vector in Hz, shape ``(F,)``, float64.
    a : torch.Tensor
        Neuronal noise parameters, shape ``(2, N)``, float64.
        ``a[0, :]`` = log amplitude per region.
        ``a[1, :]`` = log exponent per region.

    Returns
    -------
    torch.Tensor
        Neuronal noise CSD, shape ``(F, N, N)``, complex128.
        Diagonal matrix at each frequency.

    Examples
    --------
    >>> import torch
    >>> freqs = torch.linspace(1/128, 0.25, 32, dtype=torch.float64)
    >>> a = torch.zeros(2, 3, dtype=torch.float64)
    >>> Gu = neuronal_noise_csd(freqs, a)
    >>> Gu.shape  # (32, 3, 3)
    """
    # Vectorized computation over regions
    # amp: (N,), exp_val: (N,)
    amp = torch.exp(a[0, :])
    exp_val = torch.exp(a[1, :])

    # Power-law spectrum: C * amp * w^(-exp_val) * 4.0
    # G shape: (F, N)
    G = (
        _C_SCALE
        * amp[None, :]
        * freqs[:, None] ** (-exp_val[None, :])
        * 4.0
    )

    # Pack into diagonal (F, N, N) complex128 matrix
    Gu = torch.diag_embed(G.to(torch.complex128))

    return Gu


def observation_noise_csd(
    freqs: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    n_regions: int,
) -> torch.Tensor:
    """Compute observation noise cross-spectral density.

    Implements [REF-010] Eq. 7 (Friston et al. 2014):
    - Global component (fills all entries i,j):
        G_global(w) = C * exp(b[0,0]) * w^(-exp(b[1,0])/2) / 8.0
    - Regional component (diagonal only, adds to global):
        G_regional_i(w) = C * exp(c[0,i]) * w^(-exp(c[1,i])/2)

    SPM divides the exponent by 2 for observation noise, producing a
    flatter spectrum than the neuronal noise. The /8.0 factor on the
    global component matches SPM12 spm_csd_fmri_mtf.m.

    Cite: [REF-010] Eq. 7 and SPM12 spm_csd_fmri_mtf.m.

    Parameters
    ----------
    freqs : torch.Tensor
        Frequency vector in Hz, shape ``(F,)``, float64.
    b : torch.Tensor
        Global observation noise params, shape ``(2, 1)``, float64.
        ``b[0, 0]`` = log amplitude, ``b[1, 0]`` = log exponent.
    c : torch.Tensor
        Regional observation noise params, shape ``(2, N)``, float64.
        ``c[0, :]`` = log amplitude, ``c[1, :]`` = log exponent.
    n_regions : int
        Number of brain regions N.

    Returns
    -------
    torch.Tensor
        Observation noise CSD, shape ``(F, N, N)``, complex128.

    Examples
    --------
    >>> import torch
    >>> freqs = torch.linspace(1/128, 0.25, 32, dtype=torch.float64)
    >>> b = torch.zeros(2, 1, dtype=torch.float64)
    >>> c = torch.zeros(2, 3, dtype=torch.float64)
    >>> Gn = observation_noise_csd(freqs, b, c, n_regions=3)
    >>> Gn.shape  # (32, 3, 3)
    """
    F_len = freqs.shape[0]
    N = n_regions

    # Global component: C * exp(b[0,0]) * w^(-exp(b[1,0])/2) / 8.0
    # Shape: (F,)
    G_global = (
        _C_SCALE
        * torch.exp(b[0, 0])
        * freqs ** (-torch.exp(b[1, 0]) / 2.0)
        / 8.0
    )

    # Broadcast global noise to all entries (F, N, N)
    # Use ones matrix to fill all pairs
    Gn = (
        G_global[:, None, None]
        * torch.ones(N, N, dtype=freqs.dtype, device=freqs.device)
    ).to(torch.complex128)

    # Regional component: C * exp(c[0,i]) * w^(-exp(c[1,i])/2)
    # Shape: (F, N)
    amp_c = torch.exp(c[0, :])
    exp_c = torch.exp(c[1, :])
    G_regional = (
        _C_SCALE
        * amp_c[None, :]
        * freqs[:, None] ** (-exp_c[None, :] / 2.0)
    )

    # Add regional component to diagonal only
    # Create diagonal matrix from regional spectrum and add
    Gn = Gn + torch.diag_embed(G_regional.to(torch.complex128))

    return Gn


def default_noise_priors(n_regions: int) -> dict[str, torch.Tensor]:
    """Return SPM12 default prior expectations and variances for noise.

    SPM12 uses zero-mean Gaussian priors in log-space for all noise
    parameters, with variance 1/64. Total parameters: 4N + 2 where
    N is the number of regions.

    Cite: SPM12 spm_dcm_fmri_priors.m.

    Parameters
    ----------
    n_regions : int
        Number of brain regions.

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary with keys:
        - ``a_prior_mean``: shape ``(2, N)``, zeros (float64)
        - ``a_prior_var``: shape ``(2, N)``, filled with 1/64 (float64)
        - ``b_prior_mean``: shape ``(2, 1)``, zeros (float64)
        - ``b_prior_var``: shape ``(2, 1)``, filled with 1/64 (float64)
        - ``c_prior_mean``: shape ``(2, N)``, zeros (float64)
        - ``c_prior_var``: shape ``(2, N)``, filled with 1/64 (float64)

    Examples
    --------
    >>> priors = default_noise_priors(n_regions=3)
    >>> priors['a_prior_mean'].shape  # (2, 3)
    >>> priors['b_prior_var'].item()  # 0.015625 (1/64)
    """
    N = n_regions
    var_val = 1.0 / 64.0

    return {
        "a_prior_mean": torch.zeros(2, N, dtype=torch.float64),
        "a_prior_var": torch.full((2, N), var_val, dtype=torch.float64),
        "b_prior_mean": torch.zeros(2, 1, dtype=torch.float64),
        "b_prior_var": torch.full((2, 1), var_val, dtype=torch.float64),
        "c_prior_mean": torch.zeros(2, N, dtype=torch.float64),
        "c_prior_var": torch.full((2, N), var_val, dtype=torch.float64),
    }
