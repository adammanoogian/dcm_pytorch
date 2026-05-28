"""Spectral noise models for spectral DCM.

Implements the neuronal fluctuation and observation noise cross-spectral
density models from [REF-010] Eq. 5-7 (Friston, Kahan, Biswal & Razi,
2014), matching SPM12 spm_csd_fmri_mtf.m parameterization.

Two parameterization modes are supported:

**spm_fmri** (default, matches SPM12):
  - P.a (2, 1): shared neuronal noise amplitude + exponent
  - P.b (2, 1): global observation noise amplitude + exponent
  - P.c (1, N): per-region observation noise amplitude only
  - Total parameters: N + 4
  - All spectra normalized by /sum(G)

**extended** (original dcm_pytorch, for MEG/custom use):
  - P.a (2, N): per-region neuronal noise amplitude + exponent
  - P.b (2, 1): global observation noise amplitude + exponent
  - P.c (2, N): per-region observation noise amplitude + exponent
  - Total parameters: 4N + 2
  - Fixed scaling constants (C = 1/256)
"""

from __future__ import annotations

import torch


# Legacy SPM scaling constant (only used in "extended" parameterization)
_C_SCALE: float = 1.0 / 256.0


def neuronal_noise_csd(
    freqs: torch.Tensor,
    a: torch.Tensor,
    n_regions: int | None = None,
    *,
    noise_parameterization: str = "spm_fmri",
) -> torch.Tensor:
    """Compute neuronal fluctuation cross-spectral density (diagonal).

    Implements [REF-010] Eq. 5-6 (Friston et al. 2014).

    In ``spm_fmri`` mode (default), matches SPM12 spm_csd_fmri_mtf.m:
        Gu_i(w) = exp(a[0,0]) * w^(-exp(a[1,0])) / sum(w^(-exp(a[1,0])))

    In ``extended`` mode (original dcm_pytorch):
        Gu_i(w) = C * exp(a[0,i]) * w^(-exp(a[1,i])) * 4.0

    Cite: [REF-010] Eq. 5-6 and SPM12 spm_csd_fmri_mtf.m.

    Parameters
    ----------
    freqs : torch.Tensor
        Frequency vector in Hz, shape ``(F,)``, float64.
    a : torch.Tensor
        Neuronal noise parameters, float64.
        ``spm_fmri`` mode: shape ``(2, 1)`` -- shared amplitude and exponent.
        ``extended`` mode: shape ``(2, N)`` -- per-region.
    n_regions : int or None
        Number of brain regions. Required for ``spm_fmri`` mode (since
        ``a`` is shared). If None, inferred from ``a.shape[1]``.
    noise_parameterization : str
        ``"spm_fmri"`` (default): shared (2,1) params + /sum(G) normalization.
        ``"extended"``: per-region (2,N) params + fixed scaling constants.

    Returns
    -------
    torch.Tensor
        Neuronal noise CSD, shape ``(F, N, N)``, complex128.
        Diagonal matrix at each frequency.

    Examples
    --------
    >>> import torch
    >>> freqs = torch.linspace(1/128, 0.25, 32, dtype=torch.float64)
    >>> a = torch.zeros(2, 1, dtype=torch.float64)
    >>> Gu = neuronal_noise_csd(freqs, a, n_regions=3)
    >>> Gu.shape  # (32, 3, 3)
    """
    if noise_parameterization == "spm_fmri":
        # SPM12 mode: a is (2, 1) shared across all regions
        if n_regions is None:
            n_regions = a.shape[1]

        # Power-law spectrum: w^(-exp(a[1,0]))
        # G_raw shape: (F,)
        G_raw = freqs ** (-torch.exp(a[1, 0]))

        # Normalize by sum(G) -- SPM12 convention
        G_norm = G_raw / G_raw.sum()

        # Scale by amplitude
        G = torch.exp(a[0, 0]) * G_norm  # (F,)

        # Broadcast to N regions: (F, N)
        G_diag = G.unsqueeze(-1).expand(-1, n_regions)

        # Pack into diagonal (F, N, N) complex128 matrix
        Gu = torch.diag_embed(G_diag.to(torch.complex128))

    elif noise_parameterization == "extended":
        # Original dcm_pytorch mode: a is (2, N) per-region
        N = a.shape[1]
        amp = torch.exp(a[0, :])
        exp_val = torch.exp(a[1, :])

        # Power-law spectrum: C * amp * w^(-exp_val) * 4.0
        G = (
            _C_SCALE
            * amp[None, :]
            * freqs[:, None] ** (-exp_val[None, :])
            * 4.0
        )

        Gu = torch.diag_embed(G.to(torch.complex128))

    else:
        msg = (
            f"Unknown noise_parameterization: '{noise_parameterization}'. "
            f"Expected 'spm_fmri' or 'extended'."
        )
        raise ValueError(msg)

    return Gu


def observation_noise_csd(
    freqs: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    n_regions: int,
    *,
    noise_parameterization: str = "spm_fmri",
) -> torch.Tensor:
    """Compute observation noise cross-spectral density.

    Implements [REF-010] Eq. 7 (Friston et al. 2014).

    In ``spm_fmri`` mode (default), matches SPM12 spm_csd_fmri_mtf.m:
    - Global component (fills all entries i,j):
        G_global(w) = exp(b[0,0]) * w^(-exp(b[1,0])/2) / sum(G)
    - Regional component (diagonal only, adds to global):
        G_regional_i(w) = exp(c[0,i]) * w^(-exp(b[1,0])/2) / sum(G)
      Note: exponent comes from b, not c (c is amplitude-only).

    In ``extended`` mode (original dcm_pytorch):
    - Global component:
        G_global(w) = C * exp(b[0,0]) * w^(-exp(b[1,0])/2) / 8.0
    - Regional component:
        G_regional_i(w) = C * exp(c[0,i]) * w^(-exp(c[1,i])/2)

    Cite: [REF-010] Eq. 7 and SPM12 spm_csd_fmri_mtf.m.

    Parameters
    ----------
    freqs : torch.Tensor
        Frequency vector in Hz, shape ``(F,)``, float64.
    b : torch.Tensor
        Global observation noise params, shape ``(2, 1)``, float64.
        ``b[0, 0]`` = log amplitude, ``b[1, 0]`` = log exponent.
    c : torch.Tensor
        Regional observation noise params, float64.
        ``spm_fmri`` mode: shape ``(1, N)`` -- amplitude only.
        ``extended`` mode: shape ``(2, N)`` -- amplitude + exponent.
    n_regions : int
        Number of brain regions N.
    noise_parameterization : str
        ``"spm_fmri"`` (default): c is (1,N) amplitude-only + /sum(G).
        ``"extended"``: c is (2,N) with per-region exponent + fixed scaling.

    Returns
    -------
    torch.Tensor
        Observation noise CSD, shape ``(F, N, N)``, complex128.

    Examples
    --------
    >>> import torch
    >>> freqs = torch.linspace(1/128, 0.25, 32, dtype=torch.float64)
    >>> b = torch.zeros(2, 1, dtype=torch.float64)
    >>> c = torch.zeros(1, 3, dtype=torch.float64)
    >>> Gn = observation_noise_csd(freqs, b, c, n_regions=3)
    >>> Gn.shape  # (32, 3, 3)
    """
    F_len = freqs.shape[0]
    N = n_regions

    if noise_parameterization == "spm_fmri":
        # SPM12 mode: /sum(G) normalization, c is (1, N) amplitude-only

        # Global component: exp(b[0,0]) * w^(-exp(b[1,0])/2) / sum(G)
        G_global_raw = freqs ** (-torch.exp(b[1, 0]) / 2.0)  # (F,)
        G_global_norm = G_global_raw / G_global_raw.sum()
        G_global = torch.exp(b[0, 0]) * G_global_norm  # (F,)

        # Broadcast global noise to all entries (F, N, N)
        Gn = (
            G_global[:, None, None]
            * torch.ones(N, N, dtype=freqs.dtype, device=freqs.device)
        ).to(torch.complex128)

        # Regional component: exp(c[0,i]) * w^(-exp(b[1,0])/2) / sum(G)
        # Note: exponent from b[1,0], NOT from c (c is amplitude-only)
        # G_global_raw and G_global_norm are reused (same shape/exponent)
        amp_c = torch.exp(c[0, :])  # (N,)
        G_regional = amp_c[None, :] * G_global_norm[:, None]  # (F, N)

        # Add regional component to diagonal only
        Gn = Gn + torch.diag_embed(G_regional.to(torch.complex128))

    elif noise_parameterization == "extended":
        # Original dcm_pytorch mode: c is (2, N), fixed scaling constants

        # Global component: C * exp(b[0,0]) * w^(-exp(b[1,0])/2) / 8.0
        G_global = (
            _C_SCALE
            * torch.exp(b[0, 0])
            * freqs ** (-torch.exp(b[1, 0]) / 2.0)
            / 8.0
        )

        Gn = (
            G_global[:, None, None]
            * torch.ones(N, N, dtype=freqs.dtype, device=freqs.device)
        ).to(torch.complex128)

        # Regional component: C * exp(c[0,i]) * w^(-exp(c[1,i])/2)
        amp_c = torch.exp(c[0, :])
        exp_c = torch.exp(c[1, :])
        G_regional = (
            _C_SCALE
            * amp_c[None, :]
            * freqs[:, None] ** (-exp_c[None, :] / 2.0)
        )

        Gn = Gn + torch.diag_embed(G_regional.to(torch.complex128))

    else:
        msg = (
            f"Unknown noise_parameterization: '{noise_parameterization}'. "
            f"Expected 'spm_fmri' or 'extended'."
        )
        raise ValueError(msg)

    return Gn


def default_noise_priors(
    n_regions: int,
    *,
    noise_parameterization: str = "spm_fmri",
) -> dict[str, torch.Tensor]:
    """Return SPM12 default prior expectations and variances for noise.

    SPM12 uses zero-mean Gaussian priors in log-space for all noise
    parameters, with variance 1/64.

    In ``spm_fmri`` mode: Total parameters = 2 + 2 + N = N + 4.
    In ``extended`` mode: Total parameters = 2N + 2 + 2N = 4N + 2.

    Cite: SPM12 spm_dcm_fmri_priors.m.

    Parameters
    ----------
    n_regions : int
        Number of brain regions.
    noise_parameterization : str
        ``"spm_fmri"`` (default): a=(2,1), b=(2,1), c=(1,N).
        ``"extended"``: a=(2,N), b=(2,1), c=(2,N).

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary with keys:
        - ``a_prior_mean``: shape ``(2, 1)`` or ``(2, N)``
        - ``a_prior_var``: same shape, filled with 1/64
        - ``b_prior_mean``: shape ``(2, 1)``
        - ``b_prior_var``: shape ``(2, 1)``, filled with 1/64
        - ``c_prior_mean``: shape ``(1, N)`` or ``(2, N)``
        - ``c_prior_var``: same shape, filled with 1/64

    Examples
    --------
    >>> priors = default_noise_priors(n_regions=3)
    >>> priors['a_prior_mean'].shape  # (2, 1)
    >>> priors['c_prior_mean'].shape  # (1, 3)
    """
    N = n_regions
    var_val = 1.0 / 64.0

    if noise_parameterization == "spm_fmri":
        return {
            "a_prior_mean": torch.zeros(2, 1, dtype=torch.float64),
            "a_prior_var": torch.full((2, 1), var_val, dtype=torch.float64),
            "b_prior_mean": torch.zeros(2, 1, dtype=torch.float64),
            "b_prior_var": torch.full((2, 1), var_val, dtype=torch.float64),
            "c_prior_mean": torch.zeros(1, N, dtype=torch.float64),
            "c_prior_var": torch.full((1, N), var_val, dtype=torch.float64),
        }
    elif noise_parameterization == "extended":
        return {
            "a_prior_mean": torch.zeros(2, N, dtype=torch.float64),
            "a_prior_var": torch.full((2, N), var_val, dtype=torch.float64),
            "b_prior_mean": torch.zeros(2, 1, dtype=torch.float64),
            "b_prior_var": torch.full((2, 1), var_val, dtype=torch.float64),
            "c_prior_mean": torch.zeros(2, N, dtype=torch.float64),
            "c_prior_var": torch.full((2, N), var_val, dtype=torch.float64),
        }
    else:
        msg = (
            f"Unknown noise_parameterization: '{noise_parameterization}'. "
            f"Expected 'spm_fmri' or 'extended'."
        )
        raise ValueError(msg)
