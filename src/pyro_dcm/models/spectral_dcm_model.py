"""Pyro generative model for spectral DCM.

Implements the probabilistic model for spectral Dynamic Causal Modeling
(spDCM) as described in [REF-010] Eq. 3-10 (Friston, Kahan, Biswal &
Razi, 2014). The model samples effective connectivity parameters
(A_free) and noise parameters (a, b, c) from Normal priors matching
SPM12 spm_dcm_fmri_priors.m conventions, computes predicted
cross-spectral density via the spectral DCM forward model, decomposes
the complex CSD into a real-valued vector (since Pyro distributions
do not support complex128 tensors), and evaluates a Gaussian likelihood
against observed CSD data.

The complex-to-real decomposition follows the pattern documented in
04-RESEARCH.md Pattern 2: flatten the (F, N, N) complex CSD matrix
and stack real/imaginary parts into a (2*F*N*N,) float64 vector.

References
----------
[REF-010] Friston, Kahan, Biswal & Razi (2014). A DCM for resting
    state fMRI. NeuroImage, 94, 396-407. Eq. 3-10.
"""

from __future__ import annotations

import torch
import pyro
import pyro.distributions as dist

from pyro_dcm.forward_models.neural_state import parameterize_A
from pyro_dcm.forward_models.spectral_transfer import spectral_dcm_forward


def decompose_csd_for_likelihood(
    csd_complex: torch.Tensor,
) -> torch.Tensor:
    """Convert complex CSD tensor to real-valued vector for Pyro likelihood.

    Pyro's ``dist.Normal`` does not support complex128 tensors. This
    function decomposes a complex CSD matrix into a real-valued vector
    by flattening and stacking real and imaginary parts.

    Implements the workaround from 04-RESEARCH.md Pattern 2.

    Parameters
    ----------
    csd_complex : torch.Tensor
        Complex cross-spectral density, shape ``(F, N, N)``,
        dtype complex128.

    Returns
    -------
    torch.Tensor
        Real-valued vector of shape ``(2 * F * N * N,)``, dtype
        float64. First half contains real parts, second half contains
        imaginary parts.

    Examples
    --------
    >>> csd = torch.randn(32, 3, 3, dtype=torch.complex128)
    >>> vec = decompose_csd_for_likelihood(csd)
    >>> vec.shape  # (576,)
    >>> vec.dtype  # torch.float64
    """
    return torch.cat(
        [csd_complex.real.reshape(-1), csd_complex.imag.reshape(-1)]
    )


def spectral_dcm_model(
    observed_csd: torch.Tensor,
    freqs: torch.Tensor,
    a_mask: torch.Tensor,
    N: int | None = None,
) -> None:
    """Pyro generative model for spectral DCM.

    Samples connectivity parameters A_free and noise parameters
    (a, b, c), applies structural masking and the parameterize_A
    transform, computes predicted CSD via the spectral DCM forward
    model, decomposes the complex CSD to a real-valued vector, and
    evaluates a Gaussian likelihood against observed CSD data.

    Implements [REF-010] Eq. 3-10 (Friston et al. 2014).

    Parameters
    ----------
    observed_csd : torch.Tensor
        Observed cross-spectral density, shape ``(F, N, N)``,
        dtype complex128.
    freqs : torch.Tensor
        Frequency vector in Hz, shape ``(F,)``, dtype float64.
    a_mask : torch.Tensor
        Binary structural mask for A connections, shape ``(N, N)``,
        dtype float64. 1 where connection exists, 0 where absent.
    N : int or None, optional
        Number of brain regions. If None, inferred from
        ``a_mask.shape[0]``.

    Notes
    -----
    - A_free is sampled from N(0, 1/64) matching SPM12
      spm_dcm_fmri_priors.m. Structural masking zeros out absent
      connections before the parameterize_A transform.
    - Noise parameters (a, b, c) follow SPM12 conventions:
      a (2, N) neuronal, b (2, 1) global observation, c (2, N)
      regional observation.
    - The predicted CSD (complex128) is stored as a Pyro deterministic
      site for downstream analysis.
    - Only the decomposed real vector is used in the likelihood.
    - An additional CSD observation noise scale (HalfCauchy) accounts
      for model-data mismatch.

    Examples
    --------
    >>> import torch
    >>> from pyro_dcm.models import spectral_dcm_model, create_guide, run_svi
    >>> csd = torch.randn(32, 3, 3, dtype=torch.complex128)
    >>> freqs = torch.linspace(1/128, 0.25, 32, dtype=torch.float64)
    >>> a_mask = torch.ones(3, 3, dtype=torch.float64)
    >>> guide = create_guide(spectral_dcm_model, init_scale=0.01)
    >>> # result = run_svi(spectral_dcm_model, guide,
    >>> #     model_args=(csd, freqs, a_mask))
    """
    # --- Infer dimensions ---
    if N is None:
        N = a_mask.shape[0]
    F = freqs.shape[0]  # noqa: N806

    # Prior standard deviation: SPM12 convention
    prior_std = (1.0 / 64.0) ** 0.5

    # --- Sample A_free ---
    A_free = pyro.sample(
        "A_free",
        dist.Normal(
            torch.zeros(N, N, dtype=torch.float64),
            prior_std * torch.ones(N, N, dtype=torch.float64),
        ).to_event(2),
    )

    # Apply structural mask: zero absent connections
    A_free = A_free * a_mask

    # Deterministic transform: parameterize_A ensures negative diagonal
    A = pyro.deterministic("A", parameterize_A(A_free))

    # --- Sample noise parameters (SPM12 priors) ---
    # Neuronal noise: (2, N) - [log amplitude, log exponent] per region
    noise_a = pyro.sample(
        "noise_a",
        dist.Normal(
            torch.zeros(2, N, dtype=torch.float64),
            prior_std * torch.ones(2, N, dtype=torch.float64),
        ).to_event(2),
    )

    # Global observation noise: (2, 1)
    noise_b = pyro.sample(
        "noise_b",
        dist.Normal(
            torch.zeros(2, 1, dtype=torch.float64),
            prior_std * torch.ones(2, 1, dtype=torch.float64),
        ).to_event(2),
    )

    # Regional observation noise: (2, N)
    noise_c = pyro.sample(
        "noise_c",
        dist.Normal(
            torch.zeros(2, N, dtype=torch.float64),
            prior_std * torch.ones(2, N, dtype=torch.float64),
        ).to_event(2),
    )

    # --- Forward model ---
    # Compute predicted CSD via spectral DCM pipeline
    predicted_csd_complex = spectral_dcm_forward(
        A, freqs, noise_a, noise_b, noise_c
    )

    # Store complex predicted CSD as deterministic for analysis
    pyro.deterministic("predicted_csd", predicted_csd_complex)

    # --- Decompose complex CSD to real vector ---
    pred_real = decompose_csd_for_likelihood(predicted_csd_complex)
    obs_real = decompose_csd_for_likelihood(observed_csd)

    # --- CSD observation noise ---
    # HalfCauchy prior accounts for model-data mismatch
    csd_noise_scale = pyro.sample(
        "csd_noise_scale",
        dist.HalfCauchy(torch.tensor(1.0, dtype=torch.float64)),
    )

    # --- Likelihood ---
    # Gaussian on the stacked real/imaginary vector
    pyro.sample(
        "obs_csd",
        dist.Normal(pred_real, csd_noise_scale).to_event(1),
        obs=obs_real,
    )
