"""Spectral DCM simulator for synthetic CSD generation.

Generates synthetic cross-spectral density (CSD) data from a given effective
connectivity matrix A and noise parameters, using the spectral DCM forward
model. This is the spectral analog of ``task_simulator.py`` -- while the task
simulator generates BOLD time series, this simulator generates frequency-domain
CSD directly from the spectral DCM generative model.

Required by SIM-02 and essential for Phase 5 parameter recovery testing.
The simulator wraps the spectral DCM forward model (``spectral_dcm_forward``),
transfer function (``compute_transfer_function``), and noise models into a
single function that produces complete synthetic spectral data with all
intermediate quantities.

References
----------
[REF-010] Friston, Kahan, Biswal & Razi (2014). A DCM for resting state
    fMRI. NeuroImage, 94, 396-407. Eq. 3-7.
"""

from __future__ import annotations

import math

import torch

from pyro_dcm.forward_models.spectral_noise import (
    default_noise_priors,
    neuronal_noise_csd,
    observation_noise_csd,
)
from pyro_dcm.forward_models.spectral_transfer import (
    compute_transfer_function,
    default_frequency_grid,
    spectral_dcm_forward,
)


def simulate_spectral_dcm(
    A: torch.Tensor,
    noise_params: dict[str, torch.Tensor] | None = None,
    TR: float = 2.0,
    n_freqs: int = 32,
    seed: int | None = None,
) -> dict:
    """Generate synthetic CSD from the spectral DCM model.

    Implements the full spectral DCM generative model from [REF-010]
    Eq. 3-7 (Friston et al. 2014). Given an effective connectivity
    matrix A and noise parameters, computes the predicted CSD along
    with all intermediate quantities (transfer function, neuronal and
    observation noise spectra).

    Satisfies SIM-02 requirement for spectral DCM synthetic data
    generation.

    Parameters
    ----------
    A : torch.Tensor
        Effective connectivity matrix, shape ``(N, N)``, float64.
        Must have all eigenvalues with negative real parts (stable
        system). Checked with assertion.
    noise_params : dict or None, optional
        Noise parameters. If None, uses default priors (all zeros
        in log-space, from ``default_noise_priors``). If provided,
        must contain keys:

        - ``'a'``: torch.Tensor, shape ``(2, 1)``, float64.
          Shared neuronal noise [log-amplitude, log-exponent].
        - ``'b'``: torch.Tensor, shape ``(2, 1)``, float64.
          Global observation noise [log-amplitude, log-exponent].
        - ``'c'``: torch.Tensor, shape ``(1, N)``, float64.
          Regional observation noise [log-amplitude].
    TR : float, optional
        Repetition time in seconds. Default 2.0.
    n_freqs : int, optional
        Number of frequency bins. Default 32.
    seed : int or None, optional
        Random seed for reproducibility. Sets ``torch.manual_seed``
        if provided (relevant only if noise sampling is added in
        future extensions).

    Returns
    -------
    dict
        Dictionary with the following keys:

        - ``'csd'``: Predicted CSD, shape ``(F, N, N)``, complex128.
        - ``'freqs'``: Frequency vector, shape ``(F,)``, float64.
        - ``'transfer_function'``: H(w), shape ``(F, N, N)``,
          complex128.
        - ``'neuronal_noise'``: Gu(w), shape ``(F, N, N)``,
          complex128.
        - ``'observation_noise'``: Gn(w), shape ``(F, N, N)``,
          complex128.
        - ``'params'``: dict with ``A``, ``noise_params``, ``TR``,
          ``n_freqs``.

    Raises
    ------
    AssertionError
        If any eigenvalue of A has non-negative real part.

    References
    ----------
    [REF-010] Friston et al. (2014), Eq. 3-7.

    See Also
    --------
    spectral_dcm_forward : Convenience wrapper for predicted CSD.
    make_stable_A_spectral : Generate stable A matrices for testing.

    Examples
    --------
    >>> A = make_stable_A_spectral(3, seed=42)
    >>> result = simulate_spectral_dcm(A, TR=2.0, n_freqs=32)
    >>> result['csd'].shape  # (32, 3, 3)
    """
    if seed is not None:
        torch.manual_seed(seed)

    N = A.shape[0]

    # Validate A stability: all eigenvalues must have negative real parts
    eigvals = torch.linalg.eigvals(A.to(torch.complex128))
    assert (
        eigvals.real.max() < 0
    ), f"A matrix is unstable: max Re(lambda) = {eigvals.real.max().item():.6f}"

    # Set up default noise parameters if not provided
    if noise_params is None:
        priors = default_noise_priors(N)
        a = priors["a_prior_mean"]
        b = priors["b_prior_mean"]
        c = priors["c_prior_mean"]
    else:
        a = noise_params["a"]
        b = noise_params["b"]
        c = noise_params["c"]

    # Generate frequency grid
    freqs = default_frequency_grid(TR, n_freqs)

    # Compute predicted CSD via the full pipeline
    csd = spectral_dcm_forward(A, freqs, a, b, c)

    # Also compute individual components for inspection
    C_in = torch.eye(N, dtype=torch.float64, device=A.device)
    C_out = torch.eye(N, dtype=torch.float64, device=A.device)
    H = compute_transfer_function(A, C_in, C_out, freqs)
    Gu = neuronal_noise_csd(freqs, a, n_regions=N)
    Gn = observation_noise_csd(freqs, b, c, N)


    return {
        "csd": csd,
        "freqs": freqs,
        "transfer_function": H,
        "neuronal_noise": Gu,
        "observation_noise": Gn,
        "params": {
            "A": A,
            "noise_params": {"a": a, "b": b, "c": c},
            "TR": TR,
            "n_freqs": n_freqs,
        },
    }


def sample_csd_realisation(
    csd: torch.Tensor,
    n_averages: int,
    seed: int | None = None,
) -> torch.Tensor:
    """Draw a finite-sample realisation of a cross-spectral density.

    ``simulate_spectral_dcm`` returns the ANALYTIC CSD and is deterministic, so
    varying its ``seed`` changes nothing. Real CSD estimates are averages over a
    finite number of segments, and ``spm_dcm_csd_Q`` assumes exactly that: the
    scaled difference between the sample density and the predicted density is
    asymptotically complex normal with ``h = 2m + 1`` degrees of freedom for
    ``m`` averages [REF-111].

    For ``m`` independent segments the smoothed periodogram satisfies
    ``m * Ghat(w) ~ CW_N(m, G(w))`` (complex Wishart). Equivalently, draw
    ``m`` vectors ``z_k ~ CN(0, G(w))`` and average their outer products:
    ``Ghat(w) = sum_k z_k z_k^H / m``. This is unbiased (``E[Ghat] = G``)
    and its variance falls as ``1/m``. Each frequency is drawn
    independently.

    Parameters
    ----------
    csd : torch.Tensor
        Analytic cross-spectral density, shape ``(F, N, N)``, complex128.
        Each frequency slice must be Hermitian positive semi-definite.
    n_averages : int
        Number of segment averages ``m``. Must be at least ``N`` for a
        full-rank estimate; larger values give a tighter realisation
        (variance falls as ``1/m``).
    seed : int or None, optional
        Seed for the draw. The same seed reproduces the same realisation.

    Returns
    -------
    torch.Tensor
        Sampled CSD, shape ``(F, N, N)``, complex128. Hermitian and positive
        semi-definite at every frequency.

    Raises
    ------
    ValueError
        If ``n_averages`` is below the number of regions, or if a frequency
        slice is not positive semi-definite to within numerical tolerance.

    References
    ----------
    [REF-111] Camba-Mendez, G., & Kapetanios, G. (2005). Estimating the Rank of
    the Spectral Density Matrix. Journal of Time Series Analysis, 26(1), 37-48.
    The asymptotic result ``spm_dcm_csd_Q.m`` is built on.

    See Also
    --------
    simulate_spectral_dcm : Produces the analytic CSD this draws around.
    pyro_dcm.inference.csd_precision.compute_csd_precision : Consumes a sample
        spectral density and assumes this sampling model.
    """
    n_freqs, n_regions, _ = csd.shape
    if n_averages < n_regions:
        msg = (
            f"n_averages must be at least the number of regions for a "
            f"full-rank estimate: got n_averages={n_averages}, "
            f"n_regions={n_regions}"
        )
        raise ValueError(msg)

    generator = torch.Generator(device=csd.device)
    if seed is not None:
        generator.manual_seed(seed)

    # Hermitian part: the analytic CSD is Hermitian in exact arithmetic, but
    # the MAR round-trip leaves O(1e-16) asymmetry. Symmetrising here keeps the
    # eigendecomposition real-valued; it does NOT paper over a real asymmetry,
    # which the PSD check below would still catch.
    hermitian = (csd + csd.conj().transpose(-2, -1)) / 2

    eigvals, eigvecs = torch.linalg.eigh(hermitian)
    scale = hermitian.abs().max()
    tol = -1e-10 * torch.clamp(scale, min=1.0)
    if bool((eigvals < tol).any()):
        worst = eigvals.min().item()
        msg = (
            f"csd is not positive semi-definite: smallest eigenvalue "
            f"{worst:.6e}, tolerance {tol.item():.6e} (scale {scale:.6e}). "
            f"A CSD must be PSD at every frequency."
        )
        raise ValueError(msg)

    # Factor L with L @ L^H = G, from the clamped eigendecomposition.
    root = eigvecs @ torch.diag_embed(
        torch.clamp(eigvals, min=0.0).to(csd.dtype).sqrt(),
    )

    # z ~ CN(0, G): standard complex normal has real and imaginary parts of
    # variance 1/2 each, so E[z z^H] = L L^H = G.
    real = torch.randn(
        n_freqs, n_regions, n_averages,
        generator=generator, dtype=torch.float64, device=csd.device,
    )
    imag = torch.randn(
        n_freqs, n_regions, n_averages,
        generator=generator, dtype=torch.float64, device=csd.device,
    )
    white = torch.complex(real, imag) / math.sqrt(2.0)

    z = root @ white
    return z @ z.conj().transpose(-2, -1) / n_averages


def make_stable_A_spectral(
    n_regions: int,
    connection_strength: float = 0.1,
    self_connection: float = -0.5,
    seed: int | None = None,
) -> torch.Tensor:
    """Generate a stable A matrix for spectral DCM simulation.

    Creates an effective connectivity matrix with guaranteed negative
    eigenvalues, suitable for spectral DCM simulation. The diagonal
    contains self-connections (negative) and off-diagonal entries are
    random Gaussian-scaled by ``connection_strength``.

    If the initial random matrix has any eigenvalue with non-negative
    real part, the off-diagonal entries are rescaled to ensure stability.

    Parameters
    ----------
    n_regions : int
        Number of brain regions.
    connection_strength : float, optional
        Scale of off-diagonal (inter-region) connections. Default 0.1.
    self_connection : float, optional
        Self-connection strength (must be negative). Default -0.5.
    seed : int or None, optional
        Random seed for reproducibility.

    Returns
    -------
    torch.Tensor
        Effective connectivity matrix, shape ``(n_regions, n_regions)``,
        dtype float64. Guaranteed to have all eigenvalues with negative
        real parts.

    Raises
    ------
    ValueError
        If ``self_connection`` is not negative.

    Examples
    --------
    >>> A = make_stable_A_spectral(3, seed=42)
    >>> A.shape  # (3, 3)
    >>> torch.linalg.eigvals(A).real.max() < 0  # True
    """
    if self_connection >= 0:
        raise ValueError(
            f"self_connection must be negative, got {self_connection}"
        )

    if seed is not None:
        torch.manual_seed(seed)

    N = n_regions
    A = torch.zeros(N, N, dtype=torch.float64)

    # Set diagonal to self-connection
    A.diagonal().fill_(self_connection)

    # Set off-diagonal to random Gaussian * connection_strength
    off_diag = connection_strength * torch.randn(N, N, dtype=torch.float64)
    mask = ~torch.eye(N, dtype=torch.bool)
    A[mask] = off_diag[mask]

    # Verify eigenvalue stability and rescale if needed
    eigvals = torch.linalg.eigvals(A.to(torch.complex128))
    max_real = eigvals.real.max().item()

    if max_real >= 0:
        # Rescale off-diagonal to ensure stability
        # Strategy: reduce off-diagonal magnitude until stable
        for scale in [0.5, 0.25, 0.1, 0.05, 0.01]:
            A_trial = torch.zeros(N, N, dtype=torch.float64)
            A_trial.diagonal().fill_(self_connection)
            A_trial[mask] = off_diag[mask] * scale
            eigvals_trial = torch.linalg.eigvals(
                A_trial.to(torch.complex128)
            )
            if eigvals_trial.real.max().item() < 0:
                A = A_trial
                break

    return A
