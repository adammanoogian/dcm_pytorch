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

        - ``'a'``: torch.Tensor, shape ``(2, N)``, float64.
          Neuronal noise [log-amplitude, log-exponent] per region.
        - ``'b'``: torch.Tensor, shape ``(2, 1)``, float64.
          Global observation noise [log-amplitude, log-exponent].
        - ``'c'``: torch.Tensor, shape ``(2, N)``, float64.
          Regional observation noise [log-amplitude, log-exponent].
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
    Gu = neuronal_noise_csd(freqs, a)
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
