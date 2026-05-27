"""MEG-like timeseries simulator via Ornstein-Uhlenbeck SDE.

Generates multivariate MEG-like timeseries from a known effective connectivity
matrix A using the Ornstein-Uhlenbeck (OU) stochastic differential equation:

    dx = A x dt + sigma I dW

This is the time-domain realization of the linear stochastic process underlying
spectral DCM [REF-010]. The simulator produces ``(n_samples, T, N)`` timeseries
tensors suitable for training temporal autoencoders (e.g., LSTM-AE) with known
ground-truth connectivity for recovery validation.

Also provides ``make_sensorimotor_A`` for a structured 10-region sensorimotor
network (M1, S1, PMC, SMA, A1 -- bilateral) with realistic connectivity patterns.

References
----------
[REF-010] Friston, Kahan, Biswal & Razi (2014). A DCM for resting state
    fMRI. NeuroImage, 94, 396-407.
"""

from __future__ import annotations

import torch

from pyro_dcm.simulators.spectral_simulator import make_stable_A_spectral

SENSORIMOTOR_ROI_NAMES: list[str] = [
    "M1_lh",
    "M1_rh",
    "S1_lh",
    "S1_rh",
    "PMC_lh",
    "PMC_rh",
    "SMA_lh",
    "SMA_rh",
    "A1_lh",
    "A1_rh",
]
"""Region names for the 10-region sensorimotor network.

Regions: primary motor cortex (M1), primary somatosensory cortex (S1),
premotor cortex (PMC), supplementary motor area (SMA), primary auditory
cortex (A1). Each region has left (lh) and right (rh) hemispheres.
"""


def simulate_meg_timeseries(
    A: torch.Tensor,
    *,
    sfreq: float = 250.0,
    duration: float = 4.0,
    n_samples: int = 50,
    sigma: float = 1.0,
    seed: int | None = None,
) -> dict:
    """Generate multivariate MEG-like timeseries via OU process.

    Simulates ``n_samples`` independent realizations of the
    Ornstein-Uhlenbeck SDE using Euler-Maruyama integration:

        x[t+1] = x[t] + A @ x[t] * dt + sigma * sqrt(dt) * noise

    where ``dt = 1 / sfreq`` and ``noise ~ N(0, I)``.

    This is the time-domain realization of the linear stochastic model
    underlying spectral DCM [REF-010] (Friston et al. 2014).

    Parameters
    ----------
    A : torch.Tensor
        Effective connectivity matrix, shape ``(N, N)``, float64.
        Must be stable (all eigenvalues with negative real parts).
    sfreq : float, optional
        Sampling frequency in Hz. Default 250.0.
    duration : float, optional
        Duration of each trial in seconds. Default 4.0.
    n_samples : int, optional
        Number of independent realizations. Default 50.
    sigma : float, optional
        Noise standard deviation (driving noise amplitude). Default 1.0.
    seed : int or None, optional
        Random seed for reproducibility. Sets ``torch.manual_seed``
        if provided.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``'timeseries'``: torch.Tensor, shape ``(n_samples, T, N)``,
          float64. T = int(sfreq * duration).
        - ``'A'``: torch.Tensor, shape ``(N, N)``, float64.
          Ground-truth connectivity.
        - ``'sfreq'``: float. Sampling frequency.
        - ``'duration'``: float. Trial duration in seconds.
        - ``'roi_names'``: list[str] or None. Set if using
          ``make_sensorimotor_A``.
        - ``'sigma'``: float. Noise amplitude.

    Raises
    ------
    ValueError
        If A has any eigenvalue with non-negative real part.

    References
    ----------
    [REF-010] Friston et al. (2014), OU process as linear stochastic
        model for spectral DCM.

    Examples
    --------
    >>> A = make_sensorimotor_A(seed=42)
    >>> result = simulate_meg_timeseries(A, n_samples=5, duration=2.0)
    >>> result['timeseries'].shape  # torch.Size([5, 500, 10])
    """
    if seed is not None:
        torch.manual_seed(seed)

    N = A.shape[0]
    A = A.to(torch.float64)

    # Validate A stability
    eigvals = torch.linalg.eigvals(A.to(torch.complex128))
    max_real = eigvals.real.max().item()
    if max_real >= 0:
        raise ValueError(
            f"A matrix is unstable: max Re(lambda) = {max_real:.6f}, "
            f"expected < 0"
        )

    T = int(sfreq * duration)
    dt = 1.0 / sfreq
    sqrt_dt = dt**0.5

    # Euler-Maruyama integration of OU process
    # x[t+1] = x[t] + A @ x[t] * dt + sigma * sqrt(dt) * noise
    x = torch.zeros(n_samples, T, N, dtype=torch.float64)

    for t in range(T - 1):
        noise = torch.randn(n_samples, N, dtype=torch.float64)
        # x_t shape: (n_samples, N)
        x_t = x[:, t, :]
        # drift: A @ x_t^T -> (N, n_samples) -> transpose -> (n_samples, N)
        drift = (A @ x_t.T).T * dt
        diffusion = sigma * sqrt_dt * noise
        x[:, t + 1, :] = x_t + drift + diffusion

    return {
        "timeseries": x,
        "A": A.clone(),
        "sfreq": sfreq,
        "duration": duration,
        "roi_names": None,
        "sigma": sigma,
    }


def make_sensorimotor_A(
    *,
    self_connection: float = -0.5,
    intra_strength: float = 0.15,
    bilateral_strength: float = 0.08,
    feedforward_strength: float = 0.10,
    seed: int | None = None,
) -> torch.Tensor:
    """Create a 10-region sensorimotor effective connectivity matrix.

    Builds a structured 10x10 A matrix mimicking a sensorimotor network
    with the following regions (bilateral):

    - M1 (primary motor): indices 0, 1
    - S1 (primary somatosensory): indices 2, 3
    - PMC (premotor cortex): indices 4, 5
    - SMA (supplementary motor area): indices 6, 7
    - A1 (primary auditory cortex): indices 8, 9

    Connection patterns:

    - Diagonal: ``self_connection`` (negative self-inhibition)
    - M1 <-> S1 (ipsilateral): ``intra_strength`` (bidirectional)
    - PMC -> M1 (ipsilateral): ``feedforward_strength``
    - SMA -> M1 (ipsilateral): ``feedforward_strength``
    - SMA -> PMC (ipsilateral): ``feedforward_strength * 0.5``
    - A1 -> M1 (ipsilateral): ``feedforward_strength * 0.5``
    - Bilateral homotopic: ``bilateral_strength`` (bidirectional)
    - Other connections: small random noise (scale 0.02)

    Validates stability and rescales off-diagonal if needed (following
    the ``make_stable_A_spectral`` pattern).

    Parameters
    ----------
    self_connection : float, optional
        Self-connection strength (must be negative). Default -0.5.
    intra_strength : float, optional
        M1 <-> S1 ipsilateral coupling strength. Default 0.15.
    bilateral_strength : float, optional
        Homotopic bilateral coupling strength. Default 0.08.
    feedforward_strength : float, optional
        Feedforward connection strength (PMC/SMA -> M1). Default 0.10.
    seed : int or None, optional
        Random seed for reproducibility (controls random noise on
        non-structured connections).

    Returns
    -------
    torch.Tensor
        Effective connectivity matrix, shape ``(10, 10)``, dtype float64.
        Guaranteed stable (all eigenvalues with negative real parts).

    Raises
    ------
    ValueError
        If ``self_connection`` is not negative.

    Examples
    --------
    >>> A = make_sensorimotor_A(seed=42)
    >>> A.shape  # torch.Size([10, 10])
    >>> torch.linalg.eigvals(A).real.max() < 0  # True
    """
    if self_connection >= 0:
        raise ValueError(
            f"self_connection must be negative, got {self_connection}"
        )

    N = 10
    if seed is not None:
        torch.manual_seed(seed)

    A = torch.zeros(N, N, dtype=torch.float64)

    # Diagonal: self-connections
    A.diagonal().fill_(self_connection)

    # Region index mapping:
    # M1_lh=0, M1_rh=1, S1_lh=2, S1_rh=3
    # PMC_lh=4, PMC_rh=5, SMA_lh=6, SMA_rh=7
    # A1_lh=8, A1_rh=9

    # M1 <-> S1 ipsilateral (bidirectional)
    # Left: M1_lh(0) <-> S1_lh(2)
    A[0, 2] = intra_strength
    A[2, 0] = intra_strength
    # Right: M1_rh(1) <-> S1_rh(3)
    A[1, 3] = intra_strength
    A[3, 1] = intra_strength

    # PMC -> M1 ipsilateral (feedforward)
    # Left: PMC_lh(4) -> M1_lh(0)
    A[0, 4] = feedforward_strength
    # Right: PMC_rh(5) -> M1_rh(1)
    A[1, 5] = feedforward_strength

    # SMA -> M1 ipsilateral (feedforward)
    # Left: SMA_lh(6) -> M1_lh(0)
    A[0, 6] = feedforward_strength
    # Right: SMA_rh(7) -> M1_rh(1)
    A[1, 7] = feedforward_strength

    # SMA -> PMC ipsilateral (feedforward, weaker)
    # Left: SMA_lh(6) -> PMC_lh(4)
    A[4, 6] = feedforward_strength * 0.5
    # Right: SMA_rh(7) -> PMC_rh(5)
    A[5, 7] = feedforward_strength * 0.5

    # A1 -> M1 ipsilateral (auditory-motor, weaker)
    # Left: A1_lh(8) -> M1_lh(0)
    A[0, 8] = feedforward_strength * 0.5
    # Right: A1_rh(9) -> M1_rh(1)
    A[1, 9] = feedforward_strength * 0.5

    # Bilateral homotopic connections (bidirectional)
    # Pairs: (0,1), (2,3), (4,5), (6,7), (8,9)
    for lh, rh in [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]:
        A[lh, rh] = bilateral_strength
        A[rh, lh] = bilateral_strength

    # Small random noise on remaining zero off-diagonal entries
    noise = 0.02 * torch.randn(N, N, dtype=torch.float64)
    mask_zero_offdiag = (A == 0) & (~torch.eye(N, dtype=torch.bool))
    A[mask_zero_offdiag] = noise[mask_zero_offdiag]

    # Validate stability; rescale off-diagonal if needed
    eigvals = torch.linalg.eigvals(A.to(torch.complex128))
    max_real = eigvals.real.max().item()

    if max_real >= 0:
        # Progressively reduce off-diagonal magnitude
        diag_vals = A.diagonal().clone()
        offdiag_mask = ~torch.eye(N, dtype=torch.bool)
        offdiag_vals = A[offdiag_mask].clone()
        for scale in [0.8, 0.6, 0.4, 0.2, 0.1]:
            A_trial = torch.zeros(N, N, dtype=torch.float64)
            A_trial.diagonal().copy_(diag_vals)
            A_trial[offdiag_mask] = offdiag_vals * scale
            eigvals_trial = torch.linalg.eigvals(
                A_trial.to(torch.complex128)
            )
            if eigvals_trial.real.max().item() < 0:
                A = A_trial
                break

    return A


def generate_meg_dataset(
    A: torch.Tensor | None = None,
    *,
    n_roi: int = 10,
    sfreq: float = 250.0,
    duration: float = 4.0,
    n_train: int = 200,
    n_val: int = 50,
    sigma: float = 1.0,
    seed: int = 42,
) -> dict:
    """Generate train/val MEG-like datasets with ground-truth connectivity.

    Convenience function that creates a ground-truth A matrix (if not
    provided) and generates training and validation timeseries splits
    using ``simulate_meg_timeseries``.

    Parameters
    ----------
    A : torch.Tensor or None, optional
        Effective connectivity matrix, shape ``(N, N)``. If None,
        uses ``make_sensorimotor_A(seed=seed)`` for 10 regions, or
        ``make_stable_A_spectral(n_roi, seed=seed)`` otherwise.
    n_roi : int, optional
        Number of regions (only used when A is None). Default 10.
    sfreq : float, optional
        Sampling frequency in Hz. Default 250.0.
    duration : float, optional
        Trial duration in seconds. Default 4.0.
    n_train : int, optional
        Number of training samples. Default 200.
    n_val : int, optional
        Number of validation samples. Default 50.
    sigma : float, optional
        Noise amplitude. Default 1.0.
    seed : int, optional
        Random seed for A generation and data simulation. Default 42.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``'train'``: torch.Tensor, shape ``(n_train, T, N)``, float64.
        - ``'val'``: torch.Tensor, shape ``(n_val, T, N)``, float64.
        - ``'A'``: torch.Tensor, shape ``(N, N)``, float64.
          Ground-truth connectivity.
        - ``'roi_names'``: list[str] or None. Set for 10-region
          sensorimotor networks.
        - ``'sfreq'``: float. Sampling frequency.
        - ``'metadata'``: dict with all generation parameters.

    Examples
    --------
    >>> ds = generate_meg_dataset(n_train=100, n_val=25, seed=42)
    >>> ds['train'].shape  # torch.Size([100, 1000, 10])
    >>> ds['val'].shape    # torch.Size([25, 1000, 10])
    """
    roi_names = None
    if A is None:
        if n_roi == 10:
            A = make_sensorimotor_A(seed=seed)
            roi_names = list(SENSORIMOTOR_ROI_NAMES)
        else:
            A = make_stable_A_spectral(n_roi, seed=seed)

    # Generate training data (seed offset 0)
    train_result = simulate_meg_timeseries(
        A,
        sfreq=sfreq,
        duration=duration,
        n_samples=n_train,
        sigma=sigma,
        seed=seed + 1000,
    )

    # Generate validation data (different seed offset)
    val_result = simulate_meg_timeseries(
        A,
        sfreq=sfreq,
        duration=duration,
        n_samples=n_val,
        sigma=sigma,
        seed=seed + 2000,
    )

    # Propagate roi_names from train_result if set, else from our default
    if roi_names is None:
        roi_names = train_result.get("roi_names")

    return {
        "train": train_result["timeseries"],
        "val": val_result["timeseries"],
        "A": A,
        "roi_names": roi_names,
        "sfreq": sfreq,
        "metadata": {
            "n_train": n_train,
            "n_val": n_val,
            "duration": duration,
            "sigma": sigma,
            "seed": seed,
            "n_roi": A.shape[0],
        },
    }
