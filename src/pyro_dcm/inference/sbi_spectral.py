"""Simulation-based inference (SBI) for spectral DCM.

Provides a simulator, prior, and NPE training wrapper for amortized
posterior inference over effective connectivity (A matrix) parameters
in spectral DCM. Uses the ``sbi`` library for neural posterior
estimation.

The simulator wraps the spectral DCM forward model, mapping free A
parameters to flattened cross-spectral density (real + imaginary parts).
The prior follows SPM12's N(0, 1/64) convention.

References
----------
Cranmer, Brehmer & Louppe (2020). The frontier of simulation-based
inference. PNAS 117(48), 30055-30062.

Tejero-Cantero et al. (2020). sbi: A toolkit for simulation-based
inference. JOSS 5(52), 2505.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from pyro_dcm.forward_models.neural_state import parameterize_A
from pyro_dcm.forward_models.spectral_transfer import spectral_dcm_forward


def _count_free_params(a_mask: torch.Tensor) -> int:
    """Count number of free A parameters from binary mask.

    Parameters
    ----------
    a_mask : torch.Tensor
        Binary structural mask, shape ``(N, N)``.

    Returns
    -------
    int
        Number of nonzero entries in ``a_mask``.
    """
    return int(a_mask.sum().item())


def _theta_to_A_free(
    theta: torch.Tensor,
    a_mask: torch.Tensor,
) -> torch.Tensor:
    """Unpack flat theta vector into masked A_free matrix.

    Parameters
    ----------
    theta : torch.Tensor
        Free A parameters, shape ``(n_free,)``.
    a_mask : torch.Tensor
        Binary mask, shape ``(N, N)``.

    Returns
    -------
    torch.Tensor
        A_free matrix, shape ``(N, N)``, with theta values placed
        at nonzero mask positions and zeros elsewhere.
    """
    N = a_mask.shape[0]
    A_free = torch.zeros(N, N, dtype=theta.dtype, device=theta.device)
    A_free[a_mask.bool()] = theta
    return A_free


def make_spectral_dcm_simulator(
    n_regions: int,
    freqs: torch.Tensor,
    a_mask: torch.Tensor,
    noise_params: dict[str, torch.Tensor] | None = None,
    eig_clamp: float | None = -1.0 / 32.0,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create a simulator mapping A_free parameters to flattened CSD.

    The returned callable takes a 1D float64 tensor of free A parameters
    (entries where ``a_mask`` is nonzero) and returns a 1D float64 tensor
    containing the real and imaginary parts of the predicted CSD,
    concatenated.

    Parameters
    ----------
    n_regions : int
        Number of brain regions (N).
    freqs : torch.Tensor
        Frequency grid in Hz, shape ``(F,)``, float64.
    a_mask : torch.Tensor
        Binary structural mask for A connections, shape ``(N, N)``.
    noise_params : dict or None
        Optional noise parameters with keys ``'a'``, ``'b'``, ``'c'``
        mapping to tensors of shapes ``(2, N)``, ``(2, 1)``, ``(2, N)``.
        If None, uses zeros (default spectral noise).
    eig_clamp : float or None
        Maximum real eigenvalue for stability clamping. Default
        ``-1/32`` (SPM12 fMRI convention).

    Returns
    -------
    Callable[[torch.Tensor], torch.Tensor]
        Simulator function: ``theta (n_free,) -> x (2*F*N*N,)``.

    Examples
    --------
    >>> a_mask = torch.ones(3, 3, dtype=torch.float64)
    >>> freqs = default_frequency_grid(TR=2.0, n_freqs=16)
    >>> sim = make_spectral_dcm_simulator(3, freqs, a_mask)
    >>> theta = torch.zeros(9, dtype=torch.float64)
    >>> x = sim(theta)
    >>> x.shape  # (2 * 16 * 3 * 3,) = (288,)
    """
    N = n_regions
    F = freqs.shape[0]
    output_dim = 2 * F * N * N

    # Default noise: zeros
    if noise_params is None:
        a_noise = torch.zeros(2, N, dtype=torch.float64)
        b_noise = torch.zeros(2, 1, dtype=torch.float64)
        c_noise = torch.zeros(2, N, dtype=torch.float64)
    else:
        a_noise = noise_params["a"]
        b_noise = noise_params["b"]
        c_noise = noise_params["c"]

    a_mask_f64 = a_mask.to(torch.float64)

    def simulator(theta: torch.Tensor) -> torch.Tensor:
        """Map free A parameters to flattened CSD vector.

        Parameters
        ----------
        theta : torch.Tensor
            Free A parameters, shape ``(n_free,)``, float64.

        Returns
        -------
        torch.Tensor
            Flattened CSD (real + imag), shape ``(2*F*N*N,)``, float64.
        """
        A_free = _theta_to_A_free(theta.to(torch.float64), a_mask_f64)
        A = parameterize_A(A_free)

        # Check stability: all eigenvalues should have negative real part
        eigvals = torch.linalg.eigvals(A)
        if torch.any(eigvals.real > 0):
            # Unstable system -- return zeros as fallback
            return torch.zeros(output_dim, dtype=torch.float64)

        try:
            csd = spectral_dcm_forward(
                A, freqs, a_noise, b_noise, c_noise, eig_clamp=eig_clamp
            )
        except RuntimeError:
            return torch.zeros(output_dim, dtype=torch.float64)

        # Check for NaN
        if torch.any(torch.isnan(csd.real)) or torch.any(torch.isnan(csd.imag)):
            return torch.zeros(output_dim, dtype=torch.float64)

        # Flatten: real parts then imaginary parts
        return torch.cat([
            csd.real.reshape(-1),
            csd.imag.reshape(-1),
        ]).to(torch.float64)

    return simulator


def make_spectral_dcm_prior(
    n_regions: int,
    a_mask: torch.Tensor,
    prior_variance: float = 1.0 / 64.0,
) -> torch.distributions.Distribution:
    """Create prior distribution over free A parameters.

    Returns an independent normal prior matching SPM12's N(0, 1/64)
    convention for effective connectivity parameters.

    Parameters
    ----------
    n_regions : int
        Number of brain regions (unused directly, mask determines
        parameter count).
    a_mask : torch.Tensor
        Binary structural mask for A connections, shape ``(N, N)``.
    prior_variance : float
        Prior variance for each free parameter. Default ``1/64``
        matches SPM12 convention.

    Returns
    -------
    torch.distributions.Distribution
        Independent normal prior with ``event_shape = (n_free,)``.

    Examples
    --------
    >>> a_mask = torch.ones(3, 3, dtype=torch.float64)
    >>> prior = make_spectral_dcm_prior(3, a_mask)
    >>> prior.event_shape  # (9,)
    >>> sample = prior.sample()
    >>> sample.shape  # (9,)
    """
    n_free = _count_free_params(a_mask)
    prior_std = prior_variance**0.5
    base = torch.distributions.Normal(
        torch.zeros(n_free, dtype=torch.float64),
        torch.full((n_free,), prior_std, dtype=torch.float64),
    )
    return torch.distributions.Independent(base, 1)


def train_npe(
    simulator: Callable[[torch.Tensor], torch.Tensor],
    prior: torch.distributions.Distribution,
    n_simulations: int = 10_000,
    embedding_net: Any = None,
    n_rounds: int = 1,
    device: str = "cpu",
) -> Any:
    """Train Neural Posterior Estimation (NPE) on spectral DCM simulator.

    Wraps the ``sbi`` library's SNPE-C (APT) implementation. Simulates
    training data from the prior, trains a conditional density estimator,
    and returns the trained posterior object.

    Parameters
    ----------
    simulator : Callable
        Simulator function: ``theta -> x``. Should return 1D float64
        tensors.
    prior : torch.distributions.Distribution
        Prior over parameters. Must have ``sample()`` and ``log_prob()``.
    n_simulations : int
        Number of simulations for training. Default 10000.
    embedding_net : nn.Module or None
        Optional embedding network for compressing observations.
        Passed to the neural density estimator.
    n_rounds : int
        Number of sequential NPE rounds. Default 1 (amortized).
    device : str
        Device for training. Default ``"cpu"``.

    Returns
    -------
    Any
        Trained sbi posterior object (``DirectPosterior`` or similar).

    Raises
    ------
    ImportError
        If the ``sbi`` package is not installed.

    Notes
    -----
    The ``sbi`` package is an optional dependency. Install with:
    ``pip install sbi>=0.22``
    """
    try:
        from sbi.inference import SNPE, simulate_for_sbi
        from sbi.utils import BoxUniform  # noqa: F401 (availability check)
    except ImportError as e:
        msg = (
            "sbi package required for train_npe. "
            "Install with: pip install 'sbi>=0.22'"
        )
        raise ImportError(msg) from e

    # Build density estimator with optional embedding net
    density_kwargs: dict[str, Any] = {}
    if embedding_net is not None:
        density_kwargs["embedding_net"] = embedding_net

    inference = SNPE(prior=prior, device=device, **density_kwargs)

    for _round in range(n_rounds):
        theta, x = simulate_for_sbi(
            simulator, prior, num_simulations=n_simulations
        )
        inference.append_simulations(theta, x)
        _ = inference.train()

    posterior = inference.build_posterior()
    return posterior


def build_sbi_posterior(
    trained_npe: Any,
    observation: torch.Tensor,
) -> Any:
    """Condition trained NPE on an observation.

    Parameters
    ----------
    trained_npe : Any
        Trained sbi posterior object from ``train_npe``.
    observation : torch.Tensor
        Observed CSD as flattened real vector (same format as simulator
        output), shape ``(2*F*N*N,)``.

    Returns
    -------
    Any
        Conditioned posterior object supporting ``.sample()`` and
        ``.log_prob()``.
    """
    return trained_npe.set_default_x(observation.to(torch.float32))
