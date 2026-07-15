"""Latent dynamics extraction from trained LSTM autoencoders.

Provides utilities to extract latent trajectories from a trained
:class:`~pyro_dcm.neural_data_models.lstm_autoencoder.MEGAutoencoder`,
compute their cross-spectral density (CSD) at MEG frequencies, and
convert the result to the format required by spectral DCM fitting.

The pipeline is:
    raw timeseries -> LSTM-AE encoder -> latent trajectories
    latent trajectories -> empirical CSD -> spectral DCM input
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from pyro_dcm.forward_models.csd_computation import compute_empirical_csd
from pyro_dcm.neural_data_models.lstm_autoencoder import MEGAutoencoder


def extract_latent_trajectories(
    model: MEGAutoencoder,
    data: torch.Tensor | DataLoader,
    *,
    device: str = "cpu",
    batch_size: int = 64,
) -> np.ndarray:
    """Extract latent trajectories from a trained autoencoder.

    Passes data through the encoder of a trained
    :class:`MEGAutoencoder` and returns the latent representations
    as a numpy array.

    Parameters
    ----------
    model : MEGAutoencoder
        Trained autoencoder model.
    data : torch.Tensor or DataLoader
        Input timeseries. If a Tensor, shape ``(n_samples, T, N_roi)``;
        it is wrapped in a DataLoader with ``batch_size``. If a
        DataLoader, each batch must yield a tensor of shape
        ``(batch, T, N_roi)`` (or a tuple whose first element is such).
    device : str, optional
        Torch device for inference. Default ``"cpu"``.
    batch_size : int, optional
        Batch size when wrapping a Tensor in a DataLoader. Default 64.

    Returns
    -------
    np.ndarray, shape (n_samples, T, N_latent)
        Latent trajectories from the encoder, detached from the
        computation graph.

    Examples
    --------
    >>> model = MEGAutoencoder(n_roi=10, n_latent=20)
    >>> x = torch.randn(50, 1000, 10)
    >>> latents = extract_latent_trajectories(model, x)
    >>> latents.shape  # (50, 1000, 20)
    """
    if isinstance(data, torch.Tensor):
        dataset = TensorDataset(data)
        loader: DataLoader = DataLoader(dataset, batch_size=batch_size)
    else:
        loader = data

    model = model.to(device)
    model.eval()

    latent_chunks: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (tuple, list)):
                x = batch[0]
            else:
                x = batch
            x = x.to(device)
            latent = model.encode(x)
            latent_chunks.append(latent.cpu().numpy())

    return np.concatenate(latent_chunks, axis=0)


def compute_latent_csd(
    latent_trajectories: np.ndarray,
    *,
    sfreq: float = 250.0,
    fmin: float = 1.0,
    fmax: float = 45.0,
    n_freqs: int = 64,
    average_over_samples: bool = True,
) -> dict[str, np.ndarray]:
    """Compute cross-spectral density from latent trajectories.

    Uses :func:`~pyro_dcm.forward_models.csd_computation.compute_empirical_csd`
    (Welch periodogram) to estimate the CSD matrix at MEG frequencies
    from latent trajectories produced by the LSTM-AE encoder.

    Parameters
    ----------
    latent_trajectories : np.ndarray
        Latent dynamics, shape ``(n_samples, T, N_latent)`` or
        ``(T, N_latent)`` for a single trajectory.
    sfreq : float, optional
        Sampling frequency in Hz. Default 250.0.
    fmin : float, optional
        Minimum frequency in Hz. Default 1.0.
    fmax : float, optional
        Maximum frequency in Hz. Default 45.0.
    n_freqs : int, optional
        Number of frequency bins. Default 64.
    average_over_samples : bool, optional
        If ``True`` and input is 3D, average CSD across samples.
        If ``False``, return per-sample CSD. Default ``True``.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with keys:

        - ``'csd'``: np.ndarray, complex128. Shape ``(F, N, N)`` when
          averaged or 2D input, ``(n_samples, F, N, N)`` when
          ``average_over_samples=False``.
        - ``'freqs'``: np.ndarray, shape ``(F,)``, float64. Frequency
          grid in Hz.
        - ``'sfreq'``: float. Sampling frequency.
        - ``'n_latent'``: int. Number of latent dimensions.

    Examples
    --------
    >>> latents = np.random.randn(50, 1000, 20)
    >>> result = compute_latent_csd(latents, sfreq=250.0)
    >>> result['csd'].shape  # (64, 20, 20)
    """
    freqs = np.linspace(fmin, fmax, n_freqs)

    if latent_trajectories.ndim == 2:
        # Single trajectory: (T, N_latent)
        n_latent = latent_trajectories.shape[1]
        csd = compute_empirical_csd(latent_trajectories, fs=sfreq, freqs=freqs)
        return {
            "csd": csd,
            "freqs": freqs,
            "sfreq": sfreq,
            "n_latent": n_latent,
        }

    # 3D: (n_samples, T, N_latent)
    n_samples, _T, n_latent = latent_trajectories.shape

    if average_over_samples:
        # Compute CSD per sample and average
        csd_sum = np.zeros((n_freqs, n_latent, n_latent), dtype=np.complex128)
        for i in range(n_samples):
            csd_sum += compute_empirical_csd(
                latent_trajectories[i], fs=sfreq, freqs=freqs
            )
        csd = csd_sum / n_samples
        return {
            "csd": csd,
            "freqs": freqs,
            "sfreq": sfreq,
            "n_latent": n_latent,
        }

    # Per-sample CSD
    csd_all = np.zeros(
        (n_samples, n_freqs, n_latent, n_latent), dtype=np.complex128
    )
    for i in range(n_samples):
        csd_all[i] = compute_empirical_csd(
            latent_trajectories[i], fs=sfreq, freqs=freqs
        )
    return {
        "csd": csd_all,
        "freqs": freqs,
        "sfreq": sfreq,
        "n_latent": n_latent,
    }


def prepare_for_spectral_dcm(
    csd_result: dict[str, np.ndarray],
) -> dict[str, torch.Tensor]:
    """Convert CSD result to torch tensors for spectral DCM fitting.

    Takes the output of :func:`compute_latent_csd` and converts it to
    the tensor format expected by
    :func:`~pyro_dcm.models.spectral_dcm_model.spectral_dcm_model`.

    Parameters
    ----------
    csd_result : dict[str, np.ndarray]
        Output of :func:`compute_latent_csd` with keys ``'csd'``,
        ``'freqs'``, and ``'n_latent'``.

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary with keys:

        - ``'csd'``: torch.Tensor, shape ``(F, N, N)``, dtype
          complex128.
        - ``'freqs'``: torch.Tensor, shape ``(F,)``, dtype float64.
        - ``'a_mask'``: torch.Tensor, shape ``(N, N)``, dtype float64.
          All-ones mask (fully connected model).

    Examples
    --------
    >>> result = compute_latent_csd(latents, sfreq=250.0)
    >>> dcm_input = prepare_for_spectral_dcm(result)
    >>> dcm_input['csd'].dtype  # torch.complex128
    """
    csd_np = csd_result["csd"]
    freqs_np = csd_result["freqs"]
    n_latent = csd_result["n_latent"]

    csd_tensor = torch.as_tensor(csd_np, dtype=torch.complex128)
    freqs_tensor = torch.as_tensor(freqs_np, dtype=torch.float64)
    a_mask = torch.ones(n_latent, n_latent, dtype=torch.float64)

    return {
        "csd": csd_tensor,
        "freqs": freqs_tensor,
        "a_mask": a_mask,
    }
