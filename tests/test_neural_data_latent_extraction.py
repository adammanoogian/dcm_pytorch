"""Unit tests for latent extraction and CSD computation utilities.

Tests cover shape contracts for extract_latent_trajectories (Tensor and
DataLoader inputs), eval-mode enforcement, CSD computation from 2D and
3D inputs (averaged and per-sample), Hermitian symmetry, and dtype
correctness of prepare_for_spectral_dcm output.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from pyro_dcm.neural_data_models.latent_extraction import (
    compute_latent_csd,
    extract_latent_trajectories,
    prepare_for_spectral_dcm,
)
from pyro_dcm.neural_data_models.lstm_autoencoder import MEGAutoencoder

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_ROI = 6
N_LATENT = 12
T = 100
N_SAMPLES = 10
SFREQ = 250.0


@pytest.fixture()
def trained_model() -> MEGAutoencoder:
    """Return a freshly initialised MEGAutoencoder (no training needed)."""
    torch.manual_seed(0)
    return MEGAutoencoder(n_roi=N_ROI, n_latent=N_LATENT, hidden_size=32)


@pytest.fixture()
def sample_data() -> torch.Tensor:
    """Synthetic timeseries of shape (N_SAMPLES, T, N_ROI)."""
    torch.manual_seed(1)
    return torch.randn(N_SAMPLES, T, N_ROI)


# ---------------------------------------------------------------------------
# extract_latent_trajectories tests
# ---------------------------------------------------------------------------


class TestExtractLatentTrajectories:
    """Shape and behaviour tests for extract_latent_trajectories."""

    def test_extract_latent_trajectories_shape(
        self, trained_model: MEGAutoencoder, sample_data: torch.Tensor
    ) -> None:
        """Tensor input (10, 100, 6) with n_latent=12 -> (10, 100, 12)."""
        latents = extract_latent_trajectories(trained_model, sample_data)
        assert isinstance(latents, np.ndarray)
        assert latents.shape == (N_SAMPLES, T, N_LATENT)

    def test_extract_latent_trajectories_from_dataloader(
        self, trained_model: MEGAutoencoder, sample_data: torch.Tensor
    ) -> None:
        """DataLoader input produces the same shape as Tensor input."""
        dataset = TensorDataset(sample_data)
        loader = DataLoader(dataset, batch_size=4)
        latents = extract_latent_trajectories(trained_model, loader)
        assert latents.shape == (N_SAMPLES, T, N_LATENT)

    def test_extract_latent_trajectories_eval_mode(
        self, trained_model: MEGAutoencoder, sample_data: torch.Tensor
    ) -> None:
        """Model is in eval mode during extraction."""
        # Start in training mode
        trained_model.train()
        assert trained_model.training is True

        extract_latent_trajectories(trained_model, sample_data)

        # After extraction, model should still be in eval mode
        assert trained_model.training is False


# ---------------------------------------------------------------------------
# compute_latent_csd tests
# ---------------------------------------------------------------------------


class TestComputeLatentCSD:
    """Shape, averaging, and symmetry tests for compute_latent_csd."""

    def test_compute_latent_csd_2d_input(self) -> None:
        """Single trajectory (T, N) -> CSD shape (F, N, N)."""
        np.random.seed(42)
        trajectory = np.random.randn(T, N_LATENT)
        result = compute_latent_csd(
            trajectory, sfreq=SFREQ, n_freqs=32
        )
        assert result["csd"].shape == (32, N_LATENT, N_LATENT)
        assert result["freqs"].shape == (32,)
        assert result["n_latent"] == N_LATENT

    def test_compute_latent_csd_3d_averaged(self) -> None:
        """3D input with average=True -> CSD shape (F, N, N)."""
        np.random.seed(43)
        trajectories = np.random.randn(5, T, N_LATENT)
        result = compute_latent_csd(
            trajectories,
            sfreq=SFREQ,
            n_freqs=32,
            average_over_samples=True,
        )
        assert result["csd"].shape == (32, N_LATENT, N_LATENT)
        assert result["csd"].dtype == np.complex128

    def test_compute_latent_csd_3d_per_sample(self) -> None:
        """3D input with average=False -> CSD shape (n, F, N, N)."""
        np.random.seed(44)
        n = 5
        trajectories = np.random.randn(n, T, N_LATENT)
        result = compute_latent_csd(
            trajectories,
            sfreq=SFREQ,
            n_freqs=32,
            average_over_samples=False,
        )
        assert result["csd"].shape == (n, 32, N_LATENT, N_LATENT)
        assert result["csd"].dtype == np.complex128

    def test_compute_latent_csd_hermitian(self) -> None:
        """CSD[f, i, j] == conj(CSD[f, j, i]) (Hermitian symmetry)."""
        np.random.seed(45)
        trajectory = np.random.randn(T, N_LATENT)
        result = compute_latent_csd(
            trajectory, sfreq=SFREQ, n_freqs=16
        )
        csd = result["csd"]
        # Check Hermitian: csd[f] == csd[f].conj().T
        for f_idx in range(csd.shape[0]):
            np.testing.assert_allclose(
                csd[f_idx],
                csd[f_idx].conj().T,
                atol=1e-12,
                err_msg=f"CSD not Hermitian at frequency index {f_idx}",
            )


# ---------------------------------------------------------------------------
# prepare_for_spectral_dcm tests
# ---------------------------------------------------------------------------


class TestPrepareForSpectralDCM:
    """Dtype and shape tests for prepare_for_spectral_dcm."""

    def test_prepare_for_spectral_dcm_dtypes(self) -> None:
        """Output tensors have correct dtypes: complex128, float64."""
        np.random.seed(46)
        trajectory = np.random.randn(T, N_LATENT)
        csd_result = compute_latent_csd(
            trajectory, sfreq=SFREQ, n_freqs=16
        )
        dcm_input = prepare_for_spectral_dcm(csd_result)

        assert dcm_input["csd"].dtype == torch.complex128
        assert dcm_input["freqs"].dtype == torch.float64
        assert dcm_input["a_mask"].dtype == torch.float64
        assert dcm_input["a_mask"].shape == (N_LATENT, N_LATENT)
        assert dcm_input["csd"].shape == (16, N_LATENT, N_LATENT)
        assert dcm_input["freqs"].shape == (16,)
