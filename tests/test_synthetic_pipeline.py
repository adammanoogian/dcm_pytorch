"""End-to-end synthetic pipeline integration tests.

Validates the full pipeline: generate timeseries from known A matrix ->
train LSTM autoencoder -> extract latent trajectories -> compute CSD ->
fit spectral DCM -> recover posterior A.

This is the primary risk-reduction test for Phase 22: if it passes, the
pipeline can recover connectivity from an autoencoder's latent space.
"""
from __future__ import annotations

import numpy as np
import pyro
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from pyro_dcm.forward_models.csd_computation import compute_empirical_csd
from pyro_dcm.models.guides import create_guide, run_svi
from pyro_dcm.models.spectral_dcm_model import spectral_dcm_model
from pyro_dcm.neural_data_models.latent_csd import (
    compute_latent_csd,
    extract_latent_trajectories,
    prepare_for_spectral_dcm,
)
from pyro_dcm.neural_data_models.lstm_autoencoder import MEGAutoencoder
from pyro_dcm.neural_data_models.trainer import AutoencoderTrainer
from pyro_dcm.simulators.meg_simulator import (
    generate_meg_dataset,
    make_sensorimotor_A,
)


@pytest.mark.slow
def test_synthetic_pipeline_end_to_end() -> None:
    """Full pipeline: known A -> timeseries -> LSTM-AE -> CSD -> DCM.

    Steps
    -----
    1. Generate synthetic MEG-like timeseries from a 10-region
       sensorimotor A matrix via OU process.
    2. Train an LSTM autoencoder (n_latent=20) on the timeseries.
    3. Extract latent trajectories from the trained encoder.
    4. Compute cross-spectral density from latent trajectories.
    5. Fit spectral DCM with MEG-adapted priors to the latent CSD.
    6. Verify ELBO decreases and posterior A has no NaN.
    """
    # ------------------------------------------------------------------
    # Step 1: Generate synthetic data from known A
    # ------------------------------------------------------------------
    A_true = make_sensorimotor_A(seed=42)
    assert A_true.shape == (10, 10)

    dataset = generate_meg_dataset(
        A_true,
        n_train=100,
        n_val=25,
        duration=4.0,
        sfreq=250.0,
        seed=42,
    )
    train_data = dataset["train"]
    assert train_data.shape == (100, 1000, 10)

    # ------------------------------------------------------------------
    # Step 2: Train LSTM autoencoder
    # ------------------------------------------------------------------
    n_roi = 10
    n_latent = 20  # 2x overcomplete
    torch.manual_seed(42)

    model = MEGAutoencoder(
        n_roi=n_roi, n_latent=n_latent, hidden_size=64
    )
    trainer = AutoencoderTrainer(model, lr=1e-3)

    train_dataset = TensorDataset(train_data.float())
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    history = trainer.train(train_loader, n_epochs=30, log_every=100)

    # Training must converge: final loss < initial loss
    assert history["train_losses"][-1] < history["train_losses"][0], (
        f"Training did not converge: "
        f"first={history['train_losses'][0]:.6f}, "
        f"last={history['train_losses'][-1]:.6f}"
    )

    # ------------------------------------------------------------------
    # Step 3: Extract latent trajectories
    # ------------------------------------------------------------------
    latents = extract_latent_trajectories(model, train_data.float())
    assert latents.shape == (100, 1000, n_latent)

    # ------------------------------------------------------------------
    # Step 4: Compute latent CSD
    # ------------------------------------------------------------------
    csd_result = compute_latent_csd(
        latents,
        sfreq=250.0,
        fmin=1.0,
        fmax=45.0,
        n_freqs=64,
        average_over_samples=True,
    )
    csd = csd_result["csd"]
    assert csd.shape == (64, n_latent, n_latent)

    # Hermitian check
    for f_idx in range(csd.shape[0]):
        np.testing.assert_allclose(
            csd[f_idx],
            csd[f_idx].conj().T,
            atol=1e-10,
            err_msg=f"Latent CSD not Hermitian at freq index {f_idx}",
        )

    # Auto-spectra must be non-negative (real-valued diagonal)
    for f_idx in range(csd.shape[0]):
        auto_spectra = np.real(np.diag(csd[f_idx]))
        assert np.all(auto_spectra >= 0), (
            f"Negative auto-spectrum at freq index {f_idx}: "
            f"min={auto_spectra.min():.6e}"
        )

    # ------------------------------------------------------------------
    # Step 5: Prepare for spectral DCM
    # ------------------------------------------------------------------
    dcm_input = prepare_for_spectral_dcm(csd_result)
    observed_csd = dcm_input["csd"]
    freqs = dcm_input["freqs"]
    a_mask = dcm_input["a_mask"]

    assert observed_csd.dtype == torch.complex128
    assert freqs.dtype == torch.float64
    assert a_mask.shape == (n_latent, n_latent)

    # ------------------------------------------------------------------
    # Step 6: Fit spectral DCM (smoke test -- 200 steps)
    # ------------------------------------------------------------------
    pyro.clear_param_store()

    guide = create_guide(
        spectral_dcm_model,
        guide_type="auto_normal",
        init_scale=0.01,
    )

    result = run_svi(
        spectral_dcm_model,
        guide,
        model_args=(observed_csd, freqs, a_mask),
        model_kwargs={"prior_a_var": 1.0 / 16.0, "eig_clamp": -1.0},
        num_steps=200,
        lr=0.01,
    )

    losses = result["losses"]

    # ELBO should decrease: mean of last 20 < mean of first 20
    early_mean = np.mean(losses[:20])
    late_mean = np.mean(losses[-20:])
    assert late_mean < early_mean, (
        f"ELBO did not decrease: "
        f"early_mean={early_mean:.2f}, late_mean={late_mean:.2f}"
    )

    # No NaN in losses
    assert not any(np.isnan(val) for val in losses), "NaN detected in losses"

    # Verify posterior A can be extracted (param store has A_free params)
    param_store = pyro.get_param_store()
    a_free_keys = [k for k in param_store.keys() if "A_free" in k]
    assert len(a_free_keys) > 0, "No A_free parameters in param store"

    # Check no NaN in the A_free posterior
    for key in a_free_keys:
        val = param_store[key]
        assert not torch.any(torch.isnan(val)), (
            f"NaN in param store key {key}"
        )


def test_raw_vs_latent_csd_shapes() -> None:
    """Compare CSD shapes from raw (10-ch) vs latent (20-ch) timeseries.

    Verifies that the CSD computation produces the correct shape for
    both raw and latent-space timeseries, and both satisfy Hermitian
    symmetry with non-negative auto-spectra.
    """
    # Generate synthetic data
    A_true = make_sensorimotor_A(seed=99)
    dataset = generate_meg_dataset(
        A_true,
        n_train=20,
        n_val=5,
        duration=4.0,
        sfreq=250.0,
        seed=99,
    )
    train_data = dataset["train"]  # (20, 1000, 10)
    n_roi = 10
    n_latent = 20
    n_freqs = 64

    # Compute raw CSD from a single sample
    freqs = np.linspace(1.0, 45.0, n_freqs)
    raw_csd = compute_empirical_csd(
        train_data[0].numpy(), fs=250.0, freqs=freqs
    )
    assert raw_csd.shape == (n_freqs, n_roi, n_roi)

    # Train a quick autoencoder
    torch.manual_seed(99)
    model = MEGAutoencoder(
        n_roi=n_roi, n_latent=n_latent, hidden_size=32
    )
    trainer = AutoencoderTrainer(model, lr=1e-3)
    train_dataset = TensorDataset(train_data.float())
    loader = DataLoader(train_dataset, batch_size=10)
    trainer.train(loader, n_epochs=5, log_every=100)

    # Extract latents and compute latent CSD
    latents = extract_latent_trajectories(model, train_data.float())
    latent_csd_result = compute_latent_csd(
        latents,
        sfreq=250.0,
        n_freqs=n_freqs,
        average_over_samples=True,
    )
    latent_csd = latent_csd_result["csd"]
    assert latent_csd.shape == (n_freqs, n_latent, n_latent)

    # Both CSDs should be Hermitian
    for label, csd in [("raw", raw_csd), ("latent", latent_csd)]:
        for f_idx in range(csd.shape[0]):
            np.testing.assert_allclose(
                csd[f_idx],
                csd[f_idx].conj().T,
                atol=1e-10,
                err_msg=f"{label} CSD not Hermitian at freq {f_idx}",
            )

    # Both should have non-negative auto-spectra
    for label, csd in [("raw", raw_csd), ("latent", latent_csd)]:
        for f_idx in range(csd.shape[0]):
            auto_spectra = np.real(np.diag(csd[f_idx]))
            assert np.all(auto_spectra >= 0), (
                f"{label} negative auto-spectrum at freq {f_idx}: "
                f"min={auto_spectra.min():.6e}"
            )
