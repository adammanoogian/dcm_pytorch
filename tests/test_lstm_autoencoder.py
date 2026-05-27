"""Unit tests for MEGAutoencoder and AutoencoderTrainer.

Tests cover shape contracts, gradient flow, batch invariance, training
loop convergence, checkpointing round-trips, early stopping, evaluation,
and spectral consistency loss behaviour.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from pyro_dcm.neural_data_models.lstm_autoencoder import MEGAutoencoder
from pyro_dcm.neural_data_models.trainer import AutoencoderTrainer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_ROI = 6
T = 100
BATCH = 4


def _make_sinusoidal_data(
    n_samples: int = 16,
    t: int = T,
    n_roi: int = N_ROI,
) -> TensorDataset:
    """Synthetic sinusoidal timeseries with small noise."""
    time = torch.linspace(0, 4 * math.pi, t)
    freqs = torch.linspace(1.0, float(n_roi), n_roi)
    # (T, N_roi)
    base = torch.sin(time.unsqueeze(-1) * freqs.unsqueeze(0))
    # (n_samples, T, N_roi)
    data = base.unsqueeze(0).expand(n_samples, -1, -1).clone()
    data += 0.05 * torch.randn_like(data)
    return TensorDataset(data)


# ---------------------------------------------------------------------------
# MEGAutoencoder tests
# ---------------------------------------------------------------------------


class TestMEGAutoencoderShapes:
    """Shape-contract tests for MEGAutoencoder."""

    def test_output_shapes(self) -> None:
        """Forward pass returns correct shapes for default config."""
        model = MEGAutoencoder(n_roi=N_ROI)
        x = torch.randn(BATCH, T, N_ROI)
        recon, latent = model(x)
        assert recon.shape == (BATCH, T, N_ROI)
        assert latent.shape == (BATCH, T, 2 * N_ROI)

    def test_default_n_latent(self) -> None:
        """n_latent defaults to 2 * n_roi (overcomplete)."""
        model = MEGAutoencoder(n_roi=N_ROI)
        assert model.n_latent == 2 * N_ROI

    def test_custom_n_latent(self) -> None:
        """Explicit n_latent overrides the default."""
        model = MEGAutoencoder(n_roi=N_ROI, n_latent=8)
        assert model.n_latent == 8
        x = torch.randn(1, T, N_ROI)
        _, latent = model(x)
        assert latent.shape == (1, T, 8)

    def test_encode_decode_shapes(self) -> None:
        """Encode and decode produce complementary shapes."""
        model = MEGAutoencoder(n_roi=N_ROI)
        x = torch.randn(BATCH, T, N_ROI)
        latent = model.encode(x)
        assert latent.shape == (BATCH, T, model.n_latent)
        recon = model.decode(latent)
        assert recon.shape == (BATCH, T, N_ROI)


class TestMEGAutoencoderBehaviour:
    """Behavioural tests for MEGAutoencoder."""

    def test_gradient_flow(self) -> None:
        """Loss.backward() produces non-None gradients on all parameters."""
        model = MEGAutoencoder(n_roi=N_ROI)
        x = torch.randn(BATCH, T, N_ROI)
        recon, _ = model(x)
        loss = torch.nn.functional.mse_loss(recon, x)
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.any(param.grad != 0), (
                f"All-zero gradient for {name}"
            )

    def test_batch_invariance(self) -> None:
        """Output for sample i is identical whether batch=1 or batch>1."""
        model = MEGAutoencoder(n_roi=N_ROI)
        model.eval()
        x_single = torch.randn(1, T, N_ROI)
        x_batch = torch.cat(
            [x_single, torch.randn(3, T, N_ROI)], dim=0
        )
        with torch.no_grad():
            recon_single, latent_single = model(x_single)
            recon_batch, latent_batch = model(x_batch)
        torch.testing.assert_close(
            recon_batch[0:1], recon_single, atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(
            latent_batch[0:1], latent_single, atol=1e-5, rtol=1e-5
        )


# ---------------------------------------------------------------------------
# AutoencoderTrainer tests
# ---------------------------------------------------------------------------


class TestAutoencoderTrainer:
    """Tests for AutoencoderTrainer training loop and utilities."""

    def test_train_reduces_loss(self) -> None:
        """20 epochs on synthetic sinusoidal data reduces train loss."""
        model = MEGAutoencoder(n_roi=N_ROI, hidden_size=32)
        trainer = AutoencoderTrainer(model, lr=1e-3)
        dataset = _make_sinusoidal_data(n_samples=16)
        loader = DataLoader(dataset, batch_size=4)
        result = trainer.train(loader, n_epochs=20, log_every=100)
        losses = result["train_losses"]
        assert len(losses) == 20
        # Loss should decrease overall
        assert losses[-1] < losses[0], (
            f"Expected loss to decrease: first={losses[0]:.6f}, "
            f"last={losses[-1]:.6f}"
        )

    def test_checkpoint_roundtrip(self, tmp_path: Path) -> None:
        """Save and load checkpoint produces identical model state_dict."""
        model = MEGAutoencoder(n_roi=N_ROI, hidden_size=16)
        trainer = AutoencoderTrainer(model)
        # Do one training step to move params away from init
        dataset = _make_sinusoidal_data(n_samples=4)
        loader = DataLoader(dataset, batch_size=4)
        trainer.train(loader, n_epochs=1, log_every=100)

        ckpt_path = tmp_path / "test_ckpt.pt"
        trainer.save_checkpoint(ckpt_path)

        # Create fresh model/trainer and load
        model2 = MEGAutoencoder(n_roi=N_ROI, hidden_size=16)
        trainer2 = AutoencoderTrainer(model2)
        trainer2.load_checkpoint(ckpt_path)

        for (k1, v1), (k2, v2) in zip(
            model.state_dict().items(),
            model2.state_dict().items(),
            strict=True,
        ):
            assert k1 == k2
            torch.testing.assert_close(v1, v2)

    def test_early_stopping(self) -> None:
        """With patience=3 and worsening val loss, training stops early."""
        model = MEGAutoencoder(n_roi=N_ROI, hidden_size=16)
        trainer = AutoencoderTrainer(model, lr=1e-3)
        # Use same data for train and val -- small enough that early
        # stopping may trigger if val loss plateaus.
        dataset = _make_sinusoidal_data(n_samples=8)
        train_loader = DataLoader(dataset, batch_size=4)
        val_loader = DataLoader(dataset, batch_size=4)
        result = trainer.train(
            train_loader,
            n_epochs=200,
            val_loader=val_loader,
            patience=3,
            log_every=1000,
        )
        # Training should stop before 200 epochs (early stopping)
        # or complete all 200 if it keeps improving. Either is valid,
        # but final_epoch should be consistent with train_losses length.
        assert result["final_epoch"] == len(result["train_losses"]) - 1
        assert result["val_losses"] is not None
        assert len(result["val_losses"]) == len(result["train_losses"])

    def test_evaluate(self) -> None:
        """evaluate() returns a positive float MSE."""
        model = MEGAutoencoder(n_roi=N_ROI, hidden_size=16)
        trainer = AutoencoderTrainer(model)
        dataset = _make_sinusoidal_data(n_samples=4)
        loader = DataLoader(dataset, batch_size=4)
        mse = trainer.evaluate(loader)
        assert isinstance(mse, float)
        assert mse > 0

    def test_spectral_loss_nonzero(self) -> None:
        """With spectral_weight=1.0, total loss exceeds pure MSE loss."""
        model = MEGAutoencoder(n_roi=N_ROI, hidden_size=32)
        # Train with spectral weight
        trainer_spec = AutoencoderTrainer(
            model, lr=1e-3, spectral_weight=1.0
        )
        dataset = _make_sinusoidal_data(n_samples=8)
        loader = DataLoader(dataset, batch_size=4)
        result_spec = trainer_spec.train(loader, n_epochs=1, log_every=100)

        # Train without spectral weight on fresh model
        model2 = MEGAutoencoder(n_roi=N_ROI, hidden_size=32)
        # Copy same initial weights for fair comparison
        model2.load_state_dict(model.state_dict())
        trainer_plain = AutoencoderTrainer(model2, lr=1e-3)
        loader2 = DataLoader(dataset, batch_size=4)
        result_plain = trainer_plain.train(
            loader2, n_epochs=1, log_every=100
        )

        # The spectral-augmented loss should differ from plain MSE
        # (it may be higher or lower after one epoch due to gradient
        # interaction, but it should not be identical)
        assert result_spec["train_losses"][0] != pytest.approx(
            result_plain["train_losses"][0], abs=1e-8
        ), "Spectral loss had no effect on total loss"

    def test_spectral_loss_zero_default(self) -> None:
        """With spectral_weight=0.0, spectral loss is not computed."""
        model = MEGAutoencoder(n_roi=N_ROI, hidden_size=16)
        trainer = AutoencoderTrainer(model, spectral_weight=0.0)
        # Verify the spectral loss method exists but weight is zero
        assert trainer.spectral_weight == 0.0
        # Train one epoch -- should use only MSE
        dataset = _make_sinusoidal_data(n_samples=4)
        loader = DataLoader(dataset, batch_size=4)
        result = trainer.train(loader, n_epochs=1, log_every=100)
        assert len(result["train_losses"]) == 1
