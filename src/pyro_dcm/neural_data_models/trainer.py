"""Training infrastructure for MEG autoencoder models.

Provides :class:`AutoencoderTrainer` with MSE reconstruction loss, optional
spectral consistency loss (via ``torch.fft``), early stopping, validation
tracking, and checkpoint save/load.

The spectral consistency loss computes per-channel power spectral density
via ``torch.fft.rfft`` and penalises log-power differences between input
and reconstruction.  This is fully differentiable and encourages the
autoencoder to preserve spectral structure needed for downstream spectral
DCM analysis.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from pyro_dcm.neural_data_models.lstm_autoencoder import MEGAutoencoder

logger = logging.getLogger(__name__)

_EPS = 1e-10  # numerical floor for log-power computation


class AutoencoderTrainer:
    """Training wrapper for :class:`MEGAutoencoder`.

    Parameters
    ----------
    model : MEGAutoencoder
        Autoencoder model to train.
    lr : float
        Adam learning rate.
    weight_decay : float
        Adam weight decay (L2 regularisation).
    device : str
        Torch device string (``"cpu"`` or ``"cuda"``).
    checkpoint_dir : Path | None
        Directory for saving best-model checkpoints.  ``None`` disables
        automatic checkpointing during training.
    spectral_weight : float
        Weight for the spectral consistency loss term.  ``0.0`` (default)
        disables the spectral loss entirely.
    sfreq : float
        Sampling frequency in Hz.  Used only when ``spectral_weight > 0``
        for documentation purposes (the PSD comparison is scale-free in
        log-power space).

    Attributes
    ----------
    model : MEGAutoencoder
    optimizer : torch.optim.Adam
    criterion : nn.MSELoss
    device : torch.device
    spectral_weight : float
    sfreq : float
    """

    def __init__(
        self,
        model: MEGAutoencoder,
        *,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        device: str = "cpu",
        checkpoint_dir: Path | None = None,
        spectral_weight: float = 0.0,
        sfreq: float = 250.0,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.criterion = nn.MSELoss()
        self.checkpoint_dir = checkpoint_dir
        self.spectral_weight = spectral_weight
        self.sfreq = sfreq

    # ------------------------------------------------------------------
    # Spectral consistency loss
    # ------------------------------------------------------------------

    def _spectral_consistency_loss(
        self, x: torch.Tensor, x_recon: torch.Tensor
    ) -> torch.Tensor:
        """Differentiable spectral consistency loss via ``torch.fft``.

        Computes per-channel power spectral density (PSD) as
        ``|rfft(signal)|^2`` and returns the MSE between log-power spectra
        of the input and reconstruction.  Log-scale comparison is used
        because power spectra span orders of magnitude.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, T, N_roi)
            Original input timeseries.
        x_recon : torch.Tensor, shape (batch, T, N_roi)
            Reconstructed timeseries.

        Returns
        -------
        torch.Tensor
            Scalar spectral consistency loss.
        """
        # PSD via rfft: (batch, T, N_roi) -> (batch, F, N_roi)
        psd_x = torch.abs(torch.fft.rfft(x, dim=1)) ** 2
        psd_recon = torch.abs(torch.fft.rfft(x_recon, dim=1)) ** 2

        log_psd_x = torch.log(psd_x + _EPS)
        log_psd_recon = torch.log(psd_recon + _EPS)

        return torch.nn.functional.mse_loss(log_psd_recon, log_psd_x)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        *,
        n_epochs: int = 100,
        val_loader: DataLoader | None = None,
        patience: int = 10,
        log_every: int = 10,
    ) -> dict:
        """Run training loop with optional validation and early stopping.

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader yielding tensors of shape
            ``(batch, T, N_roi)`` or tuples whose first element is such
            a tensor.
        n_epochs : int
            Maximum number of training epochs.
        val_loader : DataLoader | None
            Validation data loader.  If provided, validation loss is
            computed each epoch and used for early stopping.
        patience : int
            Number of epochs without validation improvement before
            stopping.  Ignored when ``val_loader`` is ``None``.
        log_every : int
            Log training progress every *log_every* epochs.

        Returns
        -------
        dict
            Training history with keys:

            - ``train_losses`` : list[float] -- per-epoch training loss.
            - ``val_losses`` : list[float] | None -- per-epoch val loss.
            - ``best_epoch`` : int -- epoch with lowest val loss (or
              final epoch if no validation).
            - ``final_epoch`` : int -- last completed epoch (0-indexed).
        """
        train_losses: list[float] = []
        val_losses: list[float] | None = [] if val_loader is not None else None

        best_val_loss = float("inf")
        best_epoch = 0
        epochs_without_improvement = 0

        for epoch in range(n_epochs):
            # --- Training ---
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0
            for batch in train_loader:
                x = self._extract_tensor(batch).to(self.device)
                self.optimizer.zero_grad()
                recon, _ = self.model(x)
                loss = self.criterion(recon, x)
                if self.spectral_weight > 0:
                    loss = loss + self.spectral_weight * (
                        self._spectral_consistency_loss(x, recon)
                    )
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            avg_train = epoch_loss / max(n_batches, 1)
            train_losses.append(avg_train)

            # --- Validation ---
            if val_loader is not None:
                val_loss = self.evaluate(val_loader)
                val_losses.append(val_loss)  # type: ignore[union-attr]
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    epochs_without_improvement = 0
                    if self.checkpoint_dir is not None:
                        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                        self.save_checkpoint(
                            self.checkpoint_dir / "best_model.pt"
                        )
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        logger.info(
                            "Early stopping at epoch %d (patience=%d)",
                            epoch,
                            patience,
                        )
                        break
            else:
                best_epoch = epoch

            if epoch % log_every == 0:
                msg = f"Epoch {epoch:4d}  train_loss={avg_train:.6f}"
                if val_loader is not None:
                    msg += f"  val_loss={val_losses[-1]:.6f}"  # type: ignore[index]
                logger.info(msg)

        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_epoch": best_epoch,
            "final_epoch": len(train_losses) - 1,
        }

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: Path) -> None:
        """Save model and optimizer state to disk.

        Parameters
        ----------
        path : Path
            File path for the checkpoint.
        """
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epoch": getattr(self, "_current_epoch", 0),
            },
            path,
        )

    def load_checkpoint(self, path: Path) -> int:
        """Load model and optimizer state from disk.

        Parameters
        ----------
        path : Path
            File path of the checkpoint.

        Returns
        -------
        int
            Epoch number stored in the checkpoint.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint.get("epoch", 0)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, data_loader: DataLoader) -> float:
        """Compute mean MSE reconstruction loss on a dataset.

        Parameters
        ----------
        data_loader : DataLoader
            Data loader yielding input tensors.

        Returns
        -------
        float
            Mean MSE loss over all batches.
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for batch in data_loader:
                x = self._extract_tensor(batch).to(self.device)
                recon, _ = self.model(x)
                total_loss += self.criterion(recon, x).item()
                n_batches += 1
        return total_loss / max(n_batches, 1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_tensor(batch: torch.Tensor | tuple | list) -> torch.Tensor:
        """Extract input tensor from a DataLoader batch.

        Parameters
        ----------
        batch : torch.Tensor | tuple | list
            Raw batch from DataLoader.  If tuple/list, uses the first
            element.

        Returns
        -------
        torch.Tensor
            The input tensor.
        """
        if isinstance(batch, (tuple, list)):
            return batch[0]
        return batch
