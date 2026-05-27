"""LSTM autoencoder for MEG ROI timeseries.

Maps multivariate ROI timeseries to an overcomplete latent representation
and back. The encoder produces a latent trajectory at every timestep (not
just the final hidden state), because downstream spectral DCM requires
continuous latent dynamics for CSD computation.

The default overcomplete latent space (``n_latent = 2 * n_roi``) avoids a
bottleneck that might discard spectrally relevant information.
"""
from __future__ import annotations

import torch
from torch import nn


class MEGAutoencoder(nn.Module):
    """LSTM autoencoder with overcomplete latent space for MEG ROI data.

    Encodes multivariate ROI timeseries ``(batch, T, N_roi)`` into an
    overcomplete latent trajectory ``(batch, T, N_latent)`` where
    ``N_latent = 2 * N_roi`` by default, then decodes back to
    ``(batch, T, N_roi)``.

    Parameters
    ----------
    n_roi : int
        Number of input ROI channels.
    n_latent : int | None
        Latent dimension. Defaults to ``2 * n_roi`` when ``None``.
    hidden_size : int
        LSTM hidden state size for both encoder and decoder.
    n_layers : int
        Number of stacked LSTM layers.
    dropout : float
        Dropout probability between LSTM layers (only active when
        ``n_layers > 1``).

    Attributes
    ----------
    n_roi : int
        Number of ROI channels.
    n_latent : int
        Latent dimension (resolved from input or default).
    hidden_size : int
        LSTM hidden state size.
    n_layers : int
        Number of stacked LSTM layers.

    Examples
    --------
    >>> model = MEGAutoencoder(n_roi=6)
    >>> x = torch.randn(2, 100, 6)
    >>> recon, latent = model(x)
    >>> recon.shape
    torch.Size([2, 100, 6])
    >>> latent.shape
    torch.Size([2, 100, 12])
    """

    def __init__(
        self,
        n_roi: int,
        n_latent: int | None = None,
        hidden_size: int = 64,
        n_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_roi = n_roi
        self.n_latent = 2 * n_roi if n_latent is None else n_latent
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        # Encoder: ROI timeseries -> hidden -> latent
        self.encoder_lstm = nn.LSTM(
            input_size=n_roi,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.encoder_fc = nn.Linear(hidden_size, self.n_latent)

        # Decoder: latent -> hidden -> ROI timeseries
        self.decoder_fc = nn.Linear(self.n_latent, hidden_size)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=n_roi,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode ROI timeseries to latent trajectory.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, T, N_roi)
            Input ROI timeseries.

        Returns
        -------
        torch.Tensor, shape (batch, T, N_latent)
            Latent trajectory at every timestep.
        """
        hidden_seq, _ = self.encoder_lstm(x)
        latent = self.encoder_fc(hidden_seq)
        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent trajectory back to ROI timeseries.

        Parameters
        ----------
        latent : torch.Tensor, shape (batch, T, N_latent)
            Latent trajectory.

        Returns
        -------
        torch.Tensor, shape (batch, T, N_roi)
            Reconstructed ROI timeseries.
        """
        hidden_seq = self.decoder_fc(latent)
        recon, _ = self.decoder_lstm(hidden_seq)
        return recon

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: encode then decode.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, T, N_roi)
            Input ROI timeseries.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(reconstruction, latent)`` where reconstruction has shape
            ``(batch, T, N_roi)`` and latent has shape
            ``(batch, T, N_latent)``.
        """
        latent = self.encode(x)
        recon = self.decode(latent)
        return recon, latent
