"""Summary networks for amortized neural inference.

Compresses raw observed data (BOLD time series or CSD matrices) into
fixed-dimensional embedding vectors that serve as conditioning context
for normalizing flow guides.

Architecture follows the SBI literature conventions:

- **BOLD**: 1D-CNN with AdaptiveAvgPool1d handles variable-length
  time series. Temporal convolutions capture hemodynamic response shape.
- **CSD**: MLP on flattened real/imaginary decomposition. CSD is
  already a compact frequency-domain summary statistic.

References
----------
[REF-042] Radev et al. (2020). BayesFlow: Learning complex stochastic
    models with invertible neural networks.
[REF-043] Cranmer, Brehmer & Louppe (2020). The frontier of
    simulation-based inference. PNAS, 117(48), 30055-30062.
07-RESEARCH.md Section 1: Summary Network Design.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BoldSummaryNet(nn.Module):
    """1D-CNN summary network for BOLD time series.

    Compresses a ``(T, N)`` BOLD tensor into a fixed-dimensional
    embedding vector via three convolutional layers with batch
    normalization, adaptive average pooling over the temporal
    dimension, and a final linear projection.

    Handles both single-observation and batched inputs. The
    ``AdaptiveAvgPool1d`` layer ensures variable-length time series
    (different T) all produce the same output dimension.

    Parameters
    ----------
    n_regions : int
        Number of brain regions (N). This is the input channel
        dimension for the first Conv1d layer.
    embed_dim : int, optional
        Dimension of the output embedding vector. Default 128.

    Notes
    -----
    Architecture from 07-RESEARCH.md Section 1 and [REF-042]:

    - Conv1d(N, 64, k=5, pad=2) + BN + ReLU
    - Conv1d(64, 128, k=5, pad=2) + BN + ReLU
    - Conv1d(128, 256, k=5, pad=2) + BN + ReLU
    - AdaptiveAvgPool1d(1) -> squeeze temporal dim
    - Linear(256, embed_dim)

    All parameters use float64 (project convention).

    Examples
    --------
    >>> net = BoldSummaryNet(n_regions=3, embed_dim=128)
    >>> bold = torch.randn(100, 3, dtype=torch.float64)
    >>> embedding = net(bold)
    >>> embedding.shape  # (128,)
    """

    def __init__(self, n_regions: int, embed_dim: int = 128) -> None:
        super().__init__()
        self.n_regions = n_regions
        self.embed_dim = embed_dim

        # Conv layers: channels = regions -> 64 -> 128 -> 256
        self.conv1 = nn.Conv1d(
            n_regions, 64, kernel_size=5, padding=2,
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(
            64, 128, kernel_size=5, padding=2,
        )
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(
            128, 256, kernel_size=5, padding=2,
        )
        self.bn3 = nn.BatchNorm1d(256)

        self.relu = nn.ReLU()

        # Adaptive pooling squeezes temporal dimension to 1
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Final projection to embedding dimension
        self.fc = nn.Linear(256, embed_dim)

        # Convert all parameters to float64
        self.double()

    def forward(self, bold: torch.Tensor) -> torch.Tensor:
        """Compress BOLD time series to fixed-dim embedding.

        Parameters
        ----------
        bold : torch.Tensor
            BOLD time series. Shape ``(T, N)`` for a single
            observation or ``(batch, T, N)`` for a batch.
            dtype must be float64.

        Returns
        -------
        torch.Tensor
            Embedding vector. Shape ``(embed_dim,)`` if input was
            unbatched, or ``(batch, embed_dim)`` if batched.
        """
        # Handle unbatched input: (T, N) -> (1, T, N)
        unbatched = bold.dim() == 2
        if unbatched:
            bold = bold.unsqueeze(0)  # shape: (1, T, N)

        # Transpose to channels-first: (batch, T, N) -> (batch, N, T)
        x = bold.transpose(1, 2)  # shape: (batch, N, T)

        # Conv block 1
        x = self.relu(self.bn1(self.conv1(x)))
        # Conv block 2
        x = self.relu(self.bn2(self.conv2(x)))
        # Conv block 3
        x = self.relu(self.bn3(self.conv3(x)))

        # Pool over temporal dimension: (batch, 256, T) -> (batch, 256, 1)
        x = self.pool(x)
        # Squeeze: (batch, 256, 1) -> (batch, 256)
        x = x.squeeze(-1)

        # Project to embedding: (batch, 256) -> (batch, embed_dim)
        x = self.fc(x)

        # Remove batch dim if input was unbatched
        if unbatched:
            x = x.squeeze(0)  # shape: (embed_dim,)

        return x


class CsdSummaryNet(nn.Module):
    """MLP summary network for cross-spectral density matrices.

    Compresses a ``(F, N, N)`` complex CSD tensor into a
    fixed-dimensional embedding vector by decomposing to real and
    imaginary parts, flattening, and passing through a 3-layer MLP.

    Handles both single-observation and batched inputs.

    Parameters
    ----------
    n_regions : int
        Number of brain regions (N).
    n_freqs : int, optional
        Number of frequency bins (F). Default 32.
    embed_dim : int, optional
        Dimension of the output embedding vector. Default 128.

    Notes
    -----
    Architecture from 07-RESEARCH.md Section 1:

    - Input: complex ``(F, N, N)`` -> decompose real/imag -> flatten
      to ``(2*F*N*N,)``
    - Linear(2*F*N*N, 512) + ReLU
    - Linear(512, 256) + ReLU
    - Linear(256, embed_dim)

    For N=3, F=32: input dim = 2*32*3*3 = 576 (very manageable).
    All parameters use float64 (project convention).

    Examples
    --------
    >>> net = CsdSummaryNet(n_regions=3, n_freqs=32, embed_dim=128)
    >>> csd = torch.randn(32, 3, 3, dtype=torch.complex128)
    >>> embedding = net(csd)
    >>> embedding.shape  # (128,)
    """

    def __init__(
        self,
        n_regions: int,
        n_freqs: int = 32,
        embed_dim: int = 128,
    ) -> None:
        super().__init__()
        self.n_regions = n_regions
        self.n_freqs = n_freqs
        self.embed_dim = embed_dim

        # Input dimension: real + imaginary parts flattened
        input_dim = 2 * n_freqs * n_regions * n_regions

        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim),
        )

        # Convert all parameters to float64
        self.double()

    def forward(self, csd: torch.Tensor) -> torch.Tensor:
        """Compress complex CSD matrix to fixed-dim embedding.

        Parameters
        ----------
        csd : torch.Tensor
            Cross-spectral density. Shape ``(F, N, N)`` for a single
            observation or ``(batch, F, N, N)`` for a batch.
            dtype must be complex128 (or complex64).

        Returns
        -------
        torch.Tensor
            Embedding vector. Shape ``(embed_dim,)`` if input was
            unbatched, or ``(batch, embed_dim)`` if batched.
            dtype is float64.
        """
        # Handle unbatched input: (F, N, N) -> (1, F, N, N)
        unbatched = csd.dim() == 3
        if unbatched:
            csd = csd.unsqueeze(0)  # shape: (1, F, N, N)

        batch_size = csd.shape[0]

        # Decompose complex to real/imaginary and flatten
        # shape: (batch, F, N, N) -> (batch, 2*F*N*N)
        real_part = csd.real.reshape(batch_size, -1)
        imag_part = csd.imag.reshape(batch_size, -1)
        x = torch.cat([real_part, imag_part], dim=-1)

        # Ensure float64 (in case input was complex64)
        x = x.to(torch.float64)

        # Pass through MLP
        x = self.mlp(x)  # shape: (batch, embed_dim)

        # Remove batch dim if input was unbatched
        if unbatched:
            x = x.squeeze(0)  # shape: (embed_dim,)

        return x
