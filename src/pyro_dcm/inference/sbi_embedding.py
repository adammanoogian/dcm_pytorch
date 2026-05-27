"""Embedding networks for SBI on cross-spectral density data.

Provides neural network architectures that compress high-dimensional
CSD observations into low-dimensional summary statistics suitable for
neural posterior estimation (NPE).
"""

from __future__ import annotations

import torch
from torch import nn


class CSDEmbeddingNet(nn.Module):
    """Compress CSD observations to summary statistics for NPE.

    Two fully-connected layers with ReLU activation and batch
    normalization, followed by a linear projection to the embedding
    dimension. Designed to reduce the ``2*F*N*N``-dimensional CSD
    vector (real + imaginary parts) to a compact summary.

    Parameters
    ----------
    input_dim : int
        Dimension of the flattened CSD input (``2 * F * N * N``).
    embed_dim : int
        Dimension of the output embedding. Default 64.
    hidden_dim : int
        Width of hidden layers. Default 128.

    Examples
    --------
    >>> net = CSDEmbeddingNet(input_dim=288, embed_dim=64)
    >>> x = torch.randn(32, 288)
    >>> out = net(x)
    >>> out.shape  # (32, 64)
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Flattened CSD input, shape ``(batch, input_dim)``.

        Returns
        -------
        torch.Tensor
            Embedding, shape ``(batch, embed_dim)``.
        """
        return self.net(x)
