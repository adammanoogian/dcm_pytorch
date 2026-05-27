"""1D-CNN encoder network for hybrid VAE-DCM.

Maps observed timeseries to approximate posterior parameters (location
and scale) over a latent DCM parameter vector. The architecture is
similar to ``BoldSummaryNet`` but produces dual output heads for
variational inference: ``z_loc`` (posterior mean) and ``z_scale``
(posterior standard deviation).

The output layer weights are initialized near zero so that the initial
``z_loc`` is approximately zero (near the prior mean), following
standard VAE initialization practice.

References
----------
Kingma & Welling (2014). Auto-Encoding Variational Bayes.
    arXiv:1312.6114.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DCMEncoderNet(nn.Module):
    """1D-CNN encoder mapping observed timeseries to DCM posteriors.

    Maps ``(batch, T, N)`` timeseries to ``(z_loc, z_scale)`` where
    ``z_loc`` and ``z_scale`` have shape ``(batch, latent_dim)``. Uses
    a 1D-CNN backbone (similar to ``BoldSummaryNet``) with dual output
    heads.

    Parameters
    ----------
    n_regions : int
        Number of brain regions / observed channels (N). This is the
        input channel dimension for the first Conv1d layer.
    latent_dim : int
        Dimension of the latent parameter vector (output size per
        head).
    hidden_channels : list[int] or None, optional
        Channel sizes for the convolutional layers. Default
        ``[32, 64, 128]``.

    Notes
    -----
    Architecture:

    - Input: ``(batch, T, N)`` transposed to ``(batch, N, T)``
    - 3x Conv1d layers: ``N -> 32 -> 64 -> 128``, kernel_size=3,
      padding=1, ReLU activation
    - AdaptiveAvgPool1d(1) -> squeeze -> ``(batch, 128)``
    - FC layer: ``128 -> 2 * latent_dim``
    - Split into ``z_loc`` (first half) and ``z_scale_raw`` (second)
    - ``z_scale = softplus(z_scale_raw) + 1e-5``

    The output FC layer weights are initialized near zero so initial
    ``z_loc`` is approximately zero (near prior mean), following
    standard VAE practice.

    Examples
    --------
    >>> enc = DCMEncoderNet(n_regions=4, latent_dim=21)
    >>> x = torch.randn(8, 100, 4)  # batch=8, T=100, N=4
    >>> z_loc, z_scale = enc(x)
    >>> z_loc.shape
    torch.Size([8, 21])
    >>> z_scale.shape
    torch.Size([8, 21])
    """

    def __init__(
        self,
        n_regions: int,
        latent_dim: int,
        hidden_channels: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.n_regions = n_regions
        self.latent_dim = latent_dim

        if hidden_channels is None:
            hidden_channels = [32, 64, 128]

        if len(hidden_channels) < 1:
            raise ValueError(
                "hidden_channels must have at least 1 element, "
                f"got {len(hidden_channels)}"
            )

        # Build conv layers dynamically
        channels = [n_regions] + list(hidden_channels)
        conv_layers: list[nn.Module] = []
        for i in range(len(channels) - 1):
            conv_layers.append(
                nn.Conv1d(
                    channels[i],
                    channels[i + 1],
                    kernel_size=3,
                    padding=1,
                )
            )
            conv_layers.append(nn.ReLU())

        self.conv_backbone = nn.Sequential(*conv_layers)

        # Adaptive pooling squeezes temporal dimension to 1
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Output FC: maps to 2 * latent_dim (loc + scale_raw)
        final_channels = hidden_channels[-1]
        self.fc_out = nn.Linear(final_channels, 2 * latent_dim)

        # Initialize output layer weights near zero so initial
        # z_loc ~= 0 (near prior mean)
        nn.init.normal_(self.fc_out.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.fc_out.bias)

    def forward(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode timeseries to posterior location and scale.

        Parameters
        ----------
        x : torch.Tensor
            Observed timeseries. Shape ``(batch, T, N)`` for batched
            input or ``(T, N)`` for a single observation.

        Returns
        -------
        z_loc : torch.Tensor
            Posterior mean, shape ``(batch, latent_dim)`` or
            ``(latent_dim,)`` if input was unbatched.
        z_scale : torch.Tensor
            Posterior standard deviation (always positive), same shape
            as ``z_loc``.
        """
        # Handle unbatched input: (T, N) -> (1, T, N)
        unbatched = x.dim() == 2
        if unbatched:
            x = x.unsqueeze(0)

        # Transpose to channels-first: (batch, T, N) -> (batch, N, T)
        x = x.transpose(1, 2)

        # Conv backbone
        x = self.conv_backbone(x)  # (batch, C_last, T)

        # Pool over temporal dim: (batch, C_last, T) -> (batch, C_last)
        x = self.pool(x).squeeze(-1)

        # FC output: (batch, C_last) -> (batch, 2 * latent_dim)
        out = self.fc_out(x)

        # Split into loc and scale_raw
        z_loc = out[..., :self.latent_dim]
        z_scale_raw = out[..., self.latent_dim:]

        # Ensure positive scale via softplus + epsilon
        z_scale = F.softplus(z_scale_raw) + 1e-5

        # Remove batch dim if input was unbatched
        if unbatched:
            z_loc = z_loc.squeeze(0)
            z_scale = z_scale.squeeze(0)

        return z_loc, z_scale
