"""Amortized normalizing flow guide for DCM inference.

Wraps a Zuko Neural Spline Flow (NSF) and a summary network into a
Pyro-compatible guide that produces approximate posterior samples via
a single forward pass -- no iterative optimization.

The guide samples a single ``_latent`` vector from the conditional NSF,
matching the wrapper model pattern where both model and guide have
exactly one shared sample site. This solves the Pyro site-matching
problem described in 07-RESEARCH.md Pitfall 1.

References
----------
[REF-042] Radev et al. (2020). BayesFlow: Learning complex stochastic
    models with invertible neural networks.
[REF-043] Cranmer, Brehmer & Louppe (2020). The frontier of
    simulation-based inference. PNAS, 117(48), 30055-30062.
07-RESEARCH.md: Architecture patterns and Pyro integration.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pyro
import zuko.flows
from pyro.contrib.zuko import ZukoToPyro

from pyro_dcm.guides.parameter_packing import (
    SpectralDCMPacker,
    TaskDCMPacker,
)


class AmortizedFlowGuide(nn.Module):
    """Conditional NSF guide for amortized DCM inference.

    Combines a summary network (compresses observed data to a
    fixed-dimensional embedding) with a Neural Spline Flow (NSF)
    that maps the embedding to approximate posterior samples over
    packed DCM parameters.

    The guide produces a single ``_latent`` sample site via
    ``pyro.sample("_latent", ZukoToPyro(flow_dist))``. The
    corresponding wrapper model also samples ``_latent`` from a
    standard normal prior and deterministically unpacks it. This
    single-site pattern keeps Pyro's automatic ELBO working.

    Parameters
    ----------
    summary_net : nn.Module
        Summary network (``BoldSummaryNet`` or ``CsdSummaryNet``)
        that compresses observed data to an embedding vector.
    n_features : int
        Dimensionality of the packed parameter vector (output of
        the flow). Must match ``packer.n_features``.
    embed_dim : int, optional
        Summary network output dimension. Must match
        ``summary_net.embed_dim``. Default 128.
    n_transforms : int, optional
        Number of NSF autoregressive transforms. Default 5.
    n_bins : int, optional
        Number of rational-quadratic spline bins. Default 8.
    hidden_features : list[int] or None, optional
        Hidden layer sizes per transform. Default ``[256, 256]``.
    packer : TaskDCMPacker or SpectralDCMPacker
        Parameter packer for standardize/unstandardize operations.

    Notes
    -----
    The flow uses ``passes=2`` (coupling mode) for fast sampling,
    as recommended in 07-RESEARCH.md. All parameters use float64
    (project convention), achieved by calling ``.double()`` on the
    flow at construction time.

    References
    ----------
    [REF-042] Radev et al. (2020). BayesFlow.
    [REF-043] Cranmer, Brehmer & Louppe (2020). SBI frontier.

    Examples
    --------
    >>> from pyro_dcm.guides import BoldSummaryNet, TaskDCMPacker
    >>> net = BoldSummaryNet(3).double()
    >>> packer = TaskDCMPacker(3, 1, a_mask, c_mask)
    >>> guide = AmortizedFlowGuide(net, packer.n_features, packer=packer)
    >>> bold = torch.randn(100, 3, dtype=torch.float64)
    >>> samples = guide.sample_posterior(bold, n_samples=500)
    """

    def __init__(
        self,
        summary_net: nn.Module,
        n_features: int,
        embed_dim: int = 128,
        n_transforms: int = 5,
        n_bins: int = 8,
        hidden_features: list[int] | None = None,
        packer: TaskDCMPacker | SpectralDCMPacker | None = None,
    ) -> None:
        super().__init__()
        if hidden_features is None:
            hidden_features = [256, 256]
        if packer is None:
            msg = "packer is required for standardize/unstandardize"
            raise ValueError(msg)

        self.summary_net = summary_net
        self.packer = packer
        self.n_features = n_features
        self.embed_dim = embed_dim

        # Create NSF flow in coupling mode (passes=2) for fast sampling
        self.flow = zuko.flows.NSF(
            features=n_features,
            context=embed_dim,
            bins=n_bins,
            transforms=n_transforms,
            hidden_features=hidden_features,
            passes=2,
        ).double()

    def forward(
        self,
        observed_data: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        """Pyro guide function: sample ``_latent`` from the flow.

        Registers the summary network and flow as Pyro modules,
        computes the data embedding, and samples from the
        conditional NSF distribution.

        Parameters
        ----------
        observed_data : torch.Tensor
            Observed BOLD ``(T, N)`` or CSD ``(F, N, N)`` tensor.
        *args : object
            Additional arguments (passed by SVI, ignored here).
        **kwargs : object
            Additional keyword arguments (ignored).

        Returns
        -------
        torch.Tensor
            Sampled standardized latent vector, shape
            ``(n_features,)``.
        """
        pyro.module("summary_net", self.summary_net)
        pyro.module("flow", self.flow)
        embedding = self.summary_net(observed_data)
        flow_dist = self.flow(embedding)
        z_std = pyro.sample("_latent", ZukoToPyro(flow_dist))
        return z_std

    def sample_posterior(
        self,
        observed_data: torch.Tensor,
        n_samples: int = 1000,
    ) -> dict[str, torch.Tensor]:
        """Draw posterior samples via forward pass (no SVI needed).

        Parameters
        ----------
        observed_data : torch.Tensor
            Observed data tensor (BOLD or CSD).
        n_samples : int, optional
            Number of posterior samples to draw. Default 1000.

        Returns
        -------
        dict of str to torch.Tensor
            Unpacked parameter samples. Keys depend on the DCM
            variant (e.g., ``A_free``, ``C``, ``noise_prec`` for
            task DCM). Each value has leading batch dimension
            ``n_samples``.
        """
        self.eval()
        with torch.no_grad():
            embedding = self.summary_net(observed_data)
            flow_dist = self.flow(embedding)
            z_std = flow_dist.rsample((n_samples,))
            z = self.packer.unstandardize(z_std)
            return self.packer.unpack(z)
