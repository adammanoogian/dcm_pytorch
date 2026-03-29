"""Amortized neural inference guides for DCM.

This package provides normalizing-flow-based amortized inference guides
for Dynamic Causal Modeling. Key components:

- **Summary networks**: Compress variable-length BOLD/CSD observations
  into fixed-dimensional embeddings for flow conditioning.
- **Parameter packing**: Convert between named Pyro sample site dicts
  and flat standardized vectors for NSF spline transforms.

References
----------
[REF-042] Radev et al. (2020). BayesFlow: Learning complex stochastic
    models with invertible neural networks.
[REF-043] Cranmer, Brehmer & Louppe (2020). The frontier of
    simulation-based inference.
"""

from __future__ import annotations

from pyro_dcm.guides.parameter_packing import (
    SpectralDCMPacker,
    TaskDCMPacker,
)
from pyro_dcm.guides.summary_networks import BoldSummaryNet, CsdSummaryNet

__all__ = [
    "BoldSummaryNet",
    "CsdSummaryNet",
    "SpectralDCMPacker",
    "TaskDCMPacker",
]
