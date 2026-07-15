"""Inference backends for DCM.

Variational Laplace (VL) is the default inference method for all DCM
variants, matching SPM12's ``spm_nlsi_GN``. The engine is model-
agnostic via the ``ForwardModel`` protocol. SVI remains available
via ``pyro_dcm.models.guides`` for amortized inference.
"""

from __future__ import annotations

from pyro_dcm.inference.csd_precision import compute_csd_precision
from pyro_dcm.inference.vl_forward_models import (
    ForwardModel,
    LatentCircuitForward,
    SpectralDCMForward,
    TaskDCMForward,
)
from pyro_dcm.inference.variational_laplace import (
    VariationalLaplaceResult,
    extract_vl_posterior,
    extract_vl_posterior_generic,
    run_variational_laplace,
    run_variational_laplace_generic,
)

# The sbi_* modules (sbi_spectral, sbi_embedding, sbi_diagnostics) are
# intentionally NOT re-exported here: they carry the optional `sbi` dependency
# and are imported directly by the training scripts/tests that need them, e.g.
# ``from pyro_dcm.inference.sbi_spectral import train_npe``.

__all__ = [
    "ForwardModel",
    "LatentCircuitForward",
    "SpectralDCMForward",
    "TaskDCMForward",
    "VariationalLaplaceResult",
    "compute_csd_precision",
    "extract_vl_posterior",
    "extract_vl_posterior_generic",
    "run_variational_laplace",
    "run_variational_laplace_generic",
]
