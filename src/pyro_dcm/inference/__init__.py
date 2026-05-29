"""Inference backends for spectral DCM.

Variational Laplace (VL) is the default inference method for spectral
DCM, matching SPM12's ``spm_nlsi_GN``. SVI remains available via
``pyro_dcm.models.guides`` for task DCM and amortized inference.
"""

from __future__ import annotations

from pyro_dcm.inference.csd_precision import compute_csd_precision
from pyro_dcm.inference.variational_laplace import (
    run_variational_laplace,
    VariationalLaplaceResult,
)

__all__ = [
    "compute_csd_precision",
    "run_variational_laplace",
    "VariationalLaplaceResult",
]
