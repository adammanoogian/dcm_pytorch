"""Inference backends for spectral DCM.

Provides Variational Laplace (VL) as an alternative to the existing
Pyro SVI backend in ``pyro_dcm.models.guides``.
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
