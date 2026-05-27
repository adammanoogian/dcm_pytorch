"""Inference backends for spectral DCM.

Provides Variational Laplace (VL), and simulation-based inference (SBI)
backends as alternatives to the Pyro SVI backend in
``pyro_dcm.models.guides``.
"""

from __future__ import annotations

from pyro_dcm.inference.sbi_embedding import CSDEmbeddingNet
from pyro_dcm.inference.sbi_spectral import (
    build_sbi_posterior,
    make_spectral_dcm_prior,
    make_spectral_dcm_simulator,
    train_npe,
)
from pyro_dcm.inference.variational_laplace import (
    VariationalLaplaceResult,
    run_variational_laplace,
)

__all__ = [
    # Variational Laplace
    "VariationalLaplaceResult",
    "run_variational_laplace",
    # SBI: simulator + prior
    "build_sbi_posterior",
    "make_spectral_dcm_prior",
    "make_spectral_dcm_simulator",
    "train_npe",
    # SBI: embedding
    "CSDEmbeddingNet",
]
