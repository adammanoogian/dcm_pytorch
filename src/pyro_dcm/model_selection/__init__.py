"""Bayesian model selection utilities for DCM.

Provides Bayesian Model Reduction (BMR) and helpers for post hoc
model comparison from a single full-model inversion.
"""

from __future__ import annotations

from pyro_dcm.model_selection.bmr import (
    bayesian_model_reduction,
    make_reduced_prior_zero_connection,
)

__all__ = [
    "bayesian_model_reduction",
    "make_reduced_prior_zero_connection",
]
