"""Inference engines for Dynamic Causal Modeling.

Provides:
- ``variational_laplace``: Gauss-Newton Variational Laplace (SPM-style).
"""

from __future__ import annotations

from pyro_dcm.inference.variational_laplace import VLResult, variational_laplace
from pyro_dcm.inference.vl_dcm import make_task_dcm_forward

__all__ = [
    "VLResult",
    "make_task_dcm_forward",
    "variational_laplace",
]
