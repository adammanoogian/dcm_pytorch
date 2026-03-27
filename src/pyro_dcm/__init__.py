from __future__ import annotations

__version__ = "0.1.0"

# Phase 4: Pyro generative models
from pyro_dcm.models import (
    task_dcm_model,
    spectral_dcm_model,
    rdcm_model,
    create_guide,
    run_svi,
)

__all__ = [
    "__version__",
    # Phase 4: Pyro generative models
    "task_dcm_model",
    "spectral_dcm_model",
    "rdcm_model",
    "create_guide",
    "run_svi",
]
