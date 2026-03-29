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

# Phase 7: Amortized inference
from pyro_dcm.guides import AmortizedFlowGuide
from pyro_dcm.models import (
    amortized_task_dcm_model,
    amortized_spectral_dcm_model,
)

__all__ = [
    "__version__",
    # Phase 4: Pyro generative models
    "task_dcm_model",
    "spectral_dcm_model",
    "rdcm_model",
    "create_guide",
    "run_svi",
    # Phase 7: Amortized inference
    "AmortizedFlowGuide",
    "amortized_task_dcm_model",
    "amortized_spectral_dcm_model",
]
