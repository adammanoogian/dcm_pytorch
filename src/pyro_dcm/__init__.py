from __future__ import annotations

__version__ = "0.1.0"

import logging as _logging

# Library-logging discipline (PEP 282, Python HOWTO). Attach a NullHandler to
# the pyro_dcm package root so that consumers who do not configure logging
# never see "no handlers could be found" warnings when pyro_dcm emits log
# events (e.g. the pyro_dcm.stability eigenvalue monitor from
# CoupledDCMSystem). Downstream users silence via the standard stdlib:
#     logging.getLogger("pyro_dcm.stability").setLevel(logging.ERROR)
_logging.getLogger("pyro_dcm").addHandler(_logging.NullHandler())

# Generative models
from pyro_dcm.models import (
    task_dcm_model,
    spectral_dcm_model,
    rdcm_model,
)

# Inference
from pyro_dcm.models import (
    create_guide,
    run_svi,
    extract_posterior_params,
)

# Amortized inference
from pyro_dcm.guides import (
    AmortizedFlowGuide,
    BoldSummaryNet,
    CsdSummaryNet,
    TaskDCMPacker,
    SpectralDCMPacker,
)
from pyro_dcm.models import (
    amortized_task_dcm_model,
    amortized_spectral_dcm_model,
)

# Simulators
from pyro_dcm.simulators import (
    simulate_task_dcm,
    make_random_stable_A,
    make_block_stimulus,
    simulate_spectral_dcm,
    make_stable_A_spectral,
    simulate_rdcm,
    make_stable_A_rdcm,
    make_block_stimulus_rdcm,
)

# Utilities
from pyro_dcm.forward_models import parameterize_A

__all__ = [
    "__version__",
    # Generative models
    "task_dcm_model",
    "spectral_dcm_model",
    "rdcm_model",
    # Inference
    "create_guide",
    "run_svi",
    "extract_posterior_params",
    # Amortized inference
    "AmortizedFlowGuide",
    "amortized_task_dcm_model",
    "amortized_spectral_dcm_model",
    "BoldSummaryNet",
    "CsdSummaryNet",
    "TaskDCMPacker",
    "SpectralDCMPacker",
    # Simulators
    "simulate_task_dcm",
    "make_random_stable_A",
    "make_block_stimulus",
    "simulate_spectral_dcm",
    "make_stable_A_spectral",
    "simulate_rdcm",
    "make_stable_A_rdcm",
    "make_block_stimulus_rdcm",
    # Utilities
    "parameterize_A",
]
