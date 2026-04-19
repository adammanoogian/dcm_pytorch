from __future__ import annotations

import logging as _logging

from pyro_dcm.forward_models import (
    compute_effective_A,
    parameterize_A,
    parameterize_B,
)
from pyro_dcm.guides import (
    AmortizedFlowGuide,
    BoldSummaryNet,
    CsdSummaryNet,
    SpectralDCMPacker,
    TaskDCMPacker,
)
from pyro_dcm.models import (
    amortized_spectral_dcm_model,
    amortized_task_dcm_model,
    create_guide,
    extract_posterior_params,
    rdcm_model,
    run_svi,
    spectral_dcm_model,
    task_dcm_model,
)
from pyro_dcm.simulators import (
    make_block_stimulus,
    make_block_stimulus_rdcm,
    make_epoch_stimulus,
    make_event_stimulus,
    make_random_stable_A,
    make_stable_A_rdcm,
    make_stable_A_spectral,
    simulate_rdcm,
    simulate_spectral_dcm,
    simulate_task_dcm,
)
from pyro_dcm.utils import (
    PiecewiseConstantInput,
    merge_piecewise_inputs,
)

__version__ = "0.1.0"

# Library-logging discipline (PEP 282, Python HOWTO). Attach a NullHandler to
# the pyro_dcm package root so that consumers who do not configure logging
# never see "no handlers could be found" warnings when pyro_dcm emits log
# events (e.g. the pyro_dcm.stability eigenvalue monitor from
# CoupledDCMSystem). Downstream users silence via the standard stdlib:
#     logging.getLogger("pyro_dcm.stability").setLevel(logging.ERROR)
_logging.getLogger("pyro_dcm").addHandler(_logging.NullHandler())

__all__ = [
    "AmortizedFlowGuide",
    "BoldSummaryNet",
    "CsdSummaryNet",
    "PiecewiseConstantInput",
    "SpectralDCMPacker",
    "TaskDCMPacker",
    "__version__",
    "amortized_spectral_dcm_model",
    "amortized_task_dcm_model",
    "compute_effective_A",
    "create_guide",
    "extract_posterior_params",
    "make_block_stimulus",
    "make_block_stimulus_rdcm",
    "make_epoch_stimulus",
    "make_event_stimulus",
    "make_random_stable_A",
    "make_stable_A_rdcm",
    "make_stable_A_spectral",
    "merge_piecewise_inputs",
    "parameterize_A",
    "parameterize_B",
    "rdcm_model",
    "run_svi",
    "simulate_rdcm",
    "simulate_spectral_dcm",
    "simulate_task_dcm",
    "spectral_dcm_model",
    "task_dcm_model",
]
