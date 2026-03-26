from __future__ import annotations

from pyro_dcm.simulators.rdcm_simulator import (
    make_block_stimulus_rdcm,
    make_stable_A_rdcm,
    simulate_rdcm,
)
from pyro_dcm.simulators.spectral_simulator import (
    make_stable_A_spectral,
    simulate_spectral_dcm,
)
from pyro_dcm.simulators.task_simulator import (
    make_block_stimulus,
    make_random_stable_A,
    simulate_task_dcm,
)

__all__ = [
    # Phase 1: Task-based DCM simulator
    "make_block_stimulus",
    "make_random_stable_A",
    "simulate_task_dcm",
    # Phase 2: Spectral DCM simulator
    "make_stable_A_spectral",
    "simulate_spectral_dcm",
    # Phase 3: rDCM simulator
    "make_block_stimulus_rdcm",
    "make_stable_A_rdcm",
    "simulate_rdcm",
]
