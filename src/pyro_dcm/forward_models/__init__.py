from __future__ import annotations

from pyro_dcm.forward_models.balloon_model import BalloonWindkessel
from pyro_dcm.forward_models.bold_signal import bold_signal
from pyro_dcm.forward_models.coupled_system import CoupledDCMSystem
from pyro_dcm.forward_models.csd_computation import (
    bold_to_csd_torch,
    compute_empirical_csd,
    default_welch_params,
)
from pyro_dcm.forward_models.neural_state import (
    NeuralStateEquation,
    parameterize_A,
)
from pyro_dcm.forward_models.spectral_noise import (
    default_noise_priors,
    neuronal_noise_csd,
    observation_noise_csd,
)
from pyro_dcm.forward_models.spectral_transfer import (
    compute_transfer_function,
    default_frequency_grid,
    predicted_csd,
    spectral_dcm_forward,
)

__all__ = [
    # Phase 1: Neural-hemodynamic forward model
    "BalloonWindkessel",
    "CoupledDCMSystem",
    "NeuralStateEquation",
    "bold_signal",
    "parameterize_A",
    # Phase 2: Spectral transfer function + predicted CSD
    "compute_transfer_function",
    "default_frequency_grid",
    "predicted_csd",
    "spectral_dcm_forward",
    # Phase 2: Spectral noise models
    "default_noise_priors",
    "neuronal_noise_csd",
    "observation_noise_csd",
    # Phase 2: Empirical CSD computation
    "bold_to_csd_torch",
    "compute_empirical_csd",
    "default_welch_params",
]
