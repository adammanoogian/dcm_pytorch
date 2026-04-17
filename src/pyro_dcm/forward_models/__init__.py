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
    compute_effective_A,
    parameterize_A,
    parameterize_B,
)
from pyro_dcm.forward_models.spectral_noise import (
    default_noise_priors,
    neuronal_noise_csd,
    observation_noise_csd,
)
from pyro_dcm.forward_models.rdcm_forward import (
    compute_derivative_coefficients,
    create_regressors,
    euler_integrate_dcm,
    generate_bold,
    get_hrf,
    split_real_imag,
)
from pyro_dcm.forward_models.rdcm_posterior import (
    compute_free_energy_rigid,
    compute_free_energy_sparse,
    compute_rdcm_likelihood,
    get_priors_rigid,
    get_priors_sparse,
    rigid_inversion,
    sparse_inversion,
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
    "compute_effective_A",
    "parameterize_A",
    "parameterize_B",
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
    # Phase 3: Regression DCM forward pipeline
    "compute_derivative_coefficients",
    "create_regressors",
    "euler_integrate_dcm",
    "generate_bold",
    "get_hrf",
    "split_real_imag",
    # Phase 3: Regression DCM analytic posterior
    "compute_free_energy_rigid",
    "compute_free_energy_sparse",
    "compute_rdcm_likelihood",
    "get_priors_rigid",
    "get_priors_sparse",
    "rigid_inversion",
    "sparse_inversion",
]
