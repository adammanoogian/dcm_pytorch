"""Pyro generative model for task-based Dynamic Causal Modeling.

Implements the full generative process for task-DCM: sampling
connectivity parameters (A_free, C) from Normal priors, applying
structural masking and the parameterize_A transform, running the
coupled neural-hemodynamic ODE forward model, computing BOLD signal,
and evaluating a Gaussian likelihood on observed BOLD data.

Hemodynamic parameters are FIXED at SPM12 defaults (not sampled).

References
----------
[REF-001] Friston, Harrison & Penny (2003), Eq. 1 -- Neural state equation.
[REF-002] Stephan et al. (2007), Eq. 2-6 -- Balloon-Windkessel + BOLD.
[REF-040] Friston et al. (2007) -- Variational free energy / Laplace.
SPM12 source: spm_dcm_fmri_priors.m -- Prior specifications.
"""

from __future__ import annotations

import torch
import pyro
import pyro.distributions as dist

from pyro_dcm.forward_models.bold_signal import bold_signal
from pyro_dcm.forward_models.coupled_system import CoupledDCMSystem
from pyro_dcm.forward_models.neural_state import parameterize_A
from pyro_dcm.utils.ode_integrator import integrate_ode, make_initial_state


def task_dcm_model(
    observed_bold: torch.Tensor,
    stimulus: object,
    a_mask: torch.Tensor,
    c_mask: torch.Tensor,
    t_eval: torch.Tensor,
    TR: float,
    dt: float = 0.5,
) -> None:
    """Pyro generative model for task-based DCM.

    Defines the probabilistic generative process [REF-001] Eq. 1,
    [REF-002] Eq. 2-6:

    1. Sample A_free ~ N(0, 1/64), apply mask and parameterize_A.
    2. Sample C ~ N(0, 1), apply mask.
    3. Run coupled neural-hemodynamic ODE integration (rk4 fixed-step).
    4. Extract BOLD signal from hemodynamic states.
    5. Downsample predicted BOLD to TR resolution.
    6. Evaluate Gaussian likelihood on observed BOLD.

    Hemodynamic parameters (kappa, gamma, tau, alpha, E0) are FIXED
    at SPM12 defaults -- they are NOT sampled.

    Parameters
    ----------
    observed_bold : torch.Tensor
        Observed BOLD time series, shape ``(T, N)`` where T is the
        number of TR-resolution time points and N is the number of
        brain regions. dtype must be ``torch.float64``.
    stimulus : PiecewiseConstantInput
        Piecewise-constant stimulus function mapping time to input
        vector u(t). From ``pyro_dcm.utils.ode_integrator``.
    a_mask : torch.Tensor
        Binary structural mask for A matrix, shape ``(N, N)``.
        1 where connection exists, 0 where absent.
        dtype must be ``torch.float64``.
    c_mask : torch.Tensor
        Binary structural mask for C matrix, shape ``(N, M)``.
        1 where driving input exists, 0 where absent.
        dtype must be ``torch.float64``.
    t_eval : torch.Tensor
        Fine time grid for ODE integration, shape ``(T_fine,)``.
        dtype must be ``torch.float64``.
    TR : float
        Repetition time in seconds for BOLD downsampling.
    dt : float, optional
        ODE integration step size in seconds. Default 0.5, chosen
        for SVI efficiency (see 04-RESEARCH.md Open Question 4).
        Must match the spacing of ``t_eval``.

    Notes
    -----
    Prior specifications follow SPM12 ``spm_dcm_fmri_priors.m``:

    - A_free ~ N(0, 1/64) element-wise, then masked and transformed
      via ``parameterize_A`` which maps diagonal to -exp(free)/2
      [REF-001].
    - C ~ N(0, 1) element-wise, then masked [REF-001].
    - Noise precision ~ Gamma(1, 1), weakly informative.
    - Hemodynamic parameters fixed at SPM12 code defaults.

    The ODE integration uses the rk4 fixed-step method for
    predictable runtime during SVI optimization. Adaptive methods
    (dopri5) can cause variable computation graphs across SVI steps.

    Anti-patterns avoided:

    - Hemodynamic parameters are NOT sampled.
    - Adjoint method is NOT used (gradient reliability).
    - ``pyro.plate`` is NOT used around ODE time steps.
    - Complex tensors are NOT passed to ``pyro.sample``.

    References
    ----------
    [REF-001] Friston, Harrison & Penny (2003), Eq. 1.
    [REF-002] Stephan et al. (2007), Eq. 2-6.
    [REF-040] Friston et al. (2007) -- ELBO / free energy.

    Examples
    --------
    >>> import torch
    >>> from pyro_dcm.models import task_dcm_model, create_guide, run_svi
    >>> bold = torch.randn(150, 2, dtype=torch.float64)
    >>> guide = create_guide(task_dcm_model, init_scale=0.01)
    >>> # result = run_svi(task_dcm_model, guide,
    >>> #     model_args=(bold, stim, a_mask, c_mask, t_eval, 2.0, 0.5))
    """
    # --- Extract dimensions ---
    N = a_mask.shape[0]
    M = c_mask.shape[1]
    T = observed_bold.shape[0]

    # --- Sample A_free: connectivity free parameters ---
    # Prior: N(0, 1/64) matching SPM12 spm_dcm_fmri_priors.m
    A_free_prior = dist.Normal(
        torch.zeros(N, N, dtype=torch.float64),
        (1.0 / 64.0) ** 0.5 * torch.ones(N, N, dtype=torch.float64),
    ).to_event(2)
    A_free = pyro.sample("A_free", A_free_prior)
    A_free = A_free * a_mask  # Zero absent connections

    # Deterministic transform: guarantees negative diagonal [REF-001]
    A = pyro.deterministic("A", parameterize_A(A_free))

    # --- Sample C: driving input weights ---
    # Prior: N(0, 1) matching SPM12
    C_prior = dist.Normal(
        torch.zeros(N, M, dtype=torch.float64),
        torch.ones(N, M, dtype=torch.float64),
    ).to_event(2)
    C = pyro.sample("C", C_prior)
    C = C * c_mask  # Zero absent inputs

    # --- Forward model (deterministic computation) ---
    # Hemodynamic params: FIXED at SPM defaults (hemo_params=None)
    system = CoupledDCMSystem(A, C, stimulus)
    y0 = make_initial_state(N, dtype=torch.float64)
    solution = integrate_ode(
        system, y0, t_eval, method="rk4", step_size=dt,
    )
    # solution shape: (T_fine, 5*N)

    # --- Extract BOLD from hemodynamic states [REF-002] Eq. 6 ---
    lnv = solution[:, 3 * N : 4 * N]  # log blood volume
    lnq = solution[:, 4 * N : 5 * N]  # log deoxyhemoglobin
    v = torch.exp(lnv)
    q = torch.exp(lnq)
    bold_fine = bold_signal(v, q)  # shape: (T_fine, N)

    # --- Downsample to TR resolution ---
    step = round(TR / dt)
    predicted_bold = bold_fine[::step][:T]
    pyro.deterministic("predicted_bold", predicted_bold)

    # --- Noise precision (weakly informative prior) ---
    noise_prec = pyro.sample(
        "noise_prec",
        dist.Gamma(
            torch.tensor(1.0, dtype=torch.float64),
            torch.tensor(1.0, dtype=torch.float64),
        ),
    )
    noise_std = (1.0 / noise_prec).sqrt()

    # --- Gaussian likelihood on observed BOLD [REF-002] ---
    # .to_event(2) treats full (T, N) matrix as single observation
    pyro.sample(
        "obs",
        dist.Normal(predicted_bold, noise_std).to_event(2),
        obs=observed_bold,
    )
