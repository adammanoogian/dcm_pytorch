"""Pyro generative model for latent circuit Dynamic Causal Modeling.

Implements the probabilistic generative process for fitting bilinear
DCM to neural state trajectories with identity observation (no
hemodynamics). This is a FORK of the task_dcm_model pattern, using
``CoupledDCMSystem(hemodynamic=False)`` for direct observation of
neural states.

Key differences from task_dcm_model:
1. No hemodynamic parameters (no Balloon-Windkessel, no BOLD).
2. No TR downsampling -- observations are at the integration grid.
3. Prior constants calibrated for RNN-scale dynamics (wider than BOLD).
4. ``CoupledDCMSystem(hemodynamic=False)`` -- state is N-dim.
5. Observation via ``direct_observation`` with identity C_obs.

References
----------
[REF-001] Friston, Harrison & Penny (2003), Eq. 1 -- Neural state equation.
OBS-02 REQUIREMENTS-v0.6.0.md -- Standalone observation function.
MODEL-01/02/04 REQUIREMENTS-v0.6.0.md -- Pyro model requirements.
"""

from __future__ import annotations

import torch
import pyro
import pyro.distributions as dist

from pyro_dcm.forward_models.coupled_system import CoupledDCMSystem
from pyro_dcm.forward_models.latent_observation import direct_observation
from pyro_dcm.forward_models.neural_state import parameterize_A, parameterize_B
from pyro_dcm.utils.ode_integrator import (
    PiecewiseConstantInput,
    integrate_ode,
    merge_piecewise_inputs,
)


LC_A_PRIOR_VARIANCE: float = 1 / 16
"""Prior variance on A_free elements for latent circuit DCM.

Set to 1/16 as initial calibration for RNN-scale dynamics (wider than
SPM12's task-DCM 1/64 because direct neural trajectories have larger
magnitudes than percent-signal-change BOLD). Subject to empirical
recalibration in Plan 20-05 via joint prior_var x init_scale sweep
on cluster (addresses pitfall LC4 from v0.6.0 requirements).

See also: task_dcm_model.A_PRIOR_VARIANCE (1/64, BOLD-scale).
"""

LC_B_PRIOR_VARIANCE: float = 1.0
"""Prior variance on B_free elements for latent circuit DCM.

Matches task-DCM B_PRIOR_VARIANCE as initial value; subject to
recalibration. The Phase 16.1 lesson showed init_scale x prior_var
interaction is the primary failure mode for B recovery.
"""


def _validate_bilinear_args(
    b_masks: list[torch.Tensor],
    stim_mod: object,
    N: int,
) -> None:
    """Validate bilinear branch kwargs; raise on malformed inputs.

    Called inside ``latent_circuit_dcm_model`` only when ``b_masks``
    is non-empty. Validation runs BEFORE any ``pyro.sample`` call.

    Parameters
    ----------
    b_masks : list of torch.Tensor
        Non-empty list of per-modulator structural masks; each must be
        shape ``(N, N)``.
    stim_mod : PiecewiseConstantInput
        Modulator input with ``.values`` attribute of shape ``(K, J)``
        where ``J == len(b_masks)``.
    N : int
        Number of brain regions.

    Raises
    ------
    ValueError
        If ``stim_mod`` is None; if any ``b_masks[j].shape != (N, N)``;
        if ``len(b_masks) != stim_mod.values.shape[1]``.
    TypeError
        If ``stim_mod`` lacks a ``.values`` attribute.
    """
    if stim_mod is None:
        raise ValueError(
            "latent_circuit_dcm_model: stim_mod is required when "
            "b_masks is non-empty; got None."
        )
    if not hasattr(stim_mod, "values"):
        raise TypeError(
            "latent_circuit_dcm_model: stim_mod must be a "
            "PiecewiseConstantInput (with .values attr); got "
            f"{type(stim_mod).__name__}."
        )
    for j, m in enumerate(b_masks):
        if m.shape != (N, N):
            raise ValueError(
                f"latent_circuit_dcm_model: b_masks[{j}].shape="
                f"{tuple(m.shape)} must equal (N, N)=({N}, {N})."
            )
    J_stim = stim_mod.values.shape[1]
    if J_stim != len(b_masks):
        raise ValueError(
            f"latent_circuit_dcm_model: stim_mod.values.shape[1]="
            f"{J_stim} must equal len(b_masks)={len(b_masks)}."
        )


def latent_circuit_dcm_model(
    observed_trajectories: torch.Tensor,
    stimulus: PiecewiseConstantInput,
    a_mask: torch.Tensor,
    c_mask: torch.Tensor,
    t_eval: torch.Tensor,
    dt: float = 0.01,
    *,
    b_masks: list[torch.Tensor] | None = None,
    stim_mod: PiecewiseConstantInput | None = None,
) -> None:
    """Pyro generative model for latent circuit DCM.

    Defines the probabilistic generative process for fitting bilinear
    DCM directly to neural state trajectories [REF-001] Eq. 1:

    1. Sample A_free ~ N(0, LC_A_PRIOR_VARIANCE), apply mask and
       parameterize_A.
    2. Sample C ~ N(0, 1), apply mask.
    3. (Bilinear branch only) Sample per-modulator
       ``B_free_j ~ N(0, LC_B_PRIOR_VARIANCE)``, apply b_mask_j via
       ``parameterize_B``, stack into ``(J, N, N)``.
    4. Run neural ODE integration with CoupledDCMSystem(hemodynamic=False).
    5. Apply direct_observation (identity C_obs for v0.6.0).
    6. Evaluate Gaussian likelihood on observed trajectories.

    No hemodynamic parameters, no BOLD signal, no TR downsampling.

    Parameters
    ----------
    observed_trajectories : torch.Tensor
        Observed neural state trajectories, shape ``(T, N)`` where T
        is the number of time points and N is the number of latent
        dimensions. dtype must be ``torch.float64``.
    stimulus : PiecewiseConstantInput
        Piecewise-constant driving stimulus function mapping time to
        input vector u(t).
    a_mask : torch.Tensor
        Binary structural mask for A matrix, shape ``(N, N)``.
        1 where connection exists, 0 where absent.
    c_mask : torch.Tensor
        Binary structural mask for C matrix, shape ``(N, M)``.
        1 where driving input exists, 0 where absent.
    t_eval : torch.Tensor
        Fine time grid for ODE integration, shape ``(T_fine,)``.
        Must have spacing matching ``dt``.
    dt : float, optional
        ODE integration step size in seconds. Default 0.01 for
        direct observation (finer than task-DCM's 0.5).
    b_masks : list of torch.Tensor or None, optional
        Per-modulator structural masks for the bilinear B path. Each
        element is a binary ``(N, N)`` mask. ``None`` (default) and
        ``[]`` both activate the linear short-circuit.
    stim_mod : PiecewiseConstantInput or None, optional
        Modulator input function with ``.values`` shape ``(K, J)``
        where ``J == len(b_masks)``. Required when ``b_masks`` is
        non-empty.

    Notes
    -----
    Prior specifications for latent circuit DCM:

    - A_free ~ N(0, LC_A_PRIOR_VARIANCE=1/16) -- wider than task-DCM's
      1/64 because RNN-scale dynamics have larger magnitudes.
    - C ~ N(0, 1) -- same as task-DCM.
    - B_free_j ~ N(0, LC_B_PRIOR_VARIANCE=1.0) -- same as task-DCM,
      pending recalibration.
    - noise_prec ~ Gamma(1, 1) -- weakly informative.
    - C_obs = identity (not sampled) for v0.6.0.

    The ODE integration uses rk4 fixed-step for predictable runtime
    during SVI optimization.

    References
    ----------
    [REF-001] Friston, Harrison & Penny (2003), Eq. 1.
    OBS-02 REQUIREMENTS-v0.6.0.md -- direct_observation function.
    MODEL-01 REQUIREMENTS-v0.6.0.md -- Pyro model with proper priors.
    MODEL-02 REQUIREMENTS-v0.6.0.md -- Bilinear B sampling.
    MODEL-04 REQUIREMENTS-v0.6.0.md -- Identity C_obs.
    """
    # --- Extract dimensions ---
    N = a_mask.shape[0]
    M = c_mask.shape[1]
    T = observed_trajectories.shape[0]

    # Normalize len-0 b_masks to None for linear short-circuit.
    if b_masks is not None and len(b_masks) == 0:
        b_masks = None

    # Bilinear-mode validation; no-op in linear short-circuit.
    if b_masks is not None:
        _validate_bilinear_args(b_masks, stim_mod, N)

    # --- Sample A_free: connectivity free parameters ---
    # Prior: N(0, LC_A_PRIOR_VARIANCE) -- wider than task-DCM 1/64
    A_free_prior = dist.Normal(
        torch.zeros(N, N, dtype=torch.float64),
        LC_A_PRIOR_VARIANCE**0.5
        * torch.ones(N, N, dtype=torch.float64),
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

    # --- Bilinear B sampling (MODEL-02). Active only when b_masks non-empty.
    B_stacked: torch.Tensor | None = None
    merged_input_fn: PiecewiseConstantInput | None = None
    if b_masks is not None:
        B_prior_std = LC_B_PRIOR_VARIANCE**0.5
        B_free_list: list[torch.Tensor] = []
        for j, b_mask_j in enumerate(b_masks):
            B_free_j = pyro.sample(
                f"B_free_{j}",
                dist.Normal(
                    torch.zeros_like(b_mask_j),
                    B_prior_std * torch.ones_like(b_mask_j),
                ).to_event(2),
            )
            B_free_list.append(B_free_j)
        # Stack along modulator axis -> (J, N, N).
        B_free_stacked = torch.stack(B_free_list, dim=0)
        b_mask_stacked = torch.stack(list(b_masks), dim=0)
        B_stacked = parameterize_B(B_free_stacked, b_mask_stacked)
        pyro.deterministic("B", B_stacked)

        # Merge driving + modulator inputs for CoupledDCMSystem.
        drive_input = (
            stimulus
            if isinstance(stimulus, PiecewiseConstantInput)
            else PiecewiseConstantInput(
                stimulus["times"], stimulus["values"]
            )
        )
        mod_input = (
            stim_mod
            if isinstance(stim_mod, PiecewiseConstantInput)
            else PiecewiseConstantInput(
                stim_mod["times"], stim_mod["values"]
            )
        )
        merged_input_fn = merge_piecewise_inputs(drive_input, mod_input)

    # --- Forward model (deterministic computation) ---
    if B_stacked is not None:
        system = CoupledDCMSystem(
            A,
            C,
            merged_input_fn,
            hemodynamic=False,
            B=B_stacked,
            n_driving_inputs=c_mask.shape[1],
        )
    else:
        system = CoupledDCMSystem(
            A, C, stimulus, hemodynamic=False
        )

    # Initial state: zeros for neural activity (no hemodynamic states)
    y0 = torch.zeros(N, dtype=torch.float64)

    # Integrate ODE with rk4 fixed-step
    solution = integrate_ode(
        system, y0, t_eval, method="rk4", step_size=dt,
    )
    # solution shape: (T_fine, N)

    # --- Trim to match observed length (no TR downsampling) ---
    predicted = solution[:T]

    # NaN-safe guard: unstable dynamics can produce NaN/Inf.
    # Detach + zero-fill produces large finite penalty with zero
    # gradient, preventing NaN ELBO from halting SVI.
    if torch.isnan(predicted).any() or torch.isinf(predicted).any():
        predicted = torch.zeros_like(predicted).detach()
    pyro.deterministic("predicted_trajectories", predicted)

    # --- Observation via direct_observation (OBS-02) ---
    # For v0.6.0, C_obs is identity (not sampled). MODEL-04.
    C_obs = torch.eye(N, dtype=torch.float64)

    # --- Noise precision (weakly informative prior) ---
    noise_prec = pyro.sample(
        "noise_prec",
        dist.Gamma(
            torch.tensor(1.0, dtype=torch.float64),
            torch.tensor(1.0, dtype=torch.float64),
        ),
    )

    # Compute observation mean and noise std
    y_mean, noise_std = direct_observation(predicted, C_obs, noise_prec)

    # --- Gaussian likelihood on observed trajectories ---
    # .to_event(2) treats full (T, N) matrix as single observation
    pyro.sample(
        "obs",
        dist.Normal(y_mean, noise_std).to_event(2),
        obs=observed_trajectories,
    )
