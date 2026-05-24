"""Pyro generative model for latent-circuit Dynamic Causal Modeling (v0.6.0).

Implements the probabilistic generative process for fitting a bilinear DCM
to latent-state trajectories extracted from a trained RNN (e.g., via PCA on
network hidden states). The model operates in direct neural-activity space --
there is no hemodynamic model, no BOLD signal, and no TR-based downsampling.

Key differences from ``task_dcm_model``:

- **No hemodynamics.** State vector is neural activity x(t) shape (N,), not
  the 5N hemodynamic state. ``CoupledDCMSystem(hemodynamic=False)`` is used.
- **No downsampling.** Predictions are at the fine ODE grid (dt=0.01).
- **Identity observation matrix.** C_obs = I_N for v0.6.0 (pitfall LC5 --
  rotation ambiguity deferred to v0.7.0+).
- **Separate prior constants.** ``LC_A_PRIOR_VARIANCE`` and
  ``LC_B_PRIOR_VARIANCE`` are distinct from the BOLD-calibrated priors in
  ``task_dcm_model`` (pitfall LC4 -- prior scale mismatch).
- **Initial state zeros(N).** Not ``make_initial_state(N)`` which returns 5N.

References
----------
[REF-001] Friston, Harrison & Penny (2003), Eq. 1 -- Bilinear neural state
    equation: dx/dt = (A + sum_j u_j B_j) @ x + C @ u(t).
[REF-040] Friston et al. (2007) -- Variational free energy / Laplace.
.planning/phases/20-latent-circuit-forward-model/20-CONTEXT.md --
    C_obs identity constraint (LC5), prior recalibration (LC4).
.planning/STATE.md -- Decision: LC_A_PRIOR_VARIANCE separate from BOLD priors.
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


LC_A_PRIOR_VARIANCE: float = 1.0 / 16.0
"""Prior variance on A_free elements for latent-circuit DCM.

Calibrated for RNN latent-state timescales. RNN hidden states evolve
on sub-second timescales; the 1/64 BOLD-calibrated prior from
``task_dcm_model`` is too narrow for these faster dynamics. 1/16 allows
connectivity values up to ~1.0 at 2 sigma, appropriate for latent circuits
with self-inhibition of 1.0 Hz and off-diagonal strengths of 0.2 (see
``make_stable_latent_circuit_A``).

Separate from the BOLD A_free prior (1/64) to address pitfall LC4 (prior
scale mismatch between BOLD and latent-circuit timescales).

References
----------
.planning/phases/20-latent-circuit-forward-model/20-CONTEXT.md -- LC4.
.planning/STATE.md -- Decision [20-01-D4] self_inhibition=1.0 Hz.
"""

LC_B_PRIOR_VARIANCE: float = 1.0
"""Prior variance on B_free elements for latent-circuit DCM.

Matches the task-DCM B_prior convention (SPM12 pC.B = 1.0) for the bilinear
modulatory path. Latent-circuit B matrices have the same structural role as
task-DCM B matrices, and no empirical recalibration data yet motivates a
different prior (see D1 in ``task_dcm_model`` module-level docstring).

References
----------
SPM12 ``spm_dcm_fmri_priors.m`` -- pC.B specification for one-state DCM.
.planning/STATE.md -- Decision D1 (2026-04-17), pC.B = 1.0.
"""


def _validate_lc_bilinear_args(
    b_masks: list[torch.Tensor],
    stim_mod: object,
    N: int,
) -> None:
    """Validate bilinear branch kwargs for latent-circuit model.

    Mirrors the validation in ``task_dcm_model._validate_bilinear_args``
    for the latent-circuit model. Called before any ``pyro.sample`` site
    so errors are not wrapped inside a Pyro trace stack.

    Parameters
    ----------
    b_masks : list of torch.Tensor
        Non-empty list of per-modulator structural masks; each must be
        shape ``(N, N)``.
    stim_mod : PiecewiseConstantInput
        Modulator input with ``.values`` attribute of shape ``(K, J)``
        where ``J == len(b_masks)``.
    N : int
        Number of latent dimensions.

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
            "latent_circuit_dcm_model: stim_mod is required when b_masks is "
            "non-empty; got None. Construct stim_mod via make_block_stimulus "
            "or similar, then wrap in PiecewiseConstantInput."
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
            f"latent_circuit_dcm_model: stim_mod.values.shape[1]={J_stim} "
            f"must equal len(b_masks)={len(b_masks)}."
        )


def latent_circuit_dcm_model(
    observed_trajectories: torch.Tensor,
    stimulus: object,
    a_mask: torch.Tensor,
    c_mask: torch.Tensor,
    t_eval: torch.Tensor,
    dt: float = 0.01,
    *,
    b_masks: list[torch.Tensor] | None = None,
    stim_mod: object | None = None,
) -> None:
    """Pyro generative model for latent-circuit DCM.

    Defines the probabilistic generative process for fitting a bilinear DCM
    to latent-state trajectories from a trained RNN [REF-001] Eq. 1:

    1. Sample A_free ~ N(0, LC_A_PRIOR_VARIANCE), apply mask, parameterize_A.
    2. Sample C ~ N(0, 1), apply mask.
    3. (Bilinear branch only.) Sample per-modulator
       ``B_free_j ~ N(0, LC_B_PRIOR_VARIANCE)``, apply b_mask_j via
       ``parameterize_B``, stack into ``(J, N, N)``.
    4. Run CoupledDCMSystem(hemodynamic=False) ODE forward model (rk4).
    5. Apply ``direct_observation`` with identity C_obs (v0.6.0).
    6. Evaluate Gaussian likelihood on observed trajectories.

    No hemodynamic model, no BOLD signal, no TR-based downsampling.
    The ODE integrates the neural state equation directly at dt=0.01s
    resolution.

    Parameters
    ----------
    observed_trajectories : torch.Tensor
        Observed latent-state trajectories, shape ``(T, N)`` where T is
        the number of time points at dt resolution and N is the number of
        latent dimensions. dtype must be ``torch.float64``.
    stimulus : PiecewiseConstantInput or dict
        Piecewise-constant driving stimulus function. If dict, must have
        ``'times'`` and ``'values'`` keys. If a
        ``PiecewiseConstantInput``, used directly.
    a_mask : torch.Tensor
        Binary structural mask for A matrix, shape ``(N, N)``.
        1 where connection exists, 0 where absent.
        dtype must be ``torch.float64``.
    c_mask : torch.Tensor
        Binary structural mask for C matrix, shape ``(N, M)``.
        1 where driving input exists, 0 where absent.
        dtype must be ``torch.float64``.
    t_eval : torch.Tensor
        Fine time grid for ODE integration, shape ``(T,)``.
        Must match spacing ``dt``. dtype must be ``torch.float64``.
    dt : float, optional
        ODE integration step size in seconds. Default 0.01.
        Latent-circuit trajectories evolve on sub-second timescales;
        dt=0.01 is appropriate (vs dt=0.5 for BOLD in task_dcm_model).
        Must match the spacing of ``t_eval``.
    b_masks : list of torch.Tensor or None, optional
        Per-modulator structural masks for the bilinear B path. Each
        element is a binary ``(N, N)`` mask. None (default) and ``[]``
        both activate the linear short-circuit. When non-empty,
        ``stim_mod`` is required.
    stim_mod : PiecewiseConstantInput or dict or None, optional
        Modulator input function with ``.values`` shape ``(K, J)`` where
        ``J == len(b_masks)``. Only consulted when ``b_masks`` is
        non-empty.

    Notes
    -----
    **Prior specifications** (latent-circuit recalibrated, pitfall LC4):

    - A_free ~ N(0, LC_A_PRIOR_VARIANCE) = N(0, 1/16) element-wise.
      Wider than BOLD 1/64 to match RNN hidden-state timescales.
    - C ~ N(0, 1) element-wise (same as task_dcm_model).
    - (Bilinear branch only.) B_free_j ~ N(0, LC_B_PRIOR_VARIANCE) = N(0, 1).
    - noise_prec ~ Gamma(1, 1), weakly informative.

    **Observation model (v0.6.0).** C_obs is fixed at identity I_N so
    y_mean = x exactly (pitfall LC5 -- rotation ambiguity avoided; learned
    C_obs deferred to v0.7.0+). ``direct_observation(x, C_obs, noise_prec)``
    is called for the y_mean computation.

    **Sample site naming** (identical to task_dcm_model for guide
    auto-discovery): A_free, C, B_free_j (per modulator j), noise_prec, obs.
    Deterministic sites: A, predicted_trajectories, B (bilinear branch only).

    **NaN guard.** Same as task_dcm_model: when predicted trajectories
    contain NaN or Inf, they are zero-filled before the likelihood site to
    produce a large finite ELBO penalty with zero gradient.

    **Guide auto-discovery.** No factory changes are needed for
    AutoNormal, AutoLowRankMVN, or AutoIAFNormal because all sample sites
    are emitted via standard ``pyro.sample`` calls with .to_event(2) for
    matrix parameters and scalar for noise_prec. This matches the pattern
    in task_dcm_model (MODEL-06).

    References
    ----------
    [REF-001] Friston, Harrison & Penny (2003), Eq. 1 -- Neural state
        equation dx/dt = (A + sum_j u_j B_j) @ x + C @ u(t).
    [REF-040] Friston et al. (2007) -- Variational free energy / Laplace.
    .planning/phases/20-latent-circuit-forward-model/20-CONTEXT.md --
        C_obs identity constraint (LC5), prior recalibration (LC4).

    Examples
    --------
    >>> import torch
    >>> from pyro_dcm.models import create_guide, run_svi
    >>> from pyro_dcm.models.latent_circuit_dcm_model import latent_circuit_dcm_model
    >>> traj = torch.randn(200, 4, dtype=torch.float64)
    >>> guide = create_guide(latent_circuit_dcm_model, init_scale=0.01)
    >>> # result = run_svi(latent_circuit_dcm_model, guide,
    >>> #     model_args=(traj, stim, a_mask, c_mask, t_eval, 0.01))
    """
    # --- Extract dimensions ---
    N = a_mask.shape[0]
    M = c_mask.shape[1]
    T = observed_trajectories.shape[0]

    # Normalize len-0 b_masks to None (mirrors task_dcm_model linear gate).
    if b_masks is not None and len(b_masks) == 0:
        b_masks = None

    # Bilinear-mode validation; no-op in linear short-circuit.
    if b_masks is not None:
        _validate_lc_bilinear_args(b_masks, stim_mod, N)

    # --- Sample A_free: connectivity free parameters ---
    # Prior: N(0, LC_A_PRIOR_VARIANCE) -- wider than BOLD 1/64 (pitfall LC4)
    A_free_prior = dist.Normal(
        torch.zeros(N, N, dtype=torch.float64),
        LC_A_PRIOR_VARIANCE ** 0.5 * torch.ones(N, N, dtype=torch.float64),
    ).to_event(2)
    A_free = pyro.sample("A_free", A_free_prior)
    A_free = A_free * a_mask  # Zero absent connections

    # Deterministic transform: guarantees negative diagonal [REF-001]
    A = pyro.deterministic("A", parameterize_A(A_free))

    # --- Sample C: driving input weights ---
    # Prior: N(0, 1) matching task_dcm_model convention [REF-001]
    C_prior = dist.Normal(
        torch.zeros(N, M, dtype=torch.float64),
        torch.ones(N, M, dtype=torch.float64),
    ).to_event(2)
    C = pyro.sample("C", C_prior)
    C = C * c_mask  # Zero absent inputs

    # --- Bilinear B sampling. Active only when b_masks non-empty. ---
    # Naming convention is identical to task_dcm_model for guide
    # auto-discovery compatibility (MODEL-06).
    B_stacked: torch.Tensor | None = None
    merged_input_fn: object | None = None
    if b_masks is not None:
        B_prior_std = LC_B_PRIOR_VARIANCE ** 0.5
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
        # Emit deterministic site only in bilinear branch (mirrors task_dcm).
        pyro.deterministic("B", B_stacked)

        # Merge driving + modulator inputs into widened input function.
        drive_input = (
            stimulus if isinstance(stimulus, PiecewiseConstantInput)
            else PiecewiseConstantInput(stimulus["times"], stimulus["values"])
        )
        mod_input = (
            stim_mod if isinstance(stim_mod, PiecewiseConstantInput)
            else PiecewiseConstantInput(stim_mod["times"], stim_mod["values"])
        )
        merged_input_fn = merge_piecewise_inputs(drive_input, mod_input)

    # --- Forward model: latent-circuit ODE (hemodynamic=False) ---
    # [REF-001] Eq. 1: dx/dt = (A + sum_j u_j B_j) @ x + C @ u(t)
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
        # Linear short-circuit: no B kwarg, bit-exact linear path.
        # stimulus accepted as PiecewiseConstantInput or dict.
        if isinstance(stimulus, PiecewiseConstantInput):
            input_fn = stimulus
        else:
            input_fn = PiecewiseConstantInput(
                stimulus["times"], stimulus["values"]
            )
        system = CoupledDCMSystem(A, C, input_fn, hemodynamic=False)

    # Initial state: zeros(N) for neural activity at rest.
    # NOT make_initial_state(N) -- that returns 5N for hemodynamic mode.
    # [Decision 20-01-D3]
    y0 = torch.zeros(N, dtype=torch.float64)
    solution = integrate_ode(
        system, y0, t_eval, method="rk4", step_size=dt,
    )
    # solution shape: (T, N) -- latent states only

    # Predicted trajectories at fine ODE resolution (no downsampling).
    predicted_trajectories = solution[:T]

    # --- NaN-safe guard (mirrors task_dcm_model pattern) ---
    # Bilinear early-SVI draws can push max Re(eig(A_eff)) > 0, yielding
    # NaN/Inf in the predicted trajectories. Zero-fill produces a large
    # finite ELBO penalty with zero gradient (pattern from amortized_wrappers).
    if (
        torch.isnan(predicted_trajectories).any()
        or torch.isinf(predicted_trajectories).any()
    ):
        predicted_trajectories = torch.zeros_like(
            predicted_trajectories
        ).detach()
    pyro.deterministic("predicted_trajectories", predicted_trajectories)

    # --- Noise precision (weakly informative prior) ---
    noise_prec = pyro.sample(
        "noise_prec",
        dist.Gamma(
            torch.tensor(1.0, dtype=torch.float64),
            torch.tensor(1.0, dtype=torch.float64),
        ),
    )

    # --- Direct observation: identity C_obs for v0.6.0 (pitfall LC5) ---
    # C_obs = I_N is fixed (not sampled); rotation ambiguity deferred to v0.7.0+.
    C_obs = torch.eye(N, dtype=torch.float64)
    y_mean, noise_std = direct_observation(
        predicted_trajectories, C_obs, noise_prec
    )

    # --- Gaussian likelihood on observed trajectories [REF-001] ---
    # .to_event(2) treats full (T, N) matrix as a single observation.
    pyro.sample(
        "obs",
        dist.Normal(y_mean, noise_std).to_event(2),
        obs=observed_trajectories,
    )
