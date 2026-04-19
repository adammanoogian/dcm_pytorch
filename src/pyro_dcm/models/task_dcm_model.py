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
from pyro_dcm.forward_models.neural_state import parameterize_A, parameterize_B
from pyro_dcm.utils.ode_integrator import (
    PiecewiseConstantInput,
    integrate_ode,
    make_initial_state,
    merge_piecewise_inputs,
)


B_PRIOR_VARIANCE: float = 1.0
"""Prior variance on B_free elements in the task-DCM bilinear branch.

Locks D1 from the v0.3.0 milestone decisions: ``B_free ~ Normal(0, 1.0)`` per
SPM12 one-state DCM convention (``spm_dcm_fmri_priors.m`` pC.B = B). Corrects
the factually wrong YAML claim of "1/16 SPM12 convention" (audited in v0.3.0
PITFALLS.md Section B8).

Future one-state vs two-state prior alternatives are tracked in
REQUIREMENTS.md future-candidate BILIN-08 (v0.4.0+ scope).

References
----------
SPM12 ``spm_dcm_fmri_priors.m`` -- pC.B specification for one-state DCM.
.planning/STATE.md D1 -- milestone decision 2026-04-17.
.planning/research/v0.3.0/PITFALLS.md Section B8 -- YAML audit.
"""


def _validate_bilinear_args(
    b_masks: list[torch.Tensor],
    stim_mod: object,
    N: int,
) -> None:
    """Validate bilinear branch kwargs; raise on malformed inputs.

    Called inside ``task_dcm_model`` only when ``b_masks is not None`` and
    ``len(b_masks) > 0``. Validation runs BEFORE any ``pyro.sample`` call
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
        Number of brain regions (from ``a_mask.shape[0]``).

    Raises
    ------
    ValueError
        If ``stim_mod`` is None; if any ``b_masks[j].shape != (N, N)``;
        if ``len(b_masks) != stim_mod.values.shape[1]``.
    TypeError
        If ``stim_mod`` lacks a ``.values`` attribute (type-narrowing
        for ``PiecewiseConstantInput``).

    Notes
    -----
    Mirrors the validation policy from ``simulate_task_dcm`` (Phase 14) and
    ``_normalize_B_list`` (Phase 14). Error messages include the offending
    index and actual-vs-expected values per CLAUDE.md "Error messages must
    include expected vs actual values" convention.
    """
    if stim_mod is None:
        raise ValueError(
            "task_dcm_model: stim_mod is required when b_masks is non-empty; "
            "got None. Construct stim_mod via make_epoch_stimulus (preferred "
            "per v0.3.0 Pitfall B12) or make_event_stimulus, then wrap in "
            "PiecewiseConstantInput."
        )
    if not hasattr(stim_mod, "values"):
        raise TypeError(
            "task_dcm_model: stim_mod must be a PiecewiseConstantInput "
            f"(with .values attr); got {type(stim_mod).__name__}."
        )
    for j, m in enumerate(b_masks):
        if m.shape != (N, N):
            raise ValueError(
                f"task_dcm_model: b_masks[{j}].shape={tuple(m.shape)} must "
                f"equal (N, N)=({N}, {N})."
            )
    J_stim = stim_mod.values.shape[1]
    if J_stim != len(b_masks):
        raise ValueError(
            f"task_dcm_model: stim_mod.values.shape[1]={J_stim} must equal "
            f"len(b_masks)={len(b_masks)}."
        )


def task_dcm_model(
    observed_bold: torch.Tensor,
    stimulus: object,
    a_mask: torch.Tensor,
    c_mask: torch.Tensor,
    t_eval: torch.Tensor,
    TR: float,
    dt: float = 0.5,
    *,
    b_masks: list[torch.Tensor] | None = None,
    stim_mod: object | None = None,
) -> None:
    """Pyro generative model for task-based DCM.

    Defines the probabilistic generative process [REF-001] Eq. 1,
    [REF-002] Eq. 2-6:

    1. Sample A_free ~ N(0, 1/64), apply mask and parameterize_A.
    2. Sample C ~ N(0, 1), apply mask.
    3. (v0.3.0 bilinear branch only) Sample per-modulator
       ``B_free_j ~ N(0, 1.0)``, apply b_mask_j via ``parameterize_B``,
       stack into ``(J, N, N)``.
    4. Run coupled neural-hemodynamic ODE integration (rk4 fixed-step).
    5. Extract BOLD signal from hemodynamic states.
    6. Downsample predicted BOLD to TR resolution.
    7. Evaluate Gaussian likelihood on observed BOLD.

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
    b_masks : list of torch.Tensor or None, optional
        Per-modulator structural masks for the bilinear B path. Each
        element is a binary ``(N, N)`` mask where 1 marks a connection
        modulated by that modulator and 0 marks absent modulation.
        ``None`` (default) and ``[]`` both activate the linear
        short-circuit (bit-exact equivalent to the pre-Phase-15 linear
        task-DCM model). When non-empty, ``stim_mod`` is required.
    stim_mod : PiecewiseConstantInput or None, optional
        Modulator input function with ``.values`` shape ``(K, J)`` where
        ``J == len(b_masks)``. Column ``j`` pairs with ``b_masks[j]`` by
        position (implicit contract, not validated by column name).
        Only consulted when ``b_masks`` is non-empty.

    Notes
    -----
    Prior specifications follow SPM12 ``spm_dcm_fmri_priors.m``:

    - A_free ~ N(0, 1/64) element-wise, then masked and transformed
      via ``parameterize_A`` which maps diagonal to -exp(free)/2
      [REF-001].
    - C ~ N(0, 1) element-wise, then masked [REF-001].
    - (Bilinear branch only.) ``B_free_j ~ N(0, B_PRIOR_VARIANCE)``
      element-wise per modulator, then masked via ``parameterize_B``
      (D1; see module-level ``B_PRIOR_VARIANCE`` constant).
    - Noise precision ~ Gamma(1, 1), weakly informative.
    - Hemodynamic parameters fixed at SPM12 code defaults.

    **Linear short-circuit (MODEL-04).** When ``b_masks is None`` or
    ``b_masks == []``, the model is bit-exact equivalent to the
    pre-Phase-15 linear task-DCM model: no new sample sites, no new
    deterministic sites, and no ``"B"`` key in the trace. The
    ``CoupledDCMSystem`` is constructed with NO ``B=`` kwarg, inheriting
    the Phase 13 literal-expression gate at
    ``coupled_system.py:287-291``.

    **Bilinear mode (MODEL-01).** When ``b_masks`` is a non-empty list,
    each modulator is sampled via a literal
    ``pyro.sample(f"B_free_{j}", Normal(0, sqrt(B_PRIOR_VARIANCE)).to_event(2))``
    in a Python ``for`` loop. ``pyro.plate`` is NOT used around the loop
    (each modulator has its own potentially-different sparsity mask; see
    rDCM precedent at ``rdcm_model.py:101-145``). After the loop,
    ``parameterize_B`` is called ONCE on the stacked ``(J, N, N)``
    tensors. ``pyro.deterministic("B", B_stacked)`` is emitted ONLY in
    this branch (L3 guard preserves the linear trace structure).

    **NaN-safe BOLD guard.** Bilinear early-SVI samples can draw B tails
    that push ``max Re(eig(A_eff)) > 0`` (Gershgorin: B row-sum can reach
    ~2 for N=3, J=1 at +/- 1 sigma under N(0, 1.0)), yielding NaN/Inf in
    the predicted BOLD. When this is detected, ``predicted_bold`` is
    detached and zero-filled BEFORE the likelihood site, producing a
    large finite penalty with zero gradient (pattern ported from
    ``amortized_wrappers.py:143-145``). Applied in BOTH branches for
    defensive symmetry.

    The ODE integration uses the rk4 fixed-step method for
    predictable runtime during SVI optimization. Adaptive methods
    (dopri5) can cause variable computation graphs across SVI steps.

    Anti-patterns avoided:

    - Hemodynamic parameters are NOT sampled.
    - Adjoint method is NOT used (gradient reliability).
    - ``pyro.plate`` is NOT used around ODE time steps OR around the
      modulator-sampling loop.
    - Complex tensors are NOT passed to ``pyro.sample``.

    References
    ----------
    [REF-001] Friston, Harrison & Penny (2003), Eq. 1 -- bilinear
    extension with modulatory inputs.
    [REF-002] Stephan et al. (2007), Eq. 2-6.
    [REF-040] Friston et al. (2007) -- ELBO / free energy.
    SPM12 ``spm_dcm_fmri_priors.m`` -- pC.B = B one-state prior (D1).

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
    # Normalize len-0 b_masks to None so the linear short-circuit is the
    # single code path for J=0 (MODEL-04 edge case).
    if b_masks is not None and len(b_masks) == 0:
        b_masks = None
    M = c_mask.shape[1]
    T = observed_bold.shape[0]

    # Bilinear-mode validation; no-op in linear short-circuit.
    if b_masks is not None:
        _validate_bilinear_args(b_masks, stim_mod, N)

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

    # --- Bilinear B sampling (MODEL-01). Active only when b_masks non-empty. ---
    B_stacked: torch.Tensor | None = None
    merged_input_fn: object | None = None
    if b_masks is not None:
        # L1 locked: sample each B_free_j as full (N, N) Normal with
        # .to_event(2). NO pyro.plate -- matches rDCM per-iteration
        # precedent and enables auto-discovery across AutoNormal /
        # AutoLowRankMVN / AutoIAFNormal without factory changes
        # (MODEL-06).
        B_prior_std = B_PRIOR_VARIANCE ** 0.5
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
        # parameterize_B applies the mask elementwise and emits a
        # DeprecationWarning if any b_mask diagonal is non-zero
        # (MODEL-03, Phase 13 BILIN-01 source half). Called ONCE on
        # the stacked tensors.
        B_stacked = parameterize_B(B_free_stacked, b_mask_stacked)
        # L3 locked: emit pyro.deterministic("B", ...) ONLY in the
        # bilinear branch.
        pyro.deterministic("B", B_stacked)

        # Merge driving + modulator inputs into a widened
        # (M + J)-column PiecewiseConstantInput for the CoupledDCMSystem
        # bilinear gate. Accepts either a PiecewiseConstantInput
        # directly or a breakpoint dict (same contract as Phase 14).
        drive_input = (
            stimulus if isinstance(stimulus, PiecewiseConstantInput)
            else PiecewiseConstantInput(stimulus["times"], stimulus["values"])
        )
        mod_input = (
            stim_mod if isinstance(stim_mod, PiecewiseConstantInput)
            else PiecewiseConstantInput(stim_mod["times"], stim_mod["values"])
        )
        merged_input_fn = merge_piecewise_inputs(drive_input, mod_input)

    # --- Forward model (deterministic computation) ---
    # Hemodynamic params: FIXED at SPM defaults (hemo_params=None)
    if B_stacked is not None:
        # Bilinear branch: Phase 13 gate at coupled_system.py:292-300.
        system = CoupledDCMSystem(
            A, C, merged_input_fn,
            B=B_stacked,
            n_driving_inputs=c_mask.shape[1],
        )
    else:
        # Linear short-circuit: Phase 13 literal-expression gate at
        # coupled_system.py:287-291. MUST be called with NO B= kwarg
        # to inherit the bit-exact pre-Phase-15 trace structure
        # (MODEL-04, L3).
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

    # NaN-safe guard: bilinear early-SVI samples can sample B tails that
    # push max Re(eig(A_eff)) > 0 (Gershgorin: B row-sum can reach ~2
    # for N=3, J=1 at +/- 1 sigma with zero-diagonal mask), producing
    # exp(+1.5 * 10s) ~ 3e6 growth during sustained u_mod ON-epochs and
    # NaN/Inf in BOLD. Detach + zero-fill produces a large finite
    # penalty with zero gradient, preventing NaN ELBO from halting SVI
    # via run_svi's NaN guard at guides.py:335-337. Pattern ported from
    # amortized_wrappers.py:143-145 (untrained-flow divergence
    # mitigation). Applied in BOTH branches for defensive symmetry.
    if torch.isnan(predicted_bold).any() or torch.isinf(predicted_bold).any():
        predicted_bold = torch.zeros_like(predicted_bold).detach()
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
