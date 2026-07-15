"""Pyro generative model for DCM-for-evoked-responses (CMC scalp ERP).

Defines the probabilistic generative process for DCM for evoked responses
(ERP) on the canonical microcircuit (CMC): sample the named CMC free
parameters ``A`` (four extrinsic routing blocks), per-effect between-trial
``B`` modulation, driving input gain ``C``, intrinsic time constants ``T``,
intrinsic gains ``G``, sigmoid slope ``S``, and input dispersion ``R`` from
log-space Normal priors; run the parity-verified Phase 33/34/35 scalp-ERP
forward via :func:`simulate_erp_dcm`; and condition a Gaussian likelihood on
the flattened ``(Cnd, ns, Nc)`` scalp residual.

The deterministic forward is delegated ENTIRELY to ``simulate_erp_dcm`` (the
parity-gated pipeline: ``spm_gen_Q`` condition modulation ->
``spm_int_L`` exp-Euler integration of ``spm_fx_cmc`` -> ``spm_lx_erp`` scalp
projection). This module NEVER re-assembles the forward (pitfall V1): its only
job is priors + likelihood. Unlike :class:`ERPDCMForward` (where ``B`` is a
fixed observation context), ``B`` IS sampled here so the full SVI model can
recover between-trial modulation.

Prior scales are transcribed from ``spm_cmc_priors.m`` (mean 0 in the free
log-space; structural masking is applied AFTER sampling). Because the CMC
parameterises strengths as ``exp(P) * E0``, the mask-DEAD value for ``A``/``C``
is the strongly negative free value ``-32`` (``exp(-32) * E0 ~ 1e-12``), NOT 0
(``exp(0) * E0 = E0`` would be a LIVE edge). The between-trial ``B`` is an
ADDITIVE log-offset folded into ``Q.A``/``Q.G`` by ``spm_gen_Q``, so its
mask-DEAD value is 0 (no modulation).

References
----------
SPM12 ``spm_cmc_priors.m`` -- ``P.A`` var 1/16 (``:80-81``), ``P.C`` var
``mask/32`` (``:114-116``), ``P.T`` var 1/32 (``:121``), ``P.G`` var 1/32
(``:122``), ``P.S`` var 1/64 (``:124``), ``P.R`` var ``[1/16, 1/16]``
(``:133``). SPM12 ``spm_gen_erp.m`` / ``spm_gen_Q.m`` (condition-modulated
generative ERP loop). David & Friston (2003), NeuroImage 20(3):1743-1755
(neural-mass evoked responses); Bastos et al. (2012), Neuron 76(4):695-711
(canonical microcircuit).
"""

from __future__ import annotations

import pyro
import pyro.distributions as dist
import torch

from pyro_dcm.forward_models.cmc_priors import ERP_DEAD_FREE
from pyro_dcm.simulators.erp_simulator import simulate_erp_dcm

# Free log value mapping an ABSENT CMC connection (mask == 0) to a dead edge --
# single source of truth in ``cmc_priors`` (shared with ``ERPDCMForward``).
_ERP_DEAD_FREE: float = ERP_DEAD_FREE

# Provisional prior variance for the between-trial B modulation. B is NOT in
# ``cmc_prior_moments`` (Phase-34 condition modulation; FIXED in the parity gate).
# The exact ``pC.B`` for the SVI recovery path is MUST-VERIFY against
# ``spm_dcm_erp.m`` / ``spm_cmc_priors.m``; this provisional value is low-stakes
# for the fixed-B headline demo and matters only for B-modulation recovery.
B_PRIOR_VARIANCE: float = 1.0 / 8.0


def _sample_logspace(
    name: str, shape: tuple[int, ...], var: float, event_dim: int
) -> torch.Tensor:
    """Sample a zero-mean log-space CMC free parameter (``spm_cmc_priors.m``).

    All CMC free parameters share the same prior form -- ``dist.Normal(0,
    sqrt(var))`` reshaped to ``shape`` and made independent via ``.to_event``.

    Parameters
    ----------
    name : str
        Pyro sample-site name.
    shape : tuple of int
        Parameter shape.
    var : float
        Prior variance (``spm_cmc_priors.m`` scale); the std is ``sqrt(var)``.
    event_dim : int
        Number of rightmost dims to declare dependent via ``.to_event``.

    Returns
    -------
    torch.Tensor
        The sampled free parameter, shape ``shape``, dtype float64.
    """
    return pyro.sample(
        name,
        dist.Normal(
            torch.zeros(shape, dtype=torch.float64),
            var**0.5 * torch.ones(shape, dtype=torch.float64),
        ).to_event(event_dim),
    )


def _apply_dead_mask(free: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Map absent (``mask == 0``) ``A``/``C`` edges to the dead free value.

    Live positions keep the sampled free value; absent positions map to
    :data:`_ERP_DEAD_FREE` (``exp(-32) * E0 ~ 1e-12``, an off edge).

    Parameters
    ----------
    free : torch.Tensor
        Sampled free values.
    mask : torch.Tensor
        Binary presence mask, same shape as ``free``.

    Returns
    -------
    torch.Tensor
        ``free`` with absent positions set to :data:`_ERP_DEAD_FREE`.
    """
    m = (mask > 0).to(torch.float64)
    return free * m + _ERP_DEAD_FREE * (1.0 - m)


def erp_dcm_model(
    observed_scalp: torch.Tensor,
    a_masks: list[torch.Tensor],
    b_masks: list[torch.Tensor],
    c_mask: torch.Tensor,
    x_design: torch.Tensor,
    l_full: torch.Tensor,
    N: int | None = None,
    *,
    dt: float = 0.004,
    ns: int = 128,
    ons_ms: float = 60.0,
    dur_ms: float = 16.0,
    sus: float = 0.0,
) -> None:
    """Pyro generative model for DCM-for-evoked-responses (CMC scalp ERP).

    Samples the CMC free parameters from log-space Normal priors
    (``spm_cmc_priors.m`` scales, mean 0), applies the structural masks
    (``A``/``C`` absent edges -> ``-32``; ``B`` absent -> 0), delegates the
    deterministic forward to :func:`simulate_erp_dcm` (the parity-verified
    Phase 33/34/35 pipeline), and conditions a Gaussian likelihood on the
    flattened ``(Cnd, ns, Nc)`` scalp residual.

    Sample sites (all log-space, prior mean 0, ``dist.Normal(0, sqrt(var))``):

    - ``A_free`` ``(4, N, N)`` ``.to_event(3)``, var ``1/16``
      (``spm_cmc_priors.m:80-81``); masked to ``-32`` on absent edges.
    - ``B_free_{j}`` ``(N, N)`` ``.to_event(2)`` per between-trial effect
      (Python ``for`` loop, NO ``pyro.plate`` -- enables AutoGuide
      auto-discovery, MODEL-06), var ``B_PRIOR_VARIANCE`` (provisional ``1/8``,
      MUST-VERIFY vs ``spm_dcm_erp``); masked to 0 (additive log-offset).
    - ``C_free`` ``(N, M)`` ``.to_event(2)``, var ``1/32``
      (``spm_cmc_priors.m:114-116``); masked to ``-32`` on absent inputs.
    - ``T`` ``(N, 4)`` ``.to_event(2)``, var ``1/32`` (``:121``).
    - ``G`` ``(N, 4)`` ``.to_event(2)``, var ``1/32`` (``:122``).
    - ``S`` ``(N, 1)`` ``.to_event(2)``, var ``1/64`` (``:124``).
    - ``R`` ``(M, 2)`` ``.to_event(2)``, var ``1/16`` (``:133``).
    - ``scalp_noise_scale`` ``HalfCauchy(1)`` observation-noise scale.

    Unlike :class:`ERPDCMForward` (where ``B`` is fixed observation context),
    ``B`` IS sampled here so the full SVI model recovers between-trial
    modulation. The forward is NEVER re-assembled (pitfall V1):
    :func:`simulate_erp_dcm` is the only forward.

    Parameters
    ----------
    observed_scalp : torch.Tensor
        Observed scalp ERP, shape ``(Cnd, ns, Nc)``, dtype float64. Row 0 of
        ``x_design`` is the standard, row 1 the deviant.
    a_masks : list of torch.Tensor
        The four extrinsic routing-graph masks (fwd sp->ss, fwd sp->dp, bwd
        dp->sp, bwd dp->ii), each ``(N, N)`` binary, dtype float64.
    b_masks : list of torch.Tensor
        Per-effect between-trial ``B`` masks (live positions), each ``(N, N)``
        binary. One per between-trial effect (column of ``x_design``).
    c_mask : torch.Tensor
        Driving-input mask, shape ``(N, M)`` binary, dtype float64.
    x_design : torch.Tensor
        Between-trial design, shape ``(Cnd, n_effects)``.
    l_full : torch.Tensor
        Precomputed LFP lead field, shape ``(Nc, 8N)``, dtype float64.
    N : int or None, optional
        Number of sources. If None, inferred from ``a_masks[0].shape[0]``.
    dt : float, optional
        Integration step (``U.dt``) in seconds. Default 0.004.
    ns : int, optional
        Number of peristimulus samples. Default 128.
    ons_ms : float, optional
        Stimulus onset in ms (``M.ons``). Default 60.
    dur_ms : float, optional
        Gaussian dispersion in ms (``M.dur``). Default 16.
    sus : float, optional
        Sustained-input level (``M.sus``). Default 0.

    Notes
    -----
    The likelihood flattens the canonical ``(Cnd, ns, Nc)`` layout with a
    C-order ``reshape(-1)`` to match the layout used downstream (the
    ``ERPDCMForward`` / VL boundary). A ``torch.nan_to_num`` guard on the
    predicted scalp produces a large finite penalty (rather than a NaN ELBO)
    when an early-SVI draw destabilises the integrator (amortized idiom).

    References
    ----------
    SPM12 ``spm_cmc_priors.m`` (line refs above), ``spm_gen_erp.m``,
    ``spm_gen_Q.m``. David & Friston (2003); Bastos et al. (2012).
    """
    if N is None:
        N = a_masks[0].shape[0]
    M = c_mask.shape[1]

    # --- Sample A_free: four extrinsic routing blocks (var 1/16). ---
    # Live edges keep the sampled free value; absent edges -> -32 (dead).
    A_free = _sample_logspace("A_free", (4, N, N), 1.0 / 16.0, 3)
    a_free_list = [_apply_dead_mask(A_free[i], a_masks[i]) for i in range(4)]

    # --- Sample per-effect B_free_{j} (var B_PRIOR_VARIANCE; NO pyro.plate). ---
    # Literal sample sites in a Python loop enable AutoGuide auto-discovery
    # across AutoNormal / AutoLowRankMVN / AutoIAFNormal (MODEL-06), mirroring
    # the task_dcm_model bilinear branch. B is an ADDITIVE log-offset, so absent
    # positions map to 0 (no modulation), NOT the -32 dead value.
    B_free_list: list[torch.Tensor] = []
    for j, b_mask_j in enumerate(b_masks):
        B_free_j = _sample_logspace(f"B_free_{j}", (N, N), B_PRIOR_VARIANCE, 2)
        B_free_list.append(B_free_j * (b_mask_j > 0).to(torch.float64))

    # --- Sample C_free (var 1/32) + intrinsic T, G, S and input dispersion R. ---
    C_free = _sample_logspace("C_free", (N, M), 1.0 / 32.0, 2)
    c_masked = _apply_dead_mask(C_free, c_mask)
    T = _sample_logspace("T", (N, 4), 1.0 / 32.0, 2)
    G = _sample_logspace("G", (N, 4), 1.0 / 32.0, 2)
    S = _sample_logspace("S", (N, 1), 1.0 / 64.0, 2)
    R = _sample_logspace("R", (M, 2), 1.0 / 16.0, 2)

    # --- Deterministic forward (the parity-gated pipeline; pitfall V1). ---
    p: dict[str, object] = {
        "T": T,
        "G": G,
        "C": c_masked,
        "S": S,
        "R": R,
        "A": a_free_list,
        "B": B_free_list,
    }
    sim = simulate_erp_dcm(
        p,
        x_design,
        N,
        ns=ns,
        dt=dt,
        ons_ms=ons_ms,
        dur_ms=dur_ms,
        sus=sus,
        l_full=l_full,
    )
    pred = torch.nan_to_num(sim["scalp"])  # (Cnd, ns, Nc)
    pyro.deterministic("predicted_scalp", pred)

    # --- Observation noise + Gaussian likelihood on the flat scalp residual. ---
    noise_scale = pyro.sample(
        "scalp_noise_scale",
        dist.HalfCauchy(torch.tensor(1.0, dtype=torch.float64)),
    )
    pyro.sample(
        "obs_erp",
        dist.Normal(pred.reshape(-1), noise_scale).to_event(1),
        obs=observed_scalp.reshape(-1),
    )
