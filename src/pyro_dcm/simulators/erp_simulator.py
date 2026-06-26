"""Multi-source evoked-response simulator (the ``spm_gen_erp`` loop, Phase 34).

Ports the condition loop of ``spm_gen_erp.m`` (SPM12, ``$Id: spm_gen_erp.m
6427``) at network scale, composing the Phase-34 network forward
(:func:`pyro_dcm.forward_models.cmc_network_f`) + condition modulation
(:func:`pyro_dcm.forward_models.apply_condition_modulation`) with the frozen,
parity-verified Phase-33 exp-Euler integrator
(:func:`pyro_dcm.utils.local_linearization.integrate_local_linearization`) and
Gaussian evoked drive (:func:`pyro_dcm.forward_models.erp_gaussian_input`).

For each between-trial condition the free-space ``B`` modulation is applied first
(``spm_gen_erp.m:76`` -> ``spm_gen_Q``), THEN the per-condition closure is
integrated (``spm_gen_erp.m:84`` -> ``spm_int_L``). Because ``Q.A``/``Q.G``
differ per condition, the integrator re-freezes its Jacobian once per condition.
The Gaussian drive depends only on ``P.R`` (which ``B`` does not touch), so it is
computed ONCE and shared across conditions (``spm_gen_erp.m:44``).

Scope boundary (do NOT over-scope downstream): this Phase-34 simulator returns
SOURCE-level states only. The ``difference_wave`` hook differences the source
``sp`` voltage (column 2); the single-dipole lead field, the scalp projection,
and the TRUE MMN scalp difference wave are Phase 35 (LEAD-01/03). Phase 34 only
needs a non-trivial source-level difference between conditions, which exists iff
``B`` is wired.

References
----------
SPM12 ``spm_gen_erp.m`` -- peristimulus time + input grid (``:35,44``), the
condition loop (``:69-86``), per-condition ``spm_gen_Q`` + ``spm_int_L``
(``:76,84``).
"""

from __future__ import annotations

import torch
from torch import Tensor

from pyro_dcm.forward_models.cmc_neural_mass import cmc_flatten, cmc_unflatten
from pyro_dcm.forward_models.cmc_priors import cmc_steady_state
from pyro_dcm.forward_models.erp_coupled_system import (
    apply_condition_modulation,
    cmc_network_f,
)
from pyro_dcm.forward_models.erp_input import erp_gaussian_input
from pyro_dcm.forward_models.erp_leadfield import project_to_scalp
from pyro_dcm.utils.local_linearization import integrate_local_linearization

_F64 = torch.float64


def simulate_erp_dcm(
    p: dict[str, Tensor],
    x_design: Tensor,
    n: int,
    ns: int = 128,
    dt: float = 0.004,
    ons_ms: float = 60.0,
    dur_ms: float = 16.0,
    sus: float = 0.0,
    l_full: Tensor | None = None,
) -> dict[str, Tensor | None]:
    """Per-condition source-level evoked responses via the ``spm_gen_erp`` loop.

    When ``l_full`` is supplied, the source-state trajectory is additionally
    projected to scalp (Phase 35, LEAD-03) via
    :func:`pyro_dcm.forward_models.project_to_scalp` and the deviant-standard
    scalp difference wave is returned. The scalp difference wave's NON-ZEROness is
    the Phase-35 gate; the negative-going / frontal-dominance SIGN is Phase 36 (it
    needs ECD dipole orientation + MNI coords, which do not exist yet -- Fact 6).
    This function therefore does NOT pin a scalp polarity in LFP identity mode.

    Parameters
    ----------
    p : dict
        Free-parameter struct with keys ``"T"`` ``(n,4)``, ``"G"`` ``(n,4)``,
        ``"C"`` ``(n,n_inp)``, ``"S"`` ``(n,1)``, ``"R"`` ``(n_inp,2)``, ``"A"``
        (length-4 list of ``(n,n)`` free log-params), and ``"B"`` (list of
        ``(n,n)`` between-trial matrices).
    x_design : torch.Tensor
        Between-trial design, shape ``(Cnd, n_effects)``. Row 0 is treated as the
        standard, row 1 as the deviant (for the difference-wave hook).
    n : int
        Number of sources.
    ns : int, optional
        Number of peristimulus samples. Default 128.
    dt : float, optional
        Integration step in seconds (``U.dt``). Default 0.004.
    ons_ms : float, optional
        Stimulus onset in ms (``M.ons``). Default 60.
    dur_ms : float, optional
        Gaussian dispersion in ms (``M.dur``). Default 16.
    sus : float, optional
        Sustained-input level (``M.sus``). Default 0.
    l_full : torch.Tensor or None, optional
        Full per-state lead field ``(Nc, 8n)`` (e.g.
        :func:`pyro_dcm.forward_models.build_lead_field`). When supplied, the
        scalp projection keys are added.

    Returns
    -------
    dict
        ``{"states": (Cnd, ns, n, 8), "pst": (ns,), "inputs": (ns, n_inp),
        "difference_wave": (ns, n, 8) | None}``. ``difference_wave`` is
        ``states[deviant] - states[standard]`` (``None`` when ``Cnd < 2``).
        When ``l_full`` is supplied two more keys are added:
        ``"scalp": (Cnd, ns, Nc)`` and ``"difference_wave_scalp": (ns, Nc) | None``
        (``scalp[1] - scalp[0]``, ``None`` when ``Cnd < 2``). The existing
        source-state keys are byte-unchanged.
    """
    x_design = torch.as_tensor(x_design, dtype=_F64)
    if x_design.ndim == 1:
        x_design = x_design.reshape(-1, 1)

    # Peristimulus time grid in seconds (spm_gen_erp.m:35).
    pst = torch.arange(1, ns + 1, dtype=_F64) * dt - ons_ms / 1000.0
    # Condition-independent Gaussian drive (spm_gen_erp.m:44); compute once.
    inputs = erp_gaussian_input(pst, p["R"], ons_ms, dur_ms, sus)  # (ns, n_inp)

    x0 = cmc_flatten(cmc_steady_state(n))  # zeros(8n,)
    cnd = x_design.shape[0]

    states_list: list[Tensor] = []
    scalp_list: list[Tensor] = []
    for c in range(cnd):
        q = apply_condition_modulation(p, x_design[c])

        def f_c(v: Tensor, u: Tensor, q: dict[str, Tensor] = q) -> Tensor:
            return cmc_network_f(v, u, q, n)

        traj = integrate_local_linearization(f_c, x0, inputs, dt)  # (ns, 8n)
        src = torch.stack([cmc_unflatten(traj[i], n) for i in range(ns)], dim=0)
        states_list.append(src)  # (ns, n, 8)
        if l_full is not None:
            # Project the flat (ns, 8n) source trajectory to scalp (LEAD-03).
            scalp_list.append(project_to_scalp(traj, l_full))  # (ns, Nc)

    states = torch.stack(states_list, dim=0)  # (Cnd, ns, n, 8)

    out: dict[str, Tensor | None] = {
        "states": states,
        "pst": pst,
        "inputs": inputs,
    }
    # Difference wave on SOURCE states (sp voltage col 2 is the readout of
    # interest); the TRUE MMN scalp difference is the scalp keys below.
    out["difference_wave"] = states[1] - states[0] if cnd >= 2 else None

    if l_full is not None:
        scalp = torch.stack(scalp_list, dim=0)  # (Cnd, ns, Nc)
        out["scalp"] = scalp
        # Deviant - standard scalp difference (NON-ZERO iff B is wired). The SIGN
        # (negative-going / frontal) is Phase 36 -- not asserted here (Fact 6).
        out["difference_wave_scalp"] = scalp[1] - scalp[0] if cnd >= 2 else None
    return out
