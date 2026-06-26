"""Hierarchical-CMC network forward + condition-B modulation (Phase 34).

Lifts the parity-verified Phase-33 single-source canonical-microcircuit forward
(:mod:`pyro_dcm.forward_models.cmc_neural_mass`) to an ``n``-source hierarchical
network by ADDING the four extrinsic coupling terms of ``spm_fx_cmc.m`` (SPM12,
``$Id: spm_fx_cmc.m 7279``), and ports the condition-specific modulation
``spm_gen_Q.m`` (SPM12, ``$Id: spm_gen_Q.m 7279``). Three new symbols, composed
additively from the frozen Phase-33 code (which is NOT edited):

* :func:`parameterize_cmc_network` -- the ``n > 1`` extrinsic blocks
  ``A[i] = exp(P.A[i]) * E0[i]`` with the lateral ``(1 + 4L)`` reciprocal
  reduction (``spm_fx_cmc.m:68-82``).
* :func:`apply_condition_modulation` -- ``spm_gen_Q`` in free/log space: one
  between-trial matrix ``B[i]`` is folded additively into ALL four ``A`` blocks
  (``spm_gen_Q.m:47``) AND ``diag(B[i])`` into the free precision column
  ``Q.G[:,0]`` (``spm_gen_Q.m:65-67``), which drives ``G[:,6]`` (``sp``
  self-inhibition) after parameterisation -- the MMN precision mechanism.
* :func:`cmc_network_f` -- the Phase-33 intrinsic equations of motion with the
  four extrinsic ``A @ S`` terms added into the conductance numerators
  (``spm_fx_cmc.m:171,177,183,189``). Bit-exact to ``cmc_f`` at ``n = 1``.

The four extrinsic routes (``spm_fx_cmc.m:171-198``), 0-indexed for torch
(ss_I=1, sp_I=3, ii_I=5, dp_I=7; firing voltage cols sp-V=2, dp-V=6):

============  ===============  ==============  =============  ====
block         route            origin firing   target row    sign
============  ===============  ==============  =============  ====
``A[0]`` fwd  sp -> ss         ``S[:,2]``      ss (``f1``)   ``+``
``A[1]`` fwd  sp -> dp         ``S[:,2]``      dp (``f7``)   ``+``
``A[2]`` bwd  dp -> sp         ``S[:,6]``      sp (``f3``)   ``-``
``A[3]`` bwd  dp -> ii         ``S[:,6]``      ii (``f5``)   ``-``
============  ===============  ==============  =============  ====

References
----------
SPM12 ``spm_fx_cmc.m`` -- extrinsic blocks + ``E`` pairing (``:47,68-72``),
lateral ``(1+4L)`` reduction (``:79-82``), the four extrinsic -> EOM routes
(``:171,177,183,189``), C -> ss input (``:86,107,171``). ``spm_gen_Q.m`` --
``B`` -> all-``A`` folding (``:41-47``), C-effect (``:35-37``), the
``diag(B) -> Q.G(:,1)`` precision path (``:65-67``). David, O. & Friston, K.J.
(2003), NeuroImage 20, 1743-1755.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from pyro_dcm.forward_models.cmc_neural_mass import (
    E0,
    cmc_flatten,
    cmc_sigmoid,
    cmc_unflatten,
    parameterize_cmc,
)

_F64 = torch.float64

# Lateral-reduction threshold exp(-8) (spm_fx_cmc.m:79-82).
_RECIP_THR = math.exp(-8.0)

# 0-indexed conductance rows / firing voltage columns (spm_fx_cmc.m:6-14,171-198).
_SS_I, _SP_I, _II_I, _DP_I = 1, 3, 5, 7
_SP_V, _DP_V = 2, 6


def parameterize_cmc_network(p: dict[str, Tensor], n: int) -> dict[str, Tensor]:
    """Parameterise the CMC network: intrinsic ``+`` extrinsic blocks (Fact 1).

    Reuses :func:`parameterize_cmc` for ``T``/``G``/``C``/``S`` (byte-identical
    to Phase 33), then OVERRIDES the extrinsic path when ``p["A"]`` is present
    (``spm_fx_cmc.m:68-82``): ``A[i] = exp(P.A[i]) * E0[i]`` followed by the
    lateral reciprocal reduction ``A[i] /= (1 + 4L)`` with
    ``L = (A[i] > exp(-8)) & (A[i].T > exp(-8))``. When ``p["A"]`` is absent the
    extrinsic blocks stay ``zeros(4, n, n)`` (exactly as :func:`parameterize_cmc`
    returns), so at ``n = 1`` the network forward stays bit-exact to ``cmc_f``.

    Parameters
    ----------
    p : dict
        Free parameters. Keys (each optional): ``"T"`` ``(n, 4)``, ``"G"``
        ``(n, 4)``, ``"C"`` ``(n, n_inp)``, ``"S"`` ``(n, 1)``, and ``"A"`` --
        a length-4 list/stack of ``(n, n)`` free log-params (the four extrinsic
        blocks).
    n : int
        Number of sources.

    Returns
    -------
    dict
        ``{"T": (n,4), "G": (n,10), "C": (n,n_inp), "A": (4,n,n), "S": (n,1)}``.
    """
    params = parameterize_cmc(p, n)
    if "A" not in p:
        return params

    a_free = p["A"]
    blocks: list[Tensor] = []
    for i in range(4):
        a_i = torch.exp(torch.as_tensor(a_free[i], dtype=_F64)) * E0[i]
        # Reciprocal pair iff both directions exceed exp(-8) (spm_fx_cmc.m:79-82).
        recip = (a_i > _RECIP_THR) & (a_i.transpose(-2, -1) > _RECIP_THR)
        a_i = a_i / (1.0 + 4.0 * recip.to(_F64))
        blocks.append(a_i)
    params["A"] = torch.stack(blocks, dim=0)
    return params


def apply_condition_modulation(
    p: dict[str, Tensor], x_design: Tensor
) -> dict[str, Tensor]:
    """Port ``spm_gen_Q.m:24-67``: condition-specific ``B`` modulation (Fact 2).

    Operates ENTIRELY in free/log space, BEFORE :func:`parameterize_cmc_network`
    exponentiates. For each between-trial effect ``i`` with design weight ``Xi``:
    the SAME matrix ``B[i]`` is added to all four ``Q.A{j}``
    (``spm_gen_Q.m:47``), and ``diag(B[i])`` is added to the free precision
    column ``Q.G[:,0]`` (``spm_gen_Q.m:65-67``), which drives ``G[:,6]`` after
    parameterisation. The ``"B"`` key is dropped from the returned ``Q``. The
    C-effect (``spm_gen_Q.m:35-37``) fires only when ``C`` carries a second page
    (``C.ndim == 3 and C.shape[-1] >= 2``); the single-page reference net skips
    it. The NMDA ``AN``/``BN`` and ``M``/``int`` branches are absent in the plain
    CMC reference and are not ported.

    Parameters
    ----------
    p : dict
        Free parameters including ``"B"`` -- a list of ``(n, n)`` between-trial
        matrices -- and (optionally) ``"A"`` (a length-4 list of ``(n, n)`` free
        log-params; treated as ``zeros(4, n, n)`` if absent).
    x_design : torch.Tensor
        Between-trial design row for ONE condition, shape ``(n_effects,)``.

    Returns
    -------
    dict
        Condition-specific free-parameter struct ``Q`` (no ``"B"`` key), with
        ``Q["A"]`` a ``(4, n, n)`` tensor and ``Q["G"]`` the modulated ``(n, 4)``
        free intrinsic params.
    """
    x_design = torch.atleast_1d(torch.as_tensor(x_design, dtype=_F64))
    b_list = p["B"]
    n = b_list[0].shape[0]

    q: dict[str, Tensor] = {
        k: v.clone() for k, v in p.items() if k not in ("A", "B") and torch.is_tensor(v)
    }
    if "A" in p:
        a = torch.stack(
            [torch.as_tensor(p["A"][i], dtype=_F64).clone() for i in range(4)], dim=0
        )
    else:
        a = torch.zeros(4, n, n, dtype=_F64)

    g = q.get("G")
    for i, xi in enumerate(x_design):
        b_i = torch.as_tensor(b_list[i], dtype=_F64)
        for j in range(4):
            a[j] = a[j] + xi * b_i  # spm_gen_Q.m:47 (same B to all four blocks)
        if g is not None:
            # spm_gen_Q.m:65-67: diag(B) modulates the free precision column.
            g[:, 0] = g[:, 0] + xi * torch.diagonal(b_i)
    q["A"] = a
    if g is not None:
        q["G"] = g

    # C-effect (spm_gen_Q.m:35-37) only when C has a second condition page.
    c = p.get("C")
    if c is not None and c.ndim == 3 and c.shape[-1] >= 2:
        q["C"] = c[..., 0] + x_design[0] * c[..., 1]
    return q


def cmc_network_f(x_flat: Tensor, u: Tensor, p: dict[str, Tensor], n: int) -> Tensor:
    """Network equations of motion ``dx/dt`` (spm_fx_cmc.m:171-198).

    The Phase-33 intrinsic EOM body (``cmc_f``) plus the four extrinsic ``A @ S``
    terms added into the conductance numerators: forward ``+A[0]@S[:,2]`` -> ss
    and ``+A[1]@S[:,2]`` -> dp; backward ``-A[2]@S[:,6]`` -> sp and
    ``-A[3]@S[:,6]`` -> ii. At ``n = 1`` with default ``A`` (zeros) the four
    terms vanish and this reproduces ``cmc_f`` bit-exactly.

    Parameters
    ----------
    x_flat : torch.Tensor
        Flat state, shape ``(8n,)``, float64 (column-major, :func:`cmc_flatten`).
    u : torch.Tensor or float
        Per-sample exogenous input, shape ``(n_inp,)``.
    p : dict
        Free-parameter struct (see :func:`parameterize_cmc_network`).
    n : int
        Number of sources.

    Returns
    -------
    torch.Tensor
        ``dx/dt`` flat, shape ``(8n,)``, float64.

    Raises
    ------
    TypeError
        If ``x_flat`` is not float64 (expected ``torch.float64``).
    """
    if x_flat.dtype != _F64:
        raise TypeError(
            f"x_flat must be float64; expected {_F64}, got {x_flat.dtype} "
            "(the CMC network forward runs entirely in float64, CMC-07)"
        )
    x = cmc_unflatten(x_flat, n)  # (n, 8)
    params = parameterize_cmc_network(p, n)
    t, g, c, a, p_s = (
        params["T"],
        params["G"],
        params["C"],
        params["A"],
        params["S"],
    )

    s = cmc_sigmoid(x, p_s)  # (n, 8)
    u_vec = torch.atleast_1d(torch.as_tensor(u, dtype=_F64))
    big_u = (c @ u_vec) * 32.0  # exogenous input (spm_fx_cmc.m:86,107), (n,)

    # Origin firing for the extrinsic routes: fwd from sp voltage, bwd from dp.
    s_fwd = s[:, _SP_V]  # (n,)
    s_bwd = s[:, _DP_V]  # (n,)

    # Granular -- spiny stellate (spm_fx_cmc.m:171-174) + fwd A[0]@S[:,2].
    uu_g = big_u - g[:, 0] * s[:, 0] - g[:, 2] * s[:, 4] - g[:, 1] * s[:, 2]
    uu_g = uu_g + a[0] @ s_fwd
    f1 = (uu_g - 2.0 * x[:, 1] - x[:, 0] / t[:, 0]) / t[:, 0]

    # Supragranular -- superficial pyramidal (spm_fx_cmc.m:177-180) - bwd A[2]@S[:,6].
    uu_sp = g[:, 7] * s[:, 0] - g[:, 6] * s[:, 2]
    uu_sp = uu_sp - a[2] @ s_bwd
    f3 = (uu_sp - 2.0 * x[:, 3] - x[:, 2] / t[:, 1]) / t[:, 1]

    # Supragranular -- inhibitory interneurons (spm_fx_cmc.m:183-186) - bwd A[3]@S[:,6].
    uu_ii = g[:, 4] * s[:, 0] + g[:, 5] * s[:, 6] - g[:, 3] * s[:, 4]
    uu_ii = uu_ii - a[3] @ s_bwd
    f5 = (uu_ii - 2.0 * x[:, 5] - x[:, 4] / t[:, 2]) / t[:, 2]

    # Infragranular -- deep pyramidal (spm_fx_cmc.m:189-192) + fwd A[1]@S[:,2].
    uu_dp = -g[:, 9] * s[:, 6] - g[:, 8] * s[:, 4]
    uu_dp = uu_dp + a[1] @ s_fwd
    f7 = (uu_dp - 2.0 * x[:, 7] - x[:, 6] / t[:, 3]) / t[:, 3]

    # Voltages are integrals of conductances (spm_fx_cmc.m:195-198).
    f0 = x[:, 1]
    f2 = x[:, 3]
    f4 = x[:, 5]
    f6 = x[:, 7]

    f = torch.stack([f0, f1, f2, f3, f4, f5, f6, f7], dim=1)  # (n, 8)
    return cmc_flatten(f)
