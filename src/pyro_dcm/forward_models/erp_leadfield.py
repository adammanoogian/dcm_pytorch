"""Single-dipole lead field + scalp projection for DCM-for-evoked-responses (Phase 35).

Ports the SPM12 ERP observer ``spm_lx_erp.m`` (``$Id: spm_lx_erp.m 7256``) and its
spatial-model helper ``spm_erp_L.m`` (``$Id: spm_erp_L.m 7142``) into pure torch
(float64, zero new deps). Turns the parity-verified per-source CMC source-state
trajectory (Phase 33/34, ``(ns, 8n)`` column-major) into the observed scalp ERP via
a single linear map

    L_full = kron(P.J, L_spatial),   y = (states - x0) @ L_full.T

The single most important source fact (``spm_L_priors.m:108``): for CMC,
``pE.J = sparse(1,3,1,1,8)`` -> the default observed state is MATLAB column 3 ->
0-indexed index 2 = **superficial-pyramidal voltage** (sp_V), NOT index 6
(deep-pyramidal voltage). ``spm_lx_erp.m:33`` then forms ``L = kron(P.J, L_spatial)``
with ``P.J`` as the FIRST kron argument, whose ``(Nc, 8n)`` block ordering
(``P.J[s] * L_spatial`` in column block ``s``, columns ``[s*n:(s+1)*n]``) **exactly
matches the column-major state-blocked flatten** ``cmc_flatten = x.T.reshape(-1)``
(flat index ``state*n + source``, proven at N=5 in ``34-03-SUMMARY.md`` rung 2). No
transpose, no permutation.

In LFP mode (``spm_erp_L.m:105-118``) the spatial model is a trivial diagonal
``L_spatial = diag(P.L)``, default ``P.L = ones`` -> identity, so the LFP scalp ERP
is literally each source's sp-voltage trace -- the cleanest, head-model-free parity
target (the Phase-35 gate). The ECD path (:func:`ecd_spatial`, ``spm_erp_L.m:43-77``)
needs a sensor montage + MNI coords and is correctly deferred to Phase 36; this module
is built to *consume* a MATLAB-exported ECD gain from day one (no rework needed).

Citation policy: SPM source file + line only (the Zotero ``REF-ERP-*`` keys are still
unverified -- do NOT fabricate a ``[REF-xxx]`` bib key).

References
----------
SPM12 ``spm_lx_erp.m`` -- ``L = spm_erp_L(P, dipfit)`` (``:31``),
``L = kron(P.J, L)`` numeric-J branch (``:33``), the observer ``y = G*x``
(header ``:2-9``). ``spm_erp_L.m`` -- ECD branch ``L(:,i) = G(:,:,i)*P.L(:,i)``
post ``spm_cond_units`` (``:43-77``), LFP branch ``L = sparse(1:m,1:m,P.L,m,n)``
(``:105-118``). ``spm_L_priors.m`` -- CMC ``pE.J = sparse(1,3,1,1,8)`` /
``pC.J = sparse(1,[1 7],1/32,1,8)`` (``:106-109``), LFP ``pE.L = ones(1,m)``
(``:84``). Kiebel, S.J., David, O. & Friston, K.J. (2006), "Dynamic causal
modelling of evoked responses in EEG/MEG with lead field parameterization",
NeuroImage 30, 1273-1284.
"""

from __future__ import annotations

import torch
from torch import Tensor

_F64 = torch.float64

# CMC default contributing state: superficial-pyramidal VOLTAGE (sp_V), 0-indexed
# column 2 of the 8-state layout (spm_L_priors.m:108 sparse(1,3,1,1,8)).
_PJ_STATE = 2

# CMC 8-state column count (ss/sp/ii/dp x {voltage, conductance}).
_N_STATES = 8


def cmc_default_pj() -> Tensor:
    """CMC default contributing-state vector ``P.J`` (spm_L_priors.m:108).

    Returns the ``(8,)`` one-hot ``e_2`` -- value ``1.0`` at index 2
    (superficial-pyramidal voltage, sp_V), zero elsewhere. ``spm_L_priors.m:108``
    sets ``pE.J = sparse(1,3,1,1,8)`` (MATLAB column 3 -> 0-indexed index 2). The
    free ``pC.J`` indices are ``[0, 6]`` (ss_V, dp_V) with variance ``1/32``
    (``spm_L_priors.m:109``) but their prior MEAN contribution is zero -- they are
    held FIXED in v1 (the lead field is context, not a recovered parameter).

    GUARD (pitfall C5.1): index 2, NOT index 6 (deep-pyramidal voltage, the
    physiologically-inverted scalp-signal trap).

    Returns
    -------
    torch.Tensor
        One-hot ``P.J``, shape ``(8,)``, float64.
    """
    p_j = torch.zeros(_N_STATES, dtype=_F64)
    p_j[_PJ_STATE] = 1.0
    return p_j


def lfp_spatial(p_l: Tensor, n: int) -> Tensor:
    """LFP spatial lead field ``diag(P.L)``, shape ``(Nc=m, n)`` (spm_erp_L.m:112).

    ``spm_erp_L.m:112`` builds ``L = sparse(1:m, 1:m, P.L, m, n)`` -- a diagonal
    gain matrix (one channel per source). The default ``p_l = ones(n)``
    (``spm_L_priors.m:84``) yields the identity ``I_n`` with ``Nc == n``.

    Parameters
    ----------
    p_l : torch.Tensor
        LFP channel gains ``P.L``, shape ``(n,)``. Default usage passes
        ``ones(n)`` (identity).
    n : int
        Number of sources (``Nc == m == n`` in LFP mode).

    Returns
    -------
    torch.Tensor
        Diagonal spatial lead field, shape ``(n, n)``, float64.
    """
    p_l = torch.as_tensor(p_l, dtype=_F64)
    return torch.diag(p_l)


def ecd_spatial(g_ecd: Tensor, p_l: Tensor) -> Tensor:
    """ECD spatial lead field ``L[:,i] = G[:,:,i] @ P.L[:,i]`` (spm_erp_L.m:76).

    Built now, EXERCISED in Phase 36 (no fixture yet -- ECD needs a sensor montage
    + MNI coords). The physical gain ``g_ecd`` MUST already include the
    ``spm_cond_units`` unit-conditioning rescale (pitfall C5.3); Python only
    contracts it against the free dipole moments ``p_l``.

    Parameters
    ----------
    g_ecd : torch.Tensor
        MATLAB-exported physical gain, shape ``(Nc, 3, n)``, POST
        ``spm_cond_units`` (``spm_erp_L.m:74``).
    p_l : torch.Tensor
        Free dipole-moment 3-vectors per source, shape ``(3, n)``
        (``spm_L_priors.m:67``, ``E.L = 0``, ``V.L = 64``).

    Returns
    -------
    torch.Tensor
        Per-source spatial lead field ``L``, shape ``(Nc, n)``, float64.
    """
    g_ecd = torch.as_tensor(g_ecd, dtype=_F64)
    p_l = torch.as_tensor(p_l, dtype=_F64)
    # L[:, i] = G[:, :, i] @ P.L[:, i] for each source i (spm_erp_L.m:76).
    # einsum over the 3-component moment axis = the column-wise matvec.
    return torch.einsum("cdi,di->ci", g_ecd, p_l)


def build_lead_field(p_j: Tensor, l_spatial: Tensor) -> Tensor:
    """Full per-state lead field ``L_full = kron(P.J, L_spatial)`` (spm_lx_erp.m:33).

    ``P.J`` (``(8,)``) is the FIRST kron operand; ``l_spatial`` (``(Nc, n)``) the
    second. The Kronecker product is ``(Nc, 8n)``; the block at state ``s`` is
    ``P.J[s] * L_spatial`` occupying columns ``[s*n : (s+1)*n]``, so the full
    lead-field column index is ``state*n + source`` -- exactly the column-major
    ``cmc_flatten`` (``x.T.reshape(-1)``) ordering (Fact 2). A C-order ``reshape``
    flatten would put the block at ``source*8 + state`` and silently map the lead
    field to the wrong states (pitfall C5.2).

    Parameters
    ----------
    p_j : torch.Tensor
        Contributing-state vector ``P.J``, shape ``(8,)``.
    l_spatial : torch.Tensor
        Per-source spatial lead field, shape ``(Nc, n)``.

    Returns
    -------
    torch.Tensor
        Full per-state lead field ``L_full``, shape ``(Nc, 8n)``, float64.
    """
    p_j = torch.as_tensor(p_j, dtype=_F64).reshape(1, _N_STATES)
    l_spatial = torch.as_tensor(l_spatial, dtype=_F64)
    return torch.kron(p_j, l_spatial)


def project_to_scalp(
    states: Tensor, l_full: Tensor, x0: Tensor | None = None
) -> Tensor:
    """Scalp ERP ``y = (states - x0) @ L_full.T`` (spm_lx_erp.m header).

    ``spm_lx_erp`` is the observer ``y = G*x``; for a state-row trajectory the
    scalp ERP is ``(states - x0) @ L_full.T``. The baseline ``x0`` defaults to
    ``zeros(8n)`` (CMC ``M1``; the subtraction is then a no-op but is written
    explicitly for ECD generality).

    Parameters
    ----------
    states : torch.Tensor
        Source-state trajectory, shape ``(ns, 8n)`` or ``(Cnd, ns, 8n)``,
        column-major (:func:`pyro_dcm.forward_models.cmc_neural_mass.cmc_flatten`).
    l_full : torch.Tensor
        Full per-state lead field, shape ``(Nc, 8n)``.
    x0 : torch.Tensor or None, optional
        Baseline state, shape ``(8n,)``. ``None`` -> ``zeros(8n)``.

    Returns
    -------
    torch.Tensor
        Scalp ERP, shape ``(ns, Nc)`` or ``(Cnd, ns, Nc)``, float64.
    """
    states = torch.as_tensor(states, dtype=_F64)
    l_full = torch.as_tensor(l_full, dtype=_F64)
    if x0 is None:
        x0 = torch.zeros(l_full.shape[1], dtype=_F64, device=states.device)
    else:
        x0 = torch.as_tensor(x0, dtype=_F64)
    return (states - x0) @ l_full.transpose(-2, -1)
