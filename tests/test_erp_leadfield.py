"""Structural guards for the single-dipole lead field + scalp projection (Phase 35).

These tests are authored BEFORE ``erp_leadfield`` exists, so the suite is RED on
first run (ImportError) -- that is the point: the ``P.J = state index 2`` guard
and the ``kron`` column-major guard are the single most important lead-field
correctness checks (pitfalls C5.1/C5.2) and must exist before any implementation.

The guards pin, on the laptop with no MATLAB (pure-torch, sub-second, float64):

1. P.J guard (LEAD-02) -- ``cmc_default_pj()`` is the one-hot ``e_2`` (sp voltage,
   index 2), asserted ``argmax == 2`` AND ``!= 6`` (the dp-voltage inversion
   trap), summing to 1.0 (``spm_L_priors.m:108``).
2. kron column-major guard (LEAD-02) -- a distinct-valued ``L_spatial`` + a
   non-trivial ``p_j`` proves ``build_lead_field`` block ``s`` occupies columns
   ``[s*n:(s+1)*n]`` and equals ``p_j[s] * L_spatial`` (a C-order ``reshape``
   flatten would land the block at ``source*8 + state`` and FAIL) (``spm_lx_erp.m:33``).
3. LFP identity -- ``lfp_spatial(ones(n), n) == eye(n)``; ``build_lead_field(e_2,
   eye(n))`` has the sp-voltage block ``== I_n`` and all other state blocks ``== 0``
   (``spm_erp_L.m:112``).
4. Projection through the identity LFP lead field returns each source's
   sp-voltage trace; ``x0`` defaults to ``zeros(8n)`` (``spm_lx_erp.m`` header).
5. float64 guard at the lead-field + projection boundary (pitfall N1).

The CMC 8-state column layout (``cmc_neural_mass.py:20-33``), 0-indexed:
``[ss_V=0, ss_I=1, sp_V=2, sp_I=3, ii_V=4, ii_I=5, dp_V=6, dp_I=7]``.
"""

from __future__ import annotations

import torch

from pyro_dcm.forward_models.erp_leadfield import (
    build_lead_field,
    cmc_default_pj,
    lfp_spatial,
    project_to_scalp,
)

_F64 = torch.float64


def test_cmc_default_pj_is_state_index_2() -> None:
    """``P.J`` default one-hot is at state index 2 (sp voltage), NOT 6.

    ``spm_L_priors.m:108`` sets ``pE.J = sparse(1,3,1,1,8)`` -> MATLAB column 3 ->
    0-indexed index 2 = superficial-pyramidal VOLTAGE (EEG is dominated by L2/3
    superficial-pyramidal depolarisation). Index 6 (deep-pyramidal voltage) is the
    inversion trap (pitfall C5.1). Hard-assert index 2 and explicitly NOT 6.
    """
    pj = cmc_default_pj()
    assert pj.shape == (8,)
    assert pj.dtype == _F64
    assert int(pj.argmax().item()) == 2
    assert int(pj.argmax().item()) != 6
    assert torch.isclose(pj.sum(), torch.tensor(1.0, dtype=_F64))
    # Exactly one-hot: value 1.0 at index 2, zeros elsewhere.
    expected = torch.zeros(8, dtype=_F64)
    expected[2] = 1.0
    assert torch.equal(pj, expected)


def test_build_lead_field_kron_column_major() -> None:
    """``build_lead_field`` is column-major ``kron(P.J, L_spatial)`` (spm_lx_erp.m:33).

    With a DISTINCT-valued ``L_spatial`` (``arange`` reshape) and a non-trivial
    ``p_j``, the full lead field's column block ``s`` (columns ``[s*n:(s+1)*n]``)
    must equal ``p_j[s] * L_spatial`` element-wise -- the column index is
    ``state*n + source``, matching the proven column-major ``cmc_flatten``
    (``x.T.reshape(-1)``). A C-order ``reshape`` flatten would place the block at
    ``source*8 + state`` and FAIL this guard (pitfall C5.2).
    """
    nc = 3
    n = 3
    l_spatial = (torch.arange(nc * n, dtype=_F64) + 1.0).reshape(nc, n)
    p_j = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0], dtype=_F64)

    l_full = build_lead_field(p_j, l_spatial)
    assert l_full.shape == (nc, 8 * n)
    assert l_full.dtype == _F64
    for s in range(8):
        block = l_full[:, s * n : (s + 1) * n]
        assert torch.equal(block, p_j[s] * l_spatial), (
            f"state block {s} must equal p_j[{s}]*L_spatial (column-major); "
            "a C-order flatten would land it at source*8+state"
        )


def test_lfp_spatial_default_identity() -> None:
    """``lfp_spatial(ones(n), n) == eye(n)`` (spm_erp_L.m:112, default P.L = ones).

    ``spm_erp_L.m:112`` builds ``L = sparse(1:m, 1:m, P.L, m, n)`` -- a diagonal
    gain. Default ``P.L = ones`` -> identity (one channel per source, ``Nc == n``).
    """
    n = 4
    l_sp = lfp_spatial(torch.ones(n, dtype=_F64), n)
    assert l_sp.shape == (n, n)
    assert l_sp.dtype == _F64
    assert torch.equal(l_sp, torch.eye(n, dtype=_F64))

    # Non-trivial gain -> diagonal of that gain.
    gain = torch.tensor([2.0, 0.5, 3.0, 1.0], dtype=_F64)
    assert torch.equal(lfp_spatial(gain, n), torch.diag(gain))


def test_build_lead_field_lfp_identity_blocks() -> None:
    """``build_lead_field(e_2, I_n)``: sp-voltage block == I_n, all others == 0.

    In LFP identity mode the full lead field is ``kron(e_2, I_n)`` -> the only
    non-zero state block is ``s = 2`` (sp voltage), which equals ``I_n``; every
    other state block is exactly zero (``spm_lx_erp.m:31-33`` + ``spm_erp_L.m:112``).
    """
    n = 3
    pj = cmc_default_pj()
    l_full = build_lead_field(pj, lfp_spatial(torch.ones(n, dtype=_F64), n))
    assert l_full.shape == (n, 8 * n)
    for s in range(8):
        block = l_full[:, s * n : (s + 1) * n]
        if s == 2:
            assert torch.equal(block, torch.eye(n, dtype=_F64))
        else:
            assert torch.equal(block, torch.zeros(n, n, dtype=_F64))


def test_project_to_scalp_through_identity_lfp() -> None:
    """Projection through identity LFP returns each source's sp-voltage trace.

    ``project_to_scalp(states, kron(e_2, I_n))[..., j]`` must equal source ``j``'s
    superficial-pyramidal voltage column ``states[:, 2*n + j]`` (column-major flat
    index ``state*n + source`` with ``state = 2``). ``x0`` defaults to ``zeros(8n)``
    so the explicit ``(states - x0)`` subtraction is a no-op here (CMC M1).
    """
    n = 3
    ns = 7
    torch.manual_seed(0)
    states = torch.randn(ns, 8 * n, dtype=_F64)
    l_full = build_lead_field(cmc_default_pj(), lfp_spatial(torch.ones(n), n))

    y = project_to_scalp(states, l_full)
    assert y.shape == (ns, n)
    assert y.dtype == _F64
    for j in range(n):
        assert torch.equal(y[:, j], states[:, 2 * n + j])

    # x0 default == zeros: passing explicit zeros gives identical output.
    y0 = project_to_scalp(states, l_full, x0=torch.zeros(8 * n, dtype=_F64))
    assert torch.equal(y, y0)

    # A non-zero x0 baseline subtracts before projecting.
    x0 = torch.ones(8 * n, dtype=_F64)
    y_sub = project_to_scalp(states, l_full, x0=x0)
    assert torch.allclose(y_sub, project_to_scalp(states - x0, l_full))


def test_project_to_scalp_batched_conditions() -> None:
    """``project_to_scalp`` handles a ``(Cnd, ns, 8n)`` batch -> ``(Cnd, ns, Nc)``."""
    n = 2
    ns = 5
    cnd = 2
    torch.manual_seed(1)
    states = torch.randn(cnd, ns, 8 * n, dtype=_F64)
    l_full = build_lead_field(cmc_default_pj(), lfp_spatial(torch.ones(n), n))
    y = project_to_scalp(states, l_full)
    assert y.shape == (cnd, ns, n)
    for c in range(cnd):
        assert torch.equal(y[c], project_to_scalp(states[c], l_full))


def test_lead_field_outputs_float64() -> None:
    """Lead field + projection outputs are float64 at the boundary (pitfall N1)."""
    n = 2
    l_sp = lfp_spatial(torch.ones(n), n)
    l_full = build_lead_field(cmc_default_pj(), l_sp)
    assert l_sp.dtype == _F64
    assert l_full.dtype == _F64
    states = torch.zeros(3, 8 * n, dtype=_F64)
    assert project_to_scalp(states, l_full).dtype == _F64
