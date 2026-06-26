"""Topology / permutation / adapter-bundle tests for the public MMN net (Phase 36).

LAPTOP-only (forward-only / structural, float64, <30s). Pins, on the laptop with
no MATLAB:

1. TOPOLOGY EQUALITY -- :func:`build_mmn_5source_network` reproduces the locked
   ``validation.export_to_mat._MS_*`` edge lists ELEMENT-WISE (the masks are
   reconstructed independently from the edge tuples and asserted equal), so the
   public builder and the SPM12-parity-gated fixture cannot silently diverge.
2. PERMUTATION / PRECISION wiring -- :func:`mmn_cmc_params`'s ``sp_inhibition_gain``
   sets the FREE ``P.G[:,0]`` at the precision nodes, which moves the parameterised
   ``G[:,6]`` (sp self-inhibition, ``J_PERM[0] == 6``, ``spm_fx_cmc.m:151``) and
   NOT ``G[:,0]`` -- the Phase-33/34 permutation guard reused.
3. ADAPTER BUNDLE -- the bundle carries every expected key with the right shapes
   and feeds :func:`pyro_dcm.simulators.simulate_erp_dcm` to a finite trajectory
   with a NON-ZERO scalp difference wave (forward-only).
4. NEGATIVE control -- ``fwd_bwd_flag="forward"`` vs ``"backward"`` place the
   ``b_edge`` modulation on DIFFERENT extrinsic edges.
5. NO MNI coordinates are emitted (LFP scope).

References
----------
SPM12 ``spm_fx_cmc.m:151`` (the intrinsic permutation ``J_PERM``),
``spm_gen_Q.m:65-67`` (the ``diag(B) -> Q.G[:,0]`` precision path). Ranlund et al.
(2016) / Adams et al. (2013) (the swept self-inhibition / precision knob).
"""

from __future__ import annotations

import torch

from pyro_dcm.forward_models.erp_coupled_system import (
    apply_condition_modulation,
    parameterize_cmc_network,
)
from pyro_dcm.forward_models.mmn_reference import (
    build_mmn_5source_network,
    mmn_cmc_params,
)
from pyro_dcm.simulators.erp_simulator import simulate_erp_dcm
from validation.export_to_mat import (
    _MS_B_DIAG,
    _MS_B_EDGE,
    _MS_BACKWARD_EDGES,
    _MS_FORWARD_EDGES,
    _MS_INPUT_SOURCES,
    _MS_LATERAL_EDGES,
    _MS_N,
    _MS_PRECISION_NODES,
)

_F64 = torch.float64
_A1L, _A1R, _RIFG = 0, 1, 4


def _expected_edge_mask(edges: tuple[tuple[int, int], ...]) -> torch.Tensor:
    """Reconstruct a ``(5,5)`` binary presence mask from ``[to, from]`` edges."""
    mask = torch.zeros(_MS_N, _MS_N, dtype=_F64)
    for to_i, from_i in edges:
        mask[to_i, from_i] = 1.0
    return mask


def test_topology_equality_vs_ms_constants() -> None:
    """The public builder reproduces the locked ``_MS_*`` topology element-wise.

    The four ``a_masks`` (forward, forward, backward, backward), the deviant ``B``
    value matrix (``b_edge`` on every extrinsic edge + ``b_diag`` on the precision
    diag), and the input ``c_mask`` are each rebuilt independently from the
    ``validation.export_to_mat._MS_*`` edge tuples and asserted equal.
    """
    net = build_mmn_5source_network()

    exp_fwd = _expected_edge_mask(_MS_FORWARD_EDGES + _MS_LATERAL_EDGES)
    exp_bwd = _expected_edge_mask(_MS_BACKWARD_EDGES)
    a_masks = net["a_masks"]
    assert torch.equal(a_masks[0], exp_fwd)
    assert torch.equal(a_masks[1], exp_fwd)
    assert torch.equal(a_masks[2], exp_bwd)
    assert torch.equal(a_masks[3], exp_bwd)

    exp_b = torch.zeros(_MS_N, _MS_N, dtype=_F64)
    for to_i, from_i in _MS_FORWARD_EDGES + _MS_LATERAL_EDGES + _MS_BACKWARD_EDGES:
        exp_b[to_i, from_i] = _MS_B_EDGE
    for node in _MS_PRECISION_NODES:
        exp_b[node, node] = _MS_B_DIAG
    assert torch.equal(net["b_masks"][0], exp_b)

    exp_c = torch.zeros(_MS_N, 1, dtype=_F64)
    for src in _MS_INPUT_SOURCES:
        exp_c[src, 0] = 1.0
    assert torch.equal(net["c_mask"], exp_c)

    assert tuple(net["precision_nodes"]) == _MS_PRECISION_NODES
    assert net["source_names"] == ("A1L", "A1R", "STGL", "STGR", "rIFG")
    assert torch.equal(net["x_design"], torch.tensor([[0.0], [1.0]], dtype=_F64))


def test_no_mni_coords_emitted() -> None:
    """No source-coordinate / MNI field is present (LFP scope, MUST-VERIFY)."""
    net = build_mmn_5source_network()
    for key in net:
        assert "mni" not in key.lower()
        assert "coord" not in key.lower()
        assert "pos" not in key.lower()


def test_sp_inhibition_gain_moves_g6_not_g0() -> None:
    """``sp_inhibition_gain`` -> free ``P.G[:,0]`` -> parameterised ``G[:,6]``.

    The permutation guard: perturbing ``sp_inhibition_gain`` (which sets the FREE
    ``P.G[node,0]`` at the precision nodes) must move the parameterised ``G[:,6]``
    (sp self-inhibition, ``J_PERM[0] == 6``) at rIFG and leave the parameterised
    ``G[:,0]`` (NOT in ``J_PERM[:4]``) byte-unchanged -- NOT ``G[:,0]`` directly.
    """
    bundle_lo = mmn_cmc_params(0.0, 0.5, 0.5, fwd_bwd_flag="both")
    bundle_hi = mmn_cmc_params(0.4, 0.5, 0.5, fwd_bwd_flag="both")

    x_dev = bundle_lo["x_design"][1]  # deviant row
    q_lo = apply_condition_modulation(bundle_lo["p"], x_dev)
    q_hi = apply_condition_modulation(bundle_hi["p"], x_dev)
    params_lo = parameterize_cmc_network(q_lo, _MS_N)
    params_hi = parameterize_cmc_network(q_hi, _MS_N)

    # The precision knob moves the parameterised sp self-inhibition G[:,6]...
    assert not torch.allclose(params_lo["G"][_RIFG, 6], params_hi["G"][_RIFG, 6])
    assert not torch.allclose(params_lo["G"][_A1L, 6], params_hi["G"][_A1L, 6])
    assert not torch.allclose(params_lo["G"][_A1R, 6], params_hi["G"][_A1R, 6])
    # ... and NOT the parameterised G[:,0] (the permutation trap).
    assert torch.equal(params_lo["G"][:, 0], params_hi["G"][:, 0])


def test_free_pg0_set_at_precision_nodes_only() -> None:
    """``sp_inhibition_gain`` lands on the FREE ``P.G[:,0]`` at {rIFG,A1L,A1R}."""
    g = 0.37
    bundle = mmn_cmc_params(g, 0.5, 0.5)
    p_g = bundle["p"]["G"]
    assert p_g.shape == (_MS_N, 4)
    for node in (_RIFG, _A1L, _A1R):
        assert p_g[node, 0].item() == g
    # The non-precision sources (STGL=2, STGR=3) carry no precision gain.
    assert p_g[2, 0].item() == 0.0
    assert p_g[3, 0].item() == 0.0


def test_adapter_bundle_shapes_and_simulation() -> None:
    """The bundle has every key with the right shape and drives a finite sim.

    Feeding ``simulate_erp_dcm`` the bundle produces a finite source trajectory
    and a NON-ZERO scalp difference wave (the deviant-vs-standard MMN signal;
    forward-only, <30s laptop).
    """
    bundle = mmn_cmc_params(0.3, 0.5, 0.6, fwd_bwd_flag="both")

    assert set(bundle) == {"p", "a_masks", "b_masks", "c_mask", "x_design", "l_full"}
    p = bundle["p"]
    assert p["T"].shape == (_MS_N, 4)
    assert p["G"].shape == (_MS_N, 4)
    assert p["C"].shape == (_MS_N, 1)
    assert p["S"].shape == (_MS_N, 1)
    assert p["R"].shape == (1, 2)
    assert len(p["A"]) == 4
    assert all(a.shape == (_MS_N, _MS_N) for a in p["A"])
    assert len(p["B"]) == 1 and p["B"][0].shape == (_MS_N, _MS_N)
    assert bundle["l_full"].shape == (_MS_N, 8 * _MS_N)
    assert bundle["x_design"].shape == (2, 1)

    out = simulate_erp_dcm(
        bundle["p"], bundle["x_design"], _MS_N, l_full=bundle["l_full"]
    )
    assert torch.isfinite(out["states"]).all()
    diff_scalp = out["difference_wave_scalp"]
    assert diff_scalp is not None
    assert torch.isfinite(diff_scalp).all()
    assert diff_scalp.abs().max() > 0.0


def test_fwd_bwd_flag_toggles_b_edge_placement() -> None:
    """``"forward"`` vs ``"backward"`` place ``b_edge`` on DIFFERENT edges.

    The Garrido/Ranlund model-space toggle: forward carries ``b_edge`` on the
    forward+lateral edges, backward on the backward edges; the off-diagonal
    (extrinsic) B placements must differ. The precision diag is identical (it is
    set by ``a1_b_gain`` / ``rifg_b_gain``, independent of the flag).
    """
    b_fwd = mmn_cmc_params(0.3, 0.5, 0.5, fwd_bwd_flag="forward")["b_masks"][0]
    b_bwd = mmn_cmc_params(0.3, 0.5, 0.5, fwd_bwd_flag="backward")["b_masks"][0]

    assert not torch.equal(b_fwd, b_bwd)

    # Forward edge A1L->STGL == (STGL=2, A1L=0) is live in "forward", dead in
    # "backward"; backward edge rIFG->STGL == (STGL=2, rIFG=4) the reverse.
    assert b_fwd[2, 0].item() == _MS_B_EDGE and b_bwd[2, 0].item() == 0.0
    assert b_bwd[2, _RIFG].item() == _MS_B_EDGE and b_fwd[2, _RIFG].item() == 0.0
    # Both keep the precision diag (flag-independent).
    assert b_fwd[_RIFG, _RIFG].item() == 0.5
    assert b_bwd[_RIFG, _RIFG].item() == 0.5


def test_invalid_fwd_bwd_flag_raises() -> None:
    """An unknown ``fwd_bwd_flag`` raises ``ValueError`` (expected vs actual)."""
    try:
        mmn_cmc_params(0.3, 0.5, 0.5, fwd_bwd_flag="sideways")
    except ValueError as exc:
        assert "sideways" in str(exc)
    else:  # pragma: no cover - guard must raise
        raise AssertionError("mmn_cmc_params must reject an unknown fwd_bwd_flag")
