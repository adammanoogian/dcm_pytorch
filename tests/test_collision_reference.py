"""2-node visual-collision network: topology + permutation guard + forward smoke.

Forward-only, 2-node (V5/MT <-> rIPC) parity/permutation discipline mirroring
``scripts/demo_mmn_precision_sweep.py`` (``run_parity_gate`` +
``assert_permutation_guard``) scoped to the visual collision network of
:mod:`pyro_dcm.forward_models.collision_reference`.

SPM12-parity scope (honest note): the existing 5-source SPM12 machinery-parity
fixture (``validation/data/erp_leadfield_fixtures.mat``, validated to 1e-7 in the
MMN demo) already gates the LFP lead-field + CMC forward + ERP-integrator MACHINERY
this 2-node network REUSES verbatim -- that machinery is n-independent, so the
reuse IS the accepted SPM12-parity gate for this network. A NEW 2-node end-to-end
SPM12 ``collision_leadfield_fixture.mat`` is an OPTIONAL MATLAB follow-up
(Phase-133.1 RESEARCH Open Question #4) and is EXPLICITLY DEFERRED here -- this
module does NOT build or block on a new MATLAB fixture.

The single load-bearing assertion is the simulator-side monotone direction: a
HIGHER sp self-inhibition value at V5/MT yields a SMALLER |difference wave| (the
validated MMN-sweep direction). The kappa->knob INVERSION that turns this into the
schizophrenia prediction lives on the actinf adapter side (Plan 133.1-02), not
here.

All three tests are plain pytest functions AND are callable from the ``__main__``
block so the plan can verify on the laptop without invoking pytest (the formal
pytest run routes to M3 per the standing ALL-pytest-to-M3 rule).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# ``validation`` is a repo-root package, not an installed one (pyproject ships only
# ``src/pyro_dcm``); add the project root so the lazy ``_collision_scalars`` read of
# the locked ``_MS_*`` free-log / B-value constants resolves (mirrors the demo +
# generate_publication_figures shim). Harmless under pytest (already-in-path guard).
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pyro_dcm.forward_models import (  # noqa: E402
    build_collision_2node_network,
    collision_cmc_params,
)
from pyro_dcm.forward_models.erp_coupled_system import (  # noqa: E402
    parameterize_cmc_network,
)
from pyro_dcm.simulators import simulate_erp_dcm  # noqa: E402

_F64 = torch.float64
_N = 2
# LFP-identity readout -> one channel per source; V5/MT = 0, rIPC = 1.
_V5MT, _RIPC = 0, 1
# Fixed violated-condition diag(B) gain (mirrors the MMN b_diag baseline).
_VIOLATION_B_GAIN = 0.5
# rIPC sp self-inhibition held fixed while V5/MT is swept.
_IPC_FIXED = 0.5


def test_topology_2node() -> None:
    """The 2-node V5/MT<->rIPC graph: forward [1,0], backward [0,1], C into node 0."""
    net = build_collision_2node_network()
    assert tuple(net["source_names"]) == ("V5/MT", "rIPC")
    assert net["precision_nodes"] == (0, 1)

    x_design = net["x_design"]
    assert x_design.shape == (2, 1)
    assert torch.equal(x_design, torch.tensor([[0.0], [1.0]], dtype=_F64))

    a_masks = net["a_masks"]
    assert len(a_masks) == 4
    for m in a_masks:
        assert m.shape == (_N, _N)

    fwd, bwd = a_masks[0], a_masks[2]
    # Forward V5/MT -> rIPC = [to=1, from=0]; backward rIPC -> V5/MT = [to=0, from=1].
    assert fwd[1, 0] == 1.0 and fwd[0, 1] == 0.0
    assert bwd[0, 1] == 1.0 and bwd[1, 0] == 0.0
    # No lateral / no self in the forward or backward presence masks.
    assert fwd[0, 0] == 0.0 and fwd[1, 1] == 0.0
    assert bwd[0, 0] == 0.0 and bwd[1, 1] == 0.0

    c_mask = net["c_mask"]
    assert c_mask.shape == (_N, 1)
    assert c_mask[0, 0] == 1.0 and c_mask[1, 0] == 0.0  # C drives V5/MT only.
    print("[PASS] topology: 2-node V5/MT->rIPC fwd [1,0], rIPC->V5/MT bwd [0,1], "
          "no lateral, C into node 0, x_design=[[0],[1]]")


def test_permutation_guard_2node() -> None:
    """Free P.G[node,0] moves the parameterised G[:,6] (sp self-inhibition), not G[:,0].

    Mirrors ``assert_permutation_guard`` in ``demo_mmn_precision_sweep.py``:
    perturbing the free ``P.G[:,0]`` precision column must move the parameterised
    ``G[:,6]`` (via ``J_PERM[0] == 6``) and leave ``G[:,0]`` untouched -- the swept
    knob is precision / synaptic-gain, never a mis-indexed column.
    """
    nodes = list(build_collision_2node_network()["precision_nodes"])
    p0 = collision_cmc_params(0.0, 0.0, _VIOLATION_B_GAIN, "both")["p"]
    p1 = collision_cmc_params(0.7, 0.7, _VIOLATION_B_GAIN, "both")["p"]
    g0 = parameterize_cmc_network(p0, _N)["G"]  # (n, 10)
    g1 = parameterize_cmc_network(p1, _N)["G"]
    moved_g6 = (g1[nodes, 6] - g0[nodes, 6]).abs().max().item()
    moved_g0 = (g1[nodes, 0] - g0[nodes, 0]).abs().max().item()
    assert moved_g6 > 0.0, (
        "free P.G[:,0] must move parameterised G[:,6] (sp self-inhibition); "
        f"got moved_g6={moved_g6:.3e} (want >0). J_PERM[0]=6 broken."
    )
    assert moved_g0 == 0.0, (
        "free P.G[:,0] must leave parameterised G[:,0] fixed; "
        f"got moved_g0={moved_g0:.3e} (want 0). Permutation trap."
    )
    print(f"[PASS] permutation guard: P.G[:,0] -> G[:,6] moved_g6={moved_g6:.3e} "
          f"(>0), moved_g0={moved_g0:.3e} (==0)  [J_PERM[0]=6 confirmed]")


# Two self-inhibition values BOTH in the monotone high-self-inhibition regime
# (the SZ-relevant, low-kappa regime). Below ~1.0 the INPUT-node (V5/MT) evoked
# difference wave is dominated by an early evoked transient and the relation is
# NON-monotone -- exactly the documented "input-node early-transient" caveat of the
# 5-source MMN demo (where A1L also rises before it falls; only the 2-hop apex rIFG
# is monotone from gain 0). Above ~1.0 the V5/MT ascending-PE difference wave is
# robustly monotone-decreasing in self-inhibition (verified across 1.0..5.0): higher
# self-inhibition -> lower net sp gain -> smaller evoked prediction error. The gate
# therefore operates in that physically-meaningful precision regime.
_GAIN_LOW_SELFINHIB = 1.5
_GAIN_HIGH_SELFINHIB = 3.0


def test_forward_smoke_and_monotone_gain() -> None:
    """Forward smoke: finite difference_wave_scalp (ns, 2); higher gain -> smaller |dw|.

    Builds the bundle at a LOWER vs a HIGHER V5/MT sp self-inhibition value (rIPC
    held fixed), both within the monotone high-self-inhibition regime, runs the
    SPM-parity-verified forward, asserts the scalp difference wave is finite and
    shape ``(ns, 2)``, and asserts the HIGHER self-inhibition value yields the
    SMALLER peak ``|difference wave|`` at V5/MT -- the ascending-PE readout (sp
    voltage of the lower level). This is the validated monotone direction the actinf
    kappa-inversion (Plan 133.1-02) depends on: high kappa -> low self-inhibition ->
    large PE; SZ low kappa -> high self-inhibition -> blunted PE.
    """
    b_low = collision_cmc_params(
        _GAIN_LOW_SELFINHIB, _IPC_FIXED, _VIOLATION_B_GAIN, "both"
    )
    b_high = collision_cmc_params(
        _GAIN_HIGH_SELFINHIB, _IPC_FIXED, _VIOLATION_B_GAIN, "both"
    )

    sim_low = simulate_erp_dcm(b_low["p"], b_low["x_design"], _N, l_full=b_low["l_full"])
    sim_high = simulate_erp_dcm(
        b_high["p"], b_high["x_design"], _N, l_full=b_high["l_full"]
    )

    dw_low = sim_low["difference_wave_scalp"]  # (ns, Nc=2)
    dw_high = sim_high["difference_wave_scalp"]
    ns = sim_low["pst"].shape[0]
    assert dw_low.shape == (ns, _N), (
        f"expected (ns, 2)={(ns, _N)}, got {tuple(dw_low.shape)}"
    )
    assert dw_high.shape == (ns, _N)
    assert torch.isfinite(dw_low).all(), "difference_wave_scalp (low gain) non-finite"
    assert torch.isfinite(dw_high).all(), "difference_wave_scalp (high gain) non-finite"

    peak_low = dw_low[:, _V5MT].abs().max().item()
    peak_high = dw_high[:, _V5MT].abs().max().item()
    assert peak_high < peak_low, (
        "monotone gate FAILED: HIGHER sp self-inhibition value must yield a SMALLER "
        f"|difference wave| at V5/MT; got peak(self-inhib={_GAIN_LOW_SELFINHIB})="
        f"{peak_low:.4e}, peak(self-inhib={_GAIN_HIGH_SELFINHIB})={peak_high:.4e} "
        "(want higher-self-inhib < lower-self-inhib)."
    )
    print(f"[PASS] forward smoke + monotone: difference_wave_scalp finite (ns={ns}, "
          f"2); V5/MT |dw| peak(self-inhib={_GAIN_LOW_SELFINHIB})={peak_low:.4e} > "
          f"peak(self-inhib={_GAIN_HIGH_SELFINHIB})={peak_high:.4e}")


if __name__ == "__main__":
    test_topology_2node()
    test_permutation_guard_2node()
    test_forward_smoke_and_monotone_gain()
    print("[2-NODE GATE] PASSED")
