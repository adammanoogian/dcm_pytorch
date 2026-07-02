"""3-node visual-collision network: topology + permutation guard + cascade smoke.

Forward-only, 3-node (V5/MT -> SPL/IPS -> PMd) discipline mirroring
:mod:`tests.test_collision_reference` (the 2-node gate) scoped to the extended
network of :mod:`pyro_dcm.forward_models.collision_3node_reference`. The node set +
directed edges follow Long et al. 2026 (bioRxiv 10.64898/2026.06.21.732202; MEG
millisecond timing licenses the frontal node) + Zbaren et al. 2024 (Brain Struct
Funct 10.1007/s00429-024-02815-2; DCM top-down parietal->visual).

SPM12-parity scope (honest note): identical to the 2-node case -- the existing
5-source SPM12 machinery-parity fixture already gates the n-independent LFP
lead-field + CMC forward + ERP-integrator machinery this network REUSES verbatim, so
the reuse IS the accepted parity gate. A dedicated 3-node SPM12 fixture is an
OPTIONAL MATLAB follow-up and is EXPLICITLY DEFERRED here (same deferral as 2-node).

Load-bearing assertions: (1) the graph is the V5/MT->SPL->PMd feedforward chain with
top-down backward edges + input into V5/MT only; (2) free P.G[node,0] moves the
parameterised G[:,6] (sp self-inhibition) at all three nodes; (3) a HIGHER V5/MT sp
self-inhibition value yields a SMALLER V5/MT |difference wave| (the validated monotone
direction); (4) the SZ-vs-healthy blunting PROPAGATES up the hierarchy (per-node
SZ/healthy peak ratio < 1 at all three nodes) -- the 3-node scientific claim.

All tests are plain pytest functions AND callable from the ``__main__`` block so the
plan can verify on the laptop without invoking pytest (the formal pytest run routes to
M3 per the standing ALL-pytest-to-M3 rule).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# ``validation`` is a repo-root package, not an installed one; add the project root so
# the lazy ``_collision_scalars`` read of the locked ``_MS_*`` constants resolves
# (mirrors test_collision_reference).
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pyro_dcm.forward_models import (  # noqa: E402
    C3_MNI_COORDS,
    build_collision_3node_network,
    collision_3node_cmc_params,
)
from pyro_dcm.forward_models.erp_coupled_system import (  # noqa: E402
    parameterize_cmc_network,
)
from pyro_dcm.simulators import simulate_erp_dcm  # noqa: E402

_F64 = torch.float64
_N = 3
# LFP-identity readout -> one channel per source; V5/MT = 0, SPL/IPS = 1, PMd = 2.
_V5MT, _SPL, _PMD = 0, 1, 2
_SP_VOLTAGE_COL = 2  # sp voltage in the (ns, n, 8) difference_wave.
_VIOLATION_B_GAIN = 0.5
# Non-swept node self-inhibition values held fixed while V5/MT is swept.
_SPL_FIXED = 1.5
_PMD_FIXED = 1.5


def test_topology_3node() -> None:
    """The V5/MT->SPL->PMd graph: fwd [1,0]+[2,1], bwd [0,1]+[1,2], C into node 0."""
    net = build_collision_3node_network()
    assert tuple(net["source_names"]) == ("V5/MT", "SPL/IPS", "PMd")
    assert net["precision_nodes"] == (0, 1, 2)

    x_design = net["x_design"]
    assert x_design.shape == (2, 1)
    assert torch.equal(x_design, torch.tensor([[0.0], [1.0]], dtype=_F64))

    a_masks = net["a_masks"]
    assert len(a_masks) == 4
    for m in a_masks:
        assert m.shape == (_N, _N)

    fwd, bwd = a_masks[0], a_masks[2]
    # Forward: V5/MT->SPL [1,0], SPL->PMd [2,1].
    assert fwd[1, 0] == 1.0 and fwd[2, 1] == 1.0
    # Backward: SPL->V5/MT [0,1], PMd->SPL [1,2].
    assert bwd[0, 1] == 1.0 and bwd[1, 2] == 1.0
    # No skip / no reverse-direction leakage across fwd vs bwd.
    assert fwd[0, 1] == 0.0 and fwd[1, 2] == 0.0 and fwd[2, 0] == 0.0
    assert bwd[1, 0] == 0.0 and bwd[2, 1] == 0.0 and bwd[0, 2] == 0.0
    # No self-edges in the presence masks.
    for i in range(_N):
        assert fwd[i, i] == 0.0 and bwd[i, i] == 0.0

    c_mask = net["c_mask"]
    assert c_mask.shape == (_N, 1)
    assert c_mask[0, 0] == 1.0 and c_mask[1, 0] == 0.0 and c_mask[2, 0] == 0.0

    # MNI anchors present for every node (documentation contract).
    assert set(C3_MNI_COORDS) == {"V5/MT", "SPL/IPS", "PMd"}
    print("[PASS] topology: 3-node V5/MT->SPL->PMd fwd [1,0]+[2,1], bwd [0,1]+[1,2], "
          "no skip/self, C into node 0, x_design=[[0],[1]], MNI anchors present")


def test_permutation_guard_3node() -> None:
    """Free P.G[node,0] moves the parameterised G[:,6] at all three nodes, not G[:,0]."""
    nodes = list(build_collision_3node_network()["precision_nodes"])
    p0 = collision_3node_cmc_params(0.0, 0.0, 0.0, _VIOLATION_B_GAIN, "both")["p"]
    p1 = collision_3node_cmc_params(0.7, 0.7, 0.7, _VIOLATION_B_GAIN, "both")["p"]
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
          f"(>0), moved_g0={moved_g0:.3e} (==0)  [J_PERM[0]=6 confirmed at 3 nodes]")


# Both in the monotone high-self-inhibition regime (>~1.0); below ~1.0 the input node
# inherits an early evoked transient that inverts the relation (documented caveat).
_GAIN_LOW_SELFINHIB = 1.5
_GAIN_HIGH_SELFINHIB = 3.0


def _drive_peaks(v5: float, spl: float, pmd: float) -> tuple[int, list[float]]:
    """Forward-simulate and return (ns, per-node peak |sp-voltage difference wave|)."""
    b = collision_3node_cmc_params(v5, spl, pmd, _VIOLATION_B_GAIN, "both")
    sim = simulate_erp_dcm(b["p"], b["x_design"], _N, l_full=b["l_full"])
    dw = sim["difference_wave"]  # (ns, n, 8)
    dw = dw.detach() if hasattr(dw, "detach") else dw
    ns = sim["pst"].shape[0]
    assert dw.shape == (ns, _N, 8), f"expected (ns, 3, 8), got {tuple(dw.shape)}"
    assert torch.isfinite(dw).all(), "difference_wave non-finite"
    peaks = [dw[:, i, _SP_VOLTAGE_COL].abs().max().item() for i in range(_N)]
    return ns, peaks


def test_forward_smoke_and_monotone_gain_3node() -> None:
    """Forward smoke: finite (ns, 3, 8); higher V5/MT gain -> smaller V5/MT |dw|."""
    ns_low, peaks_low = _drive_peaks(_GAIN_LOW_SELFINHIB, _SPL_FIXED, _PMD_FIXED)
    ns_high, peaks_high = _drive_peaks(_GAIN_HIGH_SELFINHIB, _SPL_FIXED, _PMD_FIXED)
    assert ns_low == ns_high
    peak_low, peak_high = peaks_low[_V5MT], peaks_high[_V5MT]
    assert peak_high < peak_low, (
        "monotone gate FAILED: HIGHER V5/MT sp self-inhibition must yield a SMALLER "
        f"|difference wave| at V5/MT; got peak(self-inhib={_GAIN_LOW_SELFINHIB})="
        f"{peak_low:.4e}, peak(self-inhib={_GAIN_HIGH_SELFINHIB})={peak_high:.4e}."
    )
    print(f"[PASS] forward smoke + monotone: difference_wave finite (ns={ns_low}, 3, 8); "
          f"V5/MT |dw| peak(self-inhib={_GAIN_LOW_SELFINHIB})={peak_low:.4e} > "
          f"peak(self-inhib={_GAIN_HIGH_SELFINHIB})={peak_high:.4e}")


# Canonical healthy vs SZ self-inhibition operating points (mirrors the actinf
# adapter: healthy kappa=0.60 -> V5 si=1.50 / omega=0.40 -> SPL si=2.29; SZ
# kappa=0.20 -> V5 si=3.00 / omega=0.80 -> SPL si=3.00; PMd SET at 1.50 both).
_HEALTHY_KNOBS = (1.50, 2.29, 1.50)
_SZ_KNOBS = (3.00, 3.00, 1.50)


def test_blunting_propagates_up_hierarchy() -> None:
    """SZ (high self-inhibition) blunts the violation-evoked peak at ALL three nodes.

    The 3-node scientific claim: with kappa->V5/MT and omega->SPL both raising
    self-inhibition in the SZ regime (PMd held fixed), the per-node SZ/healthy peak
    ratio is < 1 at every node -- the blunting propagates UP the feedforward chain,
    reaching the SET-fixed frontal node purely through the ascending edges.
    """
    _, healthy_peaks = _drive_peaks(*_HEALTHY_KNOBS)
    _, sz_peaks = _drive_peaks(*_SZ_KNOBS)
    ratios = [sz_peaks[i] / healthy_peaks[i] if healthy_peaks[i] > 0 else float("nan")
              for i in range(_N)]
    for i, name in enumerate(("V5/MT", "SPL/IPS", "PMd")):
        assert ratios[i] < 1.0, (
            f"blunting gate FAILED at {name}: SZ/healthy peak ratio must be < 1 "
            f"(SZ blunted); got ratio={ratios[i]:.4f} (SZ peak={sz_peaks[i]:.3e}, "
            f"healthy peak={healthy_peaks[i]:.3e})."
        )
    # The PMd frontal node has NO self-inhibition change (SET fixed) yet is blunted
    # -> the effect is inherited through the feedforward chain, not a local knob.
    print(f"[PASS] blunting propagates: SZ/healthy peak ratio V5/MT={ratios[0]:.4f} "
          f"SPL/IPS={ratios[1]:.4f} PMd={ratios[2]:.4f} (all < 1; PMd blunted via "
          "the ascending chain despite a FIXED frontal knob)")


if __name__ == "__main__":
    test_topology_3node()
    test_permutation_guard_3node()
    test_forward_smoke_and_monotone_gain_3node()
    test_blunting_propagates_up_hierarchy()
    print("[3-NODE GATE] PASSED")
