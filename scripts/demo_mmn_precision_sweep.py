"""Headline 5-source MMN precision-sweep demo (the ``actinf_physics`` artifact).

The Pyro-DCM v0.8.0 capstone (ERPDCM-04 / ERPDCM-06). Sweeps the superficial-
pyramidal self-inhibition gain (the precision / synaptic-gain knob of Adams et
al. 2013) over the SPM12-parity-gated 5-source auditory mismatch-negativity (MMN)
network of Ranlund et al. 2016 (A1 L/R -> STG L/R -> rIFG), and turns the figure
into a *function*: a ``gain -> |MMN|`` transfer curve the downstream consumer
imports.

The credibility of the figure rests on ONE gate (S3 -- "demo not gated" pitfall):
the demo FIRST reproduces a single fixed-reference forward point against the
byte-frozen SPM12 LFP lead-field fixture (``erp_leadfield_fixtures.mat``, Phase
35) within ``<= 1e-7`` (scalp ERP AND deviant-standard difference wave). If the
fixture is missing OR the parity check fails, the demo RAISES before any figure
is written. Only then does it run the pure-forward precision sweep.

The sweep is PURE-FORWARD over the verified precision knob: the free ``P.G[:,0]``
at the precision nodes {rIFG, A1L, A1R} drives the parameterised ``G[:,6]`` (sp
self-inhibition) via the Phase-33 intrinsic permutation ``J_PERM[0] == 6``
(``spm_fx_cmc.m:151``) -- it is NEVER indexed directly (the permutation trap, the
reused Phase-33 guard). Higher self-inhibition lowers the superficial-pyramidal
gain, shrinking the prediction-error response and so the MMN (Garrido et al.
2009; Adams et al. 2013).

This module is forward-only (no fitting). The LFP readout is source-space
(``P.J = e_2`` sp-voltage, identity single-dipole lead field); the difference-wave
SIGN is established in that source space (David & Friston 2003; Kiebel et al.
2006).

References
----------
SPM12 source: ``spm_gen_erp.m:69-86`` (the per-condition evoked loop),
``spm_lx_erp.m:31-33`` (``kron(P.J, L)`` lead field), ``spm_fx_cmc.m:151`` (the
``J_PERM`` intrinsic-gain permutation), ``spm_gen_Q.m:65-67`` (the
``diag(B) -> Q.G(:,1)`` precision path). The fixed-reference parity logic
(``_reference_p`` / ``_production_scalp``) is reused from the Phase-35 LEAD-05
ladder ``tests/test_spm_erp_leadfield_validation.py`` (the source of truth). The
demo's swept papers (FLAG for Zotero -- not yet keyed): Adams, R.A. et al. (2013),
Front. Psychiatry 4, 47; Ranlund, S. et al. (2016), Hum. Brain Mapp. 37, 351-365;
Garrido, M.I. et al. (2009), Clin. Neurophysiol. 120, 453-463; David, O. &
Friston, K.J. (2003), NeuroImage 20, 1743-1755; Bastos, A.M. et al. (2012),
Neuron 76, 695-711; Kiebel, S.J. et al. (2006), NeuroImage 30, 1273-1284.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

# ``validation`` is a repo-root package, not an installed one (pyproject ships
# only ``src/pyro_dcm``); add the project root so the gate can reuse the frozen
# fixture's locked ``_MS_*`` topology constants (mirrors generate_publication_figures).
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pyro_dcm.forward_models import (  # noqa: E402
    build_mmn_5source_network,
    mmn_cmc_params,
)
from pyro_dcm.forward_models.erp_coupled_system import (  # noqa: E402
    apply_condition_modulation,
    cmc_network_f,
    parameterize_cmc_network,
)
from pyro_dcm.forward_models.erp_leadfield import (  # noqa: E402
    build_lead_field,
    cmc_default_pj,
    lfp_spatial,
    project_to_scalp,
)
from pyro_dcm.simulators import simulate_erp_dcm  # noqa: E402
from pyro_dcm.utils.local_linearization import (  # noqa: E402
    integrate_local_linearization,
)
from validation.export_to_mat import (  # noqa: E402
    _MS_A_DEAD,
    _MS_A_LIVE,
    _MS_B_DIAG,
    _MS_B_EDGE,
    _MS_BACKWARD_EDGES,
    _MS_FORWARD_EDGES,
    _MS_INPUT_SOURCES,
    _MS_LATERAL_EDGES,
    _MS_N,
    _MS_PRECISION_NODES,
    _erp_gaussian_u_grid,
    _ms_log_block,
)

_F64 = torch.float64
_N_STATES = 8

# The byte-frozen SPM12 LFP lead-field + scalp-ERP ground truth (Plan 35-02;
# validation/data/ is mutagen-ignored, so the .mat lives in git).
_FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "validation"
    / "data"
    / "erp_leadfield_fixtures.mat"
)

# The Phase-35 LEAD-05 production-path tolerance (David & Friston 2003 forward).
_PARITY_TOL = 1e-7

# Source / channel indices (LFP identity readout -> one channel per source;
# A1L=0, A1R=1, STGL=2, STGR=3, rIFG=4; build_mmn_5source_network source order).
_RIFG, _A1L, _A1R = 4, 0, 1

# The classic MMN latency window (ms): the rIFG deviant-standard deflection at
# ~100 ms (Garrido et al. 2009; Naatanen et al. 2007). The windowed-minimum sign
# + attenuation are asserted here (NOT the global |peak|, which carries an early
# evoked transient -- see the SUMMARY / the Phase-36 honest-scope note).
_MMN_WINDOW_MS = (90.0, 120.0)

# Swept self-inhibition-gain grid: the regime where the MMN deflection is present
# (beyond ~1.5 the deflection has attenuated to noise and its windowed sign flips
# -- so the sweep covers the physically-meaningful precision range, Adams 2013).
_GAIN_GRID = torch.linspace(0.0, 1.5, 7, dtype=_F64)

# Fixed deviant B gains for the demo (the baseline == the fixture reference point:
# a1_b_gain == rifg_b_gain == _MS_B_DIAG, so gain=0 reproduces the gated forward).
_A1_B_GAIN = 0.5
_RIFG_B_GAIN = 0.5


def _reference_p() -> dict[str, Any]:
    """Reconstruct the EXACT free-log-space ``P`` of the frozen LFP fixture.

    Replicates ``_reference_p`` in the Phase-35 LEAD-05 ladder
    (``tests/test_spm_erp_leadfield_validation.py``, the source of truth) by
    reusing the locked ``_MS_*`` topology constants, so torch and the frozen
    MATLAB fixtures feed the IDENTICAL ``P`` + drive (pitfall V1). Free-log
    convention: live edges ``_MS_A_LIVE`` (``exp(0)``), dead ``_MS_A_DEAD``
    (``exp(-32)``); ``B`` distinct from ``A`` (``b_edge`` on every extrinsic
    edge, ``b_diag`` on the precision diag).

    Returns
    -------
    dict
        ``{"A": list[4] (5,5), "B": list[1] (5,5), "C": (5,1), "T": (5,4),
        "G": (5,4), "S": (5,1), "R": (1,2)}`` (the locked 34-01 schema, float64).
    """
    n = _MS_N
    a_blocks = [
        _ms_log_block(_MS_FORWARD_EDGES + _MS_LATERAL_EDGES),  # A{1} sp->ss
        _ms_log_block(_MS_FORWARD_EDGES + _MS_LATERAL_EDGES),  # A{2} sp->dp
        _ms_log_block(_MS_BACKWARD_EDGES),  # A{3} dp->sp
        _ms_log_block(_MS_BACKWARD_EDGES),  # A{4} dp->ii
    ]
    b1 = np.zeros((n, n), dtype=np.float64)
    for to_i, from_i in _MS_FORWARD_EDGES + _MS_LATERAL_EDGES + _MS_BACKWARD_EDGES:
        b1[to_i, from_i] = _MS_B_EDGE
    for node in _MS_PRECISION_NODES:
        b1[node, node] = _MS_B_DIAG
    c = np.full((n, 1), _MS_A_DEAD, dtype=np.float64)
    for src in _MS_INPUT_SOURCES:
        c[src, 0] = _MS_A_LIVE
    return {
        "A": [torch.as_tensor(b, dtype=_F64) for b in a_blocks],
        "B": [torch.as_tensor(b1, dtype=_F64)],
        "C": torch.as_tensor(c, dtype=_F64),
        "T": torch.zeros(n, 4, dtype=_F64),
        "G": torch.zeros(n, 4, dtype=_F64),
        "S": torch.zeros(n, 1, dtype=_F64),
        "R": torch.zeros(1, 2, dtype=_F64),
    }


def _torch_l_full() -> torch.Tensor:
    """Build the production torch LFP lead field ``kron(P.J, diag(P.L))`` (5,40).

    ``P.L = ones(5)`` -> identity spatial map (no amplification); ``P.J = e_2``
    (sp-voltage). Identical to the Phase-35 ladder's ``_torch_l_full``.
    """
    l_spatial = lfp_spatial(torch.ones(_MS_N, dtype=_F64), _MS_N)  # (5,5) identity
    return build_lead_field(cmc_default_pj(), l_spatial)  # (5,40)


def _load_fixture(path: Path) -> dict[str, Any]:
    """Load the frozen LFP lead-field + scalp-ERP fixture (Plan 35-02).

    Parameters
    ----------
    path : pathlib.Path
        Path to ``erp_leadfield_fixtures.mat``.

    Returns
    -------
    dict
        The provenance ``meta`` scalars + ``L_full`` (5,40), ``y_scalp`` (a
        length-``Cnd`` list of ``(ns, Nc)`` tensors) and ``diff_wave`` (ns, Nc).
    """
    import scipy.io as sio

    mat = sio.loadmat(str(path))
    meta = mat["meta"]

    def _m(name: str) -> np.ndarray:
        return np.asarray(meta[name][0, 0])

    cnd = int(_m("Cnd").ravel()[0])
    y_scalp = [
        torch.as_tensor(np.asarray(mat["y_scalp"][0, c], dtype=np.float64), dtype=_F64)
        for c in range(cnd)
    ]
    return {
        "Cnd": cnd,
        "N": int(_m("N").ravel()[0]),
        "Nc": int(_m("Nc").ravel()[0]),
        "X": torch.as_tensor(_m("X"), dtype=_F64),  # (Cnd, n_effects)
        "dt": float(_m("dt").ravel()[0]),
        "ns": int(_m("ns").ravel()[0]),
        "ons": float(_m("ons").ravel()[0]),
        "dur": float(_m("dur").ravel()[0]),
        "sus": float(_m("sus").ravel()[0]),
        "L_full": torch.as_tensor(
            np.asarray(mat["L_full"], dtype=np.float64), dtype=_F64
        ),
        "y_scalp": y_scalp,
        "diff_wave": torch.as_tensor(
            np.asarray(mat["diff_wave"], dtype=np.float64), dtype=_F64
        ),
    }


def _drive(fx: dict[str, Any], n_inp: int) -> torch.Tensor:
    """Frozen condition-independent Gaussian evoked drive ``U.u`` (ns, n_inp).

    Reuses ``_erp_gaussian_u_grid`` (the numpy ``spm_erp_u`` port that generated
    the fixture) with ``P.R = 0`` so torch and SPM integrate the IDENTICAL input
    grid (pitfall V1), mirroring the Phase-35 ladder.
    """
    r = np.zeros((n_inp, 2), dtype=np.float64)
    u = _erp_gaussian_u_grid(r, fx["ns"], fx["dt"], fx["ons"], fx["dur"], fx["sus"])
    return torch.as_tensor(u, dtype=_F64)


def _production_scalp(fx: dict[str, Any]) -> torch.Tensor:
    """Run the FULL production forward (jacrev integrator -> project) per condition.

    Mirrors ``_production_scalp`` in the Phase-35 LEAD-05 ladder: for each
    condition, ``apply_condition_modulation`` -> ``integrate_local_linearization``
    (``cmc_network_f``) -> ``project_to_scalp``. Returns the ``(Cnd, ns, Nc)``
    scalp stack at the fixed reference ``P``.
    """
    p = _reference_p()
    n = fx["N"]
    n_inp = int(p["C"].shape[1])
    inputs = _drive(fx, n_inp)
    l_full = _torch_l_full()
    x0 = torch.zeros(_N_STATES * n, dtype=_F64)
    scalp_list: list[torch.Tensor] = []
    for c in range(fx["Cnd"]):
        q = apply_condition_modulation(p, fx["X"][c])
        traj = integrate_local_linearization(
            lambda v, u, q=q: cmc_network_f(v, u, q, n), x0, inputs, fx["dt"]
        )
        scalp_list.append(project_to_scalp(traj, l_full))  # (ns, Nc)
    return torch.stack(scalp_list, dim=0)  # (Cnd, ns, Nc)


def run_parity_gate(fixture_path: Path = _FIXTURE_PATH) -> dict[str, float]:
    """Fixed-reference SPM forward-parity GATE (ERPDCM-06); raises on failure.

    Reproduces a single fixed-reference forward point against the byte-frozen
    SPM12 LFP lead-field fixture: the production scalp ERP and the
    deviant-standard difference wave must reproduce the frozen ``y_scalp`` /
    ``diff_wave`` within ``<= 1e-7`` (the Phase-35 LEAD-05 tolerance; the
    reference ``P`` is byte-identical to the fixture's ``_reference_p``). This
    runs on the LAPTOP (torch-vs-frozen-array, no MATLAB).

    Parameters
    ----------
    fixture_path : pathlib.Path, optional
        Path to ``erp_leadfield_fixtures.mat``. Default ``_FIXTURE_PATH``.

    Returns
    -------
    dict
        ``{"scalp_max_diff": float, "diff_wave_max_diff": float}`` (both
        ``<= 1e-7`` on success).

    Raises
    ------
    SystemExit
        If the fixture is absent (the demo must NOT emit a figure unverified).
    RuntimeError
        If the L_full, scalp ERP, or difference wave diverges from SPM by
        ``> 1e-7`` (S3: an ungated figure is not credible).
    """
    if not fixture_path.exists():
        raise SystemExit(
            "PARITY GATE: frozen SPM12 LFP lead-field fixture absent -- refusing "
            f"to emit a figure. Expected: {fixture_path}"
        )
    fx = _load_fixture(fixture_path)

    # Lead-field exactness (LEAD-02) -- the kron(P.J, diag(P.L)) map.
    l_diff = (_torch_l_full() - fx["L_full"]).abs().max().item()
    if l_diff > 1e-12:
        raise RuntimeError(
            f"PARITY GATE: L_full diverges from SPM (max|diff|={l_diff:.3e} > 1e-12)."
        )

    scalp = _production_scalp(fx)  # (Cnd, ns, Nc)
    y_fix = torch.stack(fx["y_scalp"], dim=0)  # (Cnd, ns, Nc)
    scalp_max_diff = (scalp - y_fix).reshape(-1).abs().max().item()

    diff_torch = scalp[1] - scalp[0]  # deviant - standard
    diff_wave_max_diff = (diff_torch - fx["diff_wave"]).abs().max().item()

    tol = f"{_PARITY_TOL:.0e}"
    print(
        f"[GATE] L_full max|diff|      = {l_diff:.3e}  (tol 1e-12)\n"
        f"[GATE] scalp ERP max|diff|   = {scalp_max_diff:.3e}  (tol {tol})\n"
        f"[GATE] diff-wave max|diff|   = {diff_wave_max_diff:.3e}  (tol {tol})"
    )
    if scalp_max_diff > _PARITY_TOL or diff_wave_max_diff > _PARITY_TOL:
        raise RuntimeError(
            "PARITY GATE FAILED: production forward diverges from the frozen SPM12 "
            f"LFP fixture (scalp max|diff|={scalp_max_diff:.3e}, diff-wave "
            f"max|diff|={diff_wave_max_diff:.3e}, tol {_PARITY_TOL:.0e}). "
            "Refusing to emit a figure (S3: an ungated demo is not credible)."
        )
    print("[GATE] PASSED: fixed-reference SPM forward parity is green (<= 1e-7).")
    return {
        "scalp_max_diff": scalp_max_diff,
        "diff_wave_max_diff": diff_wave_max_diff,
    }


def assert_permutation_guard() -> dict[str, float]:
    """Reused Phase-33 guard: the FREE ``P.G[:,0]`` moves ``G[:,6]``, not ``G[:,0]``.

    Confirms the sweep hits the RIGHT knob (pitfall S1): perturbing the free
    ``P.G[node, 0]`` at the precision nodes changes the parameterised sp
    self-inhibition column ``G[:,6]`` (via the intrinsic permutation
    ``J_PERM[0] == 6``, ``spm_fx_cmc.m:151``) and leaves ``G[:,0]`` untouched --
    the swept gain is precision / synaptic-gain, never a mis-indexed column.

    Returns
    -------
    dict
        ``{"moved_g6": float, "moved_g0": float}`` (``moved_g6 > 0``,
        ``moved_g0 == 0`` on success).

    Raises
    ------
    RuntimeError
        If ``P.G[:,0]`` fails to move ``G[:,6]`` or spuriously moves ``G[:,0]``.
    """
    n = _MS_N
    nodes = list(_MS_PRECISION_NODES)
    p0 = mmn_cmc_params(0.0, _A1_B_GAIN, _RIFG_B_GAIN)["p"]
    p1 = mmn_cmc_params(0.7, _A1_B_GAIN, _RIFG_B_GAIN)["p"]
    g0 = parameterize_cmc_network(p0, n)["G"]  # (n, 10)
    g1 = parameterize_cmc_network(p1, n)["G"]
    moved_g6 = (g1[nodes, 6] - g0[nodes, 6]).abs().max().item()
    moved_g0 = (g1[nodes, 0] - g0[nodes, 0]).abs().max().item()
    if not (moved_g6 > 0.0 and moved_g0 == 0.0):
        raise RuntimeError(
            "PERMUTATION GUARD FAILED: free P.G[:,0] must move parameterised "
            f"G[:,6] (sp self-inhibition) and leave G[:,0] fixed; got "
            f"moved_g6={moved_g6:.3e} (want >0), moved_g0={moved_g0:.3e} (want 0). "
            "The swept knob is mis-indexed (J_PERM[0]=6 broken)."
        )
    print(
        f"[GUARD] P.G[:,0] -> G[:,6]: moved_g6={moved_g6:.3e} (>0), "
        f"moved_g0={moved_g0:.3e} (==0)  [J_PERM[0]=6 confirmed]"
    )
    return {"moved_g6": moved_g6, "moved_g0": moved_g0}


def _windowed_mmn_min(rifg_diff: torch.Tensor, pst_ms: torch.Tensor) -> float:
    """Signed minimum of the rIFG difference wave inside the MMN latency window.

    Parameters
    ----------
    rifg_diff : torch.Tensor
        Deviant-standard difference at the rIFG channel, shape ``(ns,)``.
    pst_ms : torch.Tensor
        Peristimulus time in ms, shape ``(ns,)``.

    Returns
    -------
    float
        ``min`` of ``rifg_diff`` over ``_MMN_WINDOW_MS`` (negative for a
        canonical MMN deflection).
    """
    lo, hi = _MMN_WINDOW_MS
    mask = (pst_ms >= lo) & (pst_ms <= hi)
    return float(rifg_diff[mask].min().item())


def run_precision_sweep() -> dict[str, Any]:
    """Pure-forward sweep of the sp self-inhibition gain (ERPDCM-04, re-scoped).

    Sweeps ``sp_inhibition_gain`` (the free ``P.G[:,0] -> G[:,6]`` knob) over
    :data:`_GAIN_GRID` on the SPM-parity-gated 5-source MMN network, routing each
    point through :func:`pyro_dcm.simulators.simulate_erp_dcm` (the parity-verified
    forward; no re-assembly). Records the rIFG ``|MMN|`` peak (the transfer
    curve), the windowed MMN-latency minimum, and the rIFG/A1 amplitude ratio.

    Honest LFP source-space scope (Option A; orchestrator decision 2026-06-26): in
    the LFP-identity readout the input node A1 dominates raw source amplitude, so
    frontal scalp dominance is NOT assertable here -- it is an ECD
    dipole-orientation phenomenon deferred to a follow-up phase (35-01-D1 / Fact
    6). The rIFG/A1 ratio is therefore RECORDED, not gated.

    Returns
    -------
    dict
        ``{"gains": (G,), "mmn_peak": (G,), "win_min": (G,), "rifg_a1_ratio":
        (G,), "pst_ms": (ns,), "diff_low"/"diff_base"/"diff_high": (ns, Nc),
        "idx_low"/"idx_base"/"idx_high": int}``. ``diff_*`` are the rIFG-channel
        difference waves at the lowest / a mid / the highest swept gain.

    Raises
    ------
    RuntimeError
        If the monotone-attenuation or windowed-negative-MMN acceptance criteria
        fail (the science is not loosened -- a failure is a real finding).
    """
    net = build_mmn_5source_network()
    assert tuple(net["source_names"]) == ("A1L", "A1R", "STGL", "STGR", "rIFG")  # type: ignore[arg-type]

    gains = _GAIN_GRID
    mmn_peak: list[float] = []
    win_min: list[float] = []
    ratio: list[float] = []
    diff_waves: list[torch.Tensor] = []
    pst_ms = torch.empty(0, dtype=_F64)
    for g in gains:
        bundle = mmn_cmc_params(float(g), _A1_B_GAIN, _RIFG_B_GAIN)
        sim = simulate_erp_dcm(
            bundle["p"], bundle["x_design"], _MS_N, l_full=bundle["l_full"]
        )
        dw = sim["difference_wave_scalp"]  # (ns, Nc); deviant - standard
        pst_ms = sim["pst"] * 1000.0
        rifg = dw[:, _RIFG]
        mmn_peak.append(float(rifg.abs().max().item()))
        win_min.append(_windowed_mmn_min(rifg, pst_ms))
        a1 = max(
            float(dw[:, _A1L].abs().max().item()),
            float(dw[:, _A1R].abs().max().item()),
        )
        ratio.append(float(rifg.abs().max().item()) / a1)
        diff_waves.append(dw.clone())

    mmn_t = torch.tensor(mmn_peak, dtype=_F64)
    win_t = torch.tensor(win_min, dtype=_F64)

    # ACCEPTANCE 1: monotone non-increasing |MMN| transfer curve (the headline).
    incr = (mmn_t[1:] - mmn_t[:-1] > 1e-15).nonzero().flatten().tolist()
    if incr:
        raise RuntimeError(
            "ACCEPTANCE FAILED (monotone attenuation): |MMN| peak is NOT "
            f"non-increasing across gain (increases at indices {incr}); "
            f"curve={mmn_peak}."
        )

    # ACCEPTANCE 2: windowed MMN-latency minimum is negative AND its magnitude
    # attenuates (non-increasing) with gain (the canonical ~100 ms deflection).
    if not bool((win_t < 0.0).all()):
        raise RuntimeError(
            "ACCEPTANCE FAILED (windowed-negative MMN): the rIFG difference wave "
            f"has no negative minimum in {_MMN_WINDOW_MS} ms at some gain; "
            f"win_min={win_min}."
        )
    win_mag = win_t.abs()
    incr_w = (win_mag[1:] - win_mag[:-1] > 1e-15).nonzero().flatten().tolist()
    if incr_w:
        raise RuntimeError(
            "ACCEPTANCE FAILED (windowed-MMN attenuation): |windowed minimum| is "
            f"NOT non-increasing across gain (increases at indices {incr_w}); "
            f"win_min={win_min}."
        )

    idx_low, idx_high = 0, len(gains) - 1
    idx_base = len(gains) // 2
    return {
        "gains": gains,
        "mmn_peak": mmn_t,
        "win_min": win_t,
        "rifg_a1_ratio": torch.tensor(ratio, dtype=_F64),
        "pst_ms": pst_ms,
        "diff_low": diff_waves[idx_low],
        "diff_base": diff_waves[idx_base],
        "diff_high": diff_waves[idx_high],
        "idx_low": idx_low,
        "idx_base": idx_base,
        "idx_high": idx_high,
    }


def main() -> None:
    """Gate -> permutation guard -> precision sweep -> assertions (Tasks 1-2)."""
    run_parity_gate()
    assert_permutation_guard()
    sweep = run_precision_sweep()
    gains = sweep["gains"].tolist()
    print("\n[SWEEP] sp_inhibition_gain -> |MMN| (rIFG, LFP source-space):")
    print("  gain   |MMN|peak    winMin(~100ms)   rIFG/A1")
    for i, g in enumerate(gains):
        print(
            f"  {g:5.3f}  {sweep['mmn_peak'][i].item():.4e}   "
            f"{sweep['win_min'][i].item():+.4e}      "
            f"{sweep['rifg_a1_ratio'][i].item():.4e}"
        )
    print(
        "\n[SCOPE] LFP-identity readout: the input node A1 dominates raw source "
        f"amplitude (rIFG/A1 ~ {sweep['rifg_a1_ratio'].max().item():.1e}). True "
        "frontal SCALP dominance is an ECD dipole-orientation effect, deferred to "
        "a follow-up phase (35-01-D1 / Fact 6) -- recorded, not asserted."
    )


if __name__ == "__main__":
    main()
