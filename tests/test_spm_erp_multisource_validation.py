"""SPM12 multi-source (5-source MMN) CMC-ERP parity ladder (EVOK-05, Phase-34 gate).

Asserts the pure-torch hierarchical-CMC network forward
(:func:`pyro_dcm.forward_models.erp_coupled_system.cmc_network_f`), the
condition-``B`` modulation
(:func:`pyro_dcm.forward_models.erp_coupled_system.apply_condition_modulation`),
and the Phase-33 exponential-Euler integrator
(:func:`pyro_dcm.utils.local_linearization.integrate_local_linearization`) are
element-wise equivalent to SPM12 AT NETWORK SCALE (``N = 5``), against the
byte-frozen multi-source fixtures generated on M3 in Plan 34-02
(``validation/data/erp_multisource_fixtures.mat``, SPM ``$Id`` ``spm_fx_cmc.m
7279`` / ``spm_gen_Q.m 7279`` / ``spm_int_L.m 7143`` / ``spm_gen_erp.m 6427``).

Parity is vs-SPM, never vs-torch (pitfall V1); tolerances are element-wise
forward agreement only -- NO absolute free energy and NO element-wise ``Cp``
(pitfall V2 / the Phase-32 270-nat lesson). The fixtures are committed to the
repo, so this suite RUNS AND PASSES on the laptop (no MATLAB needed -- the
comparison is torch-against-the-frozen-arrays). The gate therefore keys on
FIXTURE AVAILABILITY, not MATLAB availability (decision 33-03-D1): it skips only
when the ``.mat`` is missing. The ``@pytest.mark.spm`` / ``slow`` markers are
retained so the M3 sbatch can re-run it under ``-m "spm and slow"``.

Staged ladder (pitfall V5 -- a failure localises to one stage), asserted in
order:

1. ``spm_gen_Q`` algebra (the B-wiring guard, C4 / EVOK-05 part 1):
   ``apply_condition_modulation`` reproduces SPM's per-condition ``Q.A{1..4}``
   (all four free-log blocks, ``spm_gen_Q.m:47``) AND ``Q.G(:,1)`` (the
   ``diag(B) -> Q.G`` precision column, ``spm_gen_Q.m:65-67``) element-wise
   ``<= 1e-12``.
1b. A NEGATIVE assertion (EVOK-02): a variant that folds ``B`` into ``A`` but
    OMITS the ``diag(B) -> Q.G`` line produces a ``Q.G(:,1)`` that does NOT match
    the deviant fixture -- proving the precision path is load-bearing.
2. Network ``J0`` via the SAME ``spm_diff`` forward-difference scheme SPM used
   (``dx = exp(-8)``) of ``cmc_network_f`` at ``x0 = 0`` -- bit-exact to the
   fixture, proving ``cmc_network_f`` IS ``spm_fx_cmc`` at ``N > 1`` with the
   Jacobian-construction method held to SPM's (``<= 1e-10``).
3. Network ``Q_update`` via the production right-division
   :func:`pyro_dcm.utils.local_linearization._update_operator` (pitfall C2,
   ``<= 1e-9``).
4. Multi-source trajectory -- SCHEME rung: the exp-Euler loop driven by SPM's
   OWN per-condition ``Q_update`` (isolates loop ordering, pitfall C1, ``~1e-13``).
5. Multi-source trajectory -- FD-Jacobian rung: the operator built from the
   ``spm_diff``-matched ``J0`` (the end-to-end integrator with the Jacobian
   method held to SPM's, ``<= 1e-8``).
6. Multi-source trajectory -- shipped-``jacrev`` rung: the full production
   integrator (exact ``torch.func.jacrev``) MEASURED + recorded, NOT gated at
   ``1e-8``. Its floor is expected ``> 4.7e-8`` because exact AD is MORE accurate
   than SPM's ``spm_diff`` forward-difference Jacobian -- the INHERITED Phase
   33-03 Jacobian-method gate split (34-RESEARCH.md Fact 5, 33-03-SUMMARY
   decisions D2/D3): a documented numerical-method divergence, not a bug.

References
----------
SPM12 source: ``spm_gen_Q.m:24-67`` (the ``B``-folding + ``diag(B) -> Q.G(:,1)``
precision path), ``spm_fx_cmc.m:68-82,171-198`` (the extrinsic blocks + lateral
reduction + the four routes), ``spm_int_L.m:112-169`` (the exp-Euler loop +
``spm_diff`` Jacobian), ``spm_gen_erp.m:69-86`` (the per-condition evoked loop),
``spm_diff.m`` (the forward-difference Jacobian, default step ``exp(-8)``). Plan
34-02 (the frozen multi-source fixture provenance); pitfalls V1-V5 / C1-C2 /
C4 in ``.planning/research/v0.8.0/PITFALLS.md``; 34-RESEARCH.md Fact 5;
33-03-SUMMARY decisions D1/D2/D3.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from pyro_dcm.forward_models.erp_coupled_system import (
    apply_condition_modulation,
    cmc_network_f,
)
from pyro_dcm.utils.local_linearization import (
    _update_operator,
    integrate_local_linearization,
)
from validation.export_to_mat import (
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

# The frozen SPM12 multi-source ground truth (committed in Plan 34-02;
# validation/data/ is mutagen-ignored, so the byte-frozen .mat lives in git).
_FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "validation"
    / "data"
    / "erp_multisource_fixtures.mat"
)

# SPM's spm_diff default forward-difference step (the Jacobian SPM freezes for
# spm_int_L); replicating it makes the J0 / trajectory FD-Jacobian rungs bit-exact.
_SPM_DIFF_DX = math.exp(-8.0)

pytestmark = [
    pytest.mark.spm,
    pytest.mark.slow,
    pytest.mark.skipif(
        not _FIXTURE_PATH.exists(),
        reason=f"frozen SPM12 multi-source fixtures absent: {_FIXTURE_PATH}",
    ),
]


def _reference_p() -> dict[str, Any]:
    """Reconstruct the EXACT free-log-space ``P`` the Wave-2 exporter locked.

    Mirrors the default reference built in
    :func:`validation.export_to_mat.export_erp_dcm_multisource` (the 5-source
    auditory-MMN topology: sources A1L, A1R, STGL, STGR, rIFG) by reusing that
    module's locked topology constants, so torch and the frozen MATLAB fixtures
    feed the IDENTICAL ``P`` (pitfall V1). Free-log convention (34-01-D3):
    live edges ``_MS_A_LIVE`` (``exp(0)``), dead ``_MS_A_DEAD`` (``exp(-32)``);
    ``B`` distinct from ``A`` so the folding checks have teeth.

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


def _apply_no_diag(
    p: dict[str, Any], x_design: torch.Tensor
) -> dict[str, torch.Tensor]:
    """NEGATIVE control: fold ``B`` into ``A`` but OMIT the ``diag(B) -> Q.G`` line.

    A deliberately crippled copy of
    :func:`pyro_dcm.forward_models.erp_coupled_system.apply_condition_modulation`
    that drops ``spm_gen_Q.m:65-67`` only. Used to prove the precision path is
    load-bearing (EVOK-02): its ``Q.G(:,1)`` must FAIL to match the deviant
    fixture, the difference being exactly ``X * diag(B)``.
    """
    x_design = torch.atleast_1d(torch.as_tensor(x_design, dtype=_F64))
    b_list = p["B"]
    a = torch.stack(
        [torch.as_tensor(p["A"][i], dtype=_F64).clone() for i in range(4)], dim=0
    )
    g = p["G"].clone()
    for i, xi in enumerate(x_design):
        b_i = torch.as_tensor(b_list[i], dtype=_F64)
        for j in range(4):
            a[j] = a[j] + xi * b_i  # spm_gen_Q.m:47 (kept)
        # spm_gen_Q.m:65-67 (the diag(B)->Q.G precision line) DELIBERATELY omitted.
    return {"A": a, "G": g}


def _spm_diff_jacobian(
    f: Any, x0: torch.Tensor, dx: float = _SPM_DIFF_DX
) -> torch.Tensor:
    """Forward-difference Jacobian replicating ``spm_diff`` (``dx = exp(-8)``).

    ``spm_diff(@spm_fx_cmc, x0, u0, Q, M, 1)`` builds ``J0`` as a one-sided
    forward difference ``J[:, i] = (f(x0 + dx*e_i) - f(x0)) / dx`` with default
    step ``dx = exp(-8)`` -- the Jacobian SPM's ``spm_int_L`` freezes, so matching
    the scheme is what makes the network parity bit-exact (inherited from the
    Phase-33 single-source ladder).
    """
    f0 = f(x0)
    cols: list[torch.Tensor] = []
    for i in range(x0.shape[0]):
        xp = x0.clone()
        xp[i] = xp[i] + dx
        cols.append((f(xp) - f0) / dx)
    return torch.stack(cols, dim=1)


@pytest.fixture(scope="module")
def fx() -> dict[str, Any]:
    """Load the frozen multi-source SPM12 fixtures + provenance ``meta`` once."""
    import scipy.io as sio

    mat = sio.loadmat(str(_FIXTURE_PATH))
    meta = mat["meta"]

    def _m(name: str) -> np.ndarray:
        return np.asarray(meta[name][0, 0])

    return {
        "Cnd": int(_m("Cnd").ravel()[0]),
        "N": int(_m("N").ravel()[0]),
        "D": int(_m("D").ravel()[0]),
        "nargout_Mf": int(_m("nargout_Mf").ravel()[0]),
        "X": torch.as_tensor(_m("X"), dtype=_F64),  # (Cnd, n_effects)
        "x0": torch.as_tensor(_m("x0"), dtype=_F64),  # (5, 8)
        "dt": float(_m("dt").ravel()[0]),
        "ns": int(_m("ns").ravel()[0]),
        "ons": float(_m("ons").ravel()[0]),
        "dur": float(_m("dur").ravel()[0]),
        "sus": float(_m("sus").ravel()[0]),
        # Cell arrays (1, Cnd); unpack per condition with the helpers below.
        "QA": mat["QA"],
        "QG": mat["QG"],
        "J0": mat["J0"],
        "Qupd": mat["Qupd"],
        "y": mat["y"],
    }


def _qa_block(fx: dict[str, Any], c: int, j: int) -> torch.Tensor:
    """Fixture ``Q.A{j}`` for condition ``c`` -- a (5,5) free-log block."""
    return torch.as_tensor(fx["QA"][0, c][0, j], dtype=_F64)


def _qg(fx: dict[str, Any], c: int) -> torch.Tensor:
    """Fixture ``Q.G(:,1)`` precision column for condition ``c`` -- shape (5,)."""
    return torch.as_tensor(fx["QG"][0, c], dtype=_F64).reshape(-1)


def _j0(fx: dict[str, Any], c: int) -> torch.Tensor:
    """Fixture frozen network Jacobian ``J0`` for condition ``c`` -- (40,40)."""
    return torch.as_tensor(fx["J0"][0, c], dtype=_F64)


def _qupd(fx: dict[str, Any], c: int) -> torch.Tensor:
    """Fixture right-division update operator ``Qupd`` for condition ``c``."""
    return torch.as_tensor(fx["Qupd"][0, c], dtype=_F64)


def _y(fx: dict[str, Any], c: int) -> torch.Tensor:
    """Fixture multi-source source-state trajectory for condition ``c`` -- (128,40)."""
    return torch.as_tensor(fx["y"][0, c], dtype=_F64)


def _drive(fx: dict[str, Any], n_inp: int) -> torch.Tensor:
    """Shared frozen Gaussian evoked drive ``U.u`` (condition-independent).

    Reuses :func:`validation.export_to_mat._erp_gaussian_u_grid` (the numpy port
    of ``spm_erp_u.m`` that generated the fixture) with ``P.R = 0`` so torch and
    SPM integrate the IDENTICAL input grid (pitfall V1).
    """
    r = np.zeros((n_inp, 2), dtype=np.float64)
    u = _erp_gaussian_u_grid(r, fx["ns"], fx["dt"], fx["ons"], fx["dur"], fx["sus"])
    return torch.as_tensor(u, dtype=_F64)  # (ns, n_inp)


def test_pre_asserts(fx: dict[str, Any]) -> None:
    """Mandatory gate pre-conditions: ``D == 1``, ``nargout == 2``, ``x0 == 0``.

    Bakes the multi-source fixture provenance into the gate (EVOK-06): delays
    forced off (``meta.D == 1`` / ``meta.nargout_Mf == 2``), ``N == 5``, the M1
    fixed point ``x0 == zeros(5, 8)``, and float64 at the boundary. The gate is
    element-wise forward agreement only (no absolute-F, no Cp; pitfall V2).
    """
    assert fx["D"] == 1, f"meta.D must be 1 (delays off); got {fx['D']}"
    assert fx["nargout_Mf"] == 2, (
        f"meta.nargout_Mf must be 2 (D=1 wrapper); got {fx['nargout_Mf']}"
    )
    assert fx["N"] == _MS_N, f"meta.N must be 5; got {fx['N']}"
    assert torch.equal(fx["x0"], torch.zeros(_MS_N, 8, dtype=_F64)), (
        f"meta.x0 must be exactly zeros(5, 8) (M1 fixed point); got {fx['x0']}"
    )
    assert _j0(fx, 0).dtype == _F64
    assert _y(fx, 0).dtype == _F64
    assert abs(fx["dt"] - 0.004) < 1e-12, f"dt must be 0.004 s; got {fx['dt']}"


def test_rung1_spm_gen_q_algebra(fx: dict[str, Any]) -> None:
    """[RUNG 1] ``apply_condition_modulation`` vs ``spm_gen_Q`` ``QA``/``QG``.

    The single most important B-wiring guard (C4 / EVOK-05 part 1). For each
    condition: all four free-log ``Q.A{j}`` blocks (``spm_gen_Q.m:47``) AND the
    ``Q.G(:,1)`` precision column (``spm_gen_Q.m:65-67``) must match SPM
    element-wise ``<= 1e-12``.
    """
    p = _reference_p()
    max_a = 0.0
    max_g = 0.0
    for c in range(fx["Cnd"]):
        q = apply_condition_modulation(p, fx["X"][c])
        for j in range(4):
            d = (q["A"][j] - _qa_block(fx, c, j)).abs().max().item()
            max_a = max(max_a, d)
        dg = (q["G"][:, 0] - _qg(fx, c)).abs().max().item()
        max_g = max(max_g, dg)
    print(
        f"\n[RUNG 1] spm_gen_Q  Q.A max|diff| = {max_a:.3e}  "
        f"Q.G(:,1) max|diff| = {max_g:.3e}  (tol 1e-12)"
    )
    assert max_a <= 1e-12, (
        f"Q.A folding parity failed: max|diff|={max_a:.3e} > 1e-12. The B->all-A "
        "fold (spm_gen_Q.m:47) diverges from SPM."
    )
    assert max_g <= 1e-12, (
        f"Q.G(:,1) precision parity failed: max|diff|={max_g:.3e} > 1e-12. The "
        "diag(B)->Q.G(:,1) fold (spm_gen_Q.m:65-67) diverges from SPM."
    )


def test_rung1b_diag_to_g_negative(fx: dict[str, Any]) -> None:
    """[RUNG 1b] NEGATIVE: omitting ``diag(B) -> Q.G`` BREAKS the deviant match.

    Proves the precision path is load-bearing (EVOK-02): a variant that folds
    ``B`` into ``A`` but skips ``spm_gen_Q.m:65-67`` produces a ``Q.G(:,1)`` that
    does NOT match the deviant fixture; the residual is exactly ``X * diag(B)``.
    """
    p = _reference_p()
    dev = next(c for c in range(fx["Cnd"]) if fx["X"][c].abs().sum().item() > 0.0)
    q_omit = _apply_no_diag(p, fx["X"][dev])
    qg_fixture = _qg(fx, dev)
    mismatch = (q_omit["G"][:, 0] - qg_fixture).abs().max().item()
    # The correct path (with the diag line) must match -- pinned in rung 1.
    q_full = apply_condition_modulation(p, fx["X"][dev])
    match = (q_full["G"][:, 0] - qg_fixture).abs().max().item()
    expected_residual = (
        (fx["X"][dev][0].item() * torch.diagonal(p["B"][0])).abs().max().item()
    )
    print(
        f"\n[RUNG 1b] diag->G omitted: max|diff| = {mismatch:.3e} (must be >0, "
        f"= X*max|diag(B)| = {expected_residual:.3e}); with diag = {match:.3e}"
    )
    assert mismatch > 1e-3, (
        f"omit-diag variant unexpectedly matched QG (max|diff|={mismatch:.3e}); "
        "the diag(B)->Q.G(:,1) precision path is NOT load-bearing -- EVOK-02 broken."
    )
    assert abs(mismatch - expected_residual) <= 1e-12, (
        f"omit-diag residual {mismatch:.3e} != expected X*diag(B) "
        f"{expected_residual:.3e}; the negative control is mis-wired."
    )


def test_rung2_network_j0_fd_jacobian(fx: dict[str, Any]) -> None:
    """[RUNG 2] ``cmc_network_f`` ``spm_diff`` FD Jacobian at ``x0=0`` vs ``J0``.

    Proves ``cmc_network_f`` IS ``spm_fx_cmc`` at ``N > 1`` with the
    Jacobian-construction method held to SPM's (``spm_diff`` forward differences,
    ``dx = exp(-8)``; Fact 5). Tolerance ``<= 1e-10``. (Do NOT compare exact
    ``jacrev`` to this FD fixture -- that floor is ~5.6e-4, the inherited
    documented divergence.)
    """
    p = _reference_p()
    n = fx["N"]
    n_inp = int(p["C"].shape[1])
    u0 = torch.zeros(n_inp, dtype=_F64)
    x0 = torch.zeros(8 * n, dtype=_F64)
    max_diff = 0.0
    for c in range(fx["Cnd"]):
        q = apply_condition_modulation(p, fx["X"][c])
        j_fd = _spm_diff_jacobian(lambda v, q=q: cmc_network_f(v, u0, q, n), x0)
        d = (j_fd - _j0(fx, c)).abs().max().item()
        max_diff = max(max_diff, d)
    print(
        f"\n[RUNG 2] network J0 (spm_diff FD) max|diff| = {max_diff:.3e}  (tol 1e-10)"
    )
    assert max_diff <= 1e-10, (
        f"network J0 forward-parity failed: max|diff|={max_diff:.3e} > 1e-10. "
        "cmc_network_f and spm_fx_cmc disagree at N>1 (Jacobian method held to SPM)."
    )


def test_rung3_network_q_update(fx: dict[str, Any]) -> None:
    """[RUNG 3] production right-division ``_update_operator(J0)`` vs ``Qupd``.

    The exp-Euler update operator ``Q = (matrix_exp(dt*dfdx) - I) @ inv(dfdx)``
    built through the production :func:`_update_operator` (``exp(-16)`` shift ->
    ``matrix_exp`` -> right-division) on the frozen network ``J0``, at network
    scale (pitfall C2). Tolerance ``<= 1e-9``.
    """
    max_diff = 0.0
    for c in range(fx["Cnd"]):
        _, q_op = _update_operator(_j0(fx, c), fx["dt"], 1, None)
        d = (q_op - _qupd(fx, c)).abs().max().item()
        max_diff = max(max_diff, d)
    print(f"\n[RUNG 3] network Q_update max|diff| = {max_diff:.3e}  (tol 1e-9)")
    assert max_diff <= 1e-9, (
        f"network Q_update parity failed: max|diff|={max_diff:.3e} > 1e-9. The "
        "right-division orientation diverges at network scale (pitfall C2)."
    )


def test_rung4_trajectory_scheme(fx: dict[str, Any]) -> None:
    """[RUNG 4] SCHEME rung: exp-Euler loop on SPM's OWN ``Qupd`` vs ``y``.

    Driving the update operator with SPM's frozen per-condition ``Q_update``
    isolates the integration SCHEME + loop ordering (``v = v + Q @ f(v, u)``,
    NOT ``v = Q @ (v + f)``; pitfall C1) from the Jacobian-construction method --
    proving the multi-source exp-Euler loop is bit-exact to ``spm_gen_erp``'s
    ``spm_int_L`` body.

    Gated on the scale-invariant RELATIVE error ``max|diff| / max|y| <= 1e-12``
    (machine-epsilon, the bit-exact regime). Unlike the single-source Phase-33
    fixture (states ~O(0.1), absolute floor 6.6e-14), the 5-source network states
    reach ~O(40) under the ``E0 ~ 200`` extrinsic gains, so the ABSOLUTE
    accumulation floor scales with magnitude to ~3e-11 while the RELATIVE floor
    stays at machine epsilon (~8e-13) -- the correct loop-ordering invariant. The
    absolute floor is recorded for the SUMMARY. (Rung 5, with an INDEPENDENTLY
    built operator, passing at ``<= 1e-8`` independently confirms the loop is
    correct; a real ``v = Q@(v+f)`` ordering bug would be catastrophic, not
    machine-epsilon.)
    """
    p = _reference_p()
    n = fx["N"]
    n_inp = int(p["C"].shape[1])
    inputs = _drive(fx, n_inp)
    x0 = torch.zeros(8 * n, dtype=_F64)
    max_abs = 0.0
    max_rel = 0.0
    for c in range(fx["Cnd"]):
        q = apply_condition_modulation(p, fx["X"][c])
        q_upd = _qupd(fx, c)
        v = x0.clone()
        outputs: list[torch.Tensor] = []
        for i in range(fx["ns"]):
            v = v + q_upd @ cmc_network_f(v, inputs[i], q, n)
            outputs.append(v.clone())
        traj = torch.stack(outputs, dim=0)
        y_c = _y(fx, c)
        abs_d = (traj - y_c).abs().max().item()
        max_abs = max(max_abs, abs_d)
        max_rel = max(max_rel, abs_d / y_c.abs().max().item())
    print(
        f"\n[RUNG 4] trajectory (scheme, SPM's Qupd) max|diff| = {max_abs:.3e}  "
        f"rel = {max_rel:.3e}  (rel tol 1e-12, machine-epsilon)"
    )
    assert max_rel <= 1e-12, (
        f"trajectory scheme parity failed: rel={max_rel:.3e} > 1e-12 with SPM's "
        "own Q operator. The bug is loop ordering, not algebra (C1)."
    )


def test_rung5_trajectory_fd_jacobian(fx: dict[str, Any]) -> None:
    """[RUNG 5] FD-Jacobian rung: operator from the ``spm_diff``-matched ``J0``.

    Builds the update operator through the production :func:`_update_operator` on
    the ``spm_diff`` (``dx = exp(-8)``) frozen network Jacobian, then runs the
    exp-Euler loop. Proves the END-TO-END local-linearization integrator matches
    ``spm_gen_erp`` once the orthogonal Jacobian-construction method is held to
    SPM's. Tolerance ``<= 1e-8``.
    """
    p = _reference_p()
    n = fx["N"]
    n_inp = int(p["C"].shape[1])
    inputs = _drive(fx, n_inp)
    u0 = torch.zeros(n_inp, dtype=_F64)
    x0 = torch.zeros(8 * n, dtype=_F64)
    max_diff = 0.0
    for c in range(fx["Cnd"]):
        q = apply_condition_modulation(p, fx["X"][c])
        j_fd = _spm_diff_jacobian(lambda v, q=q: cmc_network_f(v, u0, q, n), x0)
        _, q_op = _update_operator(j_fd, fx["dt"], 1, None)
        v = x0.clone()
        outputs: list[torch.Tensor] = []
        for i in range(fx["ns"]):
            v = v + q_op @ cmc_network_f(v, inputs[i], q, n)
            outputs.append(v.clone())
        traj = torch.stack(outputs, dim=0)
        d = (traj - _y(fx, c)).abs().max().item()
        max_diff = max(max_diff, d)
    print(
        f"\n[RUNG 5] trajectory (full integrator, spm_diff Jacobian) max|diff| = "
        f"{max_diff:.3e}  (tol 1e-8)"
    )
    assert max_diff <= 1e-8, (
        f"trajectory full-integrator parity failed: max|diff|={max_diff:.3e} > "
        "1e-8 with SPM's spm_diff Jacobian. The operator path diverges from "
        "spm_gen_erp."
    )


def test_rung6_trajectory_jacrev_floor(fx: dict[str, Any]) -> None:
    """[RUNG 6] MEASURED floor of the SHIPPED ``jacrev`` integrator vs ``y``.

    The production :func:`integrate_local_linearization` freezes its Jacobian
    with exact ``torch.func.jacrev`` (the Wave-1 design choice for Phase-35
    differentiability), so its trajectory differs from SPM's ``spm_diff``-Jacobian
    trajectory by the PROPAGATED forward-difference truncation -- MEASURED +
    recorded, NOT gated at 1e-8. The exact-AD trajectory is MORE accurate than
    SPM's (Fact 5 / 33-03-D2/D3); the loose ceiling only pins the floor's
    magnitude (expected > 4.7e-8). This is documented, not a bug.
    """
    p = _reference_p()
    n = fx["N"]
    n_inp = int(p["C"].shape[1])
    inputs = _drive(fx, n_inp)
    x0 = torch.zeros(8 * n, dtype=_F64)
    floor = 0.0
    for c in range(fx["Cnd"]):
        q = apply_condition_modulation(p, fx["X"][c])
        traj = integrate_local_linearization(
            lambda v, u, q=q: cmc_network_f(v, u, q, n), x0, inputs, fx["dt"]
        )
        floor = max(floor, (traj - _y(fx, c)).abs().max().item())
    print(
        f"\n[RUNG 6] trajectory (shipped jacrev integrator) floor = {floor:.3e}  "
        "(propagated spm_diff FD truncation; jacrev is more accurate -- NOT gated)"
    )
    assert floor <= 1e-5, (
        f"shipped-jacrev trajectory floor {floor:.3e} unexpectedly large (> 1e-5); "
        "the propagated spm_diff truncation should be ~1e-7 (exact AD is more "
        "accurate than SPM, not a bug)."
    )
