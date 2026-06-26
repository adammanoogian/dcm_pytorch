"""SPM12 single-dipole lead-field + scalp-projection parity ladder (LEAD-05, Phase-35).

Asserts the pure-torch LFP lead field
(:func:`pyro_dcm.forward_models.erp_leadfield.build_lead_field`) and the scalp
observer
(:func:`pyro_dcm.forward_models.erp_leadfield.project_to_scalp`) -- the
``spm_lx_erp.m`` / ``spm_erp_L.m`` port -- are element-wise equivalent to SPM12 at
network scale (``N = 5``, LFP mode), against the byte-frozen LFP lead-field +
scalp-ERP fixtures generated on M3 in Plan 35-02
(``validation/data/erp_leadfield_fixtures.mat``, SPM ``$Id`` ``spm_lx_erp.m 7256`` /
``spm_erp_L.m 7142`` / ``spm_L_priors.m 7409``). The source-state forward
(``cmc_network_f`` IS ``spm_fx_cmc``) + integrator + condition-``B`` were already
proven bit-exact to SPM in the Phase 34 ladder; this gate proves the LEAD field and
projection ON TOP of that source trajectory.

Parity is vs-SPM, never vs-torch (pitfall V1); tolerances are element-wise forward
agreement only -- NO absolute free energy and NO element-wise ``Cp`` (pitfall V2 /
the Phase-32 270-nat lesson; the forward has no normalization freedom). The fixtures
are committed to the repo, so this suite RUNS AND PASSES on the laptop (no MATLAB
needed -- the comparison is torch-against-the-frozen-arrays). The gate keys on
FIXTURE AVAILABILITY, not MATLAB availability (decision 33-03-D1 / 34-03-D2): it skips
only when the ``.mat`` is missing. The ``@pytest.mark.spm`` / ``slow`` markers are
retained so the M3 sbatch can re-run it under ``-m "spm and slow"``.

Staged ladder (pitfall V5 -- a failure localises to one stage), asserted in order:

1. ``L_full`` exact (LEAD-02, the single most important lead-field guard):
   ``build_lead_field(cmc_default_pj(), lfp_spatial(ones(5), 5))`` matches the
   exported ``L_full`` element-wise ``<= 1e-12`` (expected ~0.0). PLUS the
   distinct-valued kron column-major check (block ``s`` == ``P.J[s] * L_spatial`` at
   cols ``[s*5:(s+1)*5]``; the identity block lands at the sp-voltage state ``s=2``,
   cols ``[10:15]``) and the ``P.J`` ``argmax == 2``, ``!= 6`` guard (pitfall C5.1).
2. Scalp SCHEME rung: drive the exp-Euler loop with SPM's OWN per-condition ``Qupd``
   (frozen in the byte-identical multi-source fixture, 34-02; ``P`` + drive are
   identical, pitfall V1) -> source traj ~machine-eps -> ``project_to_scalp`` ->
   matches ``y_scalp{c}`` on the scale-invariant RELATIVE error
   ``max|diff| / max|y| <= 1e-12`` (34-03-D1: N=5 states reach ~O(40), gate the
   relative floor; isolates the projection from the Jacobian-construction method).
3. Scalp FD-Jacobian rung: build the operator from the ``spm_diff``-matched ``J0``
   (the ``_spm_diff_jacobian`` helper, ``dx = exp(-8)``) -> integrate -> project ->
   matches ``y_scalp{c}`` ``<= 1e-8`` (the bit-exact-to-SPM confirmation rung).
4. THE LEAD-05 GATE (shipped-``jacrev`` PRODUCTION path, ``<= 1e-7``): the full
   production :func:`integrate_local_linearization` (exact ``torch.func.jacrev``) ->
   ``project_to_scalp`` -> matches ``y_scalp{c}`` ``<= 1e-7``. The measured scalp
   jacrev floor is PRINTED + recorded. PRODUCTION-PATH GATE DECISION: the Phase-34
   shipped jacrev SOURCE floor was 4.70e-8 (``34-03-SUMMARY`` rung 6); the default
   LFP lead field is the IDENTITY (``P.L = ones`` -> ``L_spatial = I`` -> no
   amplification), so the projection leaves the scalp jacrev floor at ~4.7e-8
   ``< 1e-7``. Therefore the production integrator is GATED DIRECTLY at ``<= 1e-7``
   (the KEY Phase-35 finding) -- unlike Phases 33/34, where 4.7e-8 > 1e-8 forced
   measure-not-gate. CAVEAT: a non-identity LFP gain ``P.L != 1`` scales the floor by
   ``max|P.L|`` -- re-MEASURE and re-confirm ``<= 1e-7`` before gating then.
5. Difference wave (LEAD-03): the production-path ``scalp[1] - scalp[0]`` matches the
   exported ``diff_wave`` (= deviant - standard) ``<= 1e-7`` AND ``max|diff_wave| > 0``
   (non-zero -- ``B`` is wired via ``_MS_B_EDGE`` / ``_MS_B_DIAG``). The
   negative-going / frontal SIGN is NOT gated here (deferred to Phase 36, Fact 6); the
   source-level sign direction is recorded as a non-gating diagnostic.

References
----------
SPM12 source: ``spm_lx_erp.m:31-33`` (``L = spm_erp_L(P, dipfit)`` then
``L = kron(P.J, L)``), ``spm_erp_L.m:105-118`` (the LFP diagonal
``L = sparse(1:m, 1:m, P.L, m, n)``), ``spm_L_priors.m:84,106-109`` (``pE.L = ones``,
``pE.J = sparse(1,3,1,1,8)``), ``spm_gen_erp.m:69-86`` (the per-condition evoked loop:
``spm_gen_Q -> spm_int_L -> ysrc * L'``), ``spm_int_L.m:112-169`` (the exp-Euler loop +
``spm_diff`` Jacobian). Plan 35-02 (the frozen LFP lead-field fixture provenance);
pitfalls V1-V5 / C1-C2 / C5 in ``.planning/research/v0.8.0/PITFALLS.md``;
35-RESEARCH.md; 34-03-SUMMARY decisions D1/D2. Kiebel, S.J., David, O. & Friston, K.J.
(2006), NeuroImage 30, 1273-1284.
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
from pyro_dcm.forward_models.erp_leadfield import (
    build_lead_field,
    cmc_default_pj,
    lfp_spatial,
    project_to_scalp,
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
_N_STATES = 8

# The frozen SPM12 LFP lead-field + scalp-ERP ground truth (committed in Plan 35-02;
# validation/data/ is mutagen-ignored, so the byte-frozen .mat lives in git).
_FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "validation"
    / "data"
    / "erp_leadfield_fixtures.mat"
)

# The byte-identical 34-02 multi-source fixture supplies SPM's OWN frozen per-condition
# Qupd for the SCHEME rung (rung 2). The exporter built BOTH from the same _MS_
# topology + drive, so the source trajectory is identical (verified: projecting the
# frozen multi-source y{c} through L_full reproduces y_scalp{c} to ~5e-14 relative).
_MS_FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "validation"
    / "data"
    / "erp_multisource_fixtures.mat"
)

# SPM's spm_diff default forward-difference step (the Jacobian SPM freezes for
# spm_int_L); replicating it makes the FD-Jacobian rung bit-exact.
_SPM_DIFF_DX = math.exp(-8.0)

pytestmark = [
    pytest.mark.spm,
    pytest.mark.slow,
    pytest.mark.skipif(
        not _FIXTURE_PATH.exists(),
        reason=f"frozen SPM12 LFP lead-field fixtures absent: {_FIXTURE_PATH}",
    ),
]


def _reference_p() -> dict[str, Any]:
    """Reconstruct the EXACT free-log-space ``P`` the Wave-2 exporter locked.

    Mirrors the default reference built in
    :func:`validation.export_to_mat.export_erp_dcm_leadfield` (the 5-source
    auditory-MMN topology A1L, A1R, STGL, STGR, rIFG) by reusing that module's
    locked topology constants, so torch and the frozen MATLAB fixtures feed the
    IDENTICAL ``P`` + drive (pitfall V1). Free-log convention (34-01-D3): live
    edges ``_MS_A_LIVE`` (``exp(0)``), dead ``_MS_A_DEAD`` (``exp(-32)``); ``B``
    distinct from ``A``. The spatial spec (``P.L`` / ``P.J``) is the lead field
    (separate from the source forward) and is built directly in the rungs.

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


def _spm_diff_jacobian(
    f: Any, x0: torch.Tensor, dx: float = _SPM_DIFF_DX
) -> torch.Tensor:
    """Forward-difference Jacobian replicating ``spm_diff`` (``dx = exp(-8)``).

    ``spm_diff(@spm_fx_cmc, x0, u0, Q, M, 1)`` builds ``J0`` as a one-sided forward
    difference ``J[:, i] = (f(x0 + dx*e_i) - f(x0)) / dx`` with default step
    ``dx = exp(-8)`` -- the Jacobian SPM's ``spm_int_L`` freezes, so matching the
    scheme is what makes the FD-Jacobian rung bit-exact (inherited from Phase 33/34).
    """
    f0 = f(x0)
    cols: list[torch.Tensor] = []
    for i in range(x0.shape[0]):
        xp = x0.clone()
        xp[i] = xp[i] + dx
        cols.append((f(xp) - f0) / dx)
    return torch.stack(cols, dim=1)


def _torch_l_full() -> torch.Tensor:
    """Build the production torch LFP lead field ``kron(P.J, diag(P.L))`` (5,40).

    ``P.L = ones(5)`` -> ``L_spatial = I_5`` (identity, no amplification);
    ``P.J = e_2`` (sp-voltage). This is the torch side rung 1 proves == the frozen
    ``L_full`` element-wise; rungs 2-5 then project through it.
    """
    l_spatial = lfp_spatial(torch.ones(_MS_N, dtype=_F64), _MS_N)  # (5,5) identity
    return build_lead_field(cmc_default_pj(), l_spatial)  # (5,40)


@pytest.fixture(scope="module")
def fx() -> dict[str, Any]:
    """Load the frozen LFP lead-field + scalp-ERP fixtures + provenance ``meta``."""
    import scipy.io as sio

    mat = sio.loadmat(str(_FIXTURE_PATH))
    meta = mat["meta"]

    def _m(name: str) -> np.ndarray:
        return np.asarray(meta[name][0, 0])

    cnd = int(_m("Cnd").ravel()[0])
    y_scalp = [
        torch.as_tensor(np.asarray(mat["y_scalp"][0, c], dtype=np.float64), dtype=_F64)
        for c in range(cnd)
    ]  # list[Cnd] of (ns, Nc)
    return {
        "Cnd": cnd,
        "N": int(_m("N").ravel()[0]),
        "Nc": int(_m("Nc").ravel()[0]),
        "D": int(_m("D").ravel()[0]),
        "nargout_Mf": int(_m("nargout_Mf").ravel()[0]),
        "dipfit_type": str(np.asarray(_m("dipfit_type")).ravel()[0]),
        "X": torch.as_tensor(_m("X"), dtype=_F64),  # (Cnd, n_effects)
        "x0": torch.as_tensor(_m("x0"), dtype=_F64),  # (5, 8)
        "P_J": torch.as_tensor(_m("P_J"), dtype=_F64).reshape(-1),  # (8,)
        "P_L": torch.as_tensor(_m("P_L"), dtype=_F64).reshape(-1),  # (5,)
        "dt": float(_m("dt").ravel()[0]),
        "ns": int(_m("ns").ravel()[0]),
        "ons": float(_m("ons").ravel()[0]),
        "dur": float(_m("dur").ravel()[0]),
        "sus": float(_m("sus").ravel()[0]),
        # The frozen scalp ERP: L_full (5,40), y_scalp list[Cnd] (ns,Nc), diff_wave.
        "L_full": torch.as_tensor(
            np.asarray(mat["L_full"], dtype=np.float64), dtype=_F64
        ),
        "y_scalp": y_scalp,
        "diff_wave": torch.as_tensor(
            np.asarray(mat["diff_wave"], dtype=np.float64), dtype=_F64
        ),
    }


def _drive(fx: dict[str, Any], n_inp: int) -> torch.Tensor:
    """Shared frozen Gaussian evoked drive ``U.u`` (condition-independent).

    Reuses :func:`validation.export_to_mat._erp_gaussian_u_grid` (the numpy port of
    ``spm_erp_u.m`` that generated the fixture) with ``P.R = 0`` so torch and SPM
    integrate the IDENTICAL input grid (pitfall V1).
    """
    r = np.zeros((n_inp, 2), dtype=np.float64)
    u = _erp_gaussian_u_grid(r, fx["ns"], fx["dt"], fx["ons"], fx["dur"], fx["sus"])
    return torch.as_tensor(u, dtype=_F64)  # (ns, n_inp)


def _ms_qupd(c: int) -> torch.Tensor:
    """SPM's OWN frozen per-condition ``Qupd`` from the multi-source fixture (40,40).

    The 34-02 exporter built the multi-source fixture from the IDENTICAL ``_MS_``
    topology + drive as the lead-field fixture, so its frozen ``Qupd{c}`` is the
    exact SPM exp-Euler operator for this problem -- used to drive the SCHEME rung
    (rung 2) to machine-eps, isolating the projection from operator construction.
    """
    import scipy.io as sio

    ms = sio.loadmat(str(_MS_FIXTURE_PATH))
    return torch.as_tensor(ms["Qupd"][0, c], dtype=_F64)


def _stack_scalp(per_condition: list[torch.Tensor]) -> torch.Tensor:
    """Stack per-condition scalp ``(ns, Nc)`` to the locked ``(Cnd, ns, Nc)`` layout.

    Honors the ``ERPDCMForward.predict`` contract (gap 4): ``torch.stack(dim=0)`` ->
    ``(Cnd, ns, Nc)`` -> C-order ``reshape(-1)`` at the flat boundary.
    """
    return torch.stack(per_condition, dim=0)  # (Cnd, ns, Nc)


def test_pre_asserts(fx: dict[str, Any]) -> None:
    """Mandatory gate pre-conditions: ``D == 1``, LFP, ``N == Nc == 5``, ``P.J`` idx 2.

    Re-asserts the lead-field fixture provenance (LEAD-05/06): delays off
    (``meta.D == 1`` / ``meta.nargout_Mf == 2``), LFP mode (``dipfit_type == 'LFP'``,
    ``N == Nc == 5``, head-model-free), the M1 fixed point ``x0 == zeros(5, 8)``,
    ``P.L == ones(5)`` (identity gain), ``P.J`` one-hot at index 2 (sp-voltage, NOT 6),
    and float64 at the boundary. Element-wise forward agreement only (pitfall V2).
    """
    assert fx["D"] == 1, f"meta.D must be 1 (delays off); got {fx['D']}"
    assert fx["nargout_Mf"] == 2, (
        f"meta.nargout_Mf must be 2 (D=1 wrapper); got {fx['nargout_Mf']}"
    )
    assert fx["dipfit_type"] == "LFP", (
        f"meta.dipfit_type must be 'LFP' (head-model-free gate); "
        f"got {fx['dipfit_type']!r}"
    )
    assert fx["N"] == _MS_N, f"meta.N must be 5; got {fx['N']}"
    assert fx["Nc"] == _MS_N, (
        f"meta.Nc must equal N=5 in LFP mode (one channel per source); got {fx['Nc']}"
    )
    assert torch.equal(fx["x0"], torch.zeros(_MS_N, 8, dtype=_F64)), (
        f"meta.x0 must be exactly zeros(5, 8) (M1 fixed point); got {fx['x0']}"
    )
    assert torch.equal(fx["P_L"], torch.ones(_MS_N, dtype=_F64)), (
        f"meta.P_L must be ones(5) (identity LFP gain); got {fx['P_L']}"
    )
    j_idx = int(torch.argmax(fx["P_J"]).item())
    assert j_idx == 2, (
        f"meta.P_J must be one-hot at index 2 (sp-voltage); argmax={j_idx} (NOT 6 -- "
        "the deep-pyramidal inverted-signal trap, pitfall C5.1)"
    )
    assert fx["P_J"][6].item() == 0.0, (
        f"meta.P_J[6] (dp-voltage) must be 0; got {fx['P_J'][6].item()}"
    )
    assert fx["L_full"].dtype == _F64
    assert fx["y_scalp"][0].dtype == _F64
    assert abs(fx["dt"] - 0.004) < 1e-12, f"dt must be 0.004 s; got {fx['dt']}"


def test_rung1_lead_field_exact(fx: dict[str, Any]) -> None:
    """[RUNG 1] ``build_lead_field`` vs frozen ``L_full`` (LEAD-02, the key guard).

    ``build_lead_field(cmc_default_pj(), lfp_spatial(ones(5), 5))`` must match the
    exported ``L_full`` element-wise ``<= 1e-12`` (expected ~0.0 -- the 35-02
    headline). PLUS the distinct-valued kron column-major check: block ``s`` ==
    ``P.J[s] * L_spatial`` at cols ``[s*5:(s+1)*5]`` (the identity block lands at the
    sp-voltage state ``s=2``, cols ``[10:15]``; a C-order flatten would mis-place it
    at ``source*8 + state``, pitfall C5.2). PLUS the ``P.J argmax == 2, != 6`` guard.
    """
    n = _MS_N
    l_spatial = lfp_spatial(torch.ones(n, dtype=_F64), n)  # (5,5) identity
    p_j = cmc_default_pj()
    l_torch = build_lead_field(p_j, l_spatial)  # (5,40)

    assert l_torch.shape == fx["L_full"].shape == (n, _N_STATES * n)
    max_diff = (l_torch - fx["L_full"]).abs().max().item()
    print(
        f"\n[RUNG 1] L_full build_lead_field vs SPM max|diff| = {max_diff:.3e}  "
        "(tol 1e-12)"
    )
    assert max_diff <= 1e-12, (
        f"L_full parity failed: max|diff|={max_diff:.3e} > 1e-12. The "
        "kron(P.J, L_spatial) lead field (spm_lx_erp.m:33) diverges from SPM."
    )

    # Kron column-major guard: block s occupies cols [s*n:(s+1)*n] == P.J[s]*L_spatial.
    for s in range(_N_STATES):
        block = fx["L_full"][:, s * n : (s + 1) * n]
        expected = p_j[s].item() * l_spatial
        bd = (block - expected).abs().max().item()
        assert bd <= 1e-12, (
            f"kron column-major broken at state block s={s}: max|diff|={bd:.3e}. "
            f"Block != P.J[{s}]*L_spatial -- the flatten is C-order not column-major "
            "(pitfall C5.2)."
        )
    # The identity block must sit at the sp-voltage state s=2 (cols [10:15]).
    assert torch.allclose(
        fx["L_full"][:, 2 * n : 3 * n], torch.eye(n, dtype=_F64), atol=1e-12
    ), "identity block not at sp-voltage state s=2 (cols [10:15])"

    # P.J guard (pitfall C5.1): the contributing state is index 2, NOT 6.
    j_idx = int(torch.argmax(p_j).item())
    assert j_idx == 2 and p_j[6].item() == 0.0, (
        f"cmc_default_pj must be one-hot at index 2 (sp-voltage); argmax={j_idx}, "
        f"P.J[6]={p_j[6].item()} (NOT index 6, the inverted-signal trap)"
    )


def test_rung2_scalp_scheme(fx: dict[str, Any]) -> None:
    """[RUNG 2] SCHEME rung: exp-Euler loop on SPM's OWN ``Qupd`` -> project -> y_scalp.

    Driving the update operator with SPM's frozen per-condition ``Qupd`` (from the
    byte-identical 34-02 multi-source fixture) reproduces SPM's source trajectory to
    machine-eps, ISOLATING the projection algebra (``project_to_scalp``) from the
    Jacobian-construction method. Gated on the scale-invariant RELATIVE error
    ``max|diff| / max|y| <= 1e-12`` (34-03-D1: N=5 states reach ~O(40), so the
    absolute float64 accumulation floor scales to ~1e-13 while the relative floor
    stays at machine epsilon). The locked ``(Cnd, ns, Nc)`` stacking is honored.
    """
    if not _MS_FIXTURE_PATH.exists():
        pytest.skip(f"multi-source fixture (SPM's Qupd) absent: {_MS_FIXTURE_PATH}")
    p = _reference_p()
    n = fx["N"]
    n_inp = int(p["C"].shape[1])
    inputs = _drive(fx, n_inp)
    l_full = _torch_l_full()
    x0 = torch.zeros(_N_STATES * n, dtype=_F64)
    scalp_list: list[torch.Tensor] = []
    for c in range(fx["Cnd"]):
        q = apply_condition_modulation(p, fx["X"][c])
        q_upd = _ms_qupd(c)
        v = x0.clone()
        outputs: list[torch.Tensor] = []
        for i in range(fx["ns"]):
            v = v + q_upd @ cmc_network_f(v, inputs[i], q, n)
            outputs.append(v.clone())
        traj = torch.stack(outputs, dim=0)  # (ns, 8n)
        scalp_list.append(project_to_scalp(traj, l_full))  # (ns, Nc)

    scalp = _stack_scalp(scalp_list)  # (Cnd, ns, Nc)
    y_fix = _stack_scalp(fx["y_scalp"])
    abs_d = (scalp - y_fix).reshape(-1).abs().max().item()
    rel = abs_d / y_fix.abs().max().item()
    print(
        f"\n[RUNG 2] scalp (scheme, SPM's Qupd) max|diff| = {abs_d:.3e}  "
        f"rel = {rel:.3e}  (rel tol 1e-12, machine-epsilon)"
    )
    assert rel <= 1e-12, (
        f"scalp scheme parity failed: rel={rel:.3e} > 1e-12 with SPM's own Q "
        "operator. The bug is in project_to_scalp / loop ordering, not the Jacobian."
    )


def test_rung3_scalp_fd_jacobian(fx: dict[str, Any]) -> None:
    """[RUNG 3] FD-Jacobian rung: ``spm_diff`` ``J0`` operator -> project -> y_scalp.

    Builds the update operator through the production :func:`_update_operator` on the
    ``spm_diff`` (``dx = exp(-8)``) frozen network Jacobian of ``cmc_network_f``, runs
    the exp-Euler loop, then projects. Proves the END-TO-END
    integrate-then-project path matches ``spm_gen_erp + spm_lx_erp`` once the
    orthogonal Jacobian-construction method is held to SPM's. Tolerance ``<= 1e-8``
    (the bit-exact-to-SPM confirmation rung).
    """
    p = _reference_p()
    n = fx["N"]
    n_inp = int(p["C"].shape[1])
    inputs = _drive(fx, n_inp)
    u0 = torch.zeros(n_inp, dtype=_F64)
    l_full = _torch_l_full()
    x0 = torch.zeros(_N_STATES * n, dtype=_F64)
    scalp_list: list[torch.Tensor] = []
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
        scalp_list.append(project_to_scalp(traj, l_full))

    scalp = _stack_scalp(scalp_list)
    y_fix = _stack_scalp(fx["y_scalp"])
    max_diff = (scalp - y_fix).reshape(-1).abs().max().item()
    print(
        f"\n[RUNG 3] scalp (full integrator, spm_diff Jacobian) max|diff| = "
        f"{max_diff:.3e}  (tol 1e-8)"
    )
    assert max_diff <= 1e-8, (
        f"scalp FD-Jacobian parity failed: max|diff|={max_diff:.3e} > 1e-8 with "
        "SPM's spm_diff Jacobian. The integrate->project path diverges from "
        "spm_gen_erp + spm_lx_erp."
    )


def _production_scalp(fx: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the FULL production path (jacrev integrator -> project) per condition.

    Returns the ``(Cnd, ns, Nc)`` scalp stack AND the fixture stack, both via the
    locked layout, so rungs 4 and 5 share one production forward.
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
    return _stack_scalp(scalp_list), _stack_scalp(fx["y_scalp"])


def test_rung4_production_path_gate(fx: dict[str, Any]) -> None:
    """[RUNG 4] THE LEAD-05 GATE: shipped-``jacrev`` production scalp ERP ``<= 1e-7``.

    The full production :func:`integrate_local_linearization` (exact
    ``torch.func.jacrev``) -> :func:`project_to_scalp` -> matches ``y_scalp{c}``
    ``<= 1e-7``. The measured scalp jacrev floor is PRINTED + recorded.

    PRODUCTION-PATH GATE DECISION (the KEY Phase-35 finding): the Phase-34 shipped
    jacrev SOURCE floor was 4.70e-8 (34-03-SUMMARY rung 6) -- the propagated
    ``spm_diff`` forward-difference truncation (exact AD is MORE accurate than SPM,
    NOT a bug, Fact 5). The default LFP lead field is the IDENTITY (``P.L = ones`` ->
    ``L_spatial = I`` -> NO amplification), so the projection leaves the scalp jacrev
    floor at ~4.7e-8 ``< 1e-7``. Therefore the production integrator is GATED DIRECTLY
    at ``<= 1e-7`` here -- unlike Phases 33/34, where 4.7e-8 > 1e-8 forced
    measure-not-gate. Rungs 2-3 remain the diagnostic localization rungs.

    CAVEAT: a non-identity LFP gain ``P.L != 1`` scales the projected floor by
    ``max|P.L|`` -- re-MEASURE the scalp jacrev floor and re-confirm ``<= 1e-7``
    before gating with a non-identity spatial model.
    """
    scalp, y_fix = _production_scalp(fx)
    floor = (scalp - y_fix).reshape(-1).abs().max().item()
    print(
        f"\n[RUNG 4] scalp (shipped jacrev PRODUCTION path) floor = {floor:.3e}  "
        "(LFP identity -> no amplification -> GATED <=1e-7; the Phase-35 finding)"
    )
    assert floor <= 1e-7, (
        f"LEAD-05 production-path gate FAILED: scalp jacrev floor {floor:.3e} > 1e-7. "
        "The propagated spm_diff truncation should sit at ~4.7e-8 under the identity "
        "LFP lead field -- a larger floor means a real divergence (or a non-identity "
        "P.L scaling it; re-measure)."
    )


def test_rung5_difference_wave(fx: dict[str, Any]) -> None:
    """[RUNG 5] Difference wave (LEAD-03): production ``scalp[1]-scalp[0]`` vs frozen.

    The production-path deviant-minus-standard scalp ERP must match the exported
    ``diff_wave`` (= ``y_scalp[1] - y_scalp[0]``) ``<= 1e-7`` AND be NON-ZERO
    (``max|diff_wave| > 0`` -- ``B`` is wired via ``_MS_B_EDGE`` / ``_MS_B_DIAG``).
    The negative-going / frontal SIGN is NOT gated (deferred to Phase 36, Fact 6);
    the source-level sign direction is printed as a non-gating diagnostic only.
    """
    scalp, _ = _production_scalp(fx)  # (Cnd, ns, Nc)
    diff_torch = scalp[1] - scalp[0]  # (ns, Nc), deviant - standard
    diff_fix = fx["diff_wave"]  # (ns, Nc)

    nonzero_mag = diff_fix.abs().max().item()
    max_diff = (diff_torch - diff_fix).abs().max().item()
    # Non-gating sign diagnostic at the precision nodes (deferred to Phase 36).
    peak_idx = int(diff_fix.abs().max(dim=0).values.argmax().item())
    peak_sign = float(
        torch.sign(diff_fix[diff_fix[:, peak_idx].abs().argmax(), peak_idx])
    )
    print(
        f"\n[RUNG 5] difference wave max|diff| = {max_diff:.3e}  (tol 1e-7); "
        f"max|diff_wave| = {nonzero_mag:.3e} (must be >0); "
        f"peak channel {peak_idx} sign = {peak_sign:+.0f} (NOT gated, Phase 36)"
    )
    assert nonzero_mag > 0.0, (
        "difference wave is identically zero -- B is not wired (EVOK-02/LEAD-03 "
        "broken); expected non-zero via _MS_B_EDGE/_MS_B_DIAG."
    )
    assert max_diff <= 1e-7, (
        f"difference-wave parity failed: max|diff|={max_diff:.3e} > 1e-7. The "
        "production deviant-standard scalp ERP diverges from the frozen diff_wave."
    )
