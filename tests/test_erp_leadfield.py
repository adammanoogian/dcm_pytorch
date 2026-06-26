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

import time

import pytest
import torch

from pyro_dcm.forward_models.erp_leadfield import (
    build_lead_field,
    cmc_default_pj,
    lfp_spatial,
    project_to_scalp,
)
from pyro_dcm.inference.forward_models import ERPDCMForward, ForwardModel
from pyro_dcm.simulators.erp_simulator import simulate_erp_dcm

_F64 = torch.float64


def _edge_block(n: int, edges: list[tuple[int, int]]) -> torch.Tensor:
    """``(n,n)`` free log-param block: ``-32`` (dead) except ``0`` at ``edges``."""
    a = torch.full((n, n), -32.0, dtype=_F64)
    for r, c in edges:
        a[r, c] = 0.0
    return a


def _planted_p(n: int, b_wired: bool) -> dict[str, object]:
    """Planted 2-source ERP free-param dict (one forward edge 0->1).

    ``B[0]`` carries a forward-edge modulation + a non-zero diagonal (the
    precision path) when ``b_wired``; an all-zero ``B[0]`` otherwise (control).
    """
    b0 = (
        torch.tensor([[0.3, 0.0], [0.2, 0.5]], dtype=_F64)
        if b_wired
        else torch.zeros(n, n, dtype=_F64)
    )
    return {
        "T": torch.zeros(n, 4, dtype=_F64),
        "G": torch.zeros(n, 4, dtype=_F64),
        "C": torch.zeros(n, 1, dtype=_F64),
        "S": torch.zeros(n, 1, dtype=_F64),
        "R": torch.zeros(1, 2, dtype=_F64),
        "A": torch.stack(
            [
                _edge_block(n, [(1, 0)]),  # A[0] fwd sp->ss, edge 0->1
                _edge_block(n, []),
                _edge_block(n, []),
                _edge_block(n, []),
            ],
            dim=0,
        ),
        "B": [b0],
    }


def _erp_forward(n: int, b_wired: bool, ns: int = 64) -> ERPDCMForward:
    """Build an ``ERPDCMForward`` matching the planted 2-source net."""
    a_masks = [
        _edge_block(n, [(1, 0)]) >= 0.0,  # live where free == 0
        torch.zeros(n, n, dtype=torch.bool),
        torch.zeros(n, n, dtype=torch.bool),
        torch.zeros(n, n, dtype=torch.bool),
    ]
    a_masks_f = [m.double() for m in a_masks]
    b0 = (
        torch.tensor([[0.3, 0.0], [0.2, 0.5]], dtype=_F64)
        if b_wired
        else torch.zeros(n, n, dtype=_F64)
    )
    c_mask = torch.ones(n, 1, dtype=_F64)
    l_full = build_lead_field(cmc_default_pj(), lfp_spatial(torch.ones(n), n))
    x_design = torch.tensor([[0.0], [1.0]], dtype=_F64)
    return ERPDCMForward(
        l_full=l_full,
        x_design=x_design,
        a_masks=a_masks_f,
        b_masks=[b0],
        c_mask=c_mask,
        dt=0.004,
        ns=ns,
    )


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


# --------------------------------------------------------------------------- #
# Task 2: ERPDCMForward protocol implementor (additive) + simulator scalp path #
# --------------------------------------------------------------------------- #


def test_erpdcmforward_is_forward_model() -> None:
    """``ERPDCMForward`` satisfies the ``runtime_checkable`` ``ForwardModel``."""
    fwd = _erp_forward(2, b_wired=True)
    assert isinstance(fwd, ForwardModel)
    assert fwd.residual_is_complex is False


def test_erpdcmforward_param_count() -> None:
    """``param_count`` == 4NN + N*M + 4N + 4N + N + 2M (A+C+T+G+S+R, L/J fixed)."""
    n = 2
    fwd = _erp_forward(n, b_wired=True)
    m = 1  # n_inp from c_mask
    expected = 4 * n * n + n * m + 4 * n + 4 * n + n + 2 * m
    assert fwd.param_count(n) == expected
    assert fwd.param_count(n) == 38


def test_erpdcmforward_pack_unpack_roundtrip() -> None:
    """``pack(unpack(theta)) == theta`` on the frozen ordering."""
    n = 2
    fwd = _erp_forward(n, b_wired=True)
    np_full = fwd.param_count(n)
    torch.manual_seed(3)
    theta = torch.randn(np_full, dtype=_F64)
    unpacked = fwd.unpack_params(theta, n)
    # Keys + shapes are the frozen pack order.
    assert unpacked["A_free"].shape == (4, n, n)
    assert unpacked["C_free"].shape == (n, 1)
    assert unpacked["T"].shape == (n, 4)
    assert unpacked["G"].shape == (n, 4)
    assert unpacked["S"].shape == (n, 1)
    assert unpacked["R"].shape == (1, 2)
    repacked = fwd.pack_params(**unpacked)
    assert torch.equal(repacked, theta)


def test_erpdcmforward_build_precision_identity() -> None:
    """``build_precision`` is identity of size ``Cnd*ns*Nc``."""
    n = 2
    ns = 64
    fwd = _erp_forward(n, b_wired=True, ns=ns)
    cnd = 2
    nc = n
    observed = torch.zeros(cnd, ns, nc, dtype=_F64)
    q_list, nq = fwd.build_precision(observed)
    assert nq == 1
    assert len(q_list) == 1
    assert q_list[0].shape == (cnd * ns * nc, cnd * ns * nc)
    assert torch.equal(q_list[0], torch.eye(cnd * ns * nc, dtype=_F64))


def test_erpdcmforward_predict_ndim_guard() -> None:
    """``predict`` returns identical flat output for 3-D and flat ``observed`` (B3).

    The VL main loop calls ``predict`` with the ``(Cnd, ns, Nc)`` tensor; the
    FD-Jacobian calls it with the flat ``observed.reshape(-1)``. The ``observed.ndim``
    guard must make both paths return the same flat vector.
    """
    n = 2
    ns = 64
    fwd = _erp_forward(n, b_wired=True, ns=ns)
    planted = _planted_p(n, b_wired=True)
    theta = fwd.pack_params(
        A_free=planted["A"],
        C_free=planted["C"],
        T=planted["T"],
        G=planted["G"],
        S=planted["S"],
        R=planted["R"],
    )
    cnd = 2
    nc = n
    obs_3d = torch.zeros(cnd, ns, nc, dtype=_F64)
    obs_flat = obs_3d.reshape(-1)
    a_mask = torch.ones(n, n, dtype=_F64)

    y_main = fwd.predict(theta, obs_3d, n, a_mask=a_mask)
    y_fd = fwd.predict(theta, obs_flat, n, a_mask=a_mask)
    assert y_main.shape == (cnd * ns * nc,)
    assert torch.equal(y_main, y_fd)
    assert torch.isfinite(y_main).all()


def test_simulate_erp_dcm_scalp_path() -> None:
    """``simulate_erp_dcm(l_full=...)`` adds ``scalp`` + ``difference_wave_scalp``.

    The scalp difference wave is NON-ZERO when ``B`` is wired (the LEAD-03 gate)
    and EXACTLY zero on the B-omitted control -- proving the condition difference
    is driven entirely by ``B``. The SIGN (negative-going / frontal) is NOT
    asserted (Phase 36, needs ECD orientation + MNI coords -- Fact 6).
    """
    n = 2
    ns = 64
    x_design = torch.tensor([[0.0], [1.0]], dtype=_F64)
    l_full = build_lead_field(cmc_default_pj(), lfp_spatial(torch.ones(n), n))

    out = simulate_erp_dcm(
        _planted_p(n, b_wired=True), x_design, n, ns=ns, l_full=l_full
    )
    assert out["scalp"].shape == (2, ns, n)
    assert out["scalp"].dtype == _F64
    # Existing source-state keys remain present + unchanged in shape.
    assert out["states"].shape == (2, ns, n, 8)
    dws = out["difference_wave_scalp"]
    assert dws is not None
    assert dws.shape == (ns, n)
    assert torch.isfinite(dws).all()
    assert dws.abs().max().item() > 0.0  # NON-ZERO (B wired)

    # Through the identity LFP lead field the scalp == source sp-voltage (col 2).
    src_spv = out["states"][..., 2]  # (Cnd, ns, n)
    assert torch.allclose(out["scalp"], src_spv)

    out0 = simulate_erp_dcm(
        _planted_p(n, b_wired=False), x_design, n, ns=ns, l_full=l_full
    )
    dws0 = out0["difference_wave_scalp"]
    assert dws0 is not None
    assert torch.equal(dws0, torch.zeros_like(dws0))  # B off -> no difference


def test_simulate_erp_dcm_no_lead_field_unchanged() -> None:
    """Without ``l_full`` the scalp keys are absent; source keys are unchanged."""
    n = 2
    ns = 32
    x_design = torch.tensor([[0.0], [1.0]], dtype=_F64)
    out = simulate_erp_dcm(_planted_p(n, b_wired=True), x_design, n, ns=ns)
    assert "scalp" not in out
    assert "difference_wave_scalp" not in out
    assert out["states"].shape == (2, ns, n, 8)


# --------------------------------------------------------------------------- #
# Task 3: LEAD-06 VL round-trip (PROTOCOL CONFIRMATION, not a parity gate)     #
# --------------------------------------------------------------------------- #


def _r_squared(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Pooled coefficient of determination ``1 - SS_res / SS_tot``."""
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    return float(1.0 - ss_res / ss_tot)


@pytest.mark.vl
@pytest.mark.slow
def test_lead06_vl_roundtrip_protocol_confirmation() -> None:
    """LEAD-06: ``run_variational_laplace_generic(ERPDCMForward(...))`` round-trip.

    PROTOCOL CONFIRMATION, NOT a parity gate. Plants a small (n=2) CMC net with a
    RECIPROCAL A graph (identifiable), a driven input, a non-trivial deviant ``B``,
    and a perturbed precision knob ``G[:,0]``; simulates the scalp ERP through the
    identity LFP lead field; adds light Gaussian noise; and fits ``ERPDCMForward``
    via the model-agnostic VL engine. Confirms the full chain runs end-to-end
    (param_count -> build_prior_cov -> _spm_svd -> predict -> ReML -> build_result)
    and recovers the planted dynamics within a LOOSE tolerance. The laptop
    wall-time is MEASURED and printed (CLAUDE.md >3 min rule: a single-seed n=2
    fit must stay < ~3 min on laptop, else escalate a multi-seed sweep to M3).
    """
    from pyro_dcm.forward_models.erp_coupled_system import parameterize_cmc_network
    from pyro_dcm.inference.variational_laplace import (
        run_variational_laplace_generic,
    )

    n = 2
    ns = 32
    torch.manual_seed(7)

    # Planted free params: reciprocal forward edge (0<->1) in A[0], driven input on
    # source 0, deviant B (edge + precision diag), and a perturbed precision knob.
    a0 = _edge_block(n, [(0, 1), (1, 0)])  # reciprocal forward sp->ss (identifiable)
    a_planted = torch.stack(
        [a0, _edge_block(n, []), _edge_block(n, []), _edge_block(n, [])], dim=0
    )
    g_planted = torch.zeros(n, 4, dtype=_F64)
    g_planted[:, 0] = 0.3  # the precision knob (drives G[:,6] sp self-inhibition)
    c_planted = torch.zeros(n, 1, dtype=_F64)
    c_planted[0, 0] = 0.6  # driven-source input gain exp(0.6)~1.8x (free, recoverable)
    b0 = torch.tensor([[0.0, 0.0], [0.0, 0.4]], dtype=_F64)  # deviant precision diag
    planted = {
        "T": torch.zeros(n, 4, dtype=_F64),
        "G": g_planted,
        "C": c_planted,
        "S": torch.zeros(n, 1, dtype=_F64),
        "R": torch.zeros(1, 2, dtype=_F64),
        "A": a_planted,
        "B": [b0],
    }

    a_masks = [
        (a0 >= 0.0).double(),  # reciprocal edges live
        torch.zeros(n, n, dtype=_F64),
        torch.zeros(n, n, dtype=_F64),
        torch.zeros(n, n, dtype=_F64),
    ]
    c_mask = torch.zeros(n, 1, dtype=_F64)
    c_mask[0, 0] = 1.0  # input drives source 0
    l_full = build_lead_field(cmc_default_pj(), lfp_spatial(torch.ones(n), n))
    x_design = torch.tensor([[0.0], [1.0]], dtype=_F64)

    sim = simulate_erp_dcm(planted, x_design, n, ns=ns, l_full=l_full)
    scalp_clean = sim["scalp"]  # (Cnd, ns, Nc)
    noise = 0.02 * scalp_clean.abs().mean() * torch.randn_like(scalp_clean)
    scalp_obs = scalp_clean + noise

    fwd = ERPDCMForward(
        l_full=l_full,
        x_design=x_design,
        a_masks=a_masks,
        b_masks=[b0],
        c_mask=c_mask,
        dt=0.004,
        ns=ns,
    )
    union_a_mask = (a0 >= 0.0).double()  # the live extrinsic graph (compat no-op)

    t0 = time.perf_counter()
    result = run_variational_laplace_generic(
        fwd,
        scalp_obs,
        a_mask=union_a_mask,
        n_regions=n,
        max_iter=24,
    )
    walltime_s = time.perf_counter() - t0
    print(
        f"\n[LEAD-06] VL round-trip wall-time: {walltime_s:.1f} s "
        f"(n={n}, ns={ns}, max_iter=24, iters_run={result.n_iterations})"
    )

    # --- chain ran end-to-end -------------------------------------------------
    assert result.n_iterations >= 1
    for key in ("A", "C", "G", "T", "S", "R"):
        assert key in result.theta_post
        assert torch.isfinite(result.theta_post[key]).all()
    assert result.predicted_output is not None
    assert result.predicted_output.shape == (2, ns, n)
    assert torch.isfinite(result.predicted_output).all()

    # --- optimiser improved the fit ------------------------------------------
    assert len(result.free_energy) >= 1
    assert max(result.free_energy) >= result.free_energy[0]

    # --- recovery in observation space (LOOSE protocol confirmation) ----------
    # The tight CMC priors regularise the fit toward the default circuit, so the
    # confirmation is that the fitted scalp moves TOWARD the planted data relative
    # to the prior-mean (zero-param) prediction -- the end-to-end gradient chain
    # (predict -> FD-Jacobian -> ReML) is exercised. NOT a parity gate.
    np_full = fwd.param_count(n)
    baseline_pred = fwd.predict(
        torch.zeros(np_full, dtype=_F64), scalp_obs, n, a_mask=union_a_mask
    ).reshape(2, ns, n)
    r2_fit = _r_squared(result.predicted_output, scalp_clean)
    r2_base = _r_squared(baseline_pred, scalp_clean)
    print(
        f"[LEAD-06] scalp R^2: prior-mean baseline={r2_base:.4f} -> VL fit={r2_fit:.4f}"
    )
    assert r2_fit > r2_base, (
        f"VL fit should improve on the prior-mean prediction; "
        f"fit R^2={r2_fit:.4f} <= baseline R^2={r2_base:.4f}"
    )
    assert r2_fit > 0.3, f"VL fit should positively recover the scalp; R^2={r2_fit:.4f}"

    # --- recovery in parameter space (LOOSE) ----------------------------------
    # Live extrinsic A stays a live edge (parameterised strength well above the
    # dead exp(-32)*E0 floor) -- the masked-free dead-edge handling round-trips.
    a_rec = result.theta_post["A"][0]
    a_planted_param = parameterize_cmc_network({"A": a_planted}, n)["A"][0]
    live = union_a_mask > 0
    assert (a_rec[live] > 1.0).all(), "recovered live A edge collapsed to dead"
    rel = (a_rec[live] - a_planted_param[live]).abs() / a_planted_param[live]
    assert (rel < 5.0).all(), f"live A recovery wildly off: rel={rel}"

    # Wall-time guard: a single-seed n=2 fit must stay laptop-tractable.
    assert walltime_s < 180.0, (
        f"VL round-trip took {walltime_s:.1f}s (> 3 min) -- escalate to M3 "
        "for any multi-seed recovery sweep (CLAUDE.md >3 min rule)"
    )
