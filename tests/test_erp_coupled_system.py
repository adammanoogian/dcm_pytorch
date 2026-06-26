"""Structural guard battery for the hierarchical-CMC network forward (Phase 34).

These tests are authored BEFORE ``erp_coupled_system`` exists, so the suite is
RED on first run (ImportError) -- that is the point: the C4 ``diag(B)->G[:,6]``
precision guard IS the MMN mechanism the milestone exists to demonstrate, so it
drives the implementation.

The guards pin, on the laptop with no MATLAB (pure-torch, sub-second):

1. C4 precision guard -- ``apply_condition_modulation`` folds ``diag(B)`` into
   the free precision column ``Q.G[:,0]`` (``spm_gen_Q.m:65-67``) so the
   parameterised ``G[:,6]`` (``sp`` self-inhibition, ``J_PERM[0]==6``) moves.
2. C4 negative test -- an omit-``diag`` variant leaves ``G[:,6]`` unchanged,
   proving the precision path is load-bearing (EVOK-02).
3. ``B`` folds additively into ALL four ``A{1..4}`` blocks (``spm_gen_Q.m:47``).
4. ``cmc_network_f(n=1)`` is bit-exact to the frozen Phase-33 ``cmc_f``.
5. Lateral ``(1+4L)`` reciprocal reduction (``spm_fx_cmc.m:79-82``).
6. Input ``C`` drives spiny-stellate only (``spm_fx_cmc.m:86,107,171``).
7. Forward/backward adjacency: fwd reads ``S[:,2]`` (sp voltage), bwd reads
   ``S[:,6]`` (dp voltage), with signs + / - (``spm_fx_cmc.m:171,177,183,189``).
8. ``float64`` guard at the network-forward boundary (CMC-07).

0-indexed conductance rows: ss_I=1, sp_I=3, ii_I=5, dp_I=7; firing voltage
columns: sp-V=2, dp-V=6.
"""

from __future__ import annotations

import pytest
import torch
from pyro_dcm.forward_models.erp_coupled_system import (
    apply_condition_modulation,
    cmc_network_f,
    parameterize_cmc_network,
)

from pyro_dcm.forward_models.cmc_neural_mass import (
    cmc_f,
    cmc_flatten,
    cmc_sigmoid,
    cmc_unflatten,
)

_F64 = torch.float64


def _net_p(n: int) -> dict[str, torch.Tensor]:
    """Free-parameter dict for an ``n``-source net, all scaling params zero.

    ``A`` is present and defaults to ``-32`` everywhere (``exp(-32)*E0`` is
    sub-``exp(-8)`` -> effectively no extrinsic edge), mirroring the SPM mask
    convention ``mask*32-32`` (``spm_cmc_priors.m:80``).
    """
    return {
        "T": torch.zeros(n, 4, dtype=_F64),
        "G": torch.zeros(n, 4, dtype=_F64),
        "C": torch.zeros(n, 1, dtype=_F64),
        "S": torch.zeros(n, 1, dtype=_F64),
        "A": torch.full((4, n, n), -32.0, dtype=_F64),
    }


def _intrinsic_p(n: int) -> dict[str, torch.Tensor]:
    """Free-parameter dict with NO ``A`` key (extrinsic blocks default to zero)."""
    return {
        "T": torch.zeros(n, 4, dtype=_F64),
        "G": torch.zeros(n, 4, dtype=_F64),
        "C": torch.zeros(n, 1, dtype=_F64),
        "S": torch.zeros(n, 1, dtype=_F64),
    }


def _apply_no_diag(
    p: dict[str, torch.Tensor], x_design: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Omit-``diag`` variant: folds ``B`` into ``A`` but SKIPS ``diag(B)->G[:,0]``.

    Used by the C4 negative test to prove the precision path (``spm_gen_Q.m:65-67``)
    is load-bearing: skipping it destroys the ``G[:,6]`` modulation.
    """
    x_design = torch.atleast_1d(torch.as_tensor(x_design, dtype=_F64))
    n = p["B"][0].shape[0]
    q = {
        k: v.clone() for k, v in p.items() if k not in ("A", "B") and torch.is_tensor(v)
    }
    if "A" in p:
        a = torch.stack(
            [torch.as_tensor(p["A"][i], dtype=_F64).clone() for i in range(4)], dim=0
        )
    else:
        a = torch.zeros(4, n, n, dtype=_F64)
    for i, xi in enumerate(x_design):
        b_i = torch.as_tensor(p["B"][i], dtype=_F64)
        for j in range(4):
            a[j] = a[j] + xi * b_i
    q["A"] = a
    return q


def test_c4_precision_guard() -> None:
    """``diag(B)`` modulates the ``G[:,6]`` precision knob (spm_gen_Q.m:65-67).

    The headline C4 guard. ``apply_condition_modulation`` adds ``X*diag(B)`` to
    the free precision column ``Q.G[:,0]``; after ``parameterize_cmc_network``
    that column drives ``G[:,6]`` (``sp`` self-inhibition, ``J_PERM[0]==6``), so
    the parameterised precision must MOVE relative to the unmodulated baseline.
    """
    n = 2
    p = _net_p(n)
    b0 = torch.tensor([[0.3, 0.1], [0.0, 0.5]], dtype=_F64)
    p["B"] = [b0]
    q = apply_condition_modulation(p, torch.tensor([1.0], dtype=_F64))

    assert torch.equal(q["G"][:, 0], p["G"][:, 0] + 1.0 * torch.diagonal(b0))

    params = parameterize_cmc_network(q, n)
    params_base = parameterize_cmc_network(p, n)
    assert not torch.allclose(params["G"][:, 6], params_base["G"][:, 6])


def test_c4_negative_omit_diag() -> None:
    """Omitting ``diag(B)->G[:,0]`` leaves ``G[:,6]`` unchanged (EVOK-02).

    The negative control: an ``apply_condition_modulation`` variant that folds
    ``B`` into ``A`` but skips the precision line produces a ``Q.G[:,0]`` that
    differs from the real port by exactly ``X*diag(B)``, and a parameterised
    ``G[:,6]`` identical to baseline -- proving the path is load-bearing.
    """
    n = 2
    p = _net_p(n)
    b0 = torch.tensor([[0.3, 0.1], [0.0, 0.5]], dtype=_F64)
    p["B"] = [b0]
    x = torch.tensor([1.0], dtype=_F64)

    q_real = apply_condition_modulation(p, x)
    q_omit = _apply_no_diag(p, x)

    assert not torch.equal(q_real["G"][:, 0], q_omit["G"][:, 0])

    params_base = parameterize_cmc_network(p, n)
    params_omit = parameterize_cmc_network(q_omit, n)
    assert torch.equal(params_omit["G"][:, 6], params_base["G"][:, 6])


def test_b_folds_into_all_four_a() -> None:
    """One ``B{i}`` adds to ALL four ``Q.A{j}`` in free log-space (spm_gen_Q.m:47)."""
    n = 2
    p = _net_p(n)
    b0 = torch.tensor([[0.3, 0.1], [0.2, 0.5]], dtype=_F64)
    p["B"] = [b0]
    q = apply_condition_modulation(p, torch.tensor([1.0], dtype=_F64))
    for j in range(4):
        assert torch.equal(q["A"][j], p["A"][j] + 1.0 * b0)


def test_network_f_n1_bit_exact() -> None:
    """``cmc_network_f(n=1) == cmc_f`` bit-exact (the 4 extrinsic terms vanish).

    With no ``A`` key the extrinsic blocks default to ``zeros(4,1,1)``, so the
    four ``A@S`` terms are exactly zero and the network forward reproduces the
    parity-gated Phase-33 ``cmc_f`` byte-for-byte (max|diff| == 0.0).
    """
    x = 0.1 * torch.ones(1, 8, dtype=_F64)
    x_flat = cmc_flatten(x)
    u = torch.tensor([32.0], dtype=_F64)
    p = _intrinsic_p(1)

    f_net = cmc_network_f(x_flat, u, p, 1)
    f_base = cmc_f(x_flat, u, p, 1)
    assert torch.equal(f_net, f_base)
    assert (f_net - f_base).abs().max().item() == 0.0


def test_lateral_reciprocal_reduction() -> None:
    """Reciprocal pairs divided by ``1+4L``; one-way edges are not (spm_fx_cmc:79).

    ``E0[0] == 200``. A reciprocal off-diagonal pair (both free params 0 ->
    ``exp(0)*200 == 200 > exp(-8)``) is divided by ``1+4*1 == 5`` -> 40; a one-way
    edge (transpose entry sub-threshold) keeps its full ``200``.
    """
    n = 2
    p_rec = {"A": torch.full((4, n, n), -32.0, dtype=_F64)}
    p_rec["A"][0, 0, 1] = 0.0
    p_rec["A"][0, 1, 0] = 0.0
    a0 = parameterize_cmc_network(p_rec, n)["A"][0]
    assert torch.allclose(a0[0, 1], torch.tensor(200.0 / 5.0, dtype=_F64))
    assert torch.allclose(a0[1, 0], torch.tensor(200.0 / 5.0, dtype=_F64))

    p_one = {"A": torch.full((4, n, n), -32.0, dtype=_F64)}
    p_one["A"][0, 0, 1] = 0.0
    a0o = parameterize_cmc_network(p_one, n)["A"][0]
    assert torch.allclose(a0o[0, 1], torch.tensor(200.0, dtype=_F64))


def test_input_drives_ss_only() -> None:
    """Exogenous input ``C@u`` enters spiny-stellate only (spm_fx_cmc.m:86,107,171).

    Perturbing ``u`` changes the ss conductance derivative row ``f[1]`` and
    leaves the other three conductance rows (sp ``f[3]``, ii ``f[5]``, dp
    ``f[7]``) untouched.
    """
    n = 2
    p = _intrinsic_p(n)
    x = 0.1 * torch.ones(n, 8, dtype=_F64)
    x_flat = cmc_flatten(x)

    f_a = cmc_unflatten(cmc_network_f(x_flat, torch.tensor([0.0], dtype=_F64), p, n), n)
    f_b = cmc_unflatten(cmc_network_f(x_flat, torch.tensor([5.0], dtype=_F64), p, n), n)
    delta = f_b - f_a
    assert torch.all(delta[:, 1] != 0.0)
    for row in (3, 5, 7):
        assert torch.equal(delta[:, row], torch.zeros(n, dtype=_F64))


def test_forward_backward_adjacency() -> None:
    """Forward reads ``S[:,2]`` (+), backward reads ``S[:,6]`` (-) (spm_fx_cmc:171).

    With a state whose ``sp`` voltage (col 2) and ``dp`` voltage (col 6) firing
    are distinguishable, the per-row extrinsic contribution must equal:
    ``f[ss=1] += A1@S[:,2]/T0``, ``f[dp=7] += A2@S[:,2]/T3`` (forward, +);
    ``f[sp=3] -= A3@S[:,6]/T1``, ``f[ii=5] -= A4@S[:,6]/T2`` (backward, -).
    """
    n = 2
    x = torch.zeros(n, 8, dtype=_F64)
    x[:, 2] = torch.tensor([0.2, 0.3], dtype=_F64)
    x[:, 6] = torch.tensor([-0.1, 0.4], dtype=_F64)
    x_flat = cmc_flatten(x)
    u = torch.tensor([0.0], dtype=_F64)

    base = _intrinsic_p(n)
    p_a = _intrinsic_p(n)
    p_a["A"] = torch.full((4, n, n), -32.0, dtype=_F64)
    for j in range(4):
        p_a["A"][j, 1, 0] = 0.0  # one forward edge source0 -> source1 per block

    params = parameterize_cmc_network(p_a, n)
    a = params["A"]
    t = params["T"]
    s = cmc_sigmoid(x, params["S"])

    f_net = cmc_unflatten(cmc_network_f(x_flat, u, p_a, n), n)
    f_base = cmc_unflatten(cmc_network_f(x_flat, u, base, n), n)
    delta = f_net - f_base

    assert torch.allclose(delta[:, 1], (a[0] @ s[:, 2]) / t[:, 0])
    assert torch.allclose(delta[:, 7], (a[1] @ s[:, 2]) / t[:, 3])
    assert torch.allclose(delta[:, 3], (-a[2] @ s[:, 6]) / t[:, 1])
    assert torch.allclose(delta[:, 5], (-a[3] @ s[:, 6]) / t[:, 2])


def test_float64_guard() -> None:
    """``cmc_network_f`` raises ``TypeError`` on a non-float64 state (CMC-07)."""
    p = _intrinsic_p(1)
    x32 = cmc_flatten(0.1 * torch.ones(1, 8, dtype=torch.float32))
    with pytest.raises(TypeError):
        cmc_network_f(x32, torch.tensor([1.0], dtype=_F64), p, 1)
