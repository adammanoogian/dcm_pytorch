"""SPM12 single-source CMC-ERP parity ladder (CMC-06, the Phase-33 gate).

Asserts the pure-torch canonical-microcircuit forward
(:func:`pyro_dcm.forward_models.cmc_neural_mass.cmc_f`) and the exponential-Euler
integrator (:func:`pyro_dcm.utils.local_linearization.integrate_local_linearization`)
are element-wise equivalent to SPM12 IN ISOLATION, against the byte-frozen
fixtures generated on M3 in Plan 33-02
(``validation/data/erp_single_source_fixtures.mat``, SPM ``$Id`` ``spm_int_L.m
7143`` / ``spm_fx_cmc.m 7279``). Parity is vs-SPM, never vs-torch (pitfall V1);
tolerances are element-wise forward agreement only -- NO absolute free energy and
NO element-wise ``Cp`` (pitfall V2 / the Phase-32 270-nat lesson).

The fixtures are committed to the repo, so this suite RUNS AND PASSES on the
laptop (no MATLAB needed -- the comparison is torch-against-the-frozen-arrays).
The gate therefore keys on FIXTURE AVAILABILITY, not MATLAB availability: it
skips only when the ``.mat`` is missing. The ``@pytest.mark.spm`` / ``slow``
markers are retained so the M3 sbatch (``cluster/sbatch/erp_parity_test.sbatch``)
also runs it under ``-m "spm and slow"``.

Staged ladder (pitfall V5 -- a failure localises to one stage), asserted in
order:

1. ``f_field``  -- ``cmc_f`` at the frozen nonzero ``(x_test, u_test)``;
   isolates every transform / sigmoid / ``J_PERM`` permutation BEFORE the
   integrator.
2a. ``J0`` via the SAME ``spm_diff`` forward-difference scheme SPM used
    (``dx = exp(-8)``); this is the TRUE forward-parity gate -- it is bit-exact
    to the fixture, proving ``cmc_f`` IS ``spm_fx_cmc``.
2b. ``J0`` via exact ``jacrev`` -- the MEASURED autodiff-vs-``spm_diff`` floor
    (the shipped integrator uses ``jacrev``; SPM uses finite differences, so the
    two Jacobians differ by the FD truncation error -- recorded, not assumed).
3. ``matrix_exp(dtJ)`` vs ``spm_expm`` ``Eexp`` -- the MEASURED Pade floor
    (pitfall V3 -- the ~1e-12 expectation is measured, never assumed).
4. ``Q_update`` -- the right-division ``(E - I) @ inv(dfdx)`` (pitfall C2).
5a. ``y_states`` driven by SPM's OWN frozen ``Q_update`` -- isolates the
    exp-Euler loop ordering (``v = v + Q @ f(v, u)``, NOT ``v = Q @ (v + f)``)
    from the Jacobian-construction method (pitfall C1).
5b. ``y_states`` from the full operator path built on SPM's ``spm_diff``
    Jacobian -- the end-to-end integrator matches SPM once the (orthogonal)
    Jacobian method is held to SPM's.
5c. ``y_states`` from the SHIPPED ``jacrev`` integrator -- the MEASURED floor of
    the propagated ``spm_diff`` truncation (the exact-AD trajectory is MORE
    accurate than SPM's).

References
----------
SPM12 source: ``spm_int_L.m:112-169`` (the exp-Euler loop + ``spm_diff``
Jacobian), ``spm_fx_cmc.m:171-198`` (equations of motion), ``spm_diff.m`` (the
forward-difference Jacobian, default step ``exp(-8)``), ``spm_expm.m`` (the
matrix exponential). Plan 33-02 (the frozen-fixture provenance); pitfalls
V1-V5 / C1-C2 in ``.planning/research/v0.8.0/PITFALLS.md``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from pyro_dcm.forward_models.cmc_neural_mass import cmc_f
from pyro_dcm.utils.local_linearization import (
    _update_operator,
    integrate_local_linearization,
)

_F64 = torch.float64

# The frozen SPM12 ground truth (committed in Plan 33-02; validation/data/ is
# mutagen-ignored, so the byte-frozen .mat lives in git).
_FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "validation"
    / "data"
    / "erp_single_source_fixtures.mat"
)

# SPM's spm_diff default forward-difference step (the Jacobian SPM freezes for
# spm_int_L); replicating it makes the J0 / y_states rungs bit-exact.
_SPM_DIFF_DX = math.exp(-8.0)
# The spm_int_L regulariser (spm_int_L.m:126): dfdx = J0 - I*exp(-16).
_EXP_SHIFT = math.exp(-16.0)

pytestmark = [
    pytest.mark.spm,
    pytest.mark.slow,
    pytest.mark.skipif(
        not _FIXTURE_PATH.exists(),
        reason=f"frozen SPM12 fixtures absent: {_FIXTURE_PATH}",
    ),
]


def _reference_p() -> dict[str, torch.Tensor]:
    """Frozen single-source reference ``P`` (all-zeros log-scale prior mean).

    The exact ``P`` struct ``export_erp_dcm`` wrote for the fixture run
    (33-01-SUMMARY / 33-02-SUMMARY): ``T`` zeros(1,4), ``G`` zeros(1,4),
    ``C`` zeros(1,1), ``S`` zeros(1,1). ``R`` gates only the input grid (frozen
    into ``meta.u_grid``) and is not consumed by :func:`cmc_f`; ``A`` / ``D`` are
    absent at ``n = 1``.
    """
    return {
        "T": torch.zeros(1, 4, dtype=_F64),
        "G": torch.zeros(1, 4, dtype=_F64),
        "C": torch.zeros(1, 1, dtype=_F64),
        "S": torch.zeros(1, 1, dtype=_F64),
    }


def _spm_diff_jacobian(
    f: Any, x0: torch.Tensor, dx: float = _SPM_DIFF_DX
) -> torch.Tensor:
    """Forward-difference Jacobian replicating ``spm_diff`` (``dx = exp(-8)``).

    ``spm_diff(@spm_fx_cmc, x0, u0, P, M, 1)`` builds ``J0`` as a one-sided
    forward difference ``J[:, i] = (f(x0 + dx*e_i) - f(x0)) / dx`` with default
    step ``dx = exp(-8)``. This is the Jacobian SPM's ``spm_int_L`` freezes, so
    matching the scheme is what makes the parity bit-exact.
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
    """Load the frozen SPM12 fixtures + provenance ``meta`` once per module."""
    import scipy.io as sio

    mat = sio.loadmat(str(_FIXTURE_PATH))
    meta = mat["meta"]

    def _m(name: str) -> np.ndarray:
        return np.asarray(meta[name][0, 0])

    return {
        "f_field": torch.as_tensor(mat["f_field"], dtype=_F64).reshape(-1),
        "J0": torch.as_tensor(mat["J0"], dtype=_F64),
        "dtJ": torch.as_tensor(mat["dtJ"], dtype=_F64),
        "Eexp": torch.as_tensor(mat["Eexp"], dtype=_F64),
        "Q_update": torch.as_tensor(mat["Q_update"], dtype=_F64),
        "y_states": torch.as_tensor(mat["y_states"], dtype=_F64),
        "x_test": torch.as_tensor(_m("x_test"), dtype=_F64).reshape(-1),
        "u_test": float(_m("u_test").ravel()[0]),
        "u_grid": torch.as_tensor(_m("u_grid"), dtype=_F64),
        "dt": float(_m("dt").ravel()[0]),
        "D": int(_m("D").ravel()[0]),
        "x0": torch.as_tensor(_m("x0"), dtype=_F64).reshape(-1),
    }


def test_pre_asserts(fx: dict[str, Any]) -> None:
    """Mandatory gate pre-conditions: ``D == 1``, ``x0 == 0``, float64.

    Bakes the fixture provenance into the gate (the integrator is only valid at
    the asserted CMC fixed point with delays off): ``meta.D == 1`` (delay
    operator forced to identity), ``meta.x0 == zeros(8)`` (the M1 fixed point),
    the frozen ``u_test == 32`` / ``dt == 0.004``, and float64 at the boundary
    (CMC-07).
    """
    assert int(fx["D"]) == 1, f"meta.D must be 1 (delays off); got {fx['D']}"
    assert torch.equal(fx["x0"], torch.zeros(8, dtype=_F64)), (
        f"meta.x0 must be exactly zeros(8) (M1 fixed point); got {fx['x0']}"
    )
    assert fx["x_test"].dtype == _F64
    assert fx["y_states"].dtype == _F64
    assert abs(fx["u_test"] - 32.0) < 1e-12, (
        f"u_test must be 32.0 (peak Gaussian, P.R=0); got {fx['u_test']}"
    )
    assert abs(fx["dt"] - 0.004) < 1e-12, (
        f"dt must be 0.004 s; got {fx['dt']}"
    )


def test_f_field_parity(fx: dict[str, Any]) -> None:
    """[RUNG 1] ``cmc_f`` at the frozen ``(x_test, u_test)`` vs ``f_field``.

    Isolates every transform / sigmoid / time-constant / ``J_PERM`` permutation
    / input-scaling BEFORE the integrator can compound anything. Tolerance
    ``<= 1e-10`` (element-wise forward agreement, V2).
    """
    p = _reference_p()
    u_test = torch.tensor([fx["u_test"]], dtype=_F64)
    f_torch = cmc_f(fx["x_test"].clone(), u_test, p, 1)
    max_diff = (f_torch - fx["f_field"]).abs().max().item()
    print(f"\n[RUNG 1] f_field max|diff| = {max_diff:.3e}  (tol 1e-10)")
    assert max_diff <= 1e-10, (
        f"f_field parity failed: max|diff|={max_diff:.3e} > 1e-10. The bug is "
        "in cmc_f's transforms/sigmoid/J_PERM/units (pre-integrator, V5)."
    )


def test_j0_forward_difference_parity(fx: dict[str, Any]) -> None:
    """[RUNG 2a] ``cmc_f`` ``spm_diff`` FD Jacobian at ``x0=0,u0=0`` vs ``J0``.

    SPM freezes ``J0`` via ``spm_diff`` (forward differences, ``dx = exp(-8)``),
    NOT analytically; replicating that scheme makes this the TRUE forward-parity
    gate -- it isolates "is ``cmc_f`` == ``spm_fx_cmc``" from the autodiff-vs-FD
    Jacobian-construction method (rung 2b). Bit-exact to the fixture.
    Tolerance ``<= 1e-10``.
    """
    p = _reference_p()
    u0 = torch.zeros(1, dtype=_F64)
    x0 = torch.zeros(8, dtype=_F64)
    assert int(fx["D"]) == 1
    assert torch.equal(x0, torch.zeros(8, dtype=_F64))
    j_fd = _spm_diff_jacobian(lambda v: cmc_f(v, u0, p, 1), x0)
    max_diff = (j_fd - fx["J0"]).abs().max().item()
    print(f"\n[RUNG 2a] J0 (spm_diff FD) max|diff| = {max_diff:.3e}  (tol 1e-10)")
    assert max_diff <= 1e-10, (
        f"J0 forward-parity failed: max|diff|={max_diff:.3e} > 1e-10. cmc_f and "
        "spm_fx_cmc disagree (Jacobian method held identical via spm_diff)."
    )


def test_j0_autodiff_vs_spm_diff_floor(fx: dict[str, Any]) -> None:
    """[RUNG 2b] MEASURED ``jacrev``-vs-``spm_diff`` ``J0`` floor.

    The SHIPPED integrator freezes its Jacobian with exact ``torch.func.jacrev``;
    SPM uses ``spm_diff`` (forward differences). The two Jacobians therefore
    differ by the ``spm_diff`` truncation error -- this is NOT a bug, the exact
    autodiff Jacobian is MORE accurate. The floor is MEASURED + recorded (the
    same "measure, don't assume" discipline as the ``matrix_exp`` floor, V3); the
    loose bound only pins its order of magnitude.
    """
    p = _reference_p()
    u0 = torch.zeros(1, dtype=_F64)
    x0 = torch.zeros(8, dtype=_F64)
    j_ad = torch.func.jacrev(lambda v: cmc_f(v, u0, p, 1))(x0)
    floor = (j_ad - fx["J0"]).abs().max().item()
    print(
        f"\n[RUNG 2b] jacrev-vs-spm_diff J0 floor = {floor:.3e}  "
        "(FD truncation, dx=exp(-8); jacrev is exact/more-accurate)"
    )
    assert floor < 1e-2, (
        f"jacrev-vs-spm_diff floor {floor:.3e} unexpectedly large (> 1e-2); the "
        "spm_diff forward-difference truncation should be ~5e-4."
    )


def test_matrix_exp_floor_measured(fx: dict[str, Any]) -> None:
    """[RUNG 3] MEASURED ``torch.matrix_exp(dtJ)`` vs ``spm_expm`` ``Eexp`` floor.

    The matrix-exponential backend floor (Pade vs ``spm_expm``) is MEASURED on
    the exported ``dtJ``, never assumed (V3 -- the MEDIUM-confidence ~1e-12 is
    measured). This value sets the small-multiple ceilings the ``Q_update`` /
    ``y_states`` rungs ride on. Recorded ceiling ``< 1e-9``.
    """
    e_torch = torch.matrix_exp(fx["dtJ"])
    floor = (e_torch - fx["Eexp"]).abs().max().item()
    print(
        f"\n[RUNG 3] matrix_exp(dtJ) vs spm_expm Eexp floor = {floor:.3e}  "
        "(tol 1e-9, MEASURED)"
    )
    assert floor < 1e-9, (
        f"matrix_exp floor {floor:.3e} >= 1e-9; torch.matrix_exp and spm_expm "
        "diverge beyond the recorded ceiling."
    )


def test_q_update_parity(fx: dict[str, Any]) -> None:
    """[RUNG 4] right-division ``Q = (E - I) @ inv(dfdx)`` vs ``Q_update``.

    Uses the EXPORTED ``Eexp`` + the regularised ``dfdx = J0 - I*exp(-16)`` so
    the rung isolates the right-division orientation (pitfall C2:
    ``solve(dfdx.T, (E-I).T).T``, never ``inv(dfdx) @ (E-I)``) from the
    ``matrix_exp`` floor. Tolerance ``<= 1e-9``.
    """
    identity = torch.eye(8, dtype=_F64)
    dfdx = fx["J0"] - identity * _EXP_SHIFT
    q_torch = torch.linalg.solve(
        dfdx.transpose(-2, -1), (fx["Eexp"] - identity).transpose(-2, -1)
    ).transpose(-2, -1)
    max_diff = (q_torch - fx["Q_update"]).abs().max().item()
    print(
        f"\n[RUNG 4] Q_update (right-division) max|diff| = "
        f"{max_diff:.3e}  (tol 1e-9)"
    )
    assert max_diff <= 1e-9, (
        f"Q_update parity failed: max|diff|={max_diff:.3e} > 1e-9. The bug is in "
        "the right-division orientation (pitfall C2)."
    )


def test_y_states_scheme_parity(fx: dict[str, Any]) -> None:
    """[RUNG 5a] exp-Euler loop driven by SPM's OWN ``Q_update`` vs ``y_states``.

    Holding the update operator fixed to SPM's frozen ``Q_update`` isolates the
    integration SCHEME + loop ordering (``v = v + Q @ f(v, u)``, NOT
    ``v = Q @ (v + f)``) from the Jacobian-construction method -- this proves the
    exp-Euler loop (pitfall C1) is bit-exact to ``spm_int_L``. Tolerance
    ``<= 1e-8``.
    """
    p = _reference_p()
    q = fx["Q_update"]
    u_grid = fx["u_grid"]
    v = torch.zeros(8, dtype=_F64)
    outputs: list[torch.Tensor] = []
    for i in range(u_grid.shape[0]):
        v = v + q @ cmc_f(v, u_grid[i], p, 1)
        outputs.append(v.clone())
    y = torch.stack(outputs, dim=0)
    max_diff = (y - fx["y_states"]).abs().max().item()
    print(
        f"\n[RUNG 5a] y_states (scheme, SPM's Q_update) max|diff| = "
        f"{max_diff:.3e}  (tol 1e-8)"
    )
    assert max_diff <= 1e-8, (
        f"y_states scheme parity failed: max|diff|={max_diff:.3e} > 1e-8 with "
        "SPM's own Q operator. The bug is loop ordering (v += Q@f vs "
        "v = Q@(v+f)), not algebra (C1)."
    )


def test_y_states_full_integrator_fd_jacobian(fx: dict[str, Any]) -> None:
    """[RUNG 5b] full operator path on SPM's ``spm_diff`` Jacobian vs ``y_states``.

    Builds the update operator through the production ``_update_operator``
    (``exp(-16)`` shift -> ``matrix_exp`` -> right-division) on SPM's
    ``spm_diff`` (``dx=exp(-8)``) frozen Jacobian, then runs the exp-Euler loop.
    Proves the END-TO-END local-linearization integrator matches ``spm_int_L``
    once the orthogonal Jacobian-construction method is held to SPM's. Tolerance
    ``<= 1e-8``.
    """
    p = _reference_p()
    u0 = torch.zeros(1, dtype=_F64)
    x0 = torch.zeros(8, dtype=_F64)
    j_fd = _spm_diff_jacobian(lambda v: cmc_f(v, u0, p, 1), x0)
    _, q = _update_operator(j_fd, fx["dt"], 1, None)
    u_grid = fx["u_grid"]
    v = x0.clone()
    outputs: list[torch.Tensor] = []
    for i in range(u_grid.shape[0]):
        v = v + q @ cmc_f(v, u_grid[i], p, 1)
        outputs.append(v.clone())
    y = torch.stack(outputs, dim=0)
    max_diff = (y - fx["y_states"]).abs().max().item()
    print(
        f"\n[RUNG 5b] y_states (full integrator, spm_diff Jacobian) max|diff| = "
        f"{max_diff:.3e}  (tol 1e-8)"
    )
    assert max_diff <= 1e-8, (
        f"y_states full-integrator parity failed: max|diff|={max_diff:.3e} > "
        "1e-8 with SPM's spm_diff Jacobian. The operator path / loop diverges "
        "from spm_int_L."
    )


def test_y_states_jacrev_integrator_floor(fx: dict[str, Any]) -> None:
    """[RUNG 5c] MEASURED floor of the SHIPPED ``jacrev`` integrator vs ``y_states``.

    The production :func:`integrate_local_linearization` freezes its Jacobian
    with exact ``jacrev`` (the Wave-1 design choice for Phase-35
    differentiability), so its trajectory differs from SPM's ``spm_diff``-Jacobian
    trajectory by the PROPAGATED FD truncation -- MEASURED + recorded, not
    assumed. The exact-AD trajectory is more accurate than SPM's; the loose bound
    only pins the propagated floor's magnitude.
    """
    p = _reference_p()
    y = integrate_local_linearization(
        lambda v, u: cmc_f(v, u, p, 1),
        torch.zeros(8, dtype=_F64),
        fx["u_grid"].clone(),
        fx["dt"],
    )
    floor = (y - fx["y_states"]).abs().max().item()
    print(
        f"\n[RUNG 5c] y_states (shipped jacrev integrator) floor = {floor:.3e}  "
        "(propagated spm_diff FD truncation; jacrev is more accurate)"
    )
    assert floor < 1e-6, (
        f"shipped-jacrev y_states floor {floor:.3e} unexpectedly large (> 1e-6); "
        "the propagated spm_diff truncation should be ~5e-8."
    )
