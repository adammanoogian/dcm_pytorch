"""C1-isolation tests for the spm_int_L exponential-Euler integrator port.

Written BEFORE ``src/pyro_dcm/utils/local_linearization.py`` (test-first, C1
isolation): these pin the right-division orientation of the update operator
``Q`` (pitfall C2), the ``exp(-16)`` regulariser ordering, float64 enforcement
(CMC-07 / N1), and the absence of any eigenvalue clipping (pitfall N2) BEFORE
any CMC dynamics can compound an integration-scheme error.

All tests are pure-torch and sub-second (laptop tier); the MATLAB
``matrix_exp`` vs ``spm_expm`` MEASUREMENT lands in Wave 3.
"""

from __future__ import annotations

import math

import pytest
import torch

from pyro_dcm.utils.local_linearization import (
    _update_operator,
    integrate_local_linearization,
)

_F64 = torch.float64


def _asymmetric_jacobian() -> torch.Tensor:
    """Return a deliberately ASYMMETRIC, invertible 4x4 float64 matrix."""
    return torch.tensor(
        [
            [-3.0, 1.0, 0.5, 0.0],
            [0.2, -2.5, 0.0, 1.0],
            [0.0, 0.7, -4.0, 0.3],
            [0.4, 0.0, 0.1, -1.5],
        ],
        dtype=_F64,
    )


def test_right_division_orientation() -> None:
    """Q is ``(E - I) @ inv(J)`` (right-division), NOT ``inv(J) @ (E - I)``.

    Pitfall C2: with a non-identity, non-symmetric delay operator ``D`` the
    propagator ``E = matrix_exp(dt * D @ dfdx)`` does NOT commute with
    ``inv(dfdx)``, so the two orderings genuinely differ (when ``D == I``,
    ``E`` is a function of ``dfdx`` and the orderings coincide, hiding the
    bug). The reference inverse is computed ONLY here in the test.
    """
    jacobian = _asymmetric_jacobian()
    dt = 0.5
    identity = torch.eye(4, dtype=_F64)
    delay = torch.tensor(
        [
            [1.0, 0.3, 0.0, 0.0],
            [0.0, 1.0, 0.2, 0.0],
            [0.0, 0.0, 1.0, 0.4],
            [0.1, 0.0, 0.0, 1.0],
        ],
        dtype=_F64,
    )
    _, q = _update_operator(jacobian, dt, 1, delay)

    dfdx = jacobian - identity * math.exp(-16.0)
    e_ref = torch.matrix_exp(dt * delay @ dfdx)
    q_right = (e_ref - identity) @ torch.linalg.inv(dfdx)
    q_wrong = torch.linalg.inv(dfdx) @ (e_ref - identity)

    assert torch.allclose(q, q_right, rtol=0.0, atol=1e-12)
    assert not torch.allclose(q, q_wrong, rtol=0.0, atol=1e-9)


def test_regulariser_before_q() -> None:
    """The ``J -> J - I*exp(-16)`` shift is applied BEFORE forming E and Q."""
    jacobian = _asymmetric_jacobian()
    dt = 0.5
    identity = torch.eye(4, dtype=_F64)
    e_op, q = _update_operator(jacobian, dt, 1, None)

    # Reference built from the SHIFTED Jacobian (same as implementation).
    dfdx = jacobian - identity * math.exp(-16.0)
    e_shift = torch.matrix_exp(dt * dfdx)
    q_shift = (e_shift - identity) @ torch.linalg.inv(dfdx)

    # Reference built from the UNSHIFTED Jacobian (what a buggy impl would do).
    e_unshift = torch.matrix_exp(dt * jacobian)
    q_unshift = (e_unshift - identity) @ torch.linalg.inv(jacobian)

    assert torch.allclose(e_op, e_shift, rtol=0.0, atol=1e-12)
    assert torch.allclose(q, q_shift, rtol=0.0, atol=1e-12)
    assert not torch.allclose(q, q_unshift, rtol=0.0, atol=1e-9)


def test_float64_enforced() -> None:
    """A float32 ``x0`` is rejected (CMC-07 / N1)."""
    jacobian = _asymmetric_jacobian()

    def f(v: torch.Tensor, _u: torch.Tensor) -> torch.Tensor:
        return jacobian.to(v.dtype) @ v

    x0_f32 = torch.zeros(4, dtype=torch.float32)
    inputs = torch.zeros(2, 1, dtype=torch.float32)
    with pytest.raises(TypeError):
        integrate_local_linearization(f, x0_f32, inputs, dt=0.1)


def test_no_eig_clip() -> None:
    """An f whose Jacobian has a POSITIVE real eigenvalue is not clipped (N2).

    With the fMRI eigenvalue-clip applied the unstable mode would decay; the
    exp-Euler integrator only applies the ``exp(-16)`` shift, so the trajectory
    must GROW.
    """
    jacobian = torch.tensor([[0.5, 0.1], [0.0, -1.0]], dtype=_F64)

    def f(v: torch.Tensor, _u: torch.Tensor) -> torch.Tensor:
        return jacobian @ v

    x0 = torch.full((2,), 0.01, dtype=_F64)
    inputs = torch.zeros(5, 1, dtype=_F64)
    traj = integrate_local_linearization(f, x0, inputs, dt=0.1)

    assert traj.dtype == _F64
    assert traj.shape == (5, 2)
    assert traj[-1].abs().max() > x0.abs().max()


def test_identity_delay_default() -> None:
    """``delay_operator=None`` uses D == identity."""
    jacobian = _asymmetric_jacobian()
    dt = 0.3
    identity = torch.eye(4, dtype=_F64)
    e_none, q_none = _update_operator(jacobian, dt, 1, None)
    e_eye, q_eye = _update_operator(jacobian, dt, 1, identity)
    assert torch.allclose(e_none, e_eye, atol=1e-15)
    assert torch.allclose(q_none, q_eye, atol=1e-15)


def test_linear_ode_matches_closed_form() -> None:
    """One step ``v1 = v0 + Q @ f(v0)`` reproduces the exp-Euler update."""
    jacobian = _asymmetric_jacobian()
    dt = 0.2

    def f(v: torch.Tensor, _u: torch.Tensor) -> torch.Tensor:
        return jacobian @ v

    x0 = torch.tensor([0.01, -0.02, 0.03, 0.0], dtype=_F64)
    inputs = torch.zeros(1, 1, dtype=_F64)
    traj = integrate_local_linearization(f, x0, inputs, dt=dt)

    e_op, _ = _update_operator(jacobian, dt, 1, None)
    expected = e_op @ x0

    assert traj.shape == (1, 4)
    assert traj.dtype == _F64
    assert torch.isfinite(traj).all()
    assert torch.allclose(traj[0], expected, atol=1e-8)
