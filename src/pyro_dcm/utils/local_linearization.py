"""Exponential-Euler (local-linearization) integrator ported from SPM12.

Ports ``spm_int_L.m`` (SPM12, ``$Id: spm_int_L.m 7143``) into pure torch: the
**frozen-Jacobian exponential-Euler** scheme of Ozaki (1992) that SPM uses to
integrate DCM-for-evoked-responses (NOT Runge-Kutta). The integrator freezes
the Jacobian ``J0 = df/dx`` at the expansion point ``x0`` (the steady state),
regularises it with an ``exp(-16)`` shift, forms the single update operator

    Q = (matrix_exp(dt * D * J0 / N) - I) * inv(J0)

once, and then advances the trajectory with ``v <- v + Q @ f(v, u)``.

This module is deliberately CMC-AGNOSTIC: it takes a callable ``f(v, u)`` and
knows nothing about the canonical-microcircuit internals. It is a NEW sibling
of :mod:`pyro_dcm.utils.ode_integrator` (the torchdiffeq Runge-Kutta wrapper)
and does NOT touch it -- the ERP forward must never be routed through an
adaptive Runge-Kutta solver (pitfall C1).

References
----------
SPM12 ``spm_int_L.m:112-169`` -- the exp-Euler loop. The regulariser and the
update operator are ``spm_int_L.m:126-127``; the integration loop is
``spm_int_L.m:132-147``; the substep count ``N`` defaults to 1
(``spm_int_L.m:70``). The right-division ``Q = (E - I) / dfdx`` is MATLAB
``mrdivide`` (right matrix division), i.e. ``(E - I) @ inv(J)``, which for an
ASYMMETRIC ``J`` differs from ``inv(J) @ (E - I)`` (pitfall C2). Ozaki, T.
(1992), "A bridge between nonlinear time series models and nonlinear
stochastic dynamical systems: a local linearization approach", Statistica
Sinica 2, 113-135.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor

_F64 = torch.float64


def _update_operator(
    jacobian: Tensor,
    dt: float,
    n_substeps: int,
    delay_operator: Tensor | None,
) -> tuple[Tensor, Tensor]:
    """Build the exp-Euler propagator ``E`` and update operator ``Q``.

    Implements ``spm_int_L.m:126-127`` verbatim: the ``exp(-16)`` regulariser
    is applied to ``jacobian`` BEFORE forming both ``E`` and ``Q``, and ``Q``
    is the RIGHT-division ``(E - I) @ inv(dfdx)`` evaluated stably via
    ``torch.linalg.solve(dfdx.T, (E - I).T).T`` (never ``torch.inverse``).

    Parameters
    ----------
    jacobian : torch.Tensor
        Frozen Jacobian ``J0 = df/dx`` at the expansion point, shape ``(d, d)``,
        float64.
    dt : float
        Integration step (seconds).
    n_substeps : int
        SPM ``N``; the exponent is scaled by ``1/n_substeps``.
    delay_operator : torch.Tensor or None
        Delay operator ``D``; ``None`` selects the identity (Phase 33).

    Returns
    -------
    e_op : torch.Tensor
        Matrix exponential ``matrix_exp(dt * D @ dfdx / n_substeps)``,
        shape ``(d, d)``.
    q_op : torch.Tensor
        Update operator ``(E - I) @ inv(dfdx)``, shape ``(d, d)``.
    """
    d = jacobian.shape[0]
    identity = torch.eye(d, dtype=jacobian.dtype, device=jacobian.device)
    # Regulariser applied BEFORE E and Q (spm_int_L.m:126).
    dfdx = jacobian - identity * math.exp(-16.0)
    delay = delay_operator if delay_operator is not None else identity
    e_op = torch.matrix_exp(dt * delay @ dfdx / n_substeps)
    # Right-division Q = (E - I) @ inv(dfdx) (spm_int_L.m:127); solve the
    # transposed system so we never form an explicit inverse (pitfall C2).
    rhs = (e_op - identity).transpose(-2, -1)
    q_op = torch.linalg.solve(dfdx.transpose(-2, -1), rhs).transpose(-2, -1)
    return e_op, q_op


def integrate_local_linearization(
    f: Callable[[Tensor, Tensor], Tensor],
    x0: Tensor,
    inputs: Tensor,
    dt: float,
    n_substeps: int = 1,
    delay_operator: Tensor | None = None,
    g: Callable[[Tensor, Tensor], Tensor] | None = None,
) -> Tensor:
    """Integrate ``dv/dt = f(v, u)`` with the SPM12 exp-Euler scheme.

    Ports ``spm_int_L.m:112-169``. The Jacobian is FROZEN at ``x0`` with the
    input held at ``u0 = 0`` (per ``spm_gen_erp.m`` removing ``M.u`` -- the
    Jacobian is taken at ``u = 0``), regularised by ``exp(-16)``, and used to
    form a single update operator ``Q`` (see :func:`_update_operator`). No
    eigenvalue clipping is applied to ``J0`` -- only the ``exp(-16)`` shift
    (pitfall N2; the fMRI stability clip must NOT leak into the CMC path).

    Parameters
    ----------
    f : Callable
        Right-hand side ``f(v, u) -> dv/dt``. ``v`` is the flat state
        ``(d,)``; ``u`` is the per-sample input ``(n_inp,)``.
    x0 : torch.Tensor
        Frozen expansion point (steady state), shape ``(d,)``, float64.
    inputs : torch.Tensor
        Per-sample inputs over time, shape ``(ns, n_inp)``, float64.
    dt : float
        Integration step (seconds), ``U.dt``.
    n_substeps : int, optional
        SPM ``N`` (``spm_int_L.m:70``). Default 1.
    delay_operator : torch.Tensor or None, optional
        Delay operator ``D``, shape ``(d, d)``. ``None`` -> identity
        (Phase 33; the single-source delay is forced to identity).
    g : Callable or None, optional
        Output map ``g(v, u)``. ``None`` -> identity (return states).

    Returns
    -------
    torch.Tensor
        Trajectory, shape ``(ns, d_out)``, float64. ``d_out == d`` when
        ``g is None``.

    Raises
    ------
    TypeError
        If ``x0`` is not float64 (expected ``torch.float64``).
    ValueError
        If ``f(x0, u0)`` is not finite.
    """
    if x0.dtype != _F64:
        raise TypeError(
            f"x0 must be float64; expected {_F64}, got {x0.dtype} "
            "(the exp-Euler integrator runs entirely in float64, CMC-07)"
        )
    u0 = torch.zeros_like(inputs[0])
    f0 = f(x0, u0)
    if not torch.isfinite(f0).all():
        raise ValueError(
            "f(x0, u0) must be finite at the frozen expansion point; "
            f"got non-finite entries: {f0}"
        )
    # Frozen Jacobian at (x0, u0); jacrev is exact and differentiable (Phase 35).
    jacobian = torch.func.jacrev(lambda v: f(v, u0))(x0)
    _, q_op = _update_operator(jacobian, dt, n_substeps, delay_operator)

    ns = inputs.shape[0]
    v = x0.clone()
    outputs: list[Tensor] = []
    for i in range(ns):
        u_i = inputs[i]
        for _ in range(n_substeps):
            v = v + q_op @ f(v, u_i)
        outputs.append(g(v, u_i) if g is not None else v)
    y = torch.stack(outputs, dim=0)
    return y.real if torch.is_complex(y) else y
