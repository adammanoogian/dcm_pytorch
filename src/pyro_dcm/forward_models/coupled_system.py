"""Coupled neural-hemodynamic ODE system for Dynamic Causal Modeling.

Implements the combined ODE right-hand side that integrates the neural
state equation [REF-001] Eq. 1 and Balloon-Windkessel hemodynamic model
[REF-002] Eq. 2-5 into a single 5N-dimensional state vector suitable
for numerical integration via torchdiffeq.

The ``CoupledDCMSystem`` is an ``nn.Module`` to satisfy the
``torchdiffeq.odeint_adjoint`` interface requirement (adjoint method
needs module parameters for backpropagation through the ODE).

State vector layout (N regions):
    [x(N), s(N), lnf(N), lnv(N), lnq(N)]

where x=neural activity, s=vasodilatory signal, lnf=log blood flow,
lnv=log blood volume, lnq=log deoxyhemoglobin.

References
----------
[REF-001] Friston, Harrison & Penny (2003), Eq. 1 — Neural state equation.
[REF-002] Stephan et al. (2007), Eq. 2-5 — Balloon-Windkessel model.
SPM12 source: spm_fx_fmri.m — Combined neural + hemodynamic ODE.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from pyro_dcm.forward_models.balloon_model import BalloonWindkessel
from pyro_dcm.forward_models.neural_state import NeuralStateEquation


class CoupledDCMSystem(nn.Module):
    """Combined neural + hemodynamic ODE system for torchdiffeq integration.

    Assembles the full ODE right-hand side for the 5N state vector by
    combining the neural state equation (dx/dt = Ax + Cu) with the
    Balloon-Windkessel hemodynamic model (ds, dlnf, dlnv, dlnq).

    The A and C matrices are stored as buffers (not parameters) because
    the Pyro generative model (Phase 4) handles parameterization via
    its own prior/guide mechanism. This module is purely a forward
    computation graph.

    Parameters
    ----------
    A : torch.Tensor
        Effective connectivity matrix, shape ``(N, N)``.
    C : torch.Tensor
        Driving input weights, shape ``(N, M)``.
    input_fn : callable
        Stimulus function mapping time ``t`` to input vector ``u(t)``,
        shape ``(M,)``. Typically a ``PiecewiseConstantInput``.
    hemo_params : dict or None, optional
        Hemodynamic parameters {kappa, gamma, tau, alpha, E0}.
        If None, uses SPM12 defaults from ``BalloonWindkessel``.

    Notes
    -----
    Implements [REF-001] Eq. 1 for the neural component and
    [REF-002] Eq. 2-5 for the hemodynamic component, matching
    SPM12 spm_fx_fmri.m.

    Examples
    --------
    >>> import torch
    >>> A = torch.tensor([[-0.5, 0.1], [0.2, -0.5]], dtype=torch.float64)
    >>> C = torch.tensor([[1.0], [0.0]], dtype=torch.float64)
    >>> input_fn = lambda t: torch.tensor([1.0], dtype=torch.float64)
    >>> system = CoupledDCMSystem(A, C, input_fn)
    >>> state = torch.zeros(10, dtype=torch.float64)  # 5*N=10
    >>> dstate = system.forward(torch.tensor(0.0), state)
    >>> dstate.shape  # (10,)
    """

    def __init__(
        self,
        A: torch.Tensor,
        C: torch.Tensor,
        input_fn: Callable[[torch.Tensor], torch.Tensor],
        hemo_params: dict[str, float] | None = None,
    ) -> None:
        super().__init__()

        # Store connectivity matrices as buffers (not parameters).
        # Pyro model will handle parameterization in Phase 4.
        self.register_buffer("A", A)
        self.register_buffer("C", C)

        self.input_fn = input_fn
        self.n_regions = A.shape[0]

        # Instantiate component models
        self.neural = NeuralStateEquation(self.A, self.C)

        if hemo_params is not None:
            self.hemo = BalloonWindkessel(**hemo_params)
        else:
            self.hemo = BalloonWindkessel()

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Compute combined ODE derivatives for the 5N state vector.

        This is the ODE right-hand side called by torchdiffeq:
        ``dy/dt = f(t, y)``.

        Implements [REF-001] Eq. 1 + [REF-002] Eq. 2-5:

        1. Unpack state into [x, s, lnf, lnv, lnq], each shape (N,).
        2. Get u(t) from input function.
        3. Compute dx/dt = Ax + Cu via NeuralStateEquation.
        4. Compute (ds, dlnf, dlnv, dlnq) via BalloonWindkessel.
        5. Return concatenated derivative vector.

        Parameters
        ----------
        t : torch.Tensor
            Current time (scalar tensor).
        state : torch.Tensor
            Full state vector, shape ``(5*N,)``.

        Returns
        -------
        torch.Tensor
            Derivative vector, shape ``(5*N,)``.
        """
        N = self.n_regions

        # 1. Unpack state vector
        x = state[:N]           # neural activity
        s = state[N:2 * N]      # vasodilatory signal
        lnf = state[2 * N:3 * N]  # log blood flow
        lnv = state[3 * N:4 * N]  # log blood volume
        lnq = state[4 * N:5 * N]  # log deoxyhemoglobin

        # 2. Get experimental input at current time
        u = self.input_fn(t)

        # 3. Neural state derivatives [REF-001] Eq. 1
        dx = self.neural.derivatives(x, u)

        # 4. Hemodynamic derivatives [REF-002] Eq. 2-5
        ds, dlnf, dlnv, dlnq = self.hemo.derivatives(x, s, lnf, lnv, lnq)

        # 5. Pack and return
        return torch.cat([dx, ds, dlnf, dlnv, dlnq])
