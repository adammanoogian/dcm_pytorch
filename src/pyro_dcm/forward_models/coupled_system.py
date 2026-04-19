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

v0.3.0 extension
----------------
The module now supports the full Friston 2003 bilinear neural state
equation ``dx/dt = (A + sum_j u_mod[j] * B[j]) @ x + C @ u_drive`` via
the optional ``B`` and ``n_driving_inputs`` kwargs. When ``B is None``
(default), the forward pass executes the literal v0.2.0 linear
expression ``self.A @ x + self.C @ u_all`` and is bit-exact against
pre-existing callers. An eigenvalue-based stability monitor
(``_maybe_check_stability``) logs WARNING to ``pyro_dcm.stability``
when ``max Re(eig(A_eff)) > 0``; the monitor never raises (D4).

References
----------
[REF-001] Friston, Harrison & Penny (2003), Eq. 1 — Neural state equation.
[REF-002] Stephan et al. (2007), Eq. 2-5 — Balloon-Windkessel model.
SPM12 source: spm_fx_fmri.m — Combined neural + hemodynamic ODE.
"""

from __future__ import annotations

import logging
from typing import Callable

import torch
from torch import nn

from pyro_dcm.forward_models.balloon_model import BalloonWindkessel
from pyro_dcm.forward_models.neural_state import (
    NeuralStateEquation,
    compute_effective_A,
)

_STABILITY_LOGGER = logging.getLogger("pyro_dcm.stability")


class CoupledDCMSystem(nn.Module):
    """Combined neural + hemodynamic ODE system for torchdiffeq integration.

    Assembles the full ODE right-hand side for the 5N state vector by
    combining the neural state equation (dx/dt = Ax + Cu in linear mode,
    or dx/dt = (A + sum_j u_mod[j] * B[j]) x + C u_drive in bilinear mode)
    with the Balloon-Windkessel hemodynamic model (ds, dlnf, dlnv, dlnq).

    The A, C, (and optional B) matrices are stored as buffers (not
    parameters) because the Pyro generative model (Phase 4 / Phase 15)
    handles parameterization via its own prior/guide mechanism. This
    module is purely a forward computation graph.

    v0.3.0 bilinear extension
    -------------------------
    Set ``B: (J, N, N)`` and ``n_driving_inputs: int`` to enable the
    bilinear path. ``input_fn(t)`` must then return a
    ``(n_driving_inputs + J,)`` vector: the first ``n_driving_inputs``
    columns are the driving inputs (consumed by ``C @ u_drive``); the
    remaining ``J`` columns are modulators (consumed by
    ``compute_effective_A(A, B, u_mod)``). When ``B is None`` (default),
    the original linear path is preserved bit-exactly.

    Stability monitor (BILIN-05): every ``stability_check_every`` RHS
    evaluations (one tick per ``forward()`` call), the monitor computes
    ``max Re(eig(A_eff))`` and emits a WARNING on the
    ``pyro_dcm.stability`` logger when the value is strictly positive.
    The monitor never raises (D4) and can be disabled with
    ``stability_check_every=0``.

    Parameters
    ----------
    A : torch.Tensor
        Effective connectivity matrix, shape ``(N, N)``.
    C : torch.Tensor
        Driving input weights. Shape ``(N, M)`` in linear mode (default),
        shape ``(N, n_driving_inputs)`` in bilinear mode.
    input_fn : callable
        Stimulus function mapping time ``t`` to the concatenated input
        vector. Linear mode: shape ``(M,)``. Bilinear mode: shape
        ``(n_driving_inputs + J,)`` where ``J == B.shape[0]``.
    hemo_params : dict or None, optional
        Hemodynamic parameters {kappa, gamma, tau, alpha, E0}.
        If None, uses SPM12 defaults from ``BalloonWindkessel``.
    B : torch.Tensor or None, optional
        Stacked modulatory matrices, shape ``(J, N, N)``. ``None``
        (default) or ``B.shape[0] == 0`` routes through the linear
        short-circuit. When supplied, is stored as a buffer and
        auto-aligned to ``A.device`` / ``A.dtype``.
    n_driving_inputs : int or None, optional
        Number of driving-input columns in ``input_fn(t)``. Required
        when ``B`` is non-empty. Raises ValueError if omitted in
        bilinear mode (explicit-split policy per CONTEXT item 4).
    stability_check_every : int, default 10
        Cadence in RHS evaluations (``forward()`` calls) between
        eigenvalue checks. Set to ``0`` to disable the monitor entirely
        (zero overhead).

    Raises
    ------
    ValueError
        If ``B`` is non-empty and ``n_driving_inputs is None``.

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
        *,
        B: torch.Tensor | None = None,
        n_driving_inputs: int | None = None,
        stability_check_every: int = 10,
    ) -> None:
        """Initialize coupled neural-hemodynamic ODE system.

        The keyword-only arguments ``B``, ``n_driving_inputs``, and
        ``stability_check_every`` extend the v0.2.0 linear system to the
        full Friston 2003 bilinear form. When ``B is None`` (default),
        the system preserves bit-exact linear behavior for all existing
        callers.

        Bilinear stacked-tensor convention (CONTEXT.md §"B-matrix
        representation"): ``B: (J, N, N)`` is the single stacked tensor
        covering J modulators.

        Input-splitting convention (CONTEXT.md §"Modulator input
        routing"): ``input_fn(t)`` returns a ``(M_driving + J_mod,)``
        vector. The first ``n_driving_inputs`` columns are the driving
        inputs consumed by ``C @ u_drive``; the remaining ``J_mod``
        columns are the modulators consumed by
        ``compute_effective_A(A, B, u_mod)``.

        Stability monitor (CONTEXT.md §"Stability monitor behavior"):
        every ``stability_check_every`` RHS evaluations (one per
        ``forward()`` call), compute ``max Re(eig(A_eff))``; if strictly
        positive, emit a WARNING to the ``pyro_dcm.stability`` logger.
        Cadence is measured in RHS evaluations, not ODE steps: rk4
        invokes ``forward()`` 4 times per ODE step, dopri5 invokes 6+
        times per accepted step (plus retries). With default
        ``stability_check_every=10``, rk4 checks stability roughly every
        2.5 ODE steps. This is an intentional approximation — a true
        per-ODE-step callback is not available in torchdiffeq. Never
        raises (D4). Set to ``0`` to disable entirely (zero overhead).

        Parameters
        ----------
        A : torch.Tensor
            Effective connectivity matrix, shape ``(N, N)``.
        C : torch.Tensor
            Driving input weights. Shape ``(N, M)`` in linear mode
            (default), shape ``(N, n_driving_inputs)`` in bilinear mode.
        input_fn : callable
            Stimulus function mapping time ``t`` to the concatenated
            input vector. Linear mode: shape ``(M,)``. Bilinear mode:
            shape ``(n_driving_inputs + J,)`` where ``J == B.shape[0]``.
        hemo_params : dict or None, optional
            Hemodynamic parameters {kappa, gamma, tau, alpha, E0}.
        B : torch.Tensor or None, optional
            Stacked modulatory matrices, shape ``(J, N, N)``. None
            (default) or ``B.shape[0] == 0`` routes through the linear
            short-circuit.
        n_driving_inputs : int or None, optional
            Number of driving-input columns in ``input_fn(t)``. Required
            when ``B`` is non-empty. Raises ValueError if omitted in
            bilinear mode (explicit-split policy per CONTEXT §Discretion
            item 4).
        stability_check_every : int, default 10
            Cadence in RHS evaluations (``forward()`` calls) between
            eigenvalue checks. Set to ``0`` to disable the monitor
            entirely (zero overhead). Counter is persistent across
            separate ``integrate_ode`` calls on the same instance.

        Raises
        ------
        ValueError
            If ``B`` is non-empty and ``n_driving_inputs is None``.
        """
        super().__init__()

        # Store A and C as buffers (existing v0.2.0 pattern).
        self.register_buffer("A", A)
        self.register_buffer("C", C)

        # Store B as a buffer when supplied; use None attribute otherwise.
        # Mixing None with buffer-slot is avoided per research Section 2.
        if B is not None:
            # Device/dtype alignment mitigates the device-drift risk
            # (13-RESEARCH Section 10.3).
            B = B.to(device=A.device, dtype=A.dtype)
            self.register_buffer("B", B)
        else:
            self.B = None

        # Enforce explicit-split policy when bilinear is active.
        if B is not None and B.shape[0] > 0 and n_driving_inputs is None:
            raise ValueError(
                "CoupledDCMSystem: n_driving_inputs is required when B is "
                "non-empty. Got B.shape="
                f"{tuple(B.shape)} but n_driving_inputs=None. Set "
                "n_driving_inputs explicitly to disambiguate driving vs "
                "modulator columns of input_fn(t)."
            )

        self.n_driving_inputs = n_driving_inputs
        self.input_fn = input_fn
        self.n_regions = A.shape[0]

        # Stability monitor configuration.
        self.stability_check_every = int(stability_check_every)
        self._step_counter = 0  # plain int, persistent across integrate_ode calls

        # Component models (neural uses the shared A, C references).
        self.neural = NeuralStateEquation(self.A, self.C)

        if hemo_params is not None:
            self.hemo = BalloonWindkessel(**hemo_params)
        else:
            self.hemo = BalloonWindkessel()

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Compute combined ODE derivatives for the 5N state vector.

        Implements [REF-001] Eq. 1 + [REF-002] Eq. 2-5. The neural RHS
        branches on ``self.B``: when ``None`` or empty-J, evaluates the
        literal v0.2.0 linear expression ``self.A @ x + self.C @ u_all``
        (CONTEXT-locked short-circuit); when non-empty, composes
        ``A_eff(t) = A + sum_j u_mod[j] * B[j]`` via
        ``compute_effective_A`` and routes
        ``A_eff @ x + self.C @ u_drive``.

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

        # 2. Get concatenated experimental input at current time
        u_all = self.input_fn(t)

        # 3. Neural state derivatives — branch on bilinear vs linear
        if self.B is None or self.B.shape[0] == 0:
            # Linear short-circuit (BILIN-03 bit-exact gate). Must
            # execute the literal v0.2.0 expression — do NOT refactor
            # to A_eff = A + 0.
            dx = self.A @ x + self.C @ u_all
        else:
            M_d = self.n_driving_inputs
            u_drive = u_all[:M_d]
            u_mod = u_all[M_d:]
            A_eff = compute_effective_A(self.A, self.B, u_mod)
            dx = A_eff @ x + self.C @ u_drive
            # Optional stability monitor (BILIN-05). Skipped when
            # stability_check_every == 0.
            self._maybe_check_stability(t, A_eff, u_mod)

        # 4. Hemodynamic derivatives [REF-002] Eq. 2-5
        ds, dlnf, dlnv, dlnq = self.hemo.derivatives(x, s, lnf, lnv, lnq)

        # 5. Pack and return
        return torch.cat([dx, ds, dlnf, dlnv, dlnq])

    def _maybe_check_stability(
        self,
        t: torch.Tensor,
        A_eff: torch.Tensor,
        u_mod: torch.Tensor,
    ) -> None:
        """Log a WARNING when ``max Re(eig(A_eff))`` is strictly positive.

        Counter-modulo cadence on RHS evaluations (one counter tick per
        ``forward()`` call). The counter increments every call and
        triggers the eigenvalue check when
        ``counter % stability_check_every == 0``.
        ``stability_check_every == 0`` disables the monitor entirely
        (zero overhead — early return before any eigenvalue
        computation).

        For fixed-step ``rk4``, each ODE step comprises 4 RHS
        evaluations; for adaptive ``dopri5``, 6+ evaluations per
        accepted step (plus retries). ``stability_check_every=10``
        therefore samples every ~2.5 ODE steps on rk4. This is an
        intentional approximation — a true per-ODE-step callback is
        not available in torchdiffeq.

        Strict gate ``max Re > 0`` per D4. Never raises — SVI divergent
        draws are expected during early inference and hard-stopping
        would corrupt gradient estimates. Users silence via::

            logging.getLogger("pyro_dcm.stability").setLevel(logging.ERROR)

        Parameters
        ----------
        t : torch.Tensor
            Current time (scalar tensor; used only for the log message).
        A_eff : torch.Tensor
            Effective connectivity at current time, shape ``(N, N)``.
        u_mod : torch.Tensor
            Modulator values at current time, shape ``(J,)``. Used only
            to compute the diagnostic ``||B·u_mod||_F`` culprit norm.
        """
        if self.stability_check_every <= 0:
            return
        self._step_counter += 1
        if self._step_counter % self.stability_check_every != 0:
            return

        # Eigenvalue computation is detached from autograd (no semantic
        # value in backprop through eigenvalues; avoids complex-gradient
        # overhead).
        with torch.no_grad():
            # torch.linalg.eigvals returns complex dtype; .real extracts
            # the real parts as the corresponding real dtype.
            eigs = torch.linalg.eigvals(A_eff.detach())
            max_re = eigs.real.max().item()
            if max_re > 0.0:
                # ||B · u_mod||_F = Frobenius norm of the stacked
                # modulator contribution sum_j u_mod[j] * B[j].
                culprit = torch.einsum("j,jnm->nm", u_mod, self.B)
                culprit_norm = culprit.norm().item()
                _STABILITY_LOGGER.warning(
                    "Stability warning at t=%.2fs: "
                    "max Re(eig(A_eff))=%+.3f; ||B·u_mod||_F=%.3f",
                    float(t.item()),
                    max_re,
                    culprit_norm,
                )
