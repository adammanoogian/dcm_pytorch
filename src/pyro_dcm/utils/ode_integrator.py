"""ODE integration utilities for Dynamic Causal Modeling.

Provides:
- ``PiecewiseConstantInput``: Handles experimental stimulus discontinuities
  critical for block-design fMRI via ``torch.searchsorted`` and grid_points.
- ``integrate_ode``: Wrapper around torchdiffeq ``odeint``/``odeint_adjoint``
  with configurable solver selection (dopri5, rk4, euler).
- ``make_initial_state``: Constructs zero-vector steady-state initial conditions
  for the 5N coupled ODE system.

References
----------
torchdiffeq API: https://github.com/rtqichen/torchdiffeq
"""

from __future__ import annotations

import torch
from torchdiffeq import odeint, odeint_adjoint


class PiecewiseConstantInput:
    """Piecewise-constant experimental stimulus function u(t).

    Maps a continuous time ``t`` to the appropriate stimulus value using
    ``torch.searchsorted``, which is critical for handling block-design
    fMRI paradigms with sharp onset/offset discontinuities.

    The ``grid_points`` property returns the onset times for use with
    adaptive ODE solvers (e.g., dopri5), which need to restart at
    discontinuities to maintain accuracy.

    Parameters
    ----------
    times : torch.Tensor
        Onset times, shape ``(K,)``, sorted ascending.
    values : torch.Tensor
        Stimulus values at each onset, shape ``(K, M)`` for M inputs.
        ``values[i]`` is active for ``times[i] <= t < times[i+1]``.

    Examples
    --------
    >>> times = torch.tensor([0.0, 10.0, 20.0, 30.0])
    >>> values = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    >>> u = PiecewiseConstantInput(times, values)
    >>> u(torch.tensor(5.0))   # returns tensor([1.0])
    >>> u(torch.tensor(15.0))  # returns tensor([0.0])
    """

    def __init__(self, times: torch.Tensor, values: torch.Tensor) -> None:
        self.times = times
        self.values = values

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """Return stimulus value at time t.

        Uses ``torch.searchsorted`` for efficient lookup. For t before
        the first onset, returns ``values[0]``. For t after the last
        onset, returns ``values[-1]``.

        Parameters
        ----------
        t : torch.Tensor
            Query time (scalar tensor).

        Returns
        -------
        torch.Tensor
            Stimulus value at time t, shape ``(M,)``.
        """
        # searchsorted(right=True) - 1 gives the index of the last onset <= t
        idx = torch.searchsorted(self.times, t.detach(), right=True) - 1
        idx = torch.clamp(idx, min=0, max=self.values.shape[0] - 1)
        return self.values[idx]

    @property
    def grid_points(self) -> torch.Tensor:
        """Return onset times for solver grid_points.

        Adaptive ODE solvers (dopri5) should restart integration at
        input discontinuities to maintain accuracy. Passing these as
        ``grid_points`` to torchdiffeq ensures the solver steps exactly
        onto each onset time.

        Returns
        -------
        torch.Tensor
            Onset times tensor, shape ``(K,)``.
        """
        return self.times


def integrate_ode(
    func: torch.nn.Module,
    y0: torch.Tensor,
    t_eval: torch.Tensor,
    method: str = "dopri5",
    rtol: float = 1e-5,
    atol: float = 1e-7,
    grid_points: torch.Tensor | None = None,
    adjoint: bool = False,
    step_size: float = 0.01,
) -> torch.Tensor:
    """Integrate an ODE system using torchdiffeq.

    Wrapper around ``torchdiffeq.odeint`` (or ``odeint_adjoint`` when
    ``adjoint=True``) with configurable solver and tolerance settings.

    For adaptive methods (dopri5), ``grid_points`` specifies times where
    the solver must step exactly (e.g., stimulus onset discontinuities).
    For fixed-step methods (euler, rk4), ``step_size`` controls the
    integration step.

    Parameters
    ----------
    func : torch.nn.Module
        ODE right-hand side, callable as ``func(t, y) -> dy/dt``.
        Must be an ``nn.Module`` for ``odeint_adjoint`` compatibility.
    y0 : torch.Tensor
        Initial state vector, shape ``(D,)``.
    t_eval : torch.Tensor
        Times at which to evaluate the solution, shape ``(T,)``,
        sorted ascending.
    method : str, optional
        ODE solver: 'dopri5' (default), 'rk4', or 'euler'.
    rtol : float, optional
        Relative tolerance for adaptive solvers. Default 1e-5.
    atol : float, optional
        Absolute tolerance for adaptive solvers. Default 1e-7.
    grid_points : torch.Tensor or None, optional
        Times where solver must step exactly (for adaptive methods).
        Typically from ``PiecewiseConstantInput.grid_points``.
    adjoint : bool, optional
        If True, use adjoint method for memory-efficient gradients.
        Default False.
    step_size : float, optional
        Step size for fixed-step methods (euler, rk4). Default 0.01.

    Returns
    -------
    torch.Tensor
        Solution tensor, shape ``(T, D)`` where T=len(t_eval),
        D=state dimension.

    Notes
    -----
    torchdiffeq API reference: https://github.com/rtqichen/torchdiffeq

    The ``grid_points`` option is only passed for adaptive methods
    (dopri5). For fixed-step methods, it is ignored since the solver
    already evaluates at regular intervals.

    References
    ----------
    Chen et al. (2018). Neural Ordinary Differential Equations. NeurIPS.

    Examples
    --------
    >>> import torch
    >>> y0 = make_initial_state(2)  # (10,) for 2 regions
    >>> t_eval = torch.arange(0, 100, 0.5, dtype=torch.float64)
    >>> # solution = integrate_ode(system, y0, t_eval, method='rk4')
    """
    # Build solver options based on method type
    adaptive_methods = {"dopri5", "dopri8", "bosh3", "adaptive_heun"}
    fixed_methods = {"euler", "rk4", "midpoint", "explicit_adams", "implicit_adams"}

    options: dict | None = None
    if method in adaptive_methods:
        if grid_points is not None:
            # torchdiffeq >= 0.2.4 uses 'jump_t' for discontinuity points
            # (solver restarts at these times) instead of the older
            # 'grid_points' API.
            options = {"jump_t": grid_points}
    elif method in fixed_methods:
        options = {"step_size": step_size}

    # Select integrator
    integrator = odeint_adjoint if adjoint else odeint

    # Build keyword arguments
    kwargs: dict = {
        "rtol": rtol,
        "atol": atol,
        "method": method,
    }
    if options is not None:
        kwargs["options"] = options

    # For adjoint mode, pass the module parameters explicitly
    if adjoint:
        kwargs["adjoint_params"] = tuple(func.parameters())

    solution = integrator(func, y0, t_eval, **kwargs)

    return solution


def make_initial_state(
    n_regions: int,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Create steady-state initial conditions for the coupled ODE system.

    Returns a zero vector of shape ``(5 * n_regions,)`` representing
    the steady state where all neural and hemodynamic variables are at
    their resting values.

    State layout (per region, N regions total):

    =========  ==============  ===================================
    Index      Variable        Description
    =========  ==============  ===================================
    [0:N]      x               Neural activity (zero = no activity)
    [N:2N]     s               Vasodilatory signal (zero = resting)
    [2N:3N]    lnf             Log blood flow (zero = f=1, resting)
    [3N:4N]    lnv             Log blood volume (zero = v=1, resting)
    [4N:5N]    lnq             Log deoxyhemoglobin (zero = q=1, resting)
    =========  ==============  ===================================

    Parameters
    ----------
    n_regions : int
        Number of brain regions.
    dtype : torch.dtype, optional
        Tensor dtype. Default ``torch.float64``.
    device : str or torch.device, optional
        Tensor device. Default 'cpu'.

    Returns
    -------
    torch.Tensor
        Zero vector of shape ``(5 * n_regions,)``.

    Examples
    --------
    >>> y0 = make_initial_state(3)
    >>> y0.shape  # (15,) for 3 regions, 5 states each
    """
    return torch.zeros(5 * n_regions, dtype=dtype, device=device)


def merge_piecewise_inputs(
    drive: PiecewiseConstantInput,
    mod: PiecewiseConstantInput,
) -> PiecewiseConstantInput:
    """Combine two piecewise-constant inputs into one widened input.

    Returns a new :class:`PiecewiseConstantInput` whose values at any
    query time ``t`` equal ``torch.cat([drive(t), mod(t)])`` along the
    column axis. The merged breakpoint set is the sorted union of the
    two input breakpoint sets (deduplicated via
    :func:`torch.unique`).

    This helper lives in :mod:`pyro_dcm.utils.ode_integrator` (next to
    :class:`PiecewiseConstantInput`) so both the simulator
    (:func:`pyro_dcm.simulators.task_simulator.simulate_task_dcm`) and
    the Pyro generative model (Phase 15) can construct the widened
    ``(M_drive + J_mod)``-column input that
    :class:`pyro_dcm.forward_models.coupled_system.CoupledDCMSystem`
    expects in its bilinear path.

    Parameters
    ----------
    drive : PiecewiseConstantInput
        Driving-input stimulus, with ``times`` shape ``(K1,)`` and
        ``values`` shape ``(K1, M)``.
    mod : PiecewiseConstantInput
        Modulatory-input stimulus, with ``times`` shape ``(K2,)`` and
        ``values`` shape ``(K2, J)``.

    Returns
    -------
    PiecewiseConstantInput
        Widened input with ``times`` shape ``(K,)`` (sorted unique
        union) and ``values`` shape ``(K, M + J)``. The first ``M``
        columns are ``drive``; the remaining ``J`` columns are ``mod``.

    Raises
    ------
    ValueError
        If ``drive.values.dtype != mod.values.dtype`` or
        ``drive.values.device != mod.values.device``. Dtype/device are
        NOT silently auto-cast; callers must align them beforehand.

    Notes
    -----
    **Correctness.** At any query time ``t*``, the merged input
    returns ``[drive(t*), mod(t*)]`` concatenated along the column
    axis. Because the merged breakpoint set contains every
    discontinuity of both inputs and
    :meth:`PiecewiseConstantInput.__call__` is left-closed, the merged
    breakpoint values at ``t_k`` equal ``drive(t_k)`` concatenated with
    ``mod(t_k)``.

    **Complexity.** ``O(K log K)`` where ``K = len(merged_times)``.
    For typical DCM runs (``K1 ~ 20`` driving blocks, ``K2 ~ 40``
    modulator events), this completes in microseconds.

    Examples
    --------
    >>> import torch
    >>> t_drive = torch.tensor([0.0, 10.0, 20.0], dtype=torch.float64)
    >>> v_drive = torch.tensor([[0.0], [1.0], [0.0]], dtype=torch.float64)
    >>> drive = PiecewiseConstantInput(t_drive, v_drive)
    >>> t_mod = torch.tensor([0.0, 5.0, 15.0], dtype=torch.float64)
    >>> v_mod = torch.tensor([[0.0], [1.0], [0.0]], dtype=torch.float64)
    >>> mod = PiecewiseConstantInput(t_mod, v_mod)
    >>> merged = merge_piecewise_inputs(drive, mod)
    >>> merged.values.shape[1]  # 1 drive column + 1 mod column
    2
    """
    t_drive = drive.times
    t_mod = mod.times
    v_drive = drive.values
    v_mod = mod.values
    dtype = v_drive.dtype
    device = v_drive.device

    # --- 1. Validate dtype / device match (no silent cast) ---
    if v_mod.dtype != dtype:
        raise ValueError(
            f"drive.values.dtype={dtype} != mod.values.dtype={v_mod.dtype}; "
            f"align dtypes before merging (no silent cast)"
        )
    if v_mod.device != device:
        raise ValueError(
            f"drive.values.device={device} != mod.values.device="
            f"{v_mod.device}; align devices before merging (no silent cast)"
        )

    # --- 2. Sorted unique union of breakpoint times ---
    all_times = torch.cat([t_drive, t_mod])
    merged_times = torch.unique(all_times, sorted=True)

    # --- 3. Evaluate drive(t_k) and mod(t_k) at each breakpoint ---
    M = v_drive.shape[1]
    J = v_mod.shape[1]
    K = merged_times.shape[0]
    merged_values = torch.empty((K, M + J), dtype=dtype, device=device)
    for k in range(K):
        t_k = merged_times[k]
        merged_values[k, :M] = drive(t_k.detach())
        merged_values[k, M:] = mod(t_k.detach())

    return PiecewiseConstantInput(merged_times, merged_values)
