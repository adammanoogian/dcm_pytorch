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
