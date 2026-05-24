"""Latent-circuit DCM simulator (v0.6.0).

Generates synthetic latent-state trajectories from a bilinear DCM
neural state equation WITHOUT hemodynamic coupling. Used for Phase 20
latent-circuit forward-model validation: the RNN latent state x(t) is
modeled as evolving under [REF-001] Eq. 1 directly.

The simulator wraps ``CoupledDCMSystem(hemodynamic=False)`` with
``torchdiffeq`` integration. The returned ``'trajectories'`` key
contains the full (T, N) latent-state trajectory at the requested
temporal resolution.

References
----------
[REF-001] Friston, Harrison & Penny (2003), Eq. 1 -- Neural state equation:
    dx/dt = (A + sum_j u_j(t) * B_j) @ x + C @ u(t).
"""

from __future__ import annotations

import torch

from pyro_dcm.forward_models.coupled_system import CoupledDCMSystem
from pyro_dcm.simulators.task_simulator import (
    _normalize_B_list,
    _normalize_stimulus_to_input_fn,
)
from pyro_dcm.utils.ode_integrator import (
    PiecewiseConstantInput,
    integrate_ode,
    merge_piecewise_inputs,
)


def make_stable_latent_circuit_A(
    n_regions: int,
    density: float = 0.5,
    off_diag_scale: float = 0.2,
    self_inhibition: float = 1.0,
    seed: int | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Generate a random stable effective connectivity matrix for latent-circuit DCM.

    Creates a sparse connectivity matrix with negative diagonal
    (self-inhibition) and random off-diagonal connections, calibrated
    for RNN latent-state timescales. The self-inhibition is larger than
    the SPM12 BOLD default (0.5 Hz) to match the faster dynamics typical
    of RNN hidden states.

    Stability is guaranteed when ``self_inhibition > off_diag_scale *
    sqrt(density * n_regions * (n_regions - 1))``.

    Parameters
    ----------
    n_regions : int
        Number of latent dimensions (N).
    density : float, optional
        Fraction of off-diagonal connections that are non-zero.
        Default 0.5.
    off_diag_scale : float, optional
        Maximum absolute value of off-diagonal connections.
        Default 0.2.
    self_inhibition : float, optional
        Self-inhibition magnitude for diagonal entries (Hz). Diagonal
        entries are set to ``-self_inhibition``. Default 1.0.
    seed : int or None, optional
        Random seed for reproducibility.
    dtype : torch.dtype, optional
        Tensor dtype. Default ``torch.float64``.

    Returns
    -------
    torch.Tensor
        Effective connectivity matrix, shape ``(N, N)``, with negative
        diagonal. All eigenvalues have strictly negative real parts when
        ``self_inhibition`` is sufficiently large relative to
        ``off_diag_scale``.

    Notes
    -----
    Implements [REF-001] Eq. 1 parameter regime. The matrix is the
    parameterized A (not free parameters), ready for direct use in
    ``simulate_latent_circuit``.

    Examples
    --------
    >>> A = make_stable_latent_circuit_A(4, seed=42)
    >>> A.shape
    torch.Size([4, 4])
    >>> torch.linalg.eigvals(A).real.max() < 0  # stable
    tensor(True)
    """
    if seed is not None:
        torch.manual_seed(seed)

    N = n_regions
    A = torch.zeros(N, N, dtype=dtype)

    # Negative self-inhibition on diagonal (guarantees stability as long
    # as off-diagonal perturbations are small enough).
    A.diagonal().fill_(-self_inhibition)

    # Generate sparse off-diagonal connections.
    n_off = N * (N - 1)
    n_connections = round(density * n_off)

    if n_connections > 0:
        off_diag_indices = [(i, j) for i in range(N) for j in range(N) if i != j]
        perm = torch.randperm(len(off_diag_indices))[:n_connections]
        strengths = off_diag_scale * torch.rand(n_connections, dtype=dtype)
        signs = 2.0 * torch.bernoulli(
            0.5 * torch.ones(n_connections, dtype=dtype)
        ) - 1.0
        for k, idx in enumerate(perm):
            i, j = off_diag_indices[idx]
            A[i, j] = strengths[k] * signs[k]

    return A


def simulate_latent_circuit(
    A: torch.Tensor,
    C: torch.Tensor,
    stimulus: dict[str, torch.Tensor] | PiecewiseConstantInput,
    duration: float = 2.0,
    dt: float = 0.01,
    SNR: float = 5.0,
    solver: str = "dopri5",
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    seed: int | None = None,
    *,
    B_list: torch.Tensor | list[torch.Tensor] | None = None,
    stimulus_mod: dict[str, torch.Tensor] | PiecewiseConstantInput | None = None,
    n_driving_inputs: int | None = None,
) -> dict:
    """Generate synthetic latent-state trajectories from a bilinear DCM.

    Integrates the neural state equation [REF-001] Eq. 1 directly,
    WITHOUT hemodynamic coupling. This is the core forward model for
    v0.6.0 latent-circuit DCM: the (T, N) trajectory represents the
    latent-state dynamics of a trained RNN modeled under a bilinear DCM.

    The simulator uses ``CoupledDCMSystem(hemodynamic=False)`` so the
    ODE state vector is shape ``(N,)`` throughout integration, returning
    a ``(T, N)`` solution matrix.

    Parameters
    ----------
    A : torch.Tensor
        Effective connectivity matrix, shape ``(N, N)``. Must have
        negative diagonal for stability.
    C : torch.Tensor
        Driving input weights, shape ``(N, M)``.
    stimulus : dict or PiecewiseConstantInput
        Experimental stimulus. If dict, must have keys ``'times'``
        (shape ``(K,)``) and ``'values'`` (shape ``(K, M)``). If already
        a ``PiecewiseConstantInput`` instance, used directly.
    duration : float, optional
        Simulation duration in seconds. Default 2.0. Latent-circuit
        trajectories are typically much shorter than BOLD (seconds not
        minutes).
    dt : float, optional
        Step size hint for fixed-step solvers and time grid. Default 0.01.
    SNR : float, optional
        Signal-to-noise ratio, ``std(signal) / std(noise)``. If <= 0,
        no noise is added. Default 5.0.
    solver : str, optional
        ODE solver: ``'dopri5'`` (default), ``'rk4'``, or ``'euler'``.
    device : str, optional
        Torch device. Default ``'cpu'``.
    dtype : torch.dtype, optional
        Tensor dtype. Default ``torch.float64``.
    seed : int or None, optional
        Random seed for reproducibility.
    B_list : torch.Tensor, list of torch.Tensor, or None, optional
        Modulatory input weights for the bilinear term
        ``sum_j u_j(t) * B_j * x``. Accepts a Python list of ``(N, N)``
        tensors or a pre-stacked ``(J, N, N)`` tensor. Empty list or
        shape ``(0, N, N)`` collapses to ``None`` (linear mode).
        Requires ``stimulus_mod`` when non-None. Keyword-only.
    stimulus_mod : dict or PiecewiseConstantInput or None, optional
        Modulatory input trajectory ``u_mod(t)``. Required when
        ``B_list`` is non-None. Keyword-only.
    n_driving_inputs : int or None, optional
        Number of driving input columns, matching ``C.shape[1]``. When
        ``None`` and ``B_list`` is supplied, defaults to ``C.shape[1]``.
        Keyword-only.

    Returns
    -------
    dict
        Dictionary with the following keys:

        - ``'trajectories'``: Noisy latent-state trajectories,
          shape ``(T, N)``. No hemodynamic or BOLD keys are present.
        - ``'trajectories_clean'``: Noise-free trajectories, shape
          ``(T, N)``.
        - ``'times'``: Evaluation time points, shape ``(T,)``.
        - ``'A'``: Effective connectivity, shape ``(N, N)``.
        - ``'C'``: Driving input weights, shape ``(N, M)``.
        - ``'B_list'``: Stacked ``(J, N, N)`` B tensor or ``None``.
        - ``'stimulus'``: Driving ``PiecewiseConstantInput`` used.
        - ``'stimulus_mod'``: Modulator ``PiecewiseConstantInput``
          or ``None`` (linear mode).
        - ``'params'``: dict with ``A``, ``C``, ``SNR``, ``duration``,
          ``solver``, ``dt``.
        - ``'simulation_diverged'``: ``bool``. ``True`` when the ODE
          overflowed (NaN/Inf in trajectories_clean).

    Notes
    -----
    Implements [REF-001] Eq. 1:
        Linear mode: dx/dt = A @ x + C @ u(t)
        Bilinear mode: dx/dt = (A + sum_j u_j(t) * B_j) @ x + C @ u(t)

    The initial state is ``torch.zeros(N)`` (zero neural activity at
    rest), consistent with the assumption that the RNN latent state
    has been mean-subtracted before fitting.

    Examples
    --------
    >>> A = make_stable_latent_circuit_A(3, seed=0)
    >>> C = torch.zeros(3, 1, dtype=torch.float64)
    >>> C[0, 0] = 0.5
    >>> from pyro_dcm.simulators.task_simulator import make_block_stimulus
    >>> stim = make_block_stimulus(n_blocks=2, block_duration=0.5,
    ...                            rest_duration=0.5)
    >>> result = simulate_latent_circuit(A, C, stim, duration=2.0, SNR=10.0)
    >>> result['trajectories'].shape  # (200, 3) at dt=0.01
    """
    if seed is not None:
        torch.manual_seed(seed)

    N = A.shape[0]

    # 1. Normalize driving stimulus.
    driving_input_fn = _normalize_stimulus_to_input_fn(stimulus, device, dtype)

    # 2. Cast A, C to device/dtype and normalize B_list.
    A_dev = A.to(device=device, dtype=dtype)
    C_dev = C.to(device=device, dtype=dtype)
    B_stacked = _normalize_B_list(B_list, device, dtype)

    # 3. Branch: linear short-circuit vs bilinear path.
    if B_stacked is None:
        input_fn = driving_input_fn
        stimulus_mod_input_fn: PiecewiseConstantInput | None = None
        system = CoupledDCMSystem(
            A_dev, C_dev, input_fn, hemodynamic=False
        )
    else:
        if stimulus_mod is None:
            raise ValueError(
                "stimulus_mod is required when B_list is non-None; got None."
            )
        stimulus_mod_input_fn = _normalize_stimulus_to_input_fn(
            stimulus_mod, device, dtype
        )
        J = B_stacked.shape[0]
        if stimulus_mod_input_fn.values.shape[1] != J:
            raise ValueError(
                f"stimulus_mod has {stimulus_mod_input_fn.values.shape[1]} "
                f"columns but B_list has J={J} modulators; they must match."
            )
        n_driv = (
            n_driving_inputs if n_driving_inputs is not None else C.shape[1]
        )
        if n_driv != C.shape[1]:
            raise ValueError(
                f"n_driving_inputs={n_driv} inconsistent with "
                f"C.shape[1]={C.shape[1]}"
            )
        input_fn = merge_piecewise_inputs(
            driving_input_fn, stimulus_mod_input_fn
        )
        system = CoupledDCMSystem(
            A_dev,
            C_dev,
            input_fn,
            hemodynamic=False,
            B=B_stacked,
            n_driving_inputs=n_driv,
        )

    # 4. Initial state: zeros of shape (N,) -- neural activity at rest.
    y0 = torch.zeros(N, dtype=dtype, device=device)

    # 5. Evaluation time grid.
    t_eval = torch.arange(0, duration, dt, dtype=dtype, device=device)

    # 6. Integrate ODE.
    grid_points = input_fn.grid_points
    solution = integrate_ode(
        system,
        y0,
        t_eval,
        method=solver,
        grid_points=grid_points,
        step_size=dt,
    )
    # solution shape: (T, N) -- latent state only, no hemodynamics.

    # 7. Divergence diagnostic.
    simulation_diverged = bool(
        torch.isnan(solution).any().item()
        or torch.isinf(solution).any().item()
    )

    # 8. Add Gaussian noise.
    if SNR > 0:
        signal_std = solution.std(dim=0)  # (N,)
        noise_std = signal_std / SNR       # (N,)
        noise = noise_std.unsqueeze(0) * torch.randn_like(solution)
        noisy_trajectories = solution + noise
    else:
        noisy_trajectories = solution.clone()

    return {
        "trajectories": noisy_trajectories,          # (T, N) noisy
        "trajectories_clean": solution,               # (T, N) noise-free
        "times": t_eval,                              # (T,)
        "A": A_dev,
        "C": C_dev,
        "B_list": B_stacked,                          # (J, N, N) or None
        "stimulus": driving_input_fn,
        "stimulus_mod": stimulus_mod_input_fn,
        "params": {
            "A": A_dev,
            "C": C_dev,
            "SNR": SNR,
            "duration": duration,
            "solver": solver,
            "dt": dt,
        },
        "simulation_diverged": simulation_diverged,
    }
