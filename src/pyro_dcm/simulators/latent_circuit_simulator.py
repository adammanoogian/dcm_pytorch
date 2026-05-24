"""End-to-end latent circuit data simulator.

Generates synthetic N-dimensional neural state trajectories from
specified bilinear connectivity parameters. This is the primary tool
for latent circuit DCM synthetic validation: parameter recovery
(Phase 20 SYNTH-01..03) and inference pipeline development.

The simulator wraps the coupled ODE system (``CoupledDCMSystem`` with
``hemodynamic=False``) and ODE integration (``integrate_ode``) into a
single function that produces ground-truth trajectories with
controllable SNR.

Unlike the task-DCM simulator (``simulate_task_dcm``), this simulator:
- Does NOT compute hemodynamic states or BOLD signal.
- Operates on an N-dimensional state vector (not 5N).
- Uses ``torch.zeros(N)`` as initial state (not ``make_initial_state``).
- Returns trajectories at the fine integration grid (no TR
  downsampling).

References
----------
[REF-001] Friston, Harrison & Penny (2003), Eq. 1 -- Neural state equation.
"""

from __future__ import annotations

from typing import Any

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


def simulate_latent_circuit(
    A: torch.Tensor,
    C: torch.Tensor,
    stimulus: dict[str, torch.Tensor] | PiecewiseConstantInput,
    duration: float = 100.0,
    dt: float = 0.01,
    SNR: float = 10.0,
    solver: str = "dopri5",
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    seed: int | None = None,
    *,
    B_list: torch.Tensor | list[torch.Tensor] | None = None,
    stimulus_mod: (
        dict[str, torch.Tensor] | PiecewiseConstantInput | None
    ) = None,
    n_driving_inputs: int | None = None,
) -> dict[str, Any]:
    """Generate synthetic neural state trajectories from a latent circuit.

    Runs the neural state equation [REF-001] Eq. 1 in direct-observation
    mode (``CoupledDCMSystem(hemodynamic=False)``), producing N-dimensional
    trajectories suitable for latent circuit DCM parameter recovery.

    Parameters
    ----------
    A : torch.Tensor
        Effective connectivity matrix, shape ``(N, N)``. Must have
        negative diagonal for stability (self-inhibition).
    C : torch.Tensor
        Driving input weights, shape ``(N, M_drive)``.
    stimulus : dict or PiecewiseConstantInput
        Driving stimulus. If dict, must have keys ``'times'``
        (shape ``(K,)``) and ``'values'`` (shape ``(K, M_drive)``).
    duration : float, optional
        Simulation duration in seconds. Default 100.0.
    dt : float, optional
        Step size for fixed-step solvers and time grid. Default 0.01.
    SNR : float, optional
        Signal-to-noise ratio, defined as ``std(signal) / std(noise)``
        averaged across regions. If SNR <= 0, no noise is added.
        Default 10.0.
    solver : str, optional
        ODE solver: ``'dopri5'`` (default), ``'rk4'``, or ``'euler'``.
    device : str, optional
        Torch device. Default ``'cpu'``.
    dtype : torch.dtype, optional
        Tensor dtype. Default ``torch.float64``.
    seed : int or None, optional
        Random seed for reproducibility. If None, no seed is set.
    B_list : torch.Tensor, list of torch.Tensor, or None, optional
        Modulatory input weights for bilinear mode. Shape ``(J, N, N)``
        or list of J ``(N, N)`` tensors. None for linear mode.
        Keyword-only.
    stimulus_mod : dict or PiecewiseConstantInput or None, optional
        Modulatory input trajectory. Required when ``B_list`` is
        non-None. Keyword-only.
    n_driving_inputs : int or None, optional
        Number of driving input columns. Defaults to ``C.shape[1]``
        when ``B_list`` is supplied. Keyword-only.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:

        - ``'trajectories'``: Noisy observed trajectories, shape
          ``(T, N)``.
        - ``'trajectories_clean'``: Clean trajectories (no noise),
          shape ``(T, N)``.
        - ``'times'``: Time vector, shape ``(T,)``.
        - ``'A'``: Ground-truth A matrix, shape ``(N, N)``.
        - ``'C'``: Ground-truth C matrix, shape ``(N, M_drive)``.
        - ``'B_list'``: Ground-truth stacked B tensor ``(J, N, N)``
          or None.
        - ``'stimulus_mod'``: Ground-truth modulatory
          ``PiecewiseConstantInput`` or None.
        - ``'noise_std'``: Scalar noise standard deviation used.
        - ``'SNR'``: Achieved SNR (per-region std ratio, averaged).

    Raises
    ------
    ValueError
        If ``B_list`` is non-None and ``stimulus_mod`` is None.
    ValueError
        If ``n_driving_inputs`` is explicit and mismatches ``C.shape[1]``.

    Notes
    -----
    Implements [REF-001] Eq. 1 in direct-observation mode:

    - Linear: ``dx/dt = A @ x + C @ u(t)``
    - Bilinear: ``dx/dt = (A + sum_j u_mod[j] * B[j]) @ x + C @ u_drive``

    Initial state is ``torch.zeros(N)`` (no hemodynamic steady-state
    assumption). Noise is added post-integration, scaled per-region to
    achieve the specified SNR.

    Examples
    --------
    >>> import torch
    >>> A = torch.tensor([[-0.5, 0.1], [0.2, -0.5]], dtype=torch.float64)
    >>> C = torch.tensor([[1.0], [0.0]], dtype=torch.float64)
    >>> stim = {"times": torch.tensor([0., 10.]), "values": torch.tensor([[1.], [0.]])}
    >>> result = simulate_latent_circuit(A, C, stim, duration=20.0)
    >>> result["trajectories"].shape  # (2000, 2)
    """
    # 1. Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)

    N = A.shape[0]

    # 2. Normalize the driving stimulus.
    driving_input_fn = _normalize_stimulus_to_input_fn(
        stimulus, device, dtype
    )

    # 3. Cast A, C to device/dtype and normalize B_list.
    A_dev = A.to(device=device, dtype=dtype)
    C_dev = C.to(device=device, dtype=dtype)
    B_stacked = _normalize_B_list(B_list, device, dtype)

    # 4. Branch: linear vs bilinear path.
    if B_stacked is None:
        # LINEAR MODE.
        input_fn = driving_input_fn
        stimulus_mod_input_fn: PiecewiseConstantInput | None = None
        system = CoupledDCMSystem(
            A_dev, C_dev, input_fn, hemodynamic=False
        )
    else:
        # BILINEAR MODE.
        if stimulus_mod is None:
            raise ValueError(
                "stimulus_mod is required when B_list is non-None; "
                "got None. Construct a modulator trajectory with "
                "make_epoch_stimulus or make_event_stimulus."
            )
        stimulus_mod_input_fn = _normalize_stimulus_to_input_fn(
            stimulus_mod, device, dtype
        )
        J = B_stacked.shape[0]
        if stimulus_mod_input_fn.values.shape[1] != J:
            raise ValueError(
                f"stimulus_mod has "
                f"{stimulus_mod_input_fn.values.shape[1]} columns but "
                f"B_list has J={J} modulators; they must match."
            )
        # Default n_driving_inputs to C.shape[1].
        n_driv = (
            n_driving_inputs
            if n_driving_inputs is not None
            else C.shape[1]
        )
        if n_driv != C.shape[1]:
            raise ValueError(
                f"n_driving_inputs={n_driv} inconsistent with "
                f"C.shape[1]={C.shape[1]}"
            )
        # Merge driving + modulator into widened PiecewiseConstantInput.
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

    # 5. Initial state: zeros for neural activity only (NOT 5N).
    y0 = torch.zeros(N, dtype=dtype, device=device)

    # 6. Create fine time grid.
    t_eval = torch.arange(0, duration, dt, dtype=dtype, device=device)

    # 7. Integrate ODE.
    grid_points = input_fn.grid_points
    solution = integrate_ode(
        system,
        y0,
        t_eval,
        method=solver,
        grid_points=grid_points,
        step_size=dt,
    )
    # solution shape: (T, N)

    # 8. Add Gaussian noise scaled to SNR.
    if SNR > 0:
        signal_std = solution.std(dim=0)  # shape (N,)
        # Avoid division by zero for quiescent regions.
        mean_signal_std = signal_std.mean()
        noise_std_scalar = (
            mean_signal_std / SNR if mean_signal_std > 0 else 0.0
        )
        if isinstance(noise_std_scalar, torch.Tensor):
            noise_std_scalar = noise_std_scalar.item()
        noise = noise_std_scalar * torch.randn_like(solution)
        noisy_trajectories = solution + noise
    else:
        noise_std_scalar = 0.0
        noisy_trajectories = solution.clone()

    # 9. Compute achieved SNR.
    if noise_std_scalar > 0:
        achieved_snr = float(solution.std().item() / noise_std_scalar)
    else:
        achieved_snr = float("inf")

    return {
        "trajectories": noisy_trajectories,
        "trajectories_clean": solution,
        "times": t_eval,
        "A": A_dev,
        "C": C_dev,
        "B_list": B_stacked,
        "stimulus_mod": stimulus_mod_input_fn,
        "noise_std": noise_std_scalar,
        "SNR": achieved_snr,
    }


def make_stable_latent_circuit_A(
    n_regions: int,
    density: float = 0.5,
    seed: int | None = None,
    *,
    self_inhibition: float = 0.5,
    strength_range: tuple[float, float] = (0.05, 0.3),
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Generate a random stable A matrix for latent circuit simulation.

    Creates a sparse effective connectivity matrix with guaranteed
    negative diagonal (self-inhibition) and random off-diagonal
    connections. Stability is verified via eigenvalue check; if the
    generated matrix is unstable, self-inhibition is increased until
    all eigenvalues have negative real parts.

    Parameters
    ----------
    n_regions : int
        Number of latent circuit nodes (brain regions or RNN units).
    density : float, optional
        Fraction of off-diagonal connections that are non-zero.
        Default 0.5.
    seed : int or None, optional
        Random seed for reproducibility.
    self_inhibition : float, optional
        Initial self-inhibition strength (Hz) for diagonal entries.
        Diagonal is set to ``-self_inhibition``. Default 0.5.
    strength_range : tuple of float, optional
        (min, max) absolute value range for off-diagonal connection
        strengths in Hz. Default ``(0.05, 0.3)``.
    dtype : torch.dtype, optional
        Tensor dtype. Default ``torch.float64``.

    Returns
    -------
    torch.Tensor
        Stable A matrix, shape ``(n_regions, n_regions)``, with all
        eigenvalues having strictly negative real parts.

    Notes
    -----
    If the initial matrix is unstable (due to strong off-diagonal
    connections exceeding self-inhibition), the diagonal is
    progressively strengthened by 0.1 Hz increments until stability
    is achieved. This ensures the returned matrix always produces
    bounded ODE trajectories.

    Examples
    --------
    >>> A = make_stable_latent_circuit_A(4, density=0.5, seed=42)
    >>> A.shape
    torch.Size([4, 4])
    >>> torch.linalg.eigvals(A).real.max() < 0
    tensor(True)
    """
    if seed is not None:
        torch.manual_seed(seed)

    N = n_regions
    A = torch.zeros(N, N, dtype=dtype)

    # Set diagonal: self-inhibition.
    A.diagonal().fill_(-self_inhibition)

    # Generate sparse off-diagonal connections.
    n_off_diag = N * (N - 1)
    n_connections = round(density * n_off_diag)

    if n_connections > 0:
        off_diag_indices = []
        for i in range(N):
            for j in range(N):
                if i != j:
                    off_diag_indices.append((i, j))

        perm = torch.randperm(len(off_diag_indices))[:n_connections]
        lo, hi = strength_range
        strengths = lo + (hi - lo) * torch.rand(
            n_connections, dtype=dtype
        )
        signs = (
            2.0
            * torch.bernoulli(
                0.5 * torch.ones(n_connections, dtype=dtype)
            )
            - 1.0
        )

        mask = torch.zeros(N, N, dtype=torch.bool)
        for idx in perm:
            i, j = off_diag_indices[idx]
            mask[i, j] = True

        A[mask] = strengths * signs

    # Ensure stability: all eigenvalues must have negative real parts.
    max_attempts = 20
    for _ in range(max_attempts):
        eigs = torch.linalg.eigvals(A)
        if eigs.real.max().item() < 0:
            break
        # Increase self-inhibition by 0.1 Hz.
        A.diagonal().add_(-0.1)

    return A
