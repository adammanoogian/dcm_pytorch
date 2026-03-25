"""End-to-end task-DCM data simulator.

Generates realistic synthetic BOLD time series from specified connectivity
and hemodynamic parameters. This is the primary tool for all downstream
testing: parameter recovery (Phase 5), SPM validation (Phase 6), and
amortized guide training (Phase 7).

The simulator wraps the coupled ODE system (``CoupledDCMSystem``), ODE
integration (``integrate_ode``), and BOLD observation equation
(``bold_signal``) into a single function that produces complete synthetic
fMRI data with controllable SNR and temporal resolution.

References
----------
[REF-001] Friston, Harrison & Penny (2003), Eq. 1 -- Neural state equation.
[REF-002] Stephan et al. (2007), Eq. 2-6 -- Balloon-Windkessel + BOLD.
SPM12 source: spm_fx_fmri.m, spm_gx_fmri.m.
"""

from __future__ import annotations

import torch

from pyro_dcm.forward_models.balloon_model import DEFAULT_HEMO_PARAMS
from pyro_dcm.forward_models.bold_signal import bold_signal
from pyro_dcm.forward_models.coupled_system import CoupledDCMSystem
from pyro_dcm.utils.ode_integrator import (
    PiecewiseConstantInput,
    integrate_ode,
    make_initial_state,
)


def simulate_task_dcm(
    A: torch.Tensor,
    C: torch.Tensor,
    stimulus: dict[str, torch.Tensor] | PiecewiseConstantInput,
    hemo_params: dict[str, float] | None = None,
    duration: float = 300.0,
    dt: float = 0.01,
    TR: float = 2.0,
    SNR: float = 5.0,
    solver: str = "dopri5",
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    seed: int | None = None,
) -> dict:
    """Generate synthetic BOLD time series from a task-DCM model.

    Runs the full forward model pipeline: neural state equation,
    Balloon-Windkessel hemodynamics, BOLD observation, downsampling to
    fMRI temporal resolution, and addition of Gaussian measurement noise.

    Parameters
    ----------
    A : torch.Tensor
        Effective connectivity matrix, shape ``(N, N)``. This should be
        the **parameterized** A matrix (with negative diagonal for
        self-inhibition), NOT the free parameters ``A_free``. Use
        ``parameterize_A(A_free)`` to convert from free parameters if
        needed.
    C : torch.Tensor
        Driving input weights, shape ``(N, M)``.
    stimulus : dict or PiecewiseConstantInput
        Experimental stimulus. If dict, must have keys ``'times'``
        (shape ``(K,)``) and ``'values'`` (shape ``(K, M)``) for
        constructing a ``PiecewiseConstantInput``. If already a
        ``PiecewiseConstantInput`` instance, used directly.
    hemo_params : dict or None, optional
        Hemodynamic parameters ``{kappa, gamma, tau, alpha, E0}``.
        If None, uses SPM12 defaults from
        ``DEFAULT_HEMO_PARAMS`` in ``balloon_model.py``.
    duration : float, optional
        Simulation duration in seconds. Default 300.0.
    dt : float, optional
        Step size hint for fixed-step solvers and fine time grid.
        Default 0.01.
    TR : float, optional
        Repetition time for BOLD downsampling in seconds. Default 2.0.
    SNR : float, optional
        Signal-to-noise ratio, defined as ``std(signal) / std(noise)``.
        If SNR <= 0, no noise is added. Default 5.0.
    solver : str, optional
        ODE solver: ``'dopri5'`` (default), ``'rk4'``, or ``'euler'``.
    device : str, optional
        Torch device for tensor creation. Default ``'cpu'``.
    dtype : torch.dtype, optional
        Tensor dtype. Default ``torch.float64``.
    seed : int or None, optional
        Random seed for reproducibility. If None, no seed is set.

    Returns
    -------
    dict
        Dictionary with the following keys:

        - ``'bold'``: Noisy BOLD time series, shape ``(T_TR, N)``.
          This is the simulated "observed" data.
        - ``'bold_clean'``: Noise-free BOLD downsampled to TR,
          shape ``(T_TR, N)``.
        - ``'bold_fine'``: Fine-grained clean BOLD before downsampling,
          shape ``(T_fine, N)``.
        - ``'neural'``: Neural activity, shape ``(T_fine, N)``.
        - ``'hemodynamic'``: dict with ``'s'``, ``'f'``, ``'v'``,
          ``'q'`` each shape ``(T_fine, N)`` in linear space.
        - ``'times_fine'``: Fine time grid, shape ``(T_fine,)``.
        - ``'times_TR'``: Downsampled time points, shape ``(T_TR,)``.
        - ``'params'``: dict with ``A``, ``C``, ``hemo_params``,
          ``SNR``, ``TR``, ``duration``, ``solver``.
        - ``'stimulus'``: The ``PiecewiseConstantInput`` used.

    Notes
    -----
    The simulator applies the full pipeline from [REF-001] Eq. 1 and
    [REF-002] Eq. 2-6:

    1. Neural dynamics: dx/dt = Ax + Cu(t)
    2. Hemodynamic ODE: ds, dlnf, dlnv, dlnq per region
    3. BOLD observation: y = V0 * (k1*(1-q) + k2*(1-q/v) + k3*(1-v))
    4. Downsample to TR resolution
    5. Add Gaussian noise scaled to requested SNR

    The A matrix must already have negative self-connections. To generate
    a random stable A matrix, use ``make_random_stable_A``.

    Examples
    --------
    >>> A = torch.tensor([[-0.5, 0.1], [0.2, -0.5]], dtype=torch.float64)
    >>> C = torch.tensor([[0.25], [0.0]], dtype=torch.float64)
    >>> stim = make_block_stimulus(n_blocks=5, block_duration=30,
    ...                            rest_duration=20)
    >>> result = simulate_task_dcm(A, C, stim, duration=250.0, SNR=10.0)
    >>> result['bold'].shape  # (125, 2) at TR=2.0
    """
    # 1. Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)

    N = A.shape[0]  # number of regions

    # 2. Create PiecewiseConstantInput from stimulus dict if needed
    if isinstance(stimulus, PiecewiseConstantInput):
        input_fn = stimulus
    else:
        times = stimulus["times"].to(device=device, dtype=dtype)
        values = stimulus["values"].to(device=device, dtype=dtype)
        input_fn = PiecewiseConstantInput(times, values)

    # 3. Resolve hemodynamic parameters
    if hemo_params is None:
        hemo_params = dict(DEFAULT_HEMO_PARAMS)

    E0 = hemo_params.get("E0", DEFAULT_HEMO_PARAMS["E0"])

    # 4. Create coupled ODE system
    A_dev = A.to(device=device, dtype=dtype)
    C_dev = C.to(device=device, dtype=dtype)
    system = CoupledDCMSystem(A_dev, C_dev, input_fn, hemo_params)

    # 5. Create initial state (steady state: all zeros in log space)
    y0 = make_initial_state(N, dtype=dtype, device=device)

    # 6. Create fine-grained evaluation time grid
    t_eval = torch.arange(0, duration, dt, dtype=dtype, device=device)

    # 7. Integrate ODE
    grid_points = input_fn.grid_points
    solution = integrate_ode(
        system,
        y0,
        t_eval,
        method=solver,
        grid_points=grid_points,
        step_size=dt,
    )
    # solution shape: (T_fine, 5*N)

    # 8. Extract states from solution
    x = solution[:, :N]                    # neural activity
    s = solution[:, N:2 * N]               # vasodilatory signal
    lnf = solution[:, 2 * N:3 * N]        # log blood flow
    lnv = solution[:, 3 * N:4 * N]        # log blood volume
    lnq = solution[:, 4 * N:5 * N]        # log deoxyhemoglobin

    # Convert to linear space for hemodynamic outputs
    f = torch.exp(lnf)
    v = torch.exp(lnv)
    q = torch.exp(lnq)

    # 9. Compute clean BOLD signal [REF-002] Eq. 6
    V0 = 0.02  # resting venous blood volume fraction (SPM12 default)
    clean_bold = bold_signal(v, q, E0=E0, V0=V0)  # shape (T_fine, N)

    # 10. Downsample to TR resolution
    step = round(TR / dt)
    indices = torch.arange(0, len(t_eval), step, device=device)
    bold_clean_ds = clean_bold[indices]    # shape (T_TR, N)
    times_TR = t_eval[indices]              # shape (T_TR,)

    # 11. Add Gaussian noise
    if SNR > 0:
        # Per-region noise: noise_std = std(signal) / SNR
        signal_std = bold_clean_ds.std(dim=0)  # shape (N,)
        noise_std = signal_std / SNR            # shape (N,)
        noise = noise_std.unsqueeze(0) * torch.randn_like(bold_clean_ds)
        noisy_bold = bold_clean_ds + noise
    else:
        noisy_bold = bold_clean_ds.clone()

    # 12. Return comprehensive output dictionary
    return {
        "bold": noisy_bold,
        "bold_clean": bold_clean_ds,
        "bold_fine": clean_bold,
        "neural": x,
        "hemodynamic": {
            "s": s,
            "f": f,
            "v": v,
            "q": q,
        },
        "times_fine": t_eval,
        "times_TR": times_TR,
        "params": {
            "A": A_dev,
            "C": C_dev,
            "hemo_params": hemo_params,
            "SNR": SNR,
            "TR": TR,
            "duration": duration,
            "solver": solver,
        },
        "stimulus": input_fn,
    }


def make_block_stimulus(
    n_blocks: int,
    block_duration: float,
    rest_duration: float,
    n_inputs: int = 1,
    dtype: torch.dtype = torch.float64,
) -> dict[str, torch.Tensor]:
    """Create a standard block-design stimulus for task-fMRI simulation.

    Generates alternating ON/OFF epochs: ``block_duration`` seconds of
    stimulus followed by ``rest_duration`` seconds of rest, repeated
    ``n_blocks`` times. The first epoch is always ON.

    Parameters
    ----------
    n_blocks : int
        Number of stimulus blocks (ON epochs).
    block_duration : float
        Duration of each ON block in seconds.
    rest_duration : float
        Duration of each OFF rest period in seconds.
    n_inputs : int, optional
        Number of stimulus inputs. Default 1. For multiple inputs,
        only input 0 is active during ON blocks.
    dtype : torch.dtype, optional
        Tensor dtype. Default ``torch.float64``.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``'times'``: Onset times, shape ``(2 * n_blocks,)``.
        - ``'values'``: Stimulus values, shape ``(2 * n_blocks, n_inputs)``.

    Examples
    --------
    >>> stim = make_block_stimulus(n_blocks=10, block_duration=30,
    ...                            rest_duration=20)
    >>> stim['times'].shape  # (20,)
    >>> stim['values'].shape  # (20, 1)
    >>> # Total duration: 10 * (30 + 20) = 500 seconds
    """
    times_list = []
    values_list = []

    for i in range(n_blocks):
        # ON epoch
        onset = i * (block_duration + rest_duration)
        times_list.append(onset)
        on_val = torch.zeros(n_inputs, dtype=dtype)
        on_val[0] = 1.0
        values_list.append(on_val)

        # OFF epoch
        offset = onset + block_duration
        times_list.append(offset)
        values_list.append(torch.zeros(n_inputs, dtype=dtype))

    times = torch.tensor(times_list, dtype=dtype)
    values = torch.stack(values_list)  # shape (2*n_blocks, n_inputs)

    return {"times": times, "values": values}


def make_random_stable_A(
    n_regions: int,
    density: float = 0.5,
    strength_range: tuple[float, float] = (0.0, 0.3),
    self_inhibition: float = 0.5,
    seed: int | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Generate a random stable effective connectivity matrix A.

    Creates a sparse connectivity matrix with guaranteed negative
    diagonal (self-inhibition) and random off-diagonal connections.

    Parameters
    ----------
    n_regions : int
        Number of brain regions.
    density : float, optional
        Fraction of off-diagonal connections that are non-zero.
        Default 0.5.
    strength_range : tuple of float, optional
        (min, max) absolute value range for off-diagonal connection
        strengths in Hz. Default ``(0.0, 0.3)``.
    self_inhibition : float, optional
        Self-inhibition strength for diagonal (Hz). Diagonal entries
        are set to ``-self_inhibition``. Default 0.5, matching SPM12
        default for ``A_free = 0`` (which gives ``-exp(0)/2 = -0.5``).
    seed : int or None, optional
        Random seed for reproducibility.
    dtype : torch.dtype, optional
        Tensor dtype. Default ``torch.float64``.

    Returns
    -------
    torch.Tensor
        Parameterized A matrix, shape ``(n_regions, n_regions)``, with
        negative diagonal. Ready for direct use in ``simulate_task_dcm``
        (NOT ``A_free``; this is already the parameterized form).

    Notes
    -----
    Off-diagonal entries can be positive (excitatory) or negative
    (inhibitory). The sign is randomly assigned with equal probability.
    The matrix is guaranteed to be stable if ``self_inhibition`` is
    sufficiently large relative to off-diagonal strengths (all
    eigenvalues will have negative real parts).

    Examples
    --------
    >>> A = make_random_stable_A(3, density=0.5, seed=42)
    >>> A.shape  # (3, 3)
    >>> torch.diagonal(A)  # all -0.5
    >>> torch.linalg.eigvals(A).real.max() < 0  # stable
    """
    if seed is not None:
        torch.manual_seed(seed)

    N = n_regions
    A = torch.zeros(N, N, dtype=dtype)

    # Set diagonal: self-inhibition
    A.diagonal().fill_(-self_inhibition)

    # Generate sparse off-diagonal connections
    n_off_diag = N * (N - 1)
    n_connections = round(density * n_off_diag)

    if n_connections > 0:
        # Create mask for off-diagonal elements
        mask = torch.zeros(N, N, dtype=torch.bool)
        off_diag_indices = []
        for i in range(N):
            for j in range(N):
                if i != j:
                    off_diag_indices.append((i, j))

        # Randomly select which connections to activate
        perm = torch.randperm(len(off_diag_indices))[:n_connections]
        for idx in perm:
            i, j = off_diag_indices[idx]
            mask[i, j] = True

        # Generate random strengths in the specified range
        lo, hi = strength_range
        strengths = lo + (hi - lo) * torch.rand(n_connections, dtype=dtype)

        # Random signs (excitatory or inhibitory)
        signs = 2.0 * torch.bernoulli(0.5 * torch.ones(n_connections, dtype=dtype)) - 1.0

        A[mask] = strengths * signs

    return A
