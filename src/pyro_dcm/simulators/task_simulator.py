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

import warnings

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


def make_event_stimulus(
    event_times: torch.Tensor | list[float],
    event_amplitudes: torch.Tensor | list[float] | list[list[float]] | float,
    duration: float,
    dt: float,
    n_inputs: int | None = None,
    dtype: torch.dtype = torch.float64,
) -> dict[str, torch.Tensor]:
    """Create a variable-amplitude stick-function stimulus (SIM-01).

    Constructs piecewise-constant breakpoints for a sequence of events,
    each a single-``dt``-wide "stick" at amplitude ``event_amplitudes[i]``.
    The output dict has the same shape contract as :func:`make_block_stimulus`
    and is directly consumable by
    :class:`pyro_dcm.utils.ode_integrator.PiecewiseConstantInput`.

    This primitive implements SIM-01 and closes ROADMAP Phase 14 Success
    Criterion 2 (variable-amplitude event stimuli via piecewise-constant
    interpolation). It is designed for the driving-input term ``u(t)`` in
    the neural state equation [REF-001] Eq. 1, where events are discrete
    experimental onsets (e.g., visual flashes, button presses).

    Parameters
    ----------
    event_times : torch.Tensor or list of float
        Event onset times in seconds, shape ``(n_events,)``. Need not be
        sorted; sorted internally before breakpoint construction. All
        values must satisfy ``0 <= t < duration``.
    event_amplitudes : torch.Tensor, list of float, list of list of float, or float
        Per-event amplitudes. Three accepted shapes:

        - Scalar (``ndim == 0``): broadcast to ``(n_events, n_inputs)``;
          defaults ``n_inputs`` to 1 if not given.
        - 1-D ``(n_events,)``: interpreted as the column-0 amplitude;
          other columns (if ``n_inputs > 1``) are zero-padded.
        - 2-D ``(n_events, n_inputs)``: used directly.
    duration : float
        Simulation duration in seconds. Must be ``> 0``.
    dt : float
        Stick-function width (and grid quantization step) in seconds.
        Must be ``> 0``. Event times are quantized via
        ``round(t / dt) * dt``.
    n_inputs : int or None, optional
        Number of stimulus columns. If ``None``, inferred from
        ``event_amplitudes`` (1 for scalar/1-D, ``amps.shape[1]`` for 2-D).
    dtype : torch.dtype, optional
        Tensor dtype. Default ``torch.float64``.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``'times'``: breakpoint times, shape ``(K,)``, sorted ascending.
        - ``'values'``: piecewise-constant values at each breakpoint,
          shape ``(K, n_inputs)``. ``values[k]`` is the amplitude active
          for ``times[k] <= t < times[k+1]`` (left-closed, matching
          :class:`PiecewiseConstantInput`).

    Raises
    ------
    ValueError
        If ``dt <= 0`` or ``duration <= 0``; if any ``event_times`` lies
        outside ``[0, duration)``; if ``event_amplitudes`` has an
        unsupported shape; if two events quantize to the same ``dt`` grid
        index (supply one event with summed amplitudes instead).

    Warns
    -----
    UserWarning
        If an event's off-transition at ``t_on + dt`` would land past
        ``duration``; the off-transition is omitted and the final window
        holds the event amplitude until end-of-sim.

    Notes
    -----
    **Pitfall B12 -- stick-function blur under rk4 mid-step sampling.**
    When used as the modulatory input ``u_mod(t)`` in a bilinear DCM
    with an rk4 integrator, a single-``dt`` stick is effectively blurred
    to ~2x its declared width because the rk4 stages sample at
    ``t, t+dt/2, t+dt/2, t+dt``. For bilinear modulators, prefer
    :func:`make_epoch_stimulus` (boxcars are dt-invariant under rk4
    mid-step sampling). Stick functions remain appropriate for the
    driving-input term ``u(t)`` in [REF-001] Eq. 1.

    **Ordering note.** Events are sorted by ``event_times`` internally,
    so callers may pass them in any order. The same permutation is
    applied to ``event_amplitudes``.

    Examples
    --------
    >>> import torch
    >>> stim = make_event_stimulus(
    ...     event_times=[1.0, 3.0, 5.0],
    ...     event_amplitudes=[0.5, 1.0, 0.7],
    ...     duration=10.0,
    ...     dt=0.01,
    ... )
    >>> stim["times"].shape
    torch.Size([7])
    >>> stim["values"].shape
    torch.Size([7, 1])
    """
    # --- 1. Validate dt / duration up front ---
    if dt <= 0 or duration <= 0:
        raise ValueError(
            f"dt={dt}, duration={duration} must be > 0"
        )

    # --- 2. Normalize event_times ---
    event_times_t = torch.as_tensor(event_times, dtype=dtype)
    if event_times_t.ndim != 1:
        raise ValueError(
            f"event_times.ndim={event_times_t.ndim}; expected 1"
        )
    n_events = event_times_t.shape[0]

    # --- 3. Normalize event_amplitudes ---
    amps = torch.as_tensor(event_amplitudes, dtype=dtype)
    if amps.ndim == 0:
        if n_inputs is None:
            n_inputs = 1
        amps = amps.expand(n_events, n_inputs).clone()
    elif amps.ndim == 1:
        if n_inputs is None:
            n_inputs = 1
        if amps.shape[0] != n_events:
            raise ValueError(
                f"event_amplitudes.shape[0]={amps.shape[0]} must match "
                f"event_times.shape[0]={n_events}"
            )
        if n_inputs == 1:
            amps = amps.unsqueeze(1)
        else:
            amps = torch.cat(
                [
                    amps.unsqueeze(1),
                    torch.zeros(n_events, n_inputs - 1, dtype=dtype),
                ],
                dim=1,
            )
    elif amps.ndim == 2:
        if n_inputs is None:
            n_inputs = amps.shape[1]
        elif amps.shape[1] != n_inputs:
            raise ValueError(
                f"event_amplitudes.shape[1]={amps.shape[1]} must match "
                f"explicit n_inputs={n_inputs}"
            )
        if amps.shape[0] != n_events:
            raise ValueError(
                f"event_amplitudes.shape[0]={amps.shape[0]} must match "
                f"event_times.shape[0]={n_events}"
            )
    else:
        raise ValueError(
            f"event_amplitudes.ndim={amps.ndim}; expected 0, 1, or 2"
        )

    # --- 4. Validate temporal domain ---
    if n_events > 0:
        out_of_range = (event_times_t < 0) | (event_times_t >= duration)
        if out_of_range.any():
            bad = event_times_t[out_of_range].tolist()
            raise ValueError(
                f"event_times {bad} out of [0, {duration})"
            )

    # --- 5. Sort events by time ---
    if n_events > 0:
        sort_idx = torch.argsort(event_times_t)
        event_times_t = event_times_t[sort_idx]
        amps = amps[sort_idx]

    # --- 6. Quantize onsets to dt grid (nearest) ---
    onset_idx = torch.round(event_times_t / dt).long()
    onset_t = onset_idx.to(dtype) * dt

    # --- 7. Detect same-grid-index collisions ---
    if n_events >= 2:
        collisions = onset_idx[1:] == onset_idx[:-1]
        if collisions.any():
            first = int(torch.nonzero(collisions, as_tuple=False)[0, 0].item())
            raise ValueError(
                f"events {first} and {first + 1} quantize to the same dt "
                f"grid index ({int(onset_idx[first].item())}); supply one "
                f"event with summed amplitudes instead"
            )

    # --- 8. Build breakpoint list ---
    times_list: list[float] = [0.0]
    values_list: list[torch.Tensor] = [torch.zeros(n_inputs, dtype=dtype)]
    truncation_warned = False
    for i in range(n_events):
        t_on = float(onset_t[i].item())
        t_off = t_on + dt
        times_list.append(t_on)
        values_list.append(amps[i].clone())
        if t_off <= duration:
            times_list.append(t_off)
            values_list.append(torch.zeros(n_inputs, dtype=dtype))
        elif not truncation_warned:
            warnings.warn(
                f"Event {i} onset at t={t_on} has off-transition at "
                f"t={t_off} > duration={duration}; truncating tail to "
                f"end-of-sim",
                UserWarning,
                stacklevel=2,
            )
            truncation_warned = True

    times = torch.tensor(times_list, dtype=dtype)
    values = torch.stack(values_list)  # shape (K, n_inputs)

    return {"times": times, "values": values}


def make_epoch_stimulus(
    event_times: torch.Tensor | list[float],
    event_durations: torch.Tensor | list[float] | float,
    event_amplitudes: torch.Tensor | list[float] | list[list[float]] | float,
    duration: float,
    dt: float,
    n_inputs: int | None = None,
    dtype: torch.dtype = torch.float64,
) -> dict[str, torch.Tensor]:
    """Create a boxcar-shaped epoch stimulus (SIM-02).

    Constructs piecewise-constant breakpoints for a sequence of epochs,
    each a sustained-amplitude rectangle from ``t_on`` to
    ``t_on + event_durations[i]`` at amplitude ``event_amplitudes[i]``.
    The output dict has the same shape contract as
    :func:`make_block_stimulus` and is directly consumable by
    :class:`pyro_dcm.utils.ode_integrator.PiecewiseConstantInput`.

    **Preferred primitive for modulatory inputs** (``stimulus_mod`` in
    :func:`simulate_task_dcm`). Unlike :func:`make_event_stimulus`
    (stick functions), boxcars are dt-invariant under rk4 mid-step
    sampling because the amplitude is held constant for the full epoch
    duration (Pitfall B12).

    Parameters
    ----------
    event_times : torch.Tensor or list of float
        Epoch onset times in seconds, shape ``(n_events,)``. Need not be
        sorted. All values must satisfy ``0 <= t < duration``.
    event_durations : torch.Tensor, list of float, or float
        Per-epoch durations in seconds. Scalar broadcasts to all events;
        1-D must match ``event_times.shape[0]``. All values must be ``> 0``.
    event_amplitudes : torch.Tensor, list of float, list of list of float, or float
        Per-epoch amplitudes. Same normalization rules as
        :func:`make_event_stimulus` (scalar / 1-D / 2-D).
    duration : float
        Simulation duration in seconds. Must be ``> 0``.
    dt : float
        Grid quantization step in seconds. Must be ``> 0``. Onset and
        offset times are quantized via ``round(t / dt) * dt``.
    n_inputs : int or None, optional
        Number of stimulus columns. If ``None``, inferred from
        ``event_amplitudes``.
    dtype : torch.dtype, optional
        Tensor dtype. Default ``torch.float64``.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``'times'``: breakpoint times, shape ``(K,)``, sorted ascending.
        - ``'values'``: piecewise-constant amplitudes at each breakpoint,
          shape ``(K, n_inputs)``.

    Raises
    ------
    ValueError
        If ``dt <= 0`` or ``duration <= 0``; if any ``event_times`` lies
        outside ``[0, duration)``; if ``event_durations`` has any value
        ``<= 0`` or a shape mismatched with ``event_times``; if
        ``event_amplitudes`` has an unsupported shape.

    Warns
    -----
    UserWarning
        If any epoch's offset ``t_on + event_durations[i]`` exceeds
        ``duration`` (epoch is clipped at end-of-sim).
    UserWarning
        If any two epochs overlap in time (amplitudes are summed; see
        Notes). Fires at most once per call.

    Notes
    -----
    **Overlap semantics (locked per v0.3.0 Phase 14 L1 decision).**
    When two epochs overlap in time, their amplitudes **sum** (e.g., two
    overlapping unit-amplitude epochs produce amplitude 2 during the
    overlap window). A single :class:`UserWarning` is emitted at
    construction time ("Overlapping epochs detected; amplitudes are
    summed."). If override-semantics are desired (event ``i+1`` cancels
    event ``i``'s tail), callers must pre-flatten the event schedule
    themselves. This decision matches the bilinear DCM neural equation
    ``A_eff(t) = A + sum_j u_j(t) * B_j``, where simultaneous modulators
    already superpose.

    **Pitfall B12 -- rk4 mid-step sampling rationale.** rk4 at
    ``step_size=dt`` evaluates ``u(t)`` at ``{t, t+dt/2, t+dt/2, t+dt}``.
    Boxcar amplitudes are constant across every window of width ``dt``,
    so all four rk4 stages see the same value inside the epoch.
    Stick-function primitives (:func:`make_event_stimulus`) do not have
    this property and therefore should not be used as modulatory inputs
    in the bilinear path.

    Examples
    --------
    >>> import torch
    >>> stim = make_epoch_stimulus(
    ...     event_times=[5.0],
    ...     event_durations=[10.0],
    ...     event_amplitudes=[1.0],
    ...     duration=30.0,
    ...     dt=0.01,
    ... )
    >>> stim["times"][:3]
    tensor([0., 5., 15.], dtype=torch.float64)
    """
    # --- 1. Validate dt / duration up front ---
    if dt <= 0 or duration <= 0:
        raise ValueError(
            f"dt={dt}, duration={duration} must be > 0"
        )

    # --- 2. Normalize event_times ---
    event_times_t = torch.as_tensor(event_times, dtype=dtype)
    if event_times_t.ndim != 1:
        raise ValueError(
            f"event_times.ndim={event_times_t.ndim}; expected 1"
        )
    n_events = event_times_t.shape[0]

    # --- 3. Normalize event_amplitudes ---
    amps = torch.as_tensor(event_amplitudes, dtype=dtype)
    if amps.ndim == 0:
        if n_inputs is None:
            n_inputs = 1
        amps = amps.expand(n_events, n_inputs).clone()
    elif amps.ndim == 1:
        if n_inputs is None:
            n_inputs = 1
        if amps.shape[0] != n_events:
            raise ValueError(
                f"event_amplitudes.shape[0]={amps.shape[0]} must match "
                f"event_times.shape[0]={n_events}"
            )
        if n_inputs == 1:
            amps = amps.unsqueeze(1)
        else:
            amps = torch.cat(
                [
                    amps.unsqueeze(1),
                    torch.zeros(n_events, n_inputs - 1, dtype=dtype),
                ],
                dim=1,
            )
    elif amps.ndim == 2:
        if n_inputs is None:
            n_inputs = amps.shape[1]
        elif amps.shape[1] != n_inputs:
            raise ValueError(
                f"event_amplitudes.shape[1]={amps.shape[1]} must match "
                f"explicit n_inputs={n_inputs}"
            )
        if amps.shape[0] != n_events:
            raise ValueError(
                f"event_amplitudes.shape[0]={amps.shape[0]} must match "
                f"event_times.shape[0]={n_events}"
            )
    else:
        raise ValueError(
            f"event_amplitudes.ndim={amps.ndim}; expected 0, 1, or 2"
        )

    # --- 4. Normalize event_durations ---
    durations_t = torch.as_tensor(event_durations, dtype=dtype)
    if durations_t.ndim == 0:
        durations_t = durations_t.expand(n_events).clone()
    elif durations_t.ndim == 1:
        if durations_t.shape[0] != n_events:
            raise ValueError(
                f"event_durations.shape[0]={durations_t.shape[0]} must "
                f"match event_times.shape[0]={n_events}"
            )
    else:
        raise ValueError(
            f"event_durations.ndim={durations_t.ndim}; expected 0 or 1"
        )
    if n_events > 0 and (durations_t <= 0).any():
        raise ValueError("event_durations must all be > 0")

    # --- 5. Validate temporal domain ---
    if n_events > 0:
        out_of_range = (event_times_t < 0) | (event_times_t >= duration)
        if out_of_range.any():
            bad = event_times_t[out_of_range].tolist()
            raise ValueError(
                f"event_times {bad} out of [0, {duration})"
            )

    # --- 6. Quantize on/off times ---
    t_on = torch.round(event_times_t / dt) * dt
    t_off_raw = t_on + durations_t
    t_off = torch.clamp(t_off_raw, max=duration)
    if n_events > 0 and (t_off_raw > duration).any():
        warnings.warn(
            "Some epochs clipped to duration",
            UserWarning,
            stacklevel=2,
        )

    # --- 7. Build delta-amp event list and sweep ---
    events: list[tuple[float, torch.Tensor]] = []
    for i in range(n_events):
        events.append((float(t_on[i].item()), amps[i].clone()))
        events.append((float(t_off[i].item()), -amps[i].clone()))
    events.sort(key=lambda x: x[0])

    times_list: list[float] = [0.0]
    values_list: list[torch.Tensor] = [torch.zeros(n_inputs, dtype=dtype)]
    current = torch.zeros(n_inputs, dtype=dtype)
    max_individual_amp = (
        amps.abs().max().item() if n_events > 0 else 0.0
    )
    overlap_warned = False
    for t_k, delta in events:
        current = current + delta
        if times_list[-1] == t_k:
            values_list[-1] = current.clone()
        else:
            times_list.append(t_k)
            values_list.append(current.clone())
        # Overlap detection: any column exceeds the max individual amp.
        if (
            not overlap_warned
            and (current.abs() > max_individual_amp + 1e-12).any()
        ):
            warnings.warn(
                "Overlapping epochs detected; amplitudes are summed. "
                "If you want override semantics, pre-flatten events.",
                UserWarning,
                stacklevel=2,
            )
            overlap_warned = True

    times = torch.tensor(times_list, dtype=dtype)
    values = torch.stack(values_list)  # shape (K, n_inputs)

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
