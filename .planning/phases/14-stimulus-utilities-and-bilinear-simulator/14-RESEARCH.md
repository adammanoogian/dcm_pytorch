# Phase 14: Stimulus Utilities & Bilinear Simulator - Research

**Researched:** 2026-04-17
**Domain:** Variable-amplitude event/epoch stimulus utilities + bilinear
extension of `simulate_task_dcm`
**Confidence:** HIGH (code-grounded in existing `task_simulator.py`,
`coupled_system.py`, `ode_integrator.py`; Phase 13 infrastructure verified)
**Scope:** Implementation-level detail layered on top of Phase 13's
`CoupledDCMSystem(B=..., n_driving_inputs=..., stability_check_every=...)`.
Does NOT rehash project-level v0.3.0 research; cites and extends.

---

## Executive Summary

Phase 14 is a narrow, additive change to one source file
(`src/pyro_dcm/simulators/task_simulator.py`), one test file
(`tests/test_task_simulator.py` extension), and one export site
(`src/pyro_dcm/simulators/__init__.py`). It adds two public utilities
(`make_event_stimulus`, `make_epoch_stimulus`), one internal helper
(`merge_piecewise_inputs`), and one signature extension to
`simulate_task_dcm` with three new keyword-only args
(`B_list`, `stimulus_mod`, `n_driving_inputs`). No change to
`CoupledDCMSystem`, `NeuralStateEquation`, `PiecewiseConstantInput`, or
Balloon/BOLD/integrator modules.

Existing `test_task_simulator.py` has **one** strict return-key assertion
at lines 79-84. That assertion must be updated additively: the
`expected_keys` set grows by `{"B_list", "stimulus_mod"}`. No other
existing test requires changes. The "linear bit-exact" invariant (SIM-03)
is a structural consequence of calling `CoupledDCMSystem(A, C, input_fn,
hemo_params)` **without any `B=` kwarg** when `B_list is None` — which
inherits Phase 13's verified linear short-circuit
(`coupled_system.py:287-291`) and `neural_state.py:322-323`).

**Primary recommendation:** break Phase 14 into **2 plans, 2 waves**:
14-01 stimulus utilities (SIM-01, SIM-02) in parallel with nothing (can
start immediately; no simulator dep), then 14-02 simulator extension
(SIM-03, SIM-04, SIM-05) which depends on 14-01 for the utility under
test at SIM-05. Tests pair with code in each plan (Phase 13 precedent).

**Key numeric anchors (for planner):**

| Thing | Value | Source |
|-------|-------|--------|
| Linear regression atol | `0` (`torch.equal`) | Structural short-circuit; no fp drift |
| dt-invariance atol (SIM-05) | `1e-4` | REQUIREMENTS.md SIM-05 verbatim |
| dt-invariance solver | `rk4` (fixed-step) | Deterministic reproducibility |
| dt-invariance grid | `dt=0.01` vs `dt=0.005` | REQUIREMENTS.md SIM-05 verbatim |
| Breakpoint storage format | `{'times': (K,), 'values': (K, J)}` | Matches `make_block_stimulus` (task_simulator.py:296-299) |
| Stick-function duration | `dt` (one step) | D2 + `PiecewiseConstantInput` semantics |

---

## 1. `make_event_stimulus` — implementation pattern

**Goal (SIM-01).** Construct variable-amplitude stick-function stimuli
via piecewise-constant representation. Returns a `dict` matching the
existing `make_block_stimulus` convention so it plugs into
`PiecewiseConstantInput(times, values)` downstream without an adapter.

### 1.1 Signature

```python
def make_event_stimulus(
    event_times: torch.Tensor | list[float],
    event_amplitudes: torch.Tensor | list[float] | list[list[float]],
    duration: float,
    dt: float,
    n_inputs: int | None = None,
    dtype: torch.dtype = torch.float64,
) -> dict[str, torch.Tensor]:
    """Create variable-amplitude stick-function stimulus [REF-001 Eq.1 context].

    Returns minimal breakpoint representation: at each event onset,
    the value steps from 0 to event_amplitudes[i]; at t_onset + dt,
    it steps back to 0. Consumable by PiecewiseConstantInput.

    WARNING (Pitfall B12): stick functions are blurred to ~2x width
    when sampled at rk4 mid-steps (t+dt/2 falls inside the amplitude
    pulse). For modulatory inputs (stimulus_mod in bilinear DCM),
    make_epoch_stimulus (boxcars) is the preferred primitive.
    """
```

### 1.2 Return format (breakpoint-minimal, NOT dense `(T, J)`)

**Decision: return the minimum-breakpoint `{'times': (K,), 'values': (K, J)}`
dict.** Matches `make_block_stimulus` (task_simulator.py:280-299) exactly.
The research question proposed two options:

- (a) Dense `(T_fine, J)` tensor with all discretization grid points.
- (b) Minimum breakpoints (K events × 2 transitions + initial zero).

**Choose (b).** Justification (HIGH confidence):

1. **`PiecewiseConstantInput.__call__` uses `torch.searchsorted`**
   (ode_integrator.py:72). Lookup is `O(log K)` — faster with fewer
   breakpoints. Dense `(T, J)` wastes memory and slows searchsorted.
2. **Existing convention.** `make_block_stimulus` returns a 20-row dict
   for a 500s-duration 10-block design, NOT a 50,000-row dense tensor.
3. **`jump_t` option for adaptive solvers** (ode_integrator.py:174).
   torchdiffeq uses `PiecewiseConstantInput.grid_points` (which IS
   `self.times` per ode_integrator.py:76-90) to know where to restart
   integration at discontinuities. Dense grids would inflate `jump_t`
   to useless per-step entries.
4. **Read the requirement text literally.** REQUIREMENTS.md SIM-01
   says "`-> (T, J)`". The phrase "(T, J)" is the *conceptual* output
   shape (time × modulator columns), which `PiecewiseConstantInput(
   times, values)` reconstructs deterministically. The *dict* format
   is the concrete implementation. Note: SIM-02 uses the same
   phrasing; both follow the same interpretation.

**Optional dense representation (addendum):** if a caller wants the
dense `(T, J)` tensor (e.g., for plotting or for unit tests verifying
the pulse width), they can re-densify via
`PiecewiseConstantInput(times, values).__call__` iterated over
`t_grid`. This is not exposed as a separate utility in Phase 14.

### 1.3 Breakpoint construction algorithm

**Input shape validation:**
- `event_times`: `(n_events,)` float tensor. If list, `torch.tensor(...,
  dtype=dtype)`. Sort ascending and remember permutation.
- `event_amplitudes`: three accepted shapes:
  1. Scalar (e.g. `1.0`) → broadcast to `(n_events, n_inputs)`; requires
     `n_inputs` to be passed explicitly.
  2. 1-D `(n_events,)` → interpreted as column-0 amplitude; `n_inputs`
     defaults to 1. If `n_inputs > 1`, unsqueeze to `(n_events, 1)` and
     zero-pad to `(n_events, n_inputs)`.
  3. 2-D `(n_events, n_inputs)` → used directly.

**Validation errors (raise `ValueError`):**
- Any `event_time < 0` or `event_time >= duration`.
- Any `event_time + dt > duration` — the stick's off-transition lands
  past end-of-sim; clip to `duration` silently per Section 2's
  precedent, BUT warn via `UserWarning` once (per phase norm: loud
  errors for input mistakes, warnings for clipping).
- `event_amplitudes` shape incompatible with `(n_events,)` or
  `(n_events, n_inputs)`.
- `dt <= 0` or `duration <= 0`.

**Algorithm (pseudocode):**

```python
def make_event_stimulus(event_times, event_amplitudes, duration, dt,
                        n_inputs=None, dtype=torch.float64):
    # --- 1. Normalize inputs ---
    event_times = torch.as_tensor(event_times, dtype=dtype)
    amps = torch.as_tensor(event_amplitudes, dtype=dtype)

    if amps.ndim == 0:            # scalar
        if n_inputs is None:
            n_inputs = 1
        amps = amps.expand(event_times.shape[0], n_inputs).clone()
    elif amps.ndim == 1:
        if n_inputs is None:
            n_inputs = 1
        if amps.shape[0] != event_times.shape[0]:
            raise ValueError(
                f"event_amplitudes.shape[0]={amps.shape[0]} must match "
                f"event_times.shape[0]={event_times.shape[0]}"
            )
        # Default: broadcast into column 0 only; zero-pad other columns.
        amps = torch.cat(
            [amps.unsqueeze(1),
             torch.zeros(event_times.shape[0], n_inputs - 1, dtype=dtype)],
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
        if amps.shape[0] != event_times.shape[0]:
            raise ValueError(
                f"event_amplitudes.shape[0]={amps.shape[0]} must match "
                f"event_times.shape[0]={event_times.shape[0]}"
            )
    else:
        raise ValueError(f"event_amplitudes.ndim={amps.ndim}; expected 0, 1, or 2")

    # --- 2. Validate temporal domain ---
    if dt <= 0 or duration <= 0:
        raise ValueError(f"dt={dt}, duration={duration} must be > 0")
    if (event_times < 0).any() or (event_times >= duration).any():
        bad = event_times[(event_times < 0) | (event_times >= duration)]
        raise ValueError(f"event_times {bad.tolist()} out of [0, {duration})")

    # --- 3. Sort by time (required for PiecewiseConstantInput) ---
    sort_idx = torch.argsort(event_times)
    event_times = event_times[sort_idx]
    amps = amps[sort_idx]

    # --- 4. Quantize onsets to the dt grid ---
    # "Nearest grid point" is the natural reading: an event at t=5.003
    # with dt=0.01 should fire at grid index 500 (= t=5.00), not index
    # 501. Use round-to-nearest (banker's) to bias ties consistently.
    onset_idx = torch.round(event_times / dt).long()                # (n_events,)
    onset_t = onset_idx.to(dtype) * dt                              # quantized time

    # --- 5. Build (times, values) with minimal breakpoints ---
    # Initial segment: [0, t_onset_0) has value 0.
    # For each event i:
    #   at t=onset_t[i]: value = amps[i]
    #   at t=onset_t[i] + dt: value = 0
    # Degenerate: if onset_t[i+1] == onset_t[i] + dt, the "back to 0"
    # breakpoint of event i is superseded by the onset of event i+1;
    # the PiecewiseConstantInput sorted-times index handles this
    # correctly because searchsorted returns the last valid index.
    # But we emit both rows for symmetry — searchsorted picks the one
    # that matters.
    times_list = [0.0]
    values_list = [torch.zeros(n_inputs, dtype=dtype)]
    for i in range(event_times.shape[0]):
        t_on = onset_t[i].item()
        t_off = t_on + dt
        if t_off > duration:
            # Onset within [duration - dt, duration): off-edge past
            # end-of-sim; emit onset only. Simulator integrates up to
            # duration; PiecewiseConstantInput returns amps[i] for the
            # final dt window — acceptable truncation.
            times_list.append(t_on)
            values_list.append(amps[i])
        else:
            times_list.append(t_on)
            values_list.append(amps[i])
            times_list.append(t_off)
            values_list.append(torch.zeros(n_inputs, dtype=dtype))

    times = torch.tensor(times_list, dtype=dtype)
    values = torch.stack(values_list)                    # (K, n_inputs)
    return {"times": times, "values": values}
```

### 1.4 Trade-offs and answered subquestions

- **Nearest vs. floor for off-grid onsets.** Round-to-nearest (preferred).
  An event declared at `t=5.003` with `dt=0.01` is physically at grid
  index 500; rounding is the least-surprise default. Floor biases all
  off-grid events earlier — consistently wrong direction.
- **Broadcast 1-D amplitudes across J columns?** Only into column 0,
  pad other columns with 0. This matches `make_block_stimulus`'s
  convention (task_simulator.py:284-288: "only input 0 is active").
  Explicit 2-D input is required for multi-column amplitudes.
- **Return `times_grid` for plotting?** Not needed — callers can build
  it via `torch.arange(0, duration, dt)`. Keeping the return format
  identical to `make_block_stimulus` is the higher priority.
- **Do we expose the dense `(T, J)` form?** No — encapsulated in the
  `PiecewiseConstantInput` wrapper. If a caller needs dense for
  plotting, they call `input_fn(t_grid)` on each point.

**Confidence:** HIGH (code pattern mirrors task_simulator.py:237-299).

---

## 2. `make_epoch_stimulus` — implementation pattern

**Goal (SIM-02).** Construct boxcar-shaped modulatory inputs
(sustained-amplitude pulses). Preferred primitive for modulatory inputs
per Pitfall B12 (rk4 mid-step blur).

### 2.1 Signature

```python
def make_epoch_stimulus(
    event_times: torch.Tensor | list[float],
    event_durations: torch.Tensor | list[float] | float,
    event_amplitudes: torch.Tensor | list[float] | list[list[float]],
    duration: float,
    dt: float,
    n_inputs: int | None = None,
    dtype: torch.dtype = torch.float64,
) -> dict[str, torch.Tensor]:
    """Create boxcar-shaped epoch stimulus.

    Preferred primitive for modulatory inputs. Unlike stick functions
    (make_event_stimulus), boxcars are dt-invariant under rk4 mid-step
    sampling because the amplitude is held constant for the full epoch
    duration.
    """
```

### 2.2 Breakpoint construction

Near-identical to `make_event_stimulus` with one change:
- For each event `i`: at `t_on`, value = amplitude; at `t_on + duration_i`,
  value = 0.

**Overlap semantics (Claude's discretion — needs a decision, no lock):**
If event `i+1` starts before event `i`'s epoch ends, what happens?
- (a) **Override:** event `i+1` onset cancels event `i`'s tail. Final
  value at overlap time is `amps[i+1]`.
- (b) **Sum:** overlapping amplitudes accumulate. Final value is
  `amps[i] + amps[i+1]`.
- (c) **Raise:** treat overlap as a user error.

**Recommendation: (b) Sum, BUT emit UserWarning at construction time
if any overlap is detected.** Rationale:
1. Neuroscience semantics: if a subject sees two modulators active
   simultaneously (e.g., attention × valence), their modulatory
   effects superpose — this is what `Σ_j u_j · B_j` means in the DCM
   neural equation (`compute_effective_A` already sums contributions).
2. `PiecewiseConstantInput` does NOT natively sum — it returns whichever
   `values[idx]` is keyed by `searchsorted`. Sum-semantics requires
   **precomputing** the summed breakpoint values.
3. Warning at construction lets users opt into override by flattening
   the event schedule themselves.

**Algorithm (pseudocode):**

```python
def make_epoch_stimulus(event_times, event_durations, event_amplitudes,
                        duration, dt, n_inputs=None, dtype=torch.float64):
    # --- 1. Normalize inputs (same as make_event_stimulus for event_times,
    #       event_amplitudes). event_durations normalized similarly to amps. ---
    event_times = torch.as_tensor(event_times, dtype=dtype)
    event_durations = torch.as_tensor(event_durations, dtype=dtype)
    if event_durations.ndim == 0:
        event_durations = event_durations.expand_as(event_times).clone()
    if event_durations.shape[0] != event_times.shape[0]:
        raise ValueError("event_durations shape must match event_times")
    if (event_durations <= 0).any():
        raise ValueError("event_durations must all be > 0")
    # (amps normalization: same as Section 1.3 step 1)

    # --- 2. Validate ---
    if (event_times < 0).any() or (event_times >= duration).any():
        raise ValueError(...)

    # --- 3. Compute epoch start/end times, quantized to dt grid ---
    t_on = torch.round(event_times / dt) * dt
    t_off_raw = t_on + event_durations
    t_off = torch.clamp(t_off_raw, max=duration)              # clip at end-of-sim
    if (t_off_raw > duration).any():
        warnings.warn("Some epochs clipped to duration", UserWarning)

    # --- 4. Build breakpoint table via sorted event-list sweep ---
    # Algorithm: collect all (t, delta_amp) pairs:
    #   - at t_on[i]: +amps[i]
    #   - at t_off[i]: -amps[i]
    # Sort by t. Accumulate running sum. Emit one row per unique
    # breakpoint time with the accumulated value.
    events = []
    for i in range(event_times.shape[0]):
        events.append((t_on[i].item(), +amps[i]))
        events.append((t_off[i].item(), -amps[i]))
    events.sort(key=lambda x: x[0])

    times_list = [0.0]
    values_list = [torch.zeros(n_inputs, dtype=dtype)]
    current = torch.zeros(n_inputs, dtype=dtype)
    # Detect overlaps for warning
    overlap_warned = False
    for t, delta in events:
        current = current + delta
        # Merge breakpoints that share the same t (exact-equality).
        # searchsorted indexes off sorted unique t; duplicates shadow each
        # other, but we still need to pick the right one.
        if times_list and times_list[-1] == t:
            values_list[-1] = current.clone()
        else:
            times_list.append(t)
            values_list.append(current.clone())
        if (current > amps.abs().max() + 1e-12).any() and not overlap_warned:
            warnings.warn(
                "Overlapping epochs detected; amplitudes are summed. "
                "If you want override semantics, pre-flatten events.",
                UserWarning,
            )
            overlap_warned = True

    times = torch.tensor(times_list, dtype=dtype)
    values = torch.stack(values_list)                        # (K, n_inputs)
    return {"times": times, "values": values}
```

### 2.3 Trade-offs and answered subquestions

- **Epoch longer than `duration - event_time`:** clip to `duration` with
  `UserWarning`. Do not raise. Users often simulate design matrices
  where the last epoch extends past the intended end.
- **Output format:** same dict as `make_event_stimulus`.
  `{'times': (K,), 'values': (K, n_inputs)}`.
- **`event_durations` as scalar vs array:** accept both. Scalar
  `expand_as(event_times)`; else must match shape.

**Confidence:** HIGH for core algorithm; MEDIUM for overlap-sum
semantics (reasonable choice but no established SPM precedent; cite
the decision in the docstring).

---

## 3. `PiecewiseConstantInput` compatibility verification

**Claim (HIGH confidence, already verified in Phase 13 research):**
`PiecewiseConstantInput(times, values)` where `times: (K,)` (sorted
ascending) and `values: (K, M)` correctly evaluates to the held value
between breakpoints. For query time `t`, it returns `values[idx]` where
`idx = searchsorted(times, t, right=True) - 1`, clamped to `[0, K-1]`.

**Source:** ode_integrator.py:50-74.

**Implication for Sections 1, 2:** Both `make_event_stimulus` and
`make_epoch_stimulus` produce breakpoints that satisfy this contract
directly. No adapter needed.

**Edge case: stick function with `dt` spacing.** Event at `t_on=1.0`,
`dt=0.01` produces breakpoints `[0.0, 1.0, 1.01]` and values
`[[0], [1], [0]]`. Query at `t=1.005`: `searchsorted([0, 1.0, 1.01],
1.005, right=True)` returns 2 → `values[1]` = 1.0. Correct.

**Edge case: two sticks `dt` apart.** Event 1 at `t=1.0`, event 2 at
`t=1.01`, amplitudes `[0.5, 0.8]`. Naive stick output:
`[0, 1.0, 1.01, 1.01, 1.02]` with values `[[0], [0.5], [0], [0.8], [0]]`.
The duplicate `1.01` breakpoint is ambiguous — `searchsorted(..., right=True) - 1`
will return the last `1.01` index (the `[0.8]` value) at `t=1.01`,
because `right=True` reports the insertion point AFTER all equal
entries. So the ordering `[0, 0.5 step, 0 step (which disappears),
0.8 step, 0 step]` collapses to `[0, 0.5, 0.8, 0]` effectively —
intended behavior (event 2 absorbs event 1's "back to 0").

**Recommendation for safety:** `make_event_stimulus` should detect
same-grid-index events (`onset_idx[i] == onset_idx[i+1]`) and raise
`ValueError("events {i} and {i+1} quantize to the same dt grid
index")`. Users who want two events at the same instant should
supply one event with summed amplitudes.

---

## 4. `simulate_task_dcm` bilinear extension — signature and semantics

### 4.1 New signature

```python
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
    *,
    B_list: torch.Tensor | list[torch.Tensor] | None = None,
    stimulus_mod: dict[str, torch.Tensor] | PiecewiseConstantInput | None = None,
    n_driving_inputs: int | None = None,
) -> dict:
```

- **Positional signature unchanged.** All bilinear args are keyword-only
  (after `*`). Prevents positional collisions with existing callers.
- **Keep the existing seed/SNR/device semantics untouched.** No new
  seeds introduced for bilinear.

### 4.2 `B_list` acceptance policy

REQUIREMENTS.md SIM-03 names the parameter `B_list`, but Phase 13's
`CoupledDCMSystem` signature takes `B: (J, N, N)` stacked tensor. These
are compatible: accept **either** a Python list of `(N, N)` tensors OR
a stacked `(J, N, N)` tensor, and normalize internally.

**Normalization rule:**
```python
if B_list is None:
    B_stacked = None
elif isinstance(B_list, (list, tuple)):
    if len(B_list) == 0:
        B_stacked = None          # empty list == linear mode
    else:
        B_stacked = torch.stack(B_list).to(device=device, dtype=dtype)
elif isinstance(B_list, torch.Tensor):
    if B_list.ndim != 3:
        raise ValueError(
            f"B_list as tensor must be 3-D (J, N, N); got ndim={B_list.ndim}"
        )
    if B_list.shape[0] == 0:
        B_stacked = None
    else:
        B_stacked = B_list.to(device=device, dtype=dtype)
else:
    raise TypeError(
        f"B_list must be None, list of (N, N) tensors, or (J, N, N) tensor; "
        f"got {type(B_list)}"
    )
```

**Rationale:**
- Accept both to match user mental models (list mirrors `b_masks: list`
  in Phase 15's planned API per FEATURES.md §TS-3).
- Normalize to stacked tensor because `CoupledDCMSystem` takes stacked
  (Phase 13 locked).
- Empty list and `(0, N, N)` tensor both collapse to `None` → linear
  short-circuit (inheriting Phase 13's `self.B is None or self.B.shape[0] == 0`
  gate at coupled_system.py:287).

### 4.3 Linear short-circuit preservation (SIM-03 primary gate)

**Critical structural invariant.** When `B_stacked is None` (after
normalization), the function MUST call `CoupledDCMSystem(A, C, input_fn,
hemo_params)` **without any `B=` or `n_driving_inputs=` kwarg.** NOT
`CoupledDCMSystem(A, C, input_fn, hemo_params, B=None, n_driving_inputs=None)`.

Phase 13's short-circuit at coupled_system.py:287-291 executes the
literal `self.A @ x + self.C @ u_all` path when `self.B is None`. By
*omitting* the kwargs entirely when linear, we route through the
identical construction as the pre-v0.3.0 code — inheriting the
bit-exact guarantee.

**Anti-pattern (DO NOT WRITE):**
```python
# Bad — even B=None causes the register_buffer branch to be skipped;
# the effective linear path is identical, but future refactors could
# diverge. Prefer unconditional omission.
system = CoupledDCMSystem(A, C, input_fn, hemo_params,
                          B=B_stacked, n_driving_inputs=n_driving_inputs)
```

**Correct pattern:**
```python
if B_stacked is None:
    # Literal v0.2.0 path; no B kwarg; inherits Phase 13 gate.
    system = CoupledDCMSystem(A_dev, C_dev, input_fn, hemo_params)
else:
    # Bilinear path; B + n_driving_inputs explicit.
    system = CoupledDCMSystem(
        A_dev, C_dev, merged_input_fn, hemo_params,
        B=B_stacked,
        n_driving_inputs=n_driving_inputs_resolved,
    )
```

### 4.4 `n_driving_inputs` policy

REQUIREMENTS doesn't lock this. Phase 13 CONTEXT locked: **raise
`ValueError` when `B` is non-empty and `n_driving_inputs is None`**
(coupled_system.py:226-233).

**Phase 14 policy: default to `C.shape[1]` when `B_list` is supplied
but `n_driving_inputs` is not.** Rationale:

1. In `simulate_task_dcm`, the user passes `C` explicitly; `C.shape[1]`
   is an unambiguous source of the driving-input column count.
2. The `CoupledDCMSystem` ValueError protects callers who bypass the
   simulator and go direct; at the simulator level, we can safely
   default because we control the widened-input construction
   (Section 5).
3. Document explicitly: "if `n_driving_inputs` is None and `B_list` is
   provided, defaults to `C.shape[1]`."

**Validation:** if user passes both `n_driving_inputs` and `C`
inconsistently (`n_driving_inputs != C.shape[1]`), raise
`ValueError`. Consistent with "fail loud on user inconsistency."

```python
n_driving_inputs_resolved = (
    n_driving_inputs if n_driving_inputs is not None else C.shape[1]
)
if n_driving_inputs_resolved != C.shape[1]:
    raise ValueError(
        f"n_driving_inputs={n_driving_inputs_resolved} inconsistent with "
        f"C.shape[1]={C.shape[1]}"
    )
```

### 4.5 Return dict extension (SIM-04)

Add `'B_list'` and `'stimulus_mod'` keys. Values:
- Linear mode: both `None`.
- Bilinear mode: `B_list` = the stacked `(J, N, N)` tensor;
  `stimulus_mod` = the constructed `PiecewiseConstantInput` (NOT the
  original dict), symmetric with the existing `'stimulus'` key at
  task_simulator.py:233.

```python
return {
    ...,                                      # existing keys
    "stimulus": input_fn,                     # driving PiecewiseConstantInput (unchanged)
    "B_list": B_stacked,                      # new; None in linear mode
    "stimulus_mod": stimulus_mod_input_fn,    # new; None in linear mode
}
```

**Existing test impact (BILIN-04 precedent):** `test_task_simulator.py`
line 79-84 asserts `set(result.keys()) == expected_keys`. This test
MUST be updated additively — add `"B_list", "stimulus_mod"` to the
set. This is a one-line change; it does not regress linear behavior.

### 4.6 Complete implementation skeleton

```python
def simulate_task_dcm(A, C, stimulus, hemo_params=None, duration=300.0,
                     dt=0.01, TR=2.0, SNR=5.0, solver="dopri5",
                     device="cpu", dtype=torch.float64, seed=None, *,
                     B_list=None, stimulus_mod=None, n_driving_inputs=None):
    # 1. Seed
    if seed is not None:
        torch.manual_seed(seed)

    N = A.shape[0]

    # 2. Normalize driving-stimulus to PiecewiseConstantInput.
    driving_input_fn = _normalize_stimulus_to_input_fn(stimulus, device, dtype)

    # 3. Normalize B_list (Section 4.2).
    B_stacked = _normalize_B_list(B_list, device, dtype)

    # 4. Build effective input_fn.
    if B_stacked is None:
        # LINEAR MODE: literal v0.2.0 path.
        input_fn = driving_input_fn
        stimulus_mod_input_fn = None
        system = CoupledDCMSystem(
            A.to(device=device, dtype=dtype),
            C.to(device=device, dtype=dtype),
            input_fn, hemo_params,
        )
    else:
        # BILINEAR MODE.
        if stimulus_mod is None:
            raise ValueError(
                "stimulus_mod is required when B_list is non-None; got None."
            )
        stimulus_mod_input_fn = _normalize_stimulus_to_input_fn(
            stimulus_mod, device, dtype
        )
        J = B_stacked.shape[0]
        # Validate stimulus_mod column count matches J.
        if stimulus_mod_input_fn.values.shape[1] != J:
            raise ValueError(
                f"stimulus_mod has {stimulus_mod_input_fn.values.shape[1]} "
                f"columns but B_list has J={J} modulators"
            )
        # Resolve n_driving_inputs (Section 4.4).
        n_driv = n_driving_inputs if n_driving_inputs is not None else C.shape[1]
        if n_driv != C.shape[1]:
            raise ValueError(
                f"n_driving_inputs={n_driv} inconsistent with C.shape[1]={C.shape[1]}"
            )
        # Merge driving + modulator into single widened PiecewiseConstantInput
        # (Section 5).
        input_fn = merge_piecewise_inputs(driving_input_fn, stimulus_mod_input_fn)
        system = CoupledDCMSystem(
            A.to(device=device, dtype=dtype),
            C.to(device=device, dtype=dtype),
            input_fn, hemo_params,
            B=B_stacked,
            n_driving_inputs=n_driv,
        )

    # 5. Integrate (unchanged from current simulator body).
    y0 = make_initial_state(N, dtype=dtype, device=device)
    t_eval = torch.arange(0, duration, dt, dtype=dtype, device=device)
    grid_points = input_fn.grid_points
    solution = integrate_ode(system, y0, t_eval, method=solver,
                             grid_points=grid_points, step_size=dt)

    # 6. Extract BOLD, downsample, add noise (unchanged from current body).
    # ... [task_simulator.py:178-208 verbatim] ...

    # 7. Return extended dict.
    return {
        "bold": noisy_bold,
        "bold_clean": bold_clean_ds,
        "bold_fine": clean_bold,
        "neural": x,
        "hemodynamic": {"s": s, "f": f, "v": v, "q": q},
        "times_fine": t_eval,
        "times_TR": times_TR,
        "params": {
            "A": A_dev, "C": C_dev, "hemo_params": hemo_params,
            "SNR": SNR, "TR": TR, "duration": duration, "solver": solver,
        },
        "stimulus": driving_input_fn,
        "B_list": B_stacked,
        "stimulus_mod": stimulus_mod_input_fn,
    }
```

**Confidence:** HIGH (pattern mirrors existing implementation with
additive conditional branches).

---

## 5. `merge_piecewise_inputs` helper — widened input construction

### 5.1 Problem statement

Phase 13 locked: `CoupledDCMSystem.forward` expects `input_fn(t)` to
return a `(M_driving + J_mod,)` vector (coupled_system.py:92-93). Phase
14 gets two separate `PiecewiseConstantInput`s (driving and modulator)
and must merge them into a single widened one.

### 5.2 Merge algorithm

**Input:**
- `drive: PiecewiseConstantInput` with `times: (K1,)`, `values: (K1, M)`.
- `mod: PiecewiseConstantInput` with `times: (K2,)`, `values: (K2, J)`.

**Output:** `merged: PiecewiseConstantInput` with `times: (K,)` (sorted
unique union), `values: (K, M + J)` where left `M` cols come from
drive and right `J` cols come from mod.

**Algorithm:**

```python
def merge_piecewise_inputs(
    drive: PiecewiseConstantInput,
    mod: PiecewiseConstantInput,
) -> PiecewiseConstantInput:
    """Combine two piecewise-constant inputs into one widened input.

    Returns a new PiecewiseConstantInput whose values at any t are
    [drive(t), mod(t)] concatenated along the column axis.
    """
    t_drive = drive.times                         # (K1,)
    t_mod = mod.times                             # (K2,)
    v_drive = drive.values                        # (K1, M)
    v_mod = mod.values                            # (K2, J)
    dtype = v_drive.dtype
    device = v_drive.device

    # 1. Sorted union of breakpoint times.
    #    Use torch.unique + sort for dedupe. Float equality is acceptable
    #    here because the caller's make_*_stimulus utilities produce
    #    times on an exact dt grid; ties merge cleanly.
    all_times = torch.cat([t_drive, t_mod])
    merged_times, _ = torch.unique(all_times, sorted=True, return_inverse=False)
    # Enforce float64 to match both inputs (caller dtype).

    # 2. For each merged breakpoint, evaluate drive(t) and mod(t).
    #    These are piecewise-constant, so the value at t is the left-closed
    #    value (matches PiecewiseConstantInput.__call__ semantics).
    #    Use the existing __call__ method per-breakpoint — O(K log K).
    M = v_drive.shape[1]
    J = v_mod.shape[1]
    merged_values = torch.empty(
        (merged_times.shape[0], M + J), dtype=dtype, device=device,
    )
    for k, t_k in enumerate(merged_times):
        merged_values[k, :M] = drive(t_k.detach())
        merged_values[k, M:] = mod(t_k.detach())

    return PiecewiseConstantInput(merged_times, merged_values)
```

### 5.3 Complexity and correctness

- **Time:** O(K log K) for the unique+sort; O(K log max(K1, K2)) for the
  per-breakpoint lookups. Total O(K log K). For typical DCM (K1 ≈ 20
  driving blocks, K2 ≈ 40 modulator events), this runs in microseconds.
- **Correctness test:** at any query time `t*`, the merged input returns
  `[drive(t*), mod(t*)]`. Because the merged breakpoint set contains
  every discontinuity of both drive and mod, and because
  `PiecewiseConstantInput.__call__` samples left-closed, the merged
  values at a breakpoint `t_k` equal `drive(t_k)` (which is the value
  active AT t_k, not just before) concatenated with `mod(t_k)`.
- **Edge case: same breakpoint in both.** If `t_drive[i] == t_mod[j]`,
  `torch.unique(sorted=True)` dedupes to one entry; both columns get
  the correct value.

### 5.4 Alternative designs considered

- **Dense re-grid on `t_eval`:** construct `(T_fine, M+J)` tensor by
  evaluating both inputs on the fine grid, then wrap. Rejected:
  (1) inflates `jump_t` for adaptive solvers, (2) wastes memory for
  sparse event schedules, (3) loses the "restart at discontinuity"
  guarantee.
- **Custom `class MergedPiecewiseInput`:** keep references to drive and
  mod, implement `__call__` as concatenation. Rejected: introduces a
  second discontinuity-function class; `PiecewiseConstantInput.grid_points`
  becomes the union of both, duplicating the merge work at each
  integrator pass. Materializing the merge once at sim-startup is
  cheaper.

### 5.5 Placement

Put `merge_piecewise_inputs` as a module-private `_merge_piecewise_inputs`
in `task_simulator.py` (closer to its single caller), OR as a public
helper in `utils/ode_integrator.py` next to `PiecewiseConstantInput`.

**Recommendation: public in `utils/ode_integrator.py`**. Reasons:

1. **Reusability:** Phase 15's Pyro model will need to build the same
   merged input_fn when SVI calls the model with `stimulus` and
   `stimulus_mod` separately (see `task_dcm_model.py` structure).
2. **Logical cohesion:** lives next to `PiecewiseConstantInput` which
   it consumes.
3. **Testability:** public helper is easier to unit-test.

Export from `pyro_dcm/utils/__init__.py` (see existing pattern there).

**Confidence:** HIGH for algorithm correctness; MEDIUM for placement
choice (either location is defensible).

---

## 6. dt-invariance test (SIM-05) construction

### 6.1 Requirement verbatim

> `dt`-invariance test: ODE integration at `dt=0.01` and `dt=0.005`
> produce equivalent BOLD within `atol=1e-4` under a fixed bilinear
> ground truth.

### 6.2 Fixture design

```python
def test_dt_invariance_bilinear():
    """SIM-05: dt=0.01 vs dt=0.005 produce equivalent BOLD (atol=1e-4).

    Under a fixed bilinear ground truth, ODE integration at two
    different step sizes should produce equivalent BOLD on the shared
    TR grid. This proves that our stimulus interpolation + ODE
    integration is dt-independent to the required tolerance.
    """
    # --- Fixed ground truth ---
    N = 3
    A = parameterize_A(torch.zeros(N, N, dtype=torch.float64))
    # A is diag(-0.5, -0.5, -0.5); guaranteed stable baseline.
    C = torch.tensor([[1.0], [0.0], [0.0]], dtype=torch.float64)
    B = torch.zeros(1, N, N, dtype=torch.float64)
    B[0, 1, 0] = 0.3                         # modulator 0 strengthens 1<-0
    B[0, 2, 1] = 0.3                         # modulator 0 strengthens 2<-1

    # --- Stimuli ---
    duration = 200.0
    TR = 2.0
    # Driving: block design, 3 blocks of 30s ON / 30s OFF
    stim_drive = make_block_stimulus(
        n_blocks=3, block_duration=30.0, rest_duration=30.0, n_inputs=1,
    )
    # Modulator: PREFER BOXCARS for dt-invariance (Pitfall B12).
    # Event at t=50s, duration=40s, amplitude=1.0.
    stim_mod = make_epoch_stimulus(
        event_times=[50.0], event_durations=[40.0],
        event_amplitudes=[1.0], duration=duration, dt=0.005,
    )

    # --- Run at two dt values, SAME solver, SAME seed, NO noise ---
    common_kwargs = dict(
        A=A, C=C, stimulus=stim_drive,
        duration=duration, TR=TR,
        SNR=-1,                              # deterministic (no noise)
        solver="rk4",                        # fixed-step for reproducibility
        dtype=torch.float64,
        seed=42,
        B_list=B, stimulus_mod=stim_mod,
    )
    result_dt01 = simulate_task_dcm(**common_kwargs, dt=0.01)
    result_dt005 = simulate_task_dcm(**common_kwargs, dt=0.005)

    # --- Compare on the shared TR grid ---
    # Both have T_TR = duration / TR = 100 rows; times_TR identical.
    assert result_dt01["times_TR"].shape == result_dt005["times_TR"].shape
    torch.testing.assert_close(
        result_dt01["times_TR"], result_dt005["times_TR"],
        atol=1e-12, rtol=0.0,
    )
    torch.testing.assert_close(
        result_dt01["bold_clean"], result_dt005["bold_clean"],
        atol=1e-4, rtol=0.0,
    )
```

### 6.3 Why `rk4` not `dopri5`?

**HIGH-confidence answer.** Adaptive solvers (`dopri5`) adjust step size
based on local error tolerance (`rtol=1e-5, atol=1e-7` in
ode_integrator.py:98-99). Running at `dt=0.01` vs `dt=0.005` doesn't
change the step size for `dopri5` — only `t_eval` density changes.
`dopri5` with different `t_eval` grids returns slightly different
results due to interpolation between adaptive steps. This confounds
the "integration dt-invariance" claim with "interpolation
equivalence."

`rk4` with `step_size=dt` is fixed-step: `dt=0.01` takes twice the
integration steps of `dt=0.005`. The result approximates the true
ODE solution with `O(dt⁴)` error. At `dt=0.01` vs `dt=0.005`, the
truncation error ratio is 16:1 — so if `dt=0.005` is accurate to
`~1e-8`, then `dt=0.01` is accurate to `~1.6e-7`, and their difference
is `~1.6e-7` — well below `atol=1e-4`.

**Alternative consideration:** if the test fails at `atol=1e-4` with
`rk4`, the likely culprit is stimulus quantization mismatch
(`make_epoch_stimulus` with `dt=0.005` embeds different breakpoints
than with `dt=0.01`). **Resolution: construct both `stim_mod` at
`dt=0.005` and reuse across both simulations.** The stimulus
breakpoints are time-domain constants independent of the simulation's
ODE step size. The test fixture above already does this.

### 6.4 Expected truncation-error magnitude

For `rk4` with local error `O(h⁵)` and global error `O(h⁴)`:
- At `h=0.01`, `h⁴ = 1e-8`.
- At `h=0.005`, `h⁴ = 6.25e-10`.
- Difference: `~1e-8`, far below `atol=1e-4`.

The 1e-4 tolerance budget is generous — room for small floating-point
accumulation across 20,000 steps (dt=0.01, duration=200s). Empirically,
the test should pass with headroom ~3 orders of magnitude.

### 6.5 Fallback if `atol=1e-4` fails

If the test empirically fails:
1. **First suspect:** stimulus mis-quantization. Verify both runs use
   the SAME `stim_drive` and `stim_mod` dicts, constructed ONCE
   outside the loop.
2. **Second suspect:** noise contamination. Verify `SNR=-1` (no noise).
3. **Third suspect:** truncation-error boundary. If fixture is
   genuinely sharp (stiff dynamics at 3σ B), relax to `atol=1e-3`
   with a comment citing this escalation path. **Do not relax without
   root-cause analysis.**

### 6.6 Should we also test linear-mode dt-invariance?

**Recommendation: include a second test `test_dt_invariance_linear` for
regression symmetry.** Linear-mode dt-invariance is a pre-Phase-14
invariant (the current simulator has this property by construction of
the same ODE). But asserting it under the Phase 14 extension
(B_list=None path) is a one-line copy of the fixture above with
`B_list` and `stimulus_mod` removed. Zero additional code cost,
extra regression coverage.

**Confidence:** HIGH for solver choice and fixture; MEDIUM for
`atol=1e-4` passing without adjustment (depends on truncation error
at specific A/B/stimulus).

---

## 7. Regression test for linear bit-exactness (SIM-03 primary gate)

### 7.1 Requirement verbatim

> When `B_list=None`, output is exactly the current linear simulator
> output (regression test required).

### 7.2 Structural argument (no drift by construction)

The short-circuit in Section 4.3 makes this a **structural** invariant,
not just a test-time property:

1. `simulate_task_dcm(..., B_list=None)` sets `B_stacked = None` via
   the normalization.
2. The function calls `CoupledDCMSystem(A_dev, C_dev, input_fn,
   hemo_params)` with **identical positional args** to the current
   implementation at task_simulator.py:158.
3. `CoupledDCMSystem.__init__` without `B` kwarg sets `self.B = None`
   (coupled_system.py:222-223).
4. `CoupledDCMSystem.forward` with `self.B is None` executes the
   literal `dx = self.A @ x + self.C @ u_all` (coupled_system.py:287-291).
5. Every subsequent operation (ODE integration, downsampling, noise)
   is byte-identical.

**Conclusion:** there is NO code path in the extended simulator that
could introduce fp drift when `B_list=None`. The regression test can
assert `torch.equal(bold_a, bold_b)` strict equality, not just
`atol=1e-10`.

### 7.3 Test fixture

```python
def test_bilinear_arg_none_matches_no_kwarg():
    """SIM-03: B_list=None output bit-exact to no-kwarg call.

    Phase 13 invariance test covered the CoupledDCMSystem level; this
    test extends it to the simulator level. Both paths must produce
    byte-identical output because the linear short-circuit avoids any
    arithmetic involving None-valued B.
    """
    A = torch.tensor([[-0.5, 0.1], [0.2, -0.5]], dtype=torch.float64)
    C = torch.tensor([[0.25], [0.0]], dtype=torch.float64)
    stim = make_block_stimulus(n_blocks=3, block_duration=20.0,
                                rest_duration=20.0)

    # Path 1: pre-Phase-14 signature (no bilinear kwargs).
    result_a = simulate_task_dcm(
        A, C, stim, duration=120.0, SNR=-1, seed=42,
    )
    # Path 2: post-Phase-14 signature with explicit None.
    result_b = simulate_task_dcm(
        A, C, stim, duration=120.0, SNR=-1, seed=42,
        B_list=None, stimulus_mod=None, n_driving_inputs=None,
    )

    # Strict bit-exact on clean BOLD (noise path seeded identically too).
    assert torch.equal(result_a["bold_clean"], result_b["bold_clean"]), (
        "B_list=None explicit path diverges from no-kwarg path"
    )
    # Noisy BOLD should also match bit-exact because seed is identical.
    assert torch.equal(result_a["bold"], result_b["bold"])
    # And neural states.
    assert torch.equal(result_a["neural"], result_b["neural"])
    # New keys are None in both.
    assert result_a["B_list"] is None and result_b["B_list"] is None
    assert result_a["stimulus_mod"] is None and result_b["stimulus_mod"] is None
```

### 7.4 Covariance with Phase 13

Phase 13's `tests/test_linear_invariance.py` and
`tests/test_coupled_system_bilinear.py::test_linear_kwarg_none_matches_no_kwarg_bit_exact`
already verify the `CoupledDCMSystem` level. SIM-03's new test is the
**simulator level** extension: simulator adds seed handling, noise, and
downsampling on top of the ODE integration. Even with these
post-integration stages, the "no kwargs passed to inner system when
linear" design extends the bit-exact guarantee.

**Confidence:** HIGH (structural argument).

---

## 8. Existing `test_task_simulator.py` test inventory

Listed for the planner to audit. Total: ~13 test methods across 7 test
classes plus utility fixtures. None of them pass `B_list` or
`stimulus_mod`; all will exercise the linear short-circuit. Only one
requires a one-line additive update (the return-keys assertion).

| Line | Class::Method | Behavior | Phase 14 action |
|------|---------------|----------|-----------------|
| 66-88 | `TestSimulatorOutputStructure::test_simulator_output_keys` | asserts `set(result.keys()) == expected_keys` with 9 literal keys | **UPDATE** — add `"B_list", "stimulus_mod"` to `expected_keys` |
| 94-117 | `TestSimulatorOutputStructure::test_simulator_output_shapes` | asserts (T_TR, N) shapes on BOLD tensors | PASS unchanged |
| 128-143 | `TestSimulatorNumerics::test_simulator_no_nan` | 300s no-NaN | PASS unchanged |
| 145-166 | `TestSimulatorNumerics::test_simulator_bold_range` | BOLD peak in 0.5–5% | PASS unchanged |
| 168-195 | `TestSimulatorNumerics::test_simulator_snr` | empirical SNR within 20% | PASS unchanged |
| 197-217 | `TestSimulatorNumerics::test_simulator_no_noise_mode` | SNR≤0 → no noise | PASS unchanged |
| 228-256 | `TestSimulatorReproducibility::test_simulator_reproducibility` | same seed → identical; diff seed → diff | PASS unchanged |
| 267-286 | `TestSimulatorMultiRegion::test_simulator_5region` | 5-region run | PASS unchanged |
| 288-331 | `TestSimulatorMultiRegion::test_simulator_500s` | 500s no-NaN, peak BOLD | PASS unchanged |
| 342-370 | `TestMakeBlockStimulus::test_make_block_stimulus` | block stimulus struct | PASS unchanged |
| 372-382 | `TestMakeBlockStimulus::test_make_block_stimulus_multi_input` | multi-input block | PASS unchanged |
| 388-396 | `TestMakeRandomStableA::test_make_random_stable_A` | A stability | PASS unchanged |
| 398-405 | `TestMakeRandomStableA::test_make_random_stable_A_diagonal` | A diagonal | PASS unchanged |
| 407-422 | `TestMakeRandomStableA::test_make_random_stable_A_density` | density | PASS unchanged |
| 424-428 | `TestMakeRandomStableA::test_make_random_stable_A_reproducibility` | same seed | PASS unchanged |
| 439-482 | `TestNeuralDynamics::test_neural_state_stable_trajectory` | neural states bounded | PASS unchanged |
| 484-518 | `TestNeuralDynamics::test_driven_region_responds` | driven > undriven | PASS unchanged |
| 529-541 | `TestStimulusPassthrough::test_piecewise_input_passthrough` | accepts `PiecewiseConstantInput` directly | PASS unchanged |

**Summary:** **1 test requires a one-line additive change** (the key set
at lines 79-83). The other **17+ tests PASS UNCHANGED** — structural
guarantee per Section 7.2.

**Other tests that call `simulate_task_dcm` indirectly** (verified via
grep):
- `tests/test_task_dcm_model.py` — uses output (`["stimulus"]`, `["bold"]`).
  Doesn't assert key-set; PASS unchanged.
- `tests/test_svi_integration.py` — uses `result["stimulus"]`; PASS unchanged.
- `tests/test_elbo_model_comparison.py` — uses `result["stimulus"]`; PASS unchanged.
- `tests/test_task_dcm_recovery.py` — uses `sim_result["stimulus"]`; PASS unchanged.

**Grep sentinels to run after the Phase 14 change:**
```bash
grep -n "set(result.keys())" tests/           # expect 1 match (updated line)
grep -n 'expected_keys' tests/                # expect 2 matches (updated + test_spectral_simulator)
grep -n 'result\[\"B_list\"\]' tests/         # expect ≥1 match (new tests)
grep -n 'result\[\"stimulus_mod\"\]' tests/   # expect ≥1 match (new tests)
```

---

## 9. Wave / plan decomposition proposal

### Recommended split: 2 plans, 2 waves

**Plan 14-01: Stimulus utilities (SIM-01, SIM-02)**
- Files modified:
  1. `src/pyro_dcm/simulators/task_simulator.py` — add
     `make_event_stimulus`, `make_epoch_stimulus` alongside existing
     `make_block_stimulus`.
  2. `src/pyro_dcm/simulators/__init__.py` — re-export new symbols in
     `__all__`.
  3. **Optional** `src/pyro_dcm/utils/ode_integrator.py` — add
     `merge_piecewise_inputs` helper here (Section 5.5 recommendation).
  4. **Optional** `src/pyro_dcm/utils/__init__.py` — re-export
     `merge_piecewise_inputs`.
- New tests: `tests/test_stimulus_utils.py`:
  - `TestMakeEventStimulus` (~6 tests): shape, scalar/1D/2D amplitudes,
    sort-by-time, validation errors, stick pulse width = dt, dict
    compatibility with `PiecewiseConstantInput`.
  - `TestMakeEpochStimulus` (~6 tests): shape, overlap sum+warning,
    clipping at duration, 1-epoch boxcar correctness, same-grid-index
    overlap, n_inputs validation.
  - `TestMergePiecewiseInputs` (~3 tests): shape, values at breakpoints
    concatenated correctly, grid_points union.
- Estimated LoC: +200 src / +280 test
- Dependencies: **none** (pure utilities; no simulator/ODE dep)
- Parallelizable: can start immediately from branch base

**Plan 14-02: Simulator bilinear extension (SIM-03, SIM-04, SIM-05)**
- Files modified:
  1. `src/pyro_dcm/simulators/task_simulator.py` — extend
     `simulate_task_dcm` signature and body per Section 4.
  2. `tests/test_task_simulator.py` line 79-83 — update `expected_keys`
     set (one-line additive).
- New tests (in `test_task_simulator.py` or new
  `test_bilinear_simulator.py`):
  - `TestBilinearSimulator::test_bilinear_arg_none_matches_no_kwarg`
    (SIM-03 regression; Section 7.3)
  - `TestBilinearSimulator::test_bilinear_output_distinguishable_from_linear`
    (SIM-03 positive): run at same seed with and without B_list; BOLD
    RMS differs.
  - `TestBilinearSimulator::test_return_dict_has_bilinear_keys` (SIM-04):
    both linear (None) and bilinear (non-None) values in keys.
  - `TestBilinearSimulator::test_dt_invariance_bilinear` (SIM-05;
    Section 6.2)
  - **Optional** `TestBilinearSimulator::test_dt_invariance_linear`
    (symmetry regression).
- Estimated LoC: +150 src / +250 test
- Dependencies: 14-01 (needs `make_event_stimulus`,
  `make_epoch_stimulus`, `merge_piecewise_inputs`)
- Parallelizable: no (strict depends on 14-01)

### Rejected alternatives

**Alternative A: 3 plans** (14-01 utils, 14-02 simulator-core, 14-03
dt-invariance). Rejected because dt-invariance test is tightly coupled
to simulator behavior; splitting would artificially fragment a single
commit-scope.

**Alternative B: 1 plan** (all 5 requirements in one commit). Rejected
because 14-01 has zero dependencies and can land independently,
minimizing merge conflict risk with parallel Phase 15 prep work.

**Alternative C: split 14-01 into 14-01a (stimulus utils) and 14-01b
(merge helper).** Rejected: merge helper is ~40 LoC and naturally
belongs with its first consumer (the simulator), but placing it in
`utils/` makes it available to Phase 15 without forcing Phase 15 to
import `simulators/`. Keep in 14-01 with utils/ placement.

### Critical path

```
Wave 1:                   Wave 2:
┌──────────────────┐      ┌──────────────────────────────┐
│ 14-01: Stim utils│──────▶│ 14-02: Simulator extension   │
│ + merge helper   │      │ + SIM-03/04/05 tests          │
└──────────────────┘      └──────────────────────────────┘
```

Total wall time: ~1 day per plan for experienced engineer. Full Phase
14 landable in ~2 days.

**Confidence:** HIGH (dependency-forced structure).

---

## 10. Pitfalls and phase-specific risks

### 10.1 Cited from project-level `PITFALLS.md`

| # | Title | Phase 14 impact | Mitigation |
|---|-------|-----------------|------------|
| **B12** | `stim_mod` interpolation at rk4 mid-steps blurs sticks | AFFECTS SIM-01 | Document `make_event_stimulus` docstring: "for modulators, prefer `make_epoch_stimulus` (boxcar). Sticks are blurred to ~2× width by rk4 mid-step sampling. See Pitfall B12." SIM-05 test uses boxcar modulator. |
| **B9** | v0.1.0 fixtures drift through bilinear path | AFFECTS SIM-03 | Structural short-circuit (Section 4.3, 7.2); test at `torch.equal`. Inherited from Phase 13 locked design. |
| **B14** | `b_mask` typing and None/empty API ambiguity | AFFECTS B_list acceptance (Section 4.2) | Normalize list ↔ tensor at simulator entry; empty list → None → linear mode. Unit tests for each path. |

### 10.2 Phase 14-specific risks not in project-level PITFALLS.md

**R1: Overlap-sum semantics in `make_epoch_stimulus`.** Sum-vs-override
is a judgment call (Section 2.3). If Phase 15's Pyro model uses these
utilities in a recovery benchmark and overlap events get summed (per
our default), the benchmark ground-truth modulator trace may differ
from what users intended. **Mitigation:** UserWarning at construction
time + explicit docstring note.

**R2: Same-grid-index events in `make_event_stimulus`.** If two events
round to the same `onset_idx` at given `dt`, the current code emits
two breakpoints with the same `times[k]` value; `searchsorted(...,
right=True) - 1` returns the LAST such index, silently discarding the
first event. **Mitigation:** raise `ValueError` at construction time
(Section 3 edge-case recommendation).

**R3: `merge_piecewise_inputs` dtype/device drift.** If drive is CPU
float64 and mod is CUDA float32, the merged input will have
ambiguous placement. **Mitigation:** enforce device/dtype match
from the first input; cast the second if mismatched. Emit warning on
cast.

**R4: `seed` reproducibility with `stimulus_mod`.** The function calls
`torch.manual_seed(seed)` at entry (task_simulator.py:136-137). No
additional random draws happen in bilinear mode — the modulator
merge is deterministic, and `B_list` is provided by the caller (not
sampled). **Conclusion:** no new seed concerns. Same seed → same BOLD
in both modes.

**R5: `CoupledDCMSystem` stability monitor noise in normal SIM-05 run.**
Phase 13 stability monitor logs WARNINGs at `max Re(eig) > 0`. The
SIM-05 fixture uses `B` with elements 0.3 and A diagonal -0.5; the
maximum Re eigenvalue of `A + u_mod * B` with u_mod=1 is bounded by
Gershgorin: `-0.5 + 0.3 + 0.3 = 0.1 > 0`. **The monitor WILL fire.**
This is not a bug but it adds noise to test logs.

**Mitigation:** silence the logger in the test fixture:
```python
@pytest.fixture(autouse=True)
def _silence_stability_logger(caplog):
    caplog.set_level(logging.ERROR, logger="pyro_dcm.stability")
```

OR choose SIM-05 fixture with stricter stability: `B[0, 1, 0] = 0.1,
B[0, 2, 1] = 0.1` gives `max Gershgorin row-sum = -0.5 + 0.1 + 0.1 =
-0.3 < 0` — monitor stays silent. **Recommendation: use the smaller B
values (0.1) to keep test logs clean, preserving SIM-05 goal
(dt-invariance at fixed ground truth).**

**R6: `stimulus_mod` required when `B_list` non-None.** If user passes
`B_list` but forgets `stimulus_mod`, what happens? Options:
- Raise `ValueError` at simulator entry ("stimulus_mod required when
  B_list is non-None").
- Default `stimulus_mod` to all-zero modulator (equivalent to linear
  dynamics with overhead).

**Recommendation: raise `ValueError`.** Defaulting to zero is a
silent-success path for a user error; fail loud.

### 10.3 Risk register summary

| Risk | Severity | Plan | Status |
|------|----------|------|--------|
| B12 stick-blur | MEDIUM | 14-01 | Docstring warning + SIM-05 uses boxcar |
| B9 linear drift | HIGH | 14-02 | Structural short-circuit (Section 4.3) |
| B14 B_list typing | MEDIUM | 14-02 | Normalize list ↔ tensor |
| R1 overlap semantics | LOW | 14-01 | UserWarning at construction |
| R2 same-grid events | LOW | 14-01 | ValueError at construction |
| R3 merge dtype/device | LOW | 14-01 | Auto-cast + warning |
| R5 monitor noise in tests | LOW | 14-02 | Use small B in SIM-05 |
| R6 missing stimulus_mod | LOW | 14-02 | ValueError at entry |

---

## Open Questions

1. **Overlap-sum vs override in `make_epoch_stimulus` (Section 2.3).**
   Recommendation is sum + UserWarning, but there's no locked decision
   from Phase 14 CONTEXT (CONTEXT.md does not exist for Phase 14 at
   research time). **Planner should lock in 14-01 plan.**

2. **Placement of `merge_piecewise_inputs` (Section 5.5).**
   `utils/ode_integrator.py` (recommended) vs private in
   `simulators/task_simulator.py`. No blocker — either works. Planner
   decides.

3. **`n_driving_inputs` default policy (Section 4.4).** Default to
   `C.shape[1]` (recommended) vs require explicit. Different from
   Phase 13's `CoupledDCMSystem` which raises; simulator adds the
   `C` context that makes default defensible. **Planner should
   document the policy divergence in 14-02 plan.**

4. **Should `make_event_stimulus` return a `times_grid` for plotting
   convenience?** (Section 1.4) Not needed structurally; adds
   ~1 line. Tiebreaker = consistency with `make_block_stimulus`
   (which does NOT return times_grid). **Defer unless user asks.**

5. **Should Plan 14-02 add a `test_dt_invariance_linear` regression
   alongside the bilinear SIM-05 test?** (Section 6.6) Zero cost,
   extra coverage. **Recommend include.**

6. **SIM-05 fixture B magnitude: 0.1 (monitor-silent) vs 0.3
   (fires monitor).** Recommendation 0.1 for clean logs (Section
   10.2 R5). **Planner decides.**

7. **Is there a CONTEXT.md for Phase 14?** Not at research time
   (`ls .planning/phases/14-stimulus-utilities-and-bilinear-simulator/`
   returns empty). Recommend running `/gsd:discuss-phase 14` before
   planning if any of the above open questions need user input.

---

## Sources

### Primary (HIGH confidence — direct code/doc reads 2026-04-17)

- `src/pyro_dcm/simulators/task_simulator.py` — 394 lines, full read.
- `src/pyro_dcm/simulators/__init__.py` — full read (exports).
- `src/pyro_dcm/utils/ode_integrator.py` — 242 lines, full read.
- `src/pyro_dcm/forward_models/coupled_system.py` — 373 lines, full read.
- `src/pyro_dcm/forward_models/neural_state.py` — 336 lines, full read.
- `src/pyro_dcm/forward_models/__init__.py` — full read.
- `tests/test_task_simulator.py` — 541 lines, full read.
- `tests/test_task_dcm_model.py` — sampled lines 50-130.
- `.planning/REQUIREMENTS.md` — SIM-01..05 lines 35-39, traceability
  table lines 108-112.
- `.planning/STATE.md` — full read; D1-D5 decisions lines 20-42.
- `.planning/phases/13-bilinear-neural-state/13-RESEARCH.md` — 1038
  lines, full read for Phase 13 patterns and pitfalls.
- `.planning/phases/13-bilinear-neural-state/13-VERIFICATION.md` —
  Phase 13 closure evidence.
- `.planning/research/v0.3.0/PITFALLS.md` — B9, B12, B14 verbatim for
  Section 10.
- `.planning/research/v0.3.0/FEATURES.md` — TS-4, TS-5 specifications
  for stimulus utils and bilinear simulator.
- `.planning/research/v0.3.0/ARCHITECTURE.md` — `simulate_task_dcm`
  change plan lines 40-48; file-by-file scope.

### Secondary (MEDIUM confidence — project-level research citations)

- `.planning/research/v0.3.0/SUMMARY.md` — cross-phase synthesis (not
  re-read; inherited from Phase 13 research).
- SPM12 `spm_fx_fmri.m` (via Pitfalls Section B1) — bilinear sum
  semantics confirming why boxcars match SPM internal behavior.

### Tertiary (LOW confidence — not load-bearing)

- torchdiffeq interpolation semantics for dopri5 at non-grid
  `t_eval` — inferred from Phase 13 research Section 1; not
  re-verified for Phase 14 (not on critical path).

---

## Metadata

**Confidence breakdown:**
- Stimulus utility algorithms (Sec 1, 2): HIGH (pattern mirrors
  `make_block_stimulus`).
- PiecewiseConstantInput compatibility (Sec 3): HIGH (code-read).
- Simulator signature extension (Sec 4): HIGH (additive, CONTEXT-free).
- `merge_piecewise_inputs` (Sec 5): HIGH algorithm; MEDIUM placement.
- dt-invariance test (Sec 6): HIGH solver choice; MEDIUM tolerance
  passing without tuning.
- Linear regression (Sec 7): HIGH structural.
- Test inventory (Sec 8): HIGH (direct grep).
- Wave decomposition (Sec 9): HIGH (dependency-forced).
- Pitfalls (Sec 10): HIGH on cited pitfalls; MEDIUM on Phase-14
  specific risk severities.

**Research date:** 2026-04-17
**Valid until:** 2026-05-17 (30 days; stable deps; no Phase 13
infrastructure churn expected).
