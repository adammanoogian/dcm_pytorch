---
phase: 13-bilinear-neural-state
plan: 03
subsystem: forward_models
tags: [pytorch, bilinear-dcm, BILIN-04, BILIN-05, BILIN-06, stability-monitor, logging, torchdiffeq]

# Dependency graph
requires:
  - phase: 13-bilinear-neural-state
    plan: 01
    provides: compute_effective_A (called from bilinear branch of CoupledDCMSystem.forward)
  - phase: 13-bilinear-neural-state
    plan: 02
    provides: NeuralStateEquation.derivatives bilinear kwargs (consumed indirectly via compute_effective_A)
  - phase: 01-neural-hemodynamic-forward-model
    provides: CoupledDCMSystem (class being extended), BalloonWindkessel, PiecewiseConstantInput
provides:
  - CoupledDCMSystem with keyword-only B, n_driving_inputs, stability_check_every kwargs (BILIN-04 integration gate)
  - _maybe_check_stability method + pyro_dcm.stability named logger (BILIN-05 monitor)
  - NullHandler attached to pyro_dcm root logger in __init__.py (library-logging discipline)
  - tests/test_coupled_system_bilinear.py (5 passing tests: bit-exact, buffer, dtype alignment, ValueError, output distinguishability)
  - tests/test_stability_monitor.py (5 passing tests: BILIN-05 caplog x4 + BILIN-06 3-sigma 500s no-NaN)
affects: [14-bilinear-simulators, 15-bilinear-pyro-model, 16-bilinear-recovery]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Buffer-vs-None hybrid storage: register_buffer('B', B) only when B is not None; self.B = None plain attribute otherwise — avoids None-as-buffer footgun (13-RESEARCH Section 2)"
    - "Device/dtype auto-alignment at construction: B.to(device=A.device, dtype=A.dtype) mitigates the mixed-precision device-drift risk (13-RESEARCH Section 10.3)"
    - "Counter-modulo cadence on RHS evaluations: self._step_counter increments per forward() call; eigenvalue check fires when counter % stability_check_every == 0"
    - "Zero-overhead disable switch: stability_check_every=0 returns from _maybe_check_stability before any eigenvalue computation"
    - "Detached + no_grad eigenvalue computation: torch.linalg.eigvals(A_eff.detach()) inside torch.no_grad() avoids complex-gradient overhead with no SVI-semantic loss"
    - "Library-logging discipline (PEP 282): NullHandler attached to the pyro_dcm root logger in package __init__.py; propagates to pyro_dcm.stability child by hierarchical semantics"
    - "pytest caplog + set_level(logger='pyro_dcm.stability'): named-logger assertion pattern for library-emitted WARNING records"

key-files:
  created:
    - tests/test_coupled_system_bilinear.py
    - tests/test_stability_monitor.py
    - .planning/phases/13-bilinear-neural-state/13-03-SUMMARY.md
  modified:
    - src/pyro_dcm/__init__.py
    - src/pyro_dcm/forward_models/coupled_system.py

key-decisions:
  - "Literal linear short-circuit preserved verbatim at the CoupledDCMSystem level: `dx = self.A @ x + self.C @ u_all` grep-matches exactly one line in coupled_system.py. No refactor into A_eff = A + 0 or a helper function."
  - "ValueError policy for missing n_driving_inputs is strict: raised at __init__ time when B is non-empty. Alternative (infer from C.shape[1]) rejected per CONTEXT Discretion item 4 — explicit is better than implicit for a bilinear-mode disambiguator."
  - "BILIN-06 C matrix is zeros(3, 1) (not zeros(3, 0)): CONTEXT specified u_drive=zeros, and keeping one driving column preserves the standard 1+J input layout and matches the adversarial 'modulator fully dominates' framing."
  - "Counter persists across separate integrate_ode calls on the same instance: self._step_counter is not reset between forward() invocations or between integrate_ode calls. This matters for reuse patterns in Phase 15 where the same CoupledDCMSystem may be called from multiple SVI iterations."
  - "Monitor silencing follows the standard stdlib idiom (logging.getLogger('pyro_dcm.stability').setLevel(logging.ERROR)); BILIN-06 uses caplog.set_level at ERROR to suppress the expected monitor firings during the 500s unstable integration."

patterns-established:
  - "BILIN-04 integration test matrix: 5 tests covering linear bit-exact (torch.equal on full (T, 5N)), buffer presence, dtype auto-alignment, ValueError on missing n_driving_inputs, and bilinear output distinguishability (RMS > 1e-6)."
  - "BILIN-05 monitor test matrix: 4 tests covering fires-on-unstable, silent-on-stable, disabled-at-zero, never-raises — together this exercises every branch of _maybe_check_stability."
  - "BILIN-06 3-sigma fixture is a self-contained reproducible worst case: deterministic tensors (no random state), 500s rk4 integration, single torch.isfinite(sol).all() assertion. Serves as the regression anchor for future changes to the bilinear path."

# Metrics
duration: ~15min
completed: 2026-04-17
---

# Phase 13 Plan 03: CoupledDCMSystem Bilinear Extension + Stability Monitor Summary

**`CoupledDCMSystem` gains keyword-only `B`, `n_driving_inputs`, and `stability_check_every` kwargs. `B is None` preserves the literal v0.2.0 linear expression `self.A @ x + self.C @ u_all` (single grep-verified source line). Bilinear path routes through `compute_effective_A` and invokes `_maybe_check_stability`, which logs WARNING to the `pyro_dcm.stability` named logger when `max Re(eig(A_eff)) > 0` and never raises (D4). BILIN-04 acceptance gate green (44/44 pre-existing task-DCM tests), BILIN-05 caplog coverage 4/4, BILIN-06 3-sigma 500s no-NaN 1/1.**

## Performance

- **Duration:** ~15 min (implementation + verification; the 245s BILIN-04 regression sweep on `test_ode_integrator.py`/`test_task_simulator.py`/`test_task_dcm_model.py` was the dominant cost, followed by the 26s BILIN-06 500s rk4 integration)
- **Started:** 2026-04-17T21:00:34Z
- **Completed:** 2026-04-17T21:15:40Z
- **Tasks:** 5 (1 logger setup, 1 source extension, 2 new test files, 1 metadata commit)
- **Files touched:** 4 (2 modified, 2 created) + 1 SUMMARY

## Accomplishments

- `src/pyro_dcm/__init__.py`: Attached `NullHandler` to the `pyro_dcm` package root logger per the stdlib "Configuring Logging for a Library" HOWTO (PEP 282). Uses an underscore-prefixed `_logging` alias that is not added to `__all__` (not a public export). Propagates to the `pyro_dcm.stability` child via standard hierarchical logger semantics.
- `src/pyro_dcm/forward_models/coupled_system.py`: Module-level `_STABILITY_LOGGER = logging.getLogger("pyro_dcm.stability")` added at the top of the file (single grep-matching line). `CoupledDCMSystem.__init__` gained keyword-only `B: Tensor | None = None`, `n_driving_inputs: int | None = None`, and `stability_check_every: int = 10`. `B` is registered as a buffer only when supplied and is auto-aligned to `A.device` / `A.dtype` at construction. `ValueError` is raised when `B` is non-empty and `n_driving_inputs is None` (explicit-split policy).
- `CoupledDCMSystem.forward` now branches on `self.B`: when `None` or empty-J, executes the literal `dx = self.A @ x + self.C @ u_all` expression (single grep-matching line); when non-empty, slices `u_all` into `u_drive = u_all[:n_driving_inputs]` and `u_mod = u_all[n_driving_inputs:]`, computes `A_eff = compute_effective_A(self.A, self.B, u_mod)`, and routes `dx = A_eff @ x + self.C @ u_drive`, then invokes `_maybe_check_stability(t, A_eff, u_mod)`.
- `_maybe_check_stability` method added. Counter-modulo cadence on RHS evaluations (one `self._step_counter` tick per `forward()` call). Early-returns when `stability_check_every <= 0` or when the counter is not on-cadence. The eigenvalue computation is wrapped in `torch.no_grad()` with `A_eff.detach()`, avoiding complex-gradient overhead. When `max Re(eig(A_eff)) > 0`, emits a WARNING record on `pyro_dcm.stability` with the CONTEXT-locked format `"Stability warning at t=%.2fs: max Re(eig(A_eff))=%+.3f; ||B·u_mod||_F=%.3f"`. Never raises (D4). `stability_check_every=0` disables entirely (zero overhead).
- `tests/test_coupled_system_bilinear.py` created with 5 passing tests in `TestCoupledDCMSystemBilinear`:
  - `test_linear_kwarg_none_matches_no_kwarg_bit_exact` — `torch.equal` on the full `(T, 5N)` solution tensor between no-kwarg and explicit-`B=None` paths.
  - `test_b_registered_as_buffer` — `'B' in dict(system.named_buffers())` and `'B' not in dict(system.named_parameters())`, shape `(1, 2, 2)` preserved.
  - `test_b_moved_to_A_device_dtype` — `float32`-supplied `B` is promoted to `A.dtype=float64` (and device-matched).
  - `test_missing_n_driving_inputs_raises` — `pytest.raises(ValueError, match="n_driving_inputs")` on non-empty `B` with no `n_driving_inputs`.
  - `test_bilinear_output_differs_from_linear` — BOLD RMS difference exceeds `1e-6` between the linear and bilinear integrations under sustained `u_mod`, confirming end-to-end path exercise.
- `tests/test_stability_monitor.py` created with 5 passing tests across 2 classes:
  - `TestStabilityMonitor` (BILIN-05, 4 tests): unstable fires WARNING, stable emits nothing, `stability_check_every=0` disables, monitor never raises under unstable dynamics.
  - `TestThreeSigmaWorstCase` (BILIN-06, 1 test): deterministic `N=3` fixture with `A = parameterize_A(zeros)` (diagonal `-0.5`), `B: (1, 3, 3)` off-diagonal `3.0` diagonal `0`, `C = zeros(3, 1)`, `u_drive = 0`, `u_mod = 1` sustained, `rk4` at `dt = 0.1` for `500 s` — single `torch.isfinite(sol).all()` assertion.
- Docstrings on `CoupledDCMSystem` class + `__init__` + `forward` + `_maybe_check_stability` updated to document the v0.3.0 kwargs, the monitor cadence semantics (RHS evaluations, not ODE steps, with the rk4 ≈ 2.5 ODE-steps annotation), the ValueError contract, and the stdlib silencing idiom.

## Task Commits

Each task committed atomically; no squashing:

1. **Task 1: NullHandler on pyro_dcm root logger** — `3e2ffa9` (`feat(13-03): add pyro_dcm.stability logger NullHandler`)
2. **Task 2: CoupledDCMSystem bilinear + stability monitor** — `956e1de` (`feat(13-03): extend CoupledDCMSystem with bilinear path + stability monitor`)
3. **Task 3: test_coupled_system_bilinear.py** — `5988dbd` (`test(13-03): add test_coupled_system_bilinear.py`)
4. **Task 4: test_stability_monitor.py (incl. BILIN-06)** — `ae9a265` (`test(13-03): add test_stability_monitor.py with BILIN-06 3-sigma 500s test`)
5. **Task 5: Plan metadata commit** — pending at the time of this summary (final `docs(13-03): complete CoupledDCMSystem + stability monitor plan` commit).

## Files Created/Modified

- `src/pyro_dcm/__init__.py` — added `import logging as _logging`, NullHandler attachment block (10 added lines, 0 removed). No changes to `__all__`.
- `src/pyro_dcm/forward_models/coupled_system.py` — 251 insertions, 28 deletions. Module docstring extended with a v0.3.0 section. Added module-level `logging` import, extended `NeuralStateEquation` import to also pull `compute_effective_A`, and the `_STABILITY_LOGGER` module-level binding. Class docstring, `__init__`, and `forward` docstrings expanded. New `_maybe_check_stability` method.
- `tests/test_coupled_system_bilinear.py` — new file, 194 lines, 5 tests in `TestCoupledDCMSystemBilinear`.
- `tests/test_stability_monitor.py` — new file, 238 lines, 5 tests across `TestStabilityMonitor` (4 BILIN-05) + `TestThreeSigmaWorstCase` (1 BILIN-06).

## Verification Evidence

- `pytest tests/test_coupled_system_bilinear.py -v` → 5/5 passing in 9.02s.
- `pytest tests/test_stability_monitor.py -v` → 5/5 passing in 26.44s (BILIN-06 dominates at ~25s for the 5000-step rk4 integration).
- `pytest tests/test_bilinear_utils.py tests/test_linear_invariance.py tests/test_coupled_system_bilinear.py tests/test_stability_monitor.py tests/test_neural_state.py -v` → **34/34 passing in 30.92s** (Phase 13 full suite + neural_state regression).
- **BILIN-04 acceptance gate (mandatory):** `pytest tests/test_ode_integrator.py tests/test_task_simulator.py tests/test_task_dcm_model.py -x -q` → **44/44 passing in 245.43s**. Zero source-line modifications to pre-existing tests.
- `grep -n "dx = self.A @ x + self.C @ u_all" src/pyro_dcm/forward_models/coupled_system.py` → exactly one match on line 291.
- `grep -n "_STABILITY_LOGGER = logging.getLogger" src/pyro_dcm/forward_models/coupled_system.py` → exactly one match on line 50.
- `grep -n "NullHandler" src/pyro_dcm/__init__.py` → two matches (the descriptive comment on line 7 and the attachment call on line 13). Both expected.
- `python -c "import logging; logging.getLogger('pyro_dcm.stability').getEffectiveLevel()"` → `30` (WARNING), confirming the hierarchy resolves correctly after the NullHandler is attached.
- Smoke test (linear mode + bilinear mode) from the plan verification — both paths finite, correct shapes.

## Decisions Made

None beyond the CONTEXT.md / RESEARCH.md / planner's-discretion items already resolved in the plan:

- **Buffer-vs-None hybrid storage** (13-RESEARCH Section 2): `register_buffer('B', B)` only when `B is not None`; `self.B = None` plain attribute otherwise. Matches the research-recommended pattern for optional tensors on `nn.Module`.
- **Counter persistence across `integrate_ode` calls**: `self._step_counter` is not reset between invocations. This is the zero-surprise behavior for callers who reuse a single `CoupledDCMSystem` instance across SVI iterations in Phase 15 — the counter is a diagnostic, not a correctness boundary.
- **BILIN-06 C-matrix shape**: `zeros(3, 1)` rather than `zeros(3, 0)`. The CONTEXT specification "`u_driving = zeros`" is satisfied by a single zero-valued driving column, which preserves the standard `1 + J` concatenated input layout and keeps the fixture consistent with realistic call sites.
- **Monitor silencing in BILIN-06**: `caplog.set_level(logging.ERROR, logger='pyro_dcm.stability')` rather than a global `disable`. This is the documented stdlib silencing idiom and doubles as a test that the silencing mechanism works as advertised.

## Deviations from Plan

None — plan executed exactly as written.

## Authentication Gates

None — no external service authentication required for this plan.

## Issues Encountered

None.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- **Phase 14 ready:** `CoupledDCMSystem(A, C, input_fn, B=..., n_driving_inputs=...)` is the public integration surface that the Phase 14 `simulate_task_dcm_bilinear` simulator will call. The auto-alignment, buffer registration, and input-split policy are all in place; Phase 14 only needs to build the widened `PiecewiseConstantInput` (driving + modulator columns) and call into `CoupledDCMSystem` without touching this plan's code.
- **Phase 15 ready:** the Pyro generative model (`task_dcm_model_bilinear`) can now sample `B_free`, apply `parameterize_B`, and feed the resulting `(J, N, N)` tensor into `CoupledDCMSystem` with `stability_check_every` set per-call (e.g. `0` during warmup, `10` during the main SVI loop). The monitor's never-raises contract (D4) is a hard requirement of Phase 15 — confirmed green by `test_monitor_never_raises`.
- **Phase 16 ready:** the BILIN-06 3-sigma test is the canonical worst-case stability anchor. Phase 16 RECOV-08 runtime benchmarks can trust that the bilinear forward path is numerically robust at the 3-sigma prior-tail boundary.
- **BILIN-04 acceptance:** closed. `CoupledDCMSystem` gains bilinear kwargs without a single source-line modification to pre-existing tests, and the literal linear short-circuit is anchored at both the `NeuralStateEquation` level (Plan 13-02) and the `CoupledDCMSystem` level (this plan).
- **BILIN-05 acceptance:** closed. Monitor emits WARNING on unstable, silent on stable, disables at zero, never raises. `pyro_dcm.stability` is the canonical named logger.
- **BILIN-06 acceptance:** closed. 3-sigma worst-case B with sustained `u_mod=1.0` integrates 500s at rk4 `dt=0.1` with zero non-finite entries across the full 5N trajectory.
- **Phase 13 status:** 3/4 plans complete (13-01, 13-02, 13-03). Plan 13-04 (end-to-end bilinear recovery smoke-test + research-pass addendum) remains; SUMMARY of 13-04 landed prior to this plan per the execution-order swap noted in STATE.md. With this plan complete, Phase 13 has delivered every BILIN requirement (01-07) and is ready to close.
- **No blockers.**

---
*Phase: 13-bilinear-neural-state*
*Completed: 2026-04-17*
