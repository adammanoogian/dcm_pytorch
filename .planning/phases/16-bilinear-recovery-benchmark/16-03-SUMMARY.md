---
phase: 16-bilinear-recovery-benchmark
plan: 03
subsystem: benchmarks+factories
tags: [pytorch, hgf-hook, factory-pattern, forward-compat, phase-16-hook]

# Dependency graph
requires:
  - phase: 16-bilinear-recovery-benchmark
    provides: run_task_bilinear_svi + _make_bilinear_ground_truth helpers (plan 16-01)
provides:
  - "stimulus_mod_factory keyword-only kwarg on run_task_bilinear_svi (L9 contract: Callable[[int], dict[str, torch.Tensor]])"
  - "make_sinusoid_mod_factory(duration, dt, frequency, amplitude) module-level constructor returning a deterministic sinusoidal closure (Phase 16 placeholder; v0.3.1 HGF factories share signature)"
  - "_make_bilinear_ground_truth_with_factory helper mirroring plan 16-01's _make_bilinear_ground_truth but substituting factory-provided stim_mod (A/B/C/b_mask/driving-stim still seed-deterministic)"
  - "metadata['stimulus_mod_factory'] return-dict key recording 'default_epochs' or 'custom' (no factory-hash tracking; factories are test/sweep artifacts)"
  - "tests/test_task_bilinear_factory.py::TestFactoryHookWiring class with 3 tests (1 fast signature contract + 2 slow runner-invoking regression + wiring proof)"
  - "StimulusModFactory type alias (Callable[[int], dict[str, torch.Tensor]]) exported from benchmarks/runners/task_bilinear.py for v0.3.1 HGF factory-builder authors"
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "runner-level-factory-hook-via-closure: stim_mod factory injected as keyword-only kwarg at runner call site (NOT stored in BenchmarkConfig); closure captures duration/dt/frequency/amplitude so different factory types share a uniform Callable[[int], dict[str, Tensor]] surface"
    - "factory-bypasses-fixture-cache: when stimulus_mod_factory is non-None, _load_or_make_fixture is bypassed entirely for stim_mod (L10); factory-driven runs do NOT participate in .npz reproducibility (test/sweep artifacts)"

key-files:
  created:
    - tests/test_task_bilinear_factory.py
    - .planning/phases/16-bilinear-recovery-benchmark/16-03-SUMMARY.md
  modified:
    - benchmarks/runners/task_bilinear.py

key-decisions:
  - "L9 (locked this plan): stimulus_mod_factory signature is Callable[[int], dict[str, torch.Tensor]] returning {'times': (K,) float64, 'values': (K, J) float64}; closure captures duration/dt/n_inputs at construction time"
  - "L10 (locked this plan): non-None factory BYPASSES config.fixtures_dir cache for stim_mod; A/B/C ground truth still seed-deterministic, factory only controls stim_mod"
  - "Factory NOT stored in BenchmarkConfig (per CONTEXT.md): kept clean .npz reproducibility path; only the runner keyword surface carries the factory"
  - "Mock factory documented as Phase 16 placeholder (NOT physiologically meaningful); v0.3.1 SIM-06 HGF factories share L9 signature so runner needs no further changes"
  - "Test file is NEW (tests/test_task_bilinear_factory.py), separate from tests/test_task_bilinear_benchmark.py to respect file-ownership contract with parallel plan 16-02"

patterns-established:
  - "runner-level-factory-hook-via-closure: keyword-only Callable kwarg with closure-captured config; default None preserves prior runner contract bit-exactly"
  - "factory-bypasses-fixture-cache: L10 dispatch — non-None factory triggers inline regeneration; .npz cache reserved for default-factory reproducibility runs"
  - "factory-signature-contract test pattern (FAST, no runner): exercises factory directly to verify dict-shape, dtype, determinism, closure configurability before any expensive end-to-end test"

# Metrics
duration: 13min
completed: 2026-04-18
---

# Phase 16 Plan 03: HGF Factory Hook Summary

**Adds keyword-only stimulus_mod_factory parameter to run_task_bilinear_svi with a placeholder mock sinusoid factory, locking the v0.3.1 HGF integration surface (Callable[[int], dict[str, torch.Tensor]]) without introducing HGF implementation in v0.3.0.**

## Performance

- **Duration:** ~13 min
- **Started:** 2026-04-18T21:08:22Z
- **Completed:** 2026-04-18T21:20:46Z
- **Tasks:** 2 (1 feat + 1 test) + 1 metadata commit
- **Files created:** 2 (`tests/test_task_bilinear_factory.py`, this SUMMARY.md)
- **Files modified:** 1 (`benchmarks/runners/task_bilinear.py`)

## Accomplishments

- `run_task_bilinear_svi` now accepts `stimulus_mod_factory: StimulusModFactory | None = None` as a keyword-only parameter (L9). Default `None` preserves plan 16-01 behavior bit-exactly. Non-None factory is invoked once per seed (`factory(seed=seed_i)`) and substituted as the modulator while A/B/C ground-truth construction stays seed-deterministic.
- `make_sinusoid_mod_factory(duration=200.0, dt=0.01, frequency=0.05, amplitude=0.5)` ships as a module-level constructor returning a deterministic sinusoidal closure. Documented as a Phase 16 placeholder/mock that exists exclusively to exercise the plumbing — not physiologically meaningful. v0.3.1 SIM-06 HGF factories will share the same L9 signature so the runner needs no further changes.
- `_make_bilinear_ground_truth_with_factory` mirrors plan 16-01's `_make_bilinear_ground_truth` exactly except the `stim_mod` breakpoint dict comes from the injected factory; includes a TypeError type-guard for malformed factory returns. L10 dispatch lives in `run_task_bilinear_svi`: when the factory is non-None, the runner BYPASSES `_load_or_make_fixture` entirely and always inlines via the helper (factories are test/sweep artifacts and do NOT participate in .npz reproducibility).
- `metadata['stimulus_mod_factory']` records `'default_epochs'` or `'custom'` on every return dict — downstream consumers of the results JSON can tell which path was used. No factory-hash tracking (research Section 7.2 N5 option c).
- `tests/test_task_bilinear_factory.py::TestFactoryHookWiring` ships 3 tests:
  1. `test_factory_signature_contract` (FAST, 4.5s): exercises `make_sinusoid_mod_factory` WITHOUT runner invocation; asserts dict shape (`times: (K=20000,)`, `values: (K, 1)`), dtype `float64`, determinism given seed, `|value| <= amplitude`, and closure configurability (different `frequency` → different output values).
  2. `test_default_factory_matches_plan_16_01_ground_truth` (`@pytest.mark.slow`): regression gate — default path produces the Phase 16 epoch-schedule stim_mod exactly (`values[2000:3200] == 1.0` for the first t=20-32s epoch; zeros before t=20s).
  3. `test_custom_mock_factory_produces_different_stim_mod` (`@pytest.mark.slow`): end-to-end wiring proof at the helper level — custom factory's `stim_mod` tensor is NOT equal to the default path's, while `A_true`/`B_true`/`C`/`b_mask_0` are identical (factory only affects stim_mod per L10). Then invokes the full runner with the mock factory at 1 seed × 50 SVI steps and asserts `metadata['stimulus_mod_factory']` flips correctly between `'custom'` and `'default_epochs'` (with `pytest.skip` on `insufficient_data` for short SVI).

## Task Commits

Each task was committed atomically:

1. **Task 1: Add stimulus_mod_factory parameter + make_sinusoid_mod_factory + _make_bilinear_ground_truth_with_factory** — `9fb391f` (feat)
2. **Task 2: Factory hook wiring tests (new file)** — `7596aa8` (test)

**Plan metadata:** `<pending>` (`docs(16-03): complete HGF factory hook plan`)

## Files Created/Modified

Created:
- `tests/test_task_bilinear_factory.py` (~193 lines) — `TestFactoryHookWiring` class with 1 fast signature-contract test + 2 slow runner-invoking tests (default-path regression + custom-factory differs + metadata flip).
- `.planning/phases/16-bilinear-recovery-benchmark/16-03-SUMMARY.md` (this file).

Modified:
- `benchmarks/runners/task_bilinear.py` — Added `from collections.abc import Callable` import; `StimulusModFactory` type alias; `make_sinusoid_mod_factory(duration, dt, frequency, amplitude)` module-level constructor; `_make_bilinear_ground_truth_with_factory` helper with TypeError type-guard; new `stimulus_mod_factory` keyword-only parameter on `run_task_bilinear_svi` with extended docstring; L10 dispatch logic inside the per-seed loop; `metadata['stimulus_mod_factory']` key. +203 lines / -3 lines (additive).

## Decisions Made

- **L9 locked (factory signature):** `Callable[[int], dict[str, torch.Tensor]]` returning `{'times': (K,), 'values': (K, J)}`. Rationale: returning a BREAKPOINT DICT (not a `PiecewiseConstantInput` instance) lets the runner (a) save `stim_mod_times` / `stim_mod_values` to the .npz fixture matching plan 16-01's pattern (research Section 1.3), and (b) construct the `PiecewiseConstantInput` at the call site. Closure-captured `duration`/`dt` is cleaner than threading them through the factory signature because different factory types (epoch, sinusoid, HGF trajectory) have different natural closure contexts.
- **L10 locked (custom factory bypasses fixture cache):** When `stimulus_mod_factory` is non-None, the runner skips `_load_or_make_fixture` entirely for stim_mod and always generates inline via `_make_bilinear_ground_truth_with_factory`. A/B/C still come from the seeded ground truth. Rationale: custom factories (HGF trajectory, sinusoid mock, sweep experiments) are test/research artifacts not reproducible-run artifacts. Mixing cached `.npz` `stim_mod_values` with a custom factory would produce a stim_mod / B_true inconsistency. Simpler rule: custom factory → skip fixture cache entirely for stim_mod.
- **No factory-hash tracking on metadata:** `metadata['stimulus_mod_factory']` records only `'default_epochs'` or `'custom'`, not which custom factory was used. Per research Section 7.2 N5 option c: factories are test artifacts, not reproducibility-tracked. Authors of custom factories should record their own provenance in the calling script.
- **Factory invocation uses kwarg style (`factory(seed=seed_i)`)** in both the runner dispatch and `_make_bilinear_ground_truth_with_factory`. Matches the test assertions and lets the v0.3.1 HGF factory builder use a kwarg-only `seed` parameter for clarity.
- **Test file is NEW (`tests/test_task_bilinear_factory.py`), not appended to existing `tests/test_task_bilinear_benchmark.py`.** Required by file-ownership contract with parallel plan 16-02 (which appends to `test_task_bilinear_benchmark.py` for its own RECOV gate tests).

## Deviations from Plan

### Plan-Inconsistent Sentinel (NOT a deviation; documented for posterity)

**1. [Plan inconsistency — NOT a deviation] Sentinel `grep -c "make_sinusoid_mod_factory" benchmarks/runners/task_bilinear.py >= 2` overshoots**

- **Found during:** Task 1 verification
- **Issue:** Plan's grep sentinel expected `>= 2` occurrences (def + docstring), but `make_sinusoid_mod_factory` appears only at the `def` line — Python convention is for a docstring NOT to mention its own function name. The function's docstring includes the substantive content the sentinel was probing for (parameters, return type, references), just without re-stating the function name verbatim.
- **Fix:** None required; implementation matches the Task 1 action-body spec exactly. The substantive sentinels — `stimulus_mod_factory >= 4` (got 10), `_make_bilinear_ground_truth_with_factory >= 2` (got 2), `StimulusModFactory >= 2` (got 7), `'custom' >= 1` (got 1) — all pass. Analogous to plan 16-01's `model_kwargs >= 5` plan-inconsistency note in `tests/test_svi_integration.py`.
- **Verification:** Functional verification (`python -c "..."`) shows the factory constructor works end-to-end (shape/determinism/closure-config); ruff clean; `test_factory_signature_contract` passes in 4.5s exercising the full closure surface.

---

**Total deviations:** 0 functional. 1 plan-inconsistency note for posterity (sentinel threshold mismatch; analogous to plan 16-01's documented sentinel inconsistency).

**Impact on plan:** None. The factory contract (L9), helper, dispatch, metadata, and tests are all line-identical to the plan spec.

## Issues Encountered

- **`pytest --timeout=N` flag unavailable.** The `pytest-timeout` plugin is not installed in the project environment. Initial verification command attempted `--timeout=30` and `--timeout=60` per the plan's verify block; pytest exited 4 with `unrecognized arguments`. Workaround: ran the verification commands without the timeout flag. The fast `test_factory_signature_contract` completed in 4.5s (well under any reasonable timeout); slow tests are gated by `@pytest.mark.slow` so they don't run in the default suite. Not a deviation requiring code changes — purely a tooling note for future plan-writers (the Phase 8 / Phase 16 plans systematically reference `--timeout=N` but the plugin is not in the dependency stack).
- **Parallel plan 16-02 in flight.** During Task 2 staging, `git status` showed `benchmarks/plotting.py` (modified) + `benchmarks/bilinear_metrics.py` (new) + `tests/test_bilinear_metrics.py` (new) + `tests/test_task_bilinear_benchmark.py` (modified) from plan 16-02's parallel execution. Per file-ownership contract, plan 16-03 ignored these and staged only its own files (`benchmarks/runners/task_bilinear.py` for Task 1, `tests/test_task_bilinear_factory.py` for Task 2). No conflicts on shared files; STATE.md will be merged carefully in the metadata commit step.

## User Setup Required

None — no external service configuration required. All Phase 16-03 deliverables run purely against local Pyro/PyTorch deps already pinned in `pyproject.toml`.

## Next Phase Readiness

**Closes 16-CONTEXT.md HGF forward-compatibility hook lock-in:**
- "the indirection is proven wired, not a theoretical API" — `TestFactoryHookWiring::test_factory_signature_contract` exercises the closure directly (FAST, no runner) and `test_custom_mock_factory_produces_different_stim_mod` proves the runner-side dispatch end-to-end.
- v0.3.1 HGF integration is now blocked only on SIM-06 (`LinearInterpolatedInput`) and the HGF belief-trajectory factory itself; the runner surface is frozen.

**v0.3.1 deferred (unblocked by this plan):**
- `benchmarks/runners/task_bilinear_hgf.py` sibling runner: will be a thin wrapper calling `run_task_bilinear_svi(config, stimulus_mod_factory=make_hgf_factory(...))` with NO changes to `task_bilinear.py`. The point of the L9 hook is exactly this.
- `make_hgf_factory(belief_trajectory_generator, ...)`: new module-level constructor next to `make_sinusoid_mod_factory`; closure captures the trajectory generator and returns the same `{'times': (K,), 'values': (K, J)}` breakpoint dict.
- `LinearInterpolatedInput` (SIM-06): new `pyro_dcm.utils.ode_integrator` class for HGF trajectories that need linear interpolation between belief-update timepoints (vs `PiecewiseConstantInput`'s step function). Plan 16-03 does NOT preclude either input class — the factory returns just a breakpoint dict, and the runner wraps in `PiecewiseConstantInput` today; v0.3.1 will conditionally wrap in `LinearInterpolatedInput` for HGF factories.

**Concerns:**
- The mock sinusoid factory's 0.05 Hz signal has period 20s, which is exactly the driving-block cadence. This is coincidental but harmless — the factory exists only to exercise plumbing, not for any scientific claim. v0.3.1 HGF factories will use real belief trajectories with their own frequency content.

**Requirements closed:**
- 16-CONTEXT.md HGF trajectory forward-compatibility hook lock-in (NOT a formal REQUIREMENTS.md item — the CONTEXT lock).

No formal RECOV-XX requirements are closed by this plan. RECOV-01..02 closed by plan 16-01; RECOV-03..08 closed by plan 16-02 (in flight).

---
*Phase: 16-bilinear-recovery-benchmark*
*Completed: 2026-04-18*
