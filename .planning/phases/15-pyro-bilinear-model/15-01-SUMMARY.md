---
phase: 15-pyro-bilinear-model
plan: 01
subsystem: models
tags: [pytorch, pyro, bilinear-dcm, task-dcm, svi, generative-model]

# Dependency graph
requires:
  - phase: 13-bilinear-neural-state
    provides: parameterize_B (stacked (J,N,N) path with DeprecationWarning on non-zero diagonal) + CoupledDCMSystem bilinear gate (coupled_system.py:287-300) + linear short-circuit literal expression (coupled_system.py:287-291)
  - phase: 14-stimulus-utilities-and-bilinear-simulator
    provides: merge_piecewise_inputs + make_epoch_stimulus (Pitfall B12 preferred boxcar primitive) + PiecewiseConstantInput breakpoint-dict contract
provides:
  - task_dcm_model extended signature with keyword-only kwargs b_masks (list[Tensor] | None = None) and stim_mod (PiecewiseConstantInput | None = None) after the * sentinel
  - Module-level constant B_PRIOR_VARIANCE = 1.0 with D1 docstring (SPM12 spm_dcm_fmri_priors.m pC.B = B one-state match; corrects factually wrong YAML '1/16' claim audited in PITFALLS.md B8)
  - Private module-level helper _validate_bilinear_args(b_masks, stim_mod, N) raising ValueError on None stim_mod / mask shape mismatch / J mismatch and TypeError on stim_mod lacking .values attr
  - Structural linear short-circuit preserving pre-Phase-15 linear trace bit-exactness: CoupledDCMSystem called with no B= kwarg when b_masks is None or [] (empty-list normalized to None at function entry)
  - Bilinear branch: per-modulator pyro.sample(f"B_free_{j}", Normal(0, 1.0).to_event(2)) Python loop (NO pyro.plate) + single stacked parameterize_B(B_free_stacked, b_mask_stacked) + pyro.deterministic("B", B_stacked) + CoupledDCMSystem(B=B_stacked, n_driving_inputs=c_mask.shape[1])
  - Merged driving + modulator PiecewiseConstantInput via merge_piecewise_inputs(drive, mod) widening to (M + J) columns for the CoupledDCMSystem bilinear gate
  - NaN-safe predicted_bold guard ported from amortized_wrappers.py:143-145 (catches torch.isinf in addition to isnan; applied in BOTH linear and bilinear branches for defensive symmetry)
  - tests/test_task_dcm_model.py gains TestBilinearStructure (8 tests) + TestBilinearSVI (1 test); all 10 pre-Phase-15 tests unchanged and passing
  - New task_bilinear_data fixture using make_epoch_stimulus (single 10s epoch at t=10s, amp 1.0) for J=1 bilinear coverage
affects: [15-02-plan, 15-03-plan, 16-recovery-benchmark]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - per-modulator-sample-loop
    - pyro-deterministic-guarded-by-branch
    - nan-safe-likelihood-guard
    - bilinear-validation-helper
    - keyword-only-additive-api-extension

key-files:
  created: []
  modified:
    - src/pyro_dcm/models/task_dcm_model.py
    - tests/test_task_dcm_model.py

key-decisions:
  - "L1 applied: B-site shape is full (N, N) Normal with .to_event(2), NOT a flat vector. Mirrors A_free / C sampling; required for AutoLowRankMVN single-vector concatenation compatibility in Plan 15-02."
  - "L2 applied: Default init_scale for bilinear SVI tests = 0.005 (half the linear 0.01), passed explicitly to AutoNormal(task_dcm_model, init_scale=0.005). NOT auto-switched inside create_guide (additive-discipline); downstream callers in Phase 16 pass explicitly."
  - "L3 applied: pyro.deterministic('B', B_stacked) emitted ONLY in the bilinear branch. Linear-mode trace (b_masks=None or []) must NOT contain a 'B' deterministic site - verified by test_linear_reduction_when_b_masks_none asserting site_names == {A_free, C, noise_prec, obs, A, predicted_bold} exactly."
  - "Empty-list normalization: b_masks=[] is converted to None at function entry (`if b_masks is not None and len(b_masks) == 0: b_masks = None`) so J=0 input takes the linear short-circuit as a single code path (MODEL-04 edge case)."

# Metrics
requirements-closed: [MODEL-01, MODEL-02, MODEL-03, MODEL-04]
duration: ~60min
completed: 2026-04-18
---

# Phase 15 Plan 01: task_dcm_model Bilinear Extension Summary

**`task_dcm_model` extended with the Friston 2003 bilinear B path behind keyword-only `b_masks` + `stim_mod` kwargs, preserving bit-exact linear trace structure when the kwargs are omitted (`b_masks is None` or `b_masks == []`) via a structural short-circuit that calls `CoupledDCMSystem` with no `B=` kwarg. Bilinear branch samples per-modulator `B_free_j ~ Normal(0, 1.0).to_event(2)` in a Python loop (no `pyro.plate`), applies `parameterize_B` once on the stacked `(J, N, N)` tensors, and emits `pyro.deterministic("B", B_stacked)` guarded inside the branch (L3). Module-level `B_PRIOR_VARIANCE = 1.0` constant (D1, SPM12 one-state match). NaN-safe `predicted_bold` guard ported from `amortized_wrappers.py:143-145` prevents Gershgorin-blowup early-SVI samples from halting training. New 9-test coverage (8 structure + 1 SVI smoke) + all 10 pre-Phase-15 tests unchanged = 19 total green.**

## Performance

- **Duration:** ~60 min (plan load -> 3 atomic commits -> regression tests)
- **Started:** 2026-04-18T13:31:00Z (phase branch baseline)
- **First task commit:** 2026-04-18T14:17:33+02:00 (23a5591)
- **Last task commit:** 2026-04-18T14:31:19+02:00 (807fb46)
- **Tasks:** 3 atomic commits (1 feat + 2 test)
- **Files modified:** 2 (src/pyro_dcm/models/task_dcm_model.py + tests/test_task_dcm_model.py)
- **Src LoC delta:** +226 / -9 in `task_dcm_model.py`
- **Test LoC delta:** +424 in `test_task_dcm_model.py` (fixture + 8 struct + 1 SVI)
- **Bilinear SVI smoke runtime:** ~29.8s (well under the 75s soft budget)
- **Full `test_task_dcm_model.py` runtime:** 74.16s (19 tests)

## Accomplishments

- Closed MODEL-01, MODEL-02, MODEL-03 (BOTH halves — Phase 13 source-side at `test_bilinear_utils.py` + Plan 15-01 stacked call-site at `test_bilinear_deprecation_warning_on_stacked_nonzero_diag`), and MODEL-04 (linear short-circuit + edge cases + SVI convergence gate).
- Linear short-circuit is **structural**, not numerical: `CoupledDCMSystem(A, C, stimulus)` with NO `B=` kwarg inherits the Phase 13 literal-expression gate at `coupled_system.py:287-291`. All 10 pre-Phase-15 tests in `TestModelStructure` / `TestNumericalStability` / `TestSVI` continue to pass unchanged — `test_model_trace_has_expected_sites` confirms the linear site set `{A_free, C, noise_prec, obs, A, predicted_bold}` is preserved byte-for-byte (L3 guard verified).
- Bilinear branch reuses Phase 13 `parameterize_B` on the stacked `(J, N, N)` tensors (one call, not per-modulator) and Phase 14 `merge_piecewise_inputs` for the widened `(M + J)`-column `PiecewiseConstantInput` fed to `CoupledDCMSystem(B=B_stacked, n_driving_inputs=c_mask.shape[1])`. SC-4 stacked-path DeprecationWarning coverage gap closed by a new test wrapping a non-zero diagonal `b_masks` with `pytest.warns(DeprecationWarning, match="non-zero diagonal")`.
- 3-region, J=1, 40-step `AutoNormal(init_scale=0.005)` SVI smoke test (`test_bilinear_svi_smoke_3region_converges`) ran in ~29.8s (<75s soft budget; Pitfall B10 3-6x slowdown assumption holds). Convergence verified via `mean(last_10) < mean(first_10)`; all 40 steps produced finite ELBO (NaN-safe guard + L2 `init_scale` together held). `caplog.set_level(logging.ERROR, logger="pyro_dcm.stability")` silences D4 WARN spam per R6.

## Task Commits

1. **Task 1: Extend task_dcm_model with bilinear B path** - `23a5591` (feat)
2. **Task 2: Bilinear structure tests** - `cd405d2` (test)
3. **Task 3: Bilinear SVI smoke test** - `807fb46` (test)

## Files Created/Modified

- `src/pyro_dcm/models/task_dcm_model.py` (modified, +226/-9):
  - New module-level import widening: `parameterize_B` added to `neural_state` import, `PiecewiseConstantInput` + `merge_piecewise_inputs` added to `ode_integrator` import (alphabetized block).
  - New module-level constant `B_PRIOR_VARIANCE: float = 1.0` with NumPy-style docstring citing D1 (SPM12 `spm_dcm_fmri_priors.m` pC.B = B one-state) and explicitly noting the YAML '1/16' claim was audited as factually wrong (PITFALLS.md B8).
  - New module-private helper `_validate_bilinear_args(b_masks, stim_mod, N)` — raises `ValueError` on None `stim_mod` / per-index mask shape mismatch / len-vs-J mismatch, raises `TypeError` if `stim_mod` lacks `.values`. Error messages include expected-vs-actual values per CLAUDE.md convention.
  - Extended `task_dcm_model` signature: keyword-only `b_masks: list[torch.Tensor] | None = None` and `stim_mod: object | None = None` after `*` sentinel. All 7 pre-existing positional parameters untouched.
  - Expanded docstring: two new Parameters entries, new "Linear short-circuit (MODEL-04)" + "Bilinear mode (MODEL-01)" + "NaN-safe BOLD guard" Notes sections, SPM12 `spm_dcm_fmri_priors.m` in References.
  - Body changes (five structural insertions):
    1. After `N = a_mask.shape[0]`: empty-list normalization `if b_masks is not None and len(b_masks) == 0: b_masks = None`.
    2. After `M = c_mask.shape[1]`: validation gate `if b_masks is not None: _validate_bilinear_args(b_masks, stim_mod, N)`.
    3. After `C = C * c_mask`: bilinear B-sampling block — per-modulator `pyro.sample(f"B_free_{j}", Normal(0, B_prior_std).to_event(2))` loop, `torch.stack` to `(J, N, N)`, single `parameterize_B(B_free_stacked, b_mask_stacked)` call, `pyro.deterministic("B", B_stacked)` (L3), `merge_piecewise_inputs(drive_input, mod_input)` widening.
    4. Replaced `system = CoupledDCMSystem(A, C, stimulus)` with branched construction: bilinear passes `B=B_stacked, n_driving_inputs=c_mask.shape[1]` kwargs; linear short-circuit passes NO `B=` kwarg (inherits Phase 13 gate).
    5. Before `pyro.deterministic("predicted_bold", predicted_bold)`: NaN/Inf guard — `if torch.isnan(predicted_bold).any() or torch.isinf(predicted_bold).any(): predicted_bold = torch.zeros_like(predicted_bold).detach()`.
- `tests/test_task_dcm_model.py` (modified, +424):
  - New `task_bilinear_data` fixture extending `task_data` with J=1 bilinear coverage: `b_masks = [b_mask_0]` where `b_mask_0[1, 0] = 1.0` (off-diagonal 1<-0 connection modulated, zero diagonal per Pitfall B5) + `stim_mod` from `make_epoch_stimulus(event_times=[10.0], event_durations=[10.0], event_amplitudes=[1.0], duration=30.0, dt=0.01, n_inputs=1)` wrapped in `PiecewiseConstantInput`. Returns superset of `task_data` with added `b_masks`, `stim_mod`, `J` keys.
  - New `TestBilinearStructure` class with 8 tests:
    - `test_B_PRIOR_VARIANCE_constant` (MODEL-02): imports constant, asserts `== 1.0`.
    - `test_linear_reduction_when_b_masks_none` (MODEL-04, L3): passes `b_masks=None, stim_mod=None`, asserts `site_names == {A_free, C, noise_prec, obs, A, predicted_bold}` exactly and no `B`/`B_free_*` sites.
    - `test_linear_reduction_when_b_masks_empty_list` (MODEL-04 []-to-None): passes `b_masks=[]`, asserts same linear site structure.
    - `test_bilinear_trace_has_B_free_sites` (MODEL-01 + L1): asserts `B_free_0` exists (NOT bare `B_free`), `B_free_0` value shape `(N, N)`, `B` deterministic site exists with shape `(J, N, N)`.
    - `test_bilinear_masking_applied` (MODEL-03 model-side): iterates all `(i, k)` where `b_mask == 0` and asserts `B[0, i, k] == 0` exactly.
    - `test_bilinear_stim_mod_required_error` (MODEL-04): `pytest.raises(ValueError, match="stim_mod is required")` when bilinear with `stim_mod=None`.
    - `test_bilinear_stim_mod_shape_mismatch_error` (MODEL-04): builds 2-column `stim_mod` with 1-element `b_masks`, asserts `pytest.raises(ValueError, match=r"stim_mod\.values\.shape\[1\]=2")`.
    - `test_bilinear_deprecation_warning_on_stacked_nonzero_diag` (MODEL-03 stacked-path / SC-4 closure): constructs a non-zero-diagonal `bad_b_mask` (`[0, 0] = 1.0`), wraps `pyro.poutine.trace(...).get_trace(...)` in `pytest.warns(DeprecationWarning, match="non-zero diagonal")`. Complements Phase 13 `test_bilinear_utils.py` `(N, N)` call with the `(J, N, N)` stacked call-site.
  - New `TestBilinearSVI` class with 1 test:
    - `_silence_stability_logger` autouse fixture sets `pyro_dcm.stability` logger level to ERROR via `caplog.set_level` (D4 + R6).
    - `test_bilinear_svi_smoke_3region_converges`: 40 SVI steps with `AutoNormal(task_dcm_model, init_scale=0.005)` (L2) + `ClippedAdam(lr=0.01, clip_norm=10.0)` + `Trace_ELBO`. Asserts every loss finite; asserts `mean(last_10) < mean(first_10)`. Runtime budget 75s is SOFT — warns (`UserWarning`) but does not fail if exceeded.

## Decisions Made

All three locked decisions applied as specified in the plan frontmatter (no in-plan decisions surfaced during execution):

- **L1** — B-site shape is full `(N, N)` `Normal(0, 1.0).to_event(2)`, NOT a flat vector. Sampling uses literal `dist.Normal(torch.zeros_like(b_mask_j), B_prior_std * torch.ones_like(b_mask_j)).to_event(2)` per-modulator, then `torch.stack(B_free_list, dim=0)` to `(J, N, N)`. This is **required** for AutoLowRankMultivariateNormal's single-vector concatenation in Plan 15-02 (`AutoContinuous._unpack_latent` reshapes by site-event-dim).
- **L2** — Bilinear SVI smoke uses `init_scale=0.005` (half the linear default `0.01`). Passed **explicitly** to `AutoNormal`; `create_guide` factory is NOT changed (additive-discipline preserved). Downstream Phase 16 recovery benchmark callers will pass `init_scale=0.005` explicitly.
- **L3** — `pyro.deterministic("B", B_stacked)` emitted ONLY inside the bilinear branch (guarded by `if b_masks is not None`). Linear-mode trace has NO `"B"` site. Verified structurally by `test_linear_reduction_when_b_masks_none` using exact set-equality `site_names == {A_free, C, noise_prec, obs, A, predicted_bold}` — preserves the pre-Phase-15 `test_model_trace_has_expected_sites` assertion.

## Deviations from Plan

None — plan executed exactly as written; no auto-fix rules triggered.

Minor sentinel variances (documented for future auditors, no action needed):

1. `B_PRIOR_VARIANCE` grep count in `task_dcm_model.py`: plan target "exactly 4 (acceptable 3-5)", achieved **5** — the docstring contains 3 references (Step-3 Notes, Bilinear Notes subsection, `pyro.sample` signature example). Initial implementation had 6; tightened one docstring reference to reach 5, which is within the acceptable range. Every occurrence is structurally justified.
2. `init_scale=0.005` grep count in `test_task_dcm_model.py`: plan target "exactly 1 (L2 literal)", achieved **3** — 1 literal `AutoNormal(..., init_scale=0.005)` call + 2 docstring references explaining the literal (class docstring + method docstring). Grep pattern is too-strict; only literal calls match the plan's stated intent. All 3 references pointing at the same L2 value.
3. Pre-existing `ruff` errors in both files (I001 import sorting + F401 unused `pyro.distributions as dist`): confirmed pre-existing via `git stash` round-trip on both files. Not touched per Phase 14 precedent (do not fix pre-existing lint in additive-edit plans).
4. `pytest-timeout` plugin not installed in the execution environment: plan's `--timeout=180` / `--timeout=120` pytest flags silently unsupported. Ran without. Bilinear SVI smoke completed in 29.8s; full `test_task_dcm_model.py` ran in 74.16s; Phase 13+14 regression (51 tests) ran in 298.50s. No runaway tests observed.

## Issues Encountered

None. All tests passed on first execution.

## User Setup Required

None — no external service configuration required.

## Downstream Contracts

### For Plan 15-02 (guide auto-discovery)

- Sample site names in bilinear mode: `B_free_0`, `B_free_1`, ..., `B_free_{J-1}` (literal f-string `f"B_free_{j}"`). No bare `B_free` site ever created.
- Site shape: each `B_free_j` is `(N, N)` with `Normal(0, 1.0).to_event(2)` — concatenable across AutoNormal / AutoLowRankMultivariateNormal / AutoIAFNormal by the existing `AutoContinuous._unpack_latent` single-vector mechanism.
- Trace fixture: `task_bilinear_data` (3 regions, J=1, 30s, dt=0.5) is available in `tests/test_task_dcm_model.py` for guide-trace inspection.
- Helper for guide tests: `TestBilinearStructure._run_bilinear_trace(task_bilinear_data)` demonstrates the `pyro.poutine.condition + pyro.poutine.trace` pattern with all bilinear sites conditioned.

### For Plan 15-03 (amortized packer refusal + extract_posterior)

- `task_dcm_model` signature now requires `b_masks` + `stim_mod` to be **keyword-only**: `TaskDCMPacker` refusal test can inspect `inspect.signature(task_dcm_model).parameters["b_masks"].kind == KEYWORD_ONLY`.
- Bilinear posterior dict keys: `B_free_0`, `B_free_1`, ..., `B_free_{J-1}` (per-modulator). The deterministic `B` site (shape `(J, N, N)`) is also available via `pyro.poutine.trace` for median-recovery analysis.
- `pyro.deterministic("B", B_stacked)` is the **single source of truth** for the masked stacked `B` tensor (already has `parameterize_B` applied).

### For Phase 16 (recovery benchmark)

- Bilinear SVI callers should pass `init_scale=0.005` explicitly to `AutoNormal` / `AutoLowRankMultivariateNormal` (L2). `create_guide` factory was deliberately NOT auto-switched.
- Phase 16 RECOV-08 3-6x slowdown assumption verified at N=3, J=1 scale: 40-step bilinear SVI smoke ran in 29.8s (wall); linear `test_svi_loss_decreases` (50 steps) ran in baseline ~60s. Ratio ~2x per-step at this scale — comfortable margin below the 6x budget upper bound.

## Next Phase Readiness

- **Plan 15-02 ready:** `B_free_j` sample sites exist and are concatenable; no further guide-factory changes needed for AutoNormal/AutoLowRankMVN/AutoIAFNormal discovery.
- **Plan 15-03 ready:** keyword-only signature supports packer refusal test; `B` deterministic site available for bilinear extract_posterior keys.
- **No blockers or concerns.** L2 `init_scale=0.005` may need tuning in Phase 16 RECOV-07 if recovery shrinkage metric fails — documented as a known follow-up; downstream callers pass explicitly so a global switch is a one-line Phase 16 change.

## Requirements Closed

| Req | Evidence |
|-----|----------|
| MODEL-01 | Per-modulator `B_free_j ~ Normal(0, 1.0).to_event(2)` loop at `task_dcm_model.py:300-314`; asserted by `test_bilinear_trace_has_B_free_sites` |
| MODEL-02 | `B_PRIOR_VARIANCE: float = 1.0` at `task_dcm_model.py:36` with D1 docstring; asserted by `test_B_PRIOR_VARIANCE_constant` |
| MODEL-03 (N, N) | Phase 13 `tests/test_bilinear_utils.py` (pre-existing, unchanged) |
| MODEL-03 (J, N, N) stacked | Single `parameterize_B(B_free_stacked, b_mask_stacked)` call at `task_dcm_model.py:317`; SC-4 DeprecationWarning asserted by `test_bilinear_deprecation_warning_on_stacked_nonzero_diag` wrapping a `pyro.poutine.trace` call |
| MODEL-04 | Linear short-circuit (None + [] paths) asserted by `test_linear_reduction_when_b_masks_none` + `test_linear_reduction_when_b_masks_empty_list`; two ValueError paths by `test_bilinear_stim_mod_required_error` + `test_bilinear_stim_mod_shape_mismatch_error`; SVI convergence gate by `test_bilinear_svi_smoke_3region_converges` |

## Grep Sentinels (final, after all three commits)

Source file (`src/pyro_dcm/models/task_dcm_model.py`):
- `B_PRIOR_VARIANCE`: 5 (def + 3 docstring + sqrt literal in `B_prior_std = B_PRIOR_VARIANCE ** 0.5`) — within acceptable 3-5 range
- `f"B_free_{j}"`: 2 (docstring example + literal sample call)
- `merge_piecewise_inputs`: 2 (import + branch call)
- `pyro.deterministic("B"`: 3 (2 docstring + 1 call)
- `CoupledDCMSystem(A, C, stimulus)`: 1 (linear short-circuit)
- `CoupledDCMSystem(`: 2 (linear + bilinear)
- `_validate_bilinear_args`: 2 (def + call)
- `torch.isnan(predicted_bold)`: 1 (NaN guard)

Test file (`tests/test_task_dcm_model.py`):
- `B_PRIOR_VARIANCE`: 7
- `B_free_0`: 7
- `test_linear_reduction_when_b_masks`: 2
- `pytest.raises(ValueError, match="stim_mod is required")`: 1
- `test_bilinear_deprecation_warning_on_stacked_nonzero_diag`: 2 (docstring + def)
- `pytest.warns(DeprecationWarning`: 1
- `init_scale=0.005`: 3 (1 literal call + 2 docstring)
- `_silence_stability_logger`: 1
- `test_bilinear_svi_smoke_3region_converges`: 1
- `class TestBilinearStructure`: 1
- `class TestBilinearSVI`: 1
- `pyro_dcm.stability`: 2

## Verification Evidence

```
pytest tests/test_task_dcm_model.py -v
  -> 19 passed in 74.16s
      (10 pre-existing TestModelStructure/TestNumericalStability/TestSVI
       + 8 TestBilinearStructure
       + 1 TestBilinearSVI)

pytest tests/test_linear_invariance.py tests/test_coupled_system_bilinear.py \
       tests/test_bilinear_utils.py tests/test_bilinear_simulator.py \
       tests/test_stimulus_utils.py -q
  -> 51 passed in 298.50s

ruff check src/pyro_dcm/models/task_dcm_model.py
  -> 1 error (I001 import-sort, pre-existing; confirmed via git stash round-trip)

ruff check tests/test_task_dcm_model.py
  -> 2 errors (I001 + F401 unused `pyro.distributions`, both pre-existing;
                confirmed via git stash round-trip)
```

---
*Phase: 15-pyro-bilinear-model*
*Plan: 01*
*Completed: 2026-04-18*
