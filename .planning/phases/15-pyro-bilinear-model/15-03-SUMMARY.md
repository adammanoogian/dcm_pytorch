---
phase: 15-pyro-bilinear-model
plan: 03
subsystem: models+guides
tags: [pytorch, pyro, bilinear-dcm, amortized, packer-refusal, posterior-extraction, model-05, model-07]

# Dependency graph
requires:
  - phase: 15-pyro-bilinear-model
    plan: 01
    provides: "bilinear task_dcm_model with B_free_j sample sites + pyro.deterministic('B', ...) site + B_PRIOR_VARIANCE constant"
provides:
  - "TaskDCMPacker.pack() NotImplementedError guard on 'B_free_*' keys (defense-in-depth packer-level refusal; references v0.3.1)"
  - "amortized_task_dcm_model keyword-only b_masks/stim_mod kwargs with NotImplementedError guard on non-empty b_masks (primary user-visible refusal surface; references v0.3.1)"
  - "extract_posterior_params docstring Notes paragraph + Examples block documenting bilinear keys B_free_0..B_free_{J-1} and 'B' deterministic site with cross-Pyro-version return_sites portability note"
  - "tests/test_parameter_packing.py::TestTaskDCMPackerBilinearRefusal (2 tests: refusal with v0.3.1 match + linear regression)"
  - "tests/test_amortized_task_dcm.py::TestAmortizedRefusesBilinear (2 tests: refusal with v0.3.1 match + linear None/[] regression)"
  - "tests/test_posterior_extraction.py::TestExtractPosteriorBilinear::test_extract_posterior_includes_bilinear_sites (1 test: 20-step bilinear SVI + Predictive with explicit return_sites + site-agnostic extract_posterior_params follow-up)"
affects: [16-recovery-benchmark]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "defense-in-depth-refusal: two independent NotImplementedError guards (packer.pack + wrapper entry) both reference v0.3.1 per D5; catches direct callers AND wrapper callers"
    - "v-next-deferral-pattern: keyword-only kwargs added to future-proof API surface but refused at entry with literal next-version string (v0.3.1) in error message"
    - "site-agnostic-posterior-extraction: extract_posterior_params iterates samples.items() and requires zero code change to surface new dynamic pyro.sample / pyro.deterministic sites"
    - "explicit-return-sites-for-portable-deterministic-assertions: Predictive(return_sites=[...]) used in tests to decouple test correctness from Pyro-version-dependent default inclusion of deterministic sites"

key-files:
  created: []
  modified:
    - "src/pyro_dcm/guides/parameter_packing.py (15-line guard at top of TaskDCMPacker.pack; SpectralDCMPacker untouched)"
    - "src/pyro_dcm/models/amortized_wrappers.py (signature extension + 11-line guard at top of amortized_task_dcm_model + 18-line docstring expansion; amortized_spectral_dcm_model untouched)"
    - "src/pyro_dcm/models/guides.py (docstring-only: 17-line Notes paragraph + 8-line Examples block in extract_posterior_params; no code change)"
    - "tests/test_amortized_task_dcm.py (TestAmortizedRefusesBilinear class appended; 2 new tests; pre-existing tests unchanged)"
    - "tests/test_parameter_packing.py (TestTaskDCMPackerBilinearRefusal class appended; 2 new tests; pre-existing tests unchanged)"
    - "tests/test_posterior_extraction.py (TestExtractPosteriorBilinear class appended with autouse caplog fixture; 1 new test; pre-existing tests unchanged)"

key-decisions:
  - "D5 applied: amortized bilinear inference deferred to v0.3.1; both refusal sites reference the literal string 'v0.3.1'"
  - "Defense-in-depth pattern chosen over single-point refusal: packer-level catches direct callers (offline/research use); wrapper-level is primary user surface"
  - "SpectralDCMPacker + amortized_spectral_dcm_model explicitly out of scope (bilinear DCM is task-only; spectral unaffected)"
  - "extract_posterior_params is code-unchanged (site-agnostic design from original 08-05 work); only docstring augmented with bilinear guidance"
  - "Test uses bare SVI loop (NOT run_svi) because run_svi takes positional model_args tuple and cannot forward task_dcm_model's keyword-only b_masks/stim_mod kwargs"
  - "Explicit return_sites=['A_free','C','noise_prec','B_free_0','B'] in test Predictive call for cross-Pyro-version portability (Plan 15-03 checker Blocker 2 resolution)"
  - "run_svi intentionally NOT imported in new bilinear test (F401 hygiene; Plan 15-03 checker Major 3 resolution)"

patterns-established:
  - "defense-in-depth-refusal: pair packer-level and wrapper-level NotImplementedError guards with identical v{next-version} string"
  - "v-next-deferral-pattern: keyword-only API expansion that refuses at entry with literal version reference"
  - "site-agnostic-posterior-extraction: zero-code-change surface for new pyro.sample / pyro.deterministic sites"
  - "explicit-return-sites-for-portable-deterministic-assertions: decouple test correctness from Pyro-version-dependent default behavior"

# Metrics
duration: 12min
completed: 2026-04-18
---

# Phase 15 Plan 15-03: Bilinear Refusal + Posterior Extraction Summary

**Defense-in-depth bilinear refusal: TaskDCMPacker.pack (direct-caller surface) and amortized_task_dcm_model (wrapper surface) both raise NotImplementedError with literal 'v0.3.1' per D5; extract_posterior_params docstring documents B_free_j + B keys; 5 new tests close MODEL-05 + MODEL-07 with 81/81 Phase-15 suite green.**

## Performance

- **Duration:** ~12 min wall-clock (Task 1: ~1 min, Task 2: ~1 min, Task 3: ~10 min including test authoring + 23s bilinear SVI test runtime)
- **Started:** 2026-04-18
- **Completed:** 2026-04-18
- **Tasks:** 3 (plus this summary + STATE.md update)
- **Files modified:** 6 (3 src + 3 test)

## Accomplishments

- **MODEL-07 defense-in-depth refusal (both surfaces):**
  - `TaskDCMPacker.pack` raises `NotImplementedError` on any `B_free_*` key (params-dict scan at method entry)
  - `amortized_task_dcm_model` signature gains keyword-only `b_masks: list[torch.Tensor] | None = None` and `stim_mod: object | None = None` after `*` sentinel; non-empty `b_masks` raises `NotImplementedError` BEFORE any other work; `None` and `[]` both pass through to linear body (API symmetry with `task_dcm_model` from Plan 15-01)
  - Both error messages include the literal string `"v0.3.1"` per D5; grep-verified
- **MODEL-05 documentation + executable gate:**
  - `extract_posterior_params` docstring gains Notes paragraph and Examples block documenting `B_free_0..B_free_{J-1}` raw keys (always present) and `B` deterministic site (shape `(J, N, N)`, conditional on `return_sites`)
  - No code change in function body; the function was already site-agnostic via `samples.items()` iteration (original 08-05 design)
  - `TestExtractPosteriorBilinear::test_extract_posterior_includes_bilinear_sites` runs 20 SVI steps on bilinear `task_dcm_model` via a bare SVI loop and uses `Predictive` with explicit `return_sites=['A_free','C','noise_prec','B_free_0','B']` for portable cross-Pyro-version assertion on both stochastic `B_free_0` (shape tail `(N, N)`) and deterministic `B` (shape tail `(J, N, N)`); supplementary `extract_posterior_params` check asserts only the always-present `B_free_0`
- **5 new tests, zero pre-existing test breakage:**
  - `tests/test_amortized_task_dcm.py::TestAmortizedRefusesBilinear`: 2 tests (refusal + linear None/[] regression)
  - `tests/test_parameter_packing.py::TestTaskDCMPackerBilinearRefusal`: 2 tests (refusal + linear pack/unpack regression after guard added)
  - `tests/test_posterior_extraction.py::TestExtractPosteriorBilinear`: 1 test (bilinear SVI + dual-path posterior extraction)
- **SpectralDCMPacker + amortized_spectral_dcm_model unmodified (bilinear DCM is task-only; spectral unaffected by MODEL-07 scope)**

## Task Commits

Each task was committed atomically:

1. **Task 1: Add bilinear refusal guards to TaskDCMPacker.pack + amortized_task_dcm_model** - `6c68b10` (feat)
2. **Task 2: Extend extract_posterior_params docstring with bilinear-keys documentation** - `66cab62` (docs)
3. **Task 3: Add refusal + posterior-extraction tests across 3 test files** - `b9928c2` (test)

**Plan metadata commit:** (pending after this SUMMARY.md + STATE.md update)

## Files Created/Modified

- `src/pyro_dcm/guides/parameter_packing.py` - `TaskDCMPacker.pack` gains 15-line bilinear refusal guard; method body otherwise unchanged; `SpectralDCMPacker` untouched.
- `src/pyro_dcm/models/amortized_wrappers.py` - `amortized_task_dcm_model` signature gains `*, b_masks=None, stim_mod=None` kwargs; 11-line refusal guard at method entry (fires BEFORE `_sample_latent_and_unpack`); docstring gains 2 Parameters entries + "Bilinear support" Notes paragraph; `_run_task_forward_model`, `_sample_latent_and_unpack`, and `amortized_spectral_dcm_model` untouched.
- `src/pyro_dcm/models/guides.py` - `extract_posterior_params` docstring gains 17-line Notes paragraph ("Bilinear task DCM sites (v0.3.0+)") + 8-line Examples block showing `posterior['B_free_0']` and conditional `posterior['B']` access. No code change.
- `tests/test_amortized_task_dcm.py` - Appended `TestAmortizedRefusesBilinear` class (2 tests). All 6 pre-existing tests (5 non-slow + 1 slow) unchanged.
- `tests/test_parameter_packing.py` - Appended `TestTaskDCMPackerBilinearRefusal` class (2 tests). All 9 pre-existing tests unchanged.
- `tests/test_posterior_extraction.py` - Appended `TestExtractPosteriorBilinear` class (1 test) with `caplog.set_level(logging.ERROR, logger='pyro_dcm.stability')` autouse fixture. All 13 pre-existing tests unchanged.

## Decisions Made

- **Defense-in-depth refusal (per research Section 6):** Both packer-level and wrapper-level guards exist independently. Packer catches direct callers building bilinear params dicts for offline use; wrapper is primary user surface. Alternative (single-point refusal at wrapper only) rejected because it would silently accept bilinear params in offline research workflows that bypass the wrapper.
- **`del stim_mod` in wrapper:** Added after the bilinear guard to mark the kwarg as intentionally unused in linear mode (suppresses potential ARG linting and makes intent explicit); kwarg exists purely for API symmetry with `task_dcm_model`.
- **`-f` stage needed for `src/pyro_dcm/models/*`:** The repo's `.gitignore` has a top-level `models/` entry intended for ML model weights (Python tracking convention). Tracked source files under `src/pyro_dcm/models/` require `git add -f` to stage. Applied in Task 1 and Task 2. Pre-existing Phase 15-01 commits used same approach.
- **Test posterior path uses bare SVI loop + explicit Predictive `return_sites`:** Two decisions in one: (1) `run_svi` intentionally NOT imported in the new test because its positional `model_args` tuple cannot forward `task_dcm_model`'s keyword-only `b_masks`/`stim_mod` kwargs — bare SVI loop is the only path; (2) `Predictive(return_sites=['A_free','C','noise_prec','B_free_0','B'])` used instead of default `return_sites=None` so the `'B'` deterministic-site assertion is portable across Pyro 1.9+ patch versions regardless of default inclusion behavior (Plan 15-03 checker Blocker 2 resolution).

## Deviations from Plan

None - plan executed exactly as written. All three task commits, all five new tests, all docstring additions match the plan's `<action>` and `<done>` sections. Grep sentinels met (see Issues Encountered for sentinel variance notes).

**Minor sentinel variances (documented, acceptable):**
- `NotImplementedError` count in `amortized_wrappers.py`: 3 (plan target "exactly 1"). 1 is the actual `raise NotImplementedError(` call; 2 are docstring references in the "Bilinear support" Notes paragraph. Same precedent as Plan 15-01 sentinel variances (docstring references count toward literal-pattern greps).
- `v0.3.1` count in `amortized_wrappers.py`: 5 (plan said ">= 1"). 1 in the raise message; 4 in docstring (Parameters + Notes expansions). Well above minimum.

**Total deviations:** 0 auto-fixed — plan was executable as specified.
**Impact on plan:** None.

## Issues Encountered

- **Pre-existing ruff lint:** `ruff check` reports 3 errors in `guides.py` (I001 import block, UP035 `Callable` from `collections.abc`, F401 unused `pyro.distributions`), 1 error in `amortized_wrappers.py` (I001), and 6 errors across the 3 modified test files (I001, F401, B007, E741). All verified pre-existing via `git stash` round-trip. Per Phase 14-02 / Plan 15-01 precedent, pre-existing lint is not fixed in additive plans. No NEW ruff errors introduced by Plan 15-03.
- **`UserWarning: std(): degrees of freedom is <= 0`:** Emitted by `TaskDCMPacker.fit_standardization` when called with a single-item dataset in `test_amortized_wrapper_linear_mode_unchanged`. The 1e-6 clamp handles the std=0 case correctly; warning is benign (std-correction is irrelevant because all packed features are identical across the single dataset element). Test passes cleanly.
- **Plan 15-02 running in parallel:** Noted in orchestrator prompt. Plan 15-02 touched only `tests/test_guide_factory.py` — disjoint from this plan's files. During Task 1 staging, `git status` showed `test_guide_factory.py` modified; correctly excluded from staging via file-specific `git add` calls. Plan 15-02 completed during my Task 3 (commit `9b796c0`), then its metadata commit (`e1d986b`); both picked up naturally without interfering with this plan's commit sequence.
- **`.gitignore` top-level `models/` entry:** Matches both ML model-weight directories AND `src/pyro_dcm/models/` source directory. Required `git add -f` to stage `amortized_wrappers.py` and `guides.py`. Not a bug (other Plan 15 commits used same approach per `git log src/pyro_dcm/models/task_dcm_model.py`); candidate for future chore to refine `.gitignore` pattern to only root-level or weight directories.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- **Phase 15 complete: 7/7 requirements closed** (MODEL-01..07). Plan 15-01 closed MODEL-01 through MODEL-04; Plan 15-02 closed MODEL-06; this plan closes MODEL-05 and MODEL-07.
- **Ready for Phase 16 (recovery-benchmark):**
  - Phase 16 recovery benchmark can pull per-modulator B_j medians via `extract_posterior_params(guide, args, model=bilinear_task_dcm_model, num_samples=1000)` and index `posterior['B_free_{j}']['mean']` (raw, always available) or use explicit `Predictive(return_sites=[..., 'B'])` for masked `(J, N, N)` tensor.
  - Bilinear SVI path (`create_guide(task_dcm_model, init_scale=0.005) + bare SVI loop with b_masks/stim_mod kwargs`) is the v0.3.0 canonical path for DCM.V1 acceptance (D5: amortized refused).
- **Known follow-ups (v0.3.1 scope, NOT blocking Phase 16):**
  - AMORT-01: extend `TaskDCMPacker` to dynamically include `B_free_j` per-modulator sections (`n_features = N*N + N*M + 1 + J*N*N`) + `fit_standardization` over bilinear datasets
  - AMORT-02: extend `amortized_task_dcm_model` to sample and unpack bilinear terms + route through `parameterize_B` + `CoupledDCMSystem(..., B=..., n_driving_inputs=...)` bilinear branch
  - AMORT-03: warm-start refusal → permissive acceptance; training-pipeline updates for bilinear datasets
- **No blockers for Phase 16.** Plan's refusal error messages explicitly direct v0.3.1-seeking users to the SVI path.

---
*Phase: 15-pyro-bilinear-model*
*Completed: 2026-04-18*
