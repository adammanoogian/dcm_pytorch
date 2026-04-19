---
phase: 16-bilinear-recovery-benchmark
plan: 02
subsystem: benchmarks+metrics
tags: [pytorch, matplotlib, recov-03, recov-04, recov-05, recov-06, recov-07, recov-08, forest-plot, acceptance-gate]

# Dependency graph
requires:
  - phase: 16-bilinear-recovery-benchmark
    plan: 01
    provides: run_task_bilinear_svi runner + posterior_list contract (a_rmse_bilinear_list, a_rmse_linear_list, posterior_list[i]['B_free_0'].{mean,std,samples}, b_true_list, time_bilinear_list, time_linear_list)
provides:
  - "benchmarks/bilinear_metrics.py: 5 pure helpers (compute_b_rmse_magnitude, compute_sign_recovery_nonzero, compute_coverage_of_zero, compute_shrinkage, compute_a_rmse_relative) + compute_acceptance_gates entry point; L5 pooled aggregation in RECOV-05/06; L7 95% CI level"
  - "Module constants SIGMA_PRIOR=1.0 and RECOV_0{3..8}_* thresholds matching .planning/REQUIREMENTS.md (with one documented Rule-1 auto-fix on RECOV_06_NULL_MASK: 0.5 -> 0.1 to avoid mis-classifying the 0.3/0.4 non-null elements as null)"
  - "benchmarks/plotting.py::plot_bilinear_b_forest (9-row forest plot, per-seed-median jittered scatter + cross-seed median+IQR + B_true reference dot + inline shrinkage annotation per L6/L7) saving b_forest_recovery.png"
  - "benchmarks/plotting.py::plot_acceptance_gates_table (6-row pass/fail table: 4 RECOV gates + RECOV-07 shrinkage info + RECOV-08 wall-time row with 10x flag) saving acceptance_gates.png"
  - "benchmarks/plotting.py::generate_all_figures dispatches to both new functions when ('task_bilinear', 'svi') entry present (supports both tuple and str-tuple keys for JSON roundtrip)"
  - "tests/test_bilinear_metrics.py: 11 unit tests in TestMetricHelpers (5) + TestAcceptanceGates (6) covering all helpers + FIX-2 AND-combination regression + insufficient_data guard + 10x flag + shrinkage info"
  - "tests/test_task_bilinear_benchmark.py::TestTaskBilinearAcceptance::test_acceptance_gates_pass_at_10_seeds (@pytest.mark.slow) — THE Phase 16 acceptance gate at full_config n_datasets=10 + n_svi_steps=500 with FIX-1 n_success>=10 guard, FIX-4 convergence caveat documented, figure artifacts written to tmp_path"
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "pooled-multi-element-aggregation: RECOV-05 and RECOV-06 aggregate over (seed, element) pairs pooled, not mean-of-per-seed-means (L5; avoids pathological {0, 0.5, 1.0} discretization at 2 free elements per seed)"
    - "single-source-acceptance-computation: compute_acceptance_gates(runner_result) is the ONLY entry point for pass/fail — consumed by both the slow acceptance test and the plot_acceptance_gates_table figure, so the test and the figure never disagree"
    - "per-seed-median-plus-cross-seed-IQR-forest: L6 two-stage aggregation for 9 elements x 10 seeds = 90 raw samples; keeps the forest readable at 9 rows while preserving per-seed variance via IQR error bars"

key-files:
  created:
    - benchmarks/bilinear_metrics.py
    - tests/test_bilinear_metrics.py
    - .planning/phases/16-bilinear-recovery-benchmark/16-02-SUMMARY.md
  modified:
    - benchmarks/plotting.py
    - tests/test_task_bilinear_benchmark.py

key-decisions:
  - "L5 (inherited from this plan): RECOV-05 sign_recovery_nonzero and RECOV-06 coverage_of_zero use POOLED counting over (seed, element) pairs, not per-seed means. Research Section 4.3, 4.4 rationale — 2 non-zero B elements per seed makes per-seed sign recovery live in {0, 0.5, 1.0}, a pathological distribution for mean-of-means."
  - "L6 (this plan): forest plot aggregates per-seed posterior MEDIANS (one point per seed per element), then plots per-element cross-seed median + IQR. 9 elements x 10 seeds = 90 raw samples would be too cluttered as a single strip plot; L6 keeps the figure readable at 9 rows while preserving per-seed variance."
  - "L7 (this plan): forest plot CI level = 95%, matching RECOV-06 coverage_of_zero's 95% CI. Using any other level (e.g. 90%) would introduce a definitional mismatch between the figure and the acceptance criterion."
  - "L8 (this plan): acceptance-gate test is @pytest.mark.slow-gated. Runs full_config at 10 seeds with n_svi_steps=500 override. Includes FIX-4 convergence caveat: 500 steps is a research-recommended starting point extrapolated from Phase 15 smoke (40 steps on 30s BOLD), NOT a validated convergence budget at 200s BOLD; if RECOV-04/05 fails under 500 steps, retry with 1500 before filing a bug."
  - "L9 (this plan): RECOV-06 85% threshold stays HARD under AutoNormal. No in-scope sidebar or softer fallback. If RECOV-06 fails due to mean-field posterior-correlation underestimation, the Phase 16 SUMMARY documents the observed coverage and flags the milestone as blocked pending v0.3.1 AutoLowRankMVN fallback. Preserves v0.3.0 acceptance criterion unchanged."
  - "FIX-1 (orchestrator revision, in Task 3 acceptance test): assert n_success >= 10 BEFORE trusting gates. Without this, an 8/10-dataset run would silently pass gates computed on only 8 seeds while the test name claims '10 seeds'."
  - "FIX-2 (orchestrator revision, in TestAcceptanceGates): added test_all_pass_false_when_only_recov_03_fails as a regression gate for AND-combination logic in compute_acceptance_gates. Constructs a synthetic result where RECOV-04/05/06 all pass but RECOV-03 fails and asserts all_pass is False."
  - "FIX-5 (this plan, in compute_acceptance_gates): RECOV-06 hard 85% threshold preserved; documented as known-failure-path per L9. compute_coverage_of_zero docstring explicitly records the mean-field limitation."

patterns-established:
  - "pooled-multi-element-aggregation: aggregate metrics over (seed, element) pairs when per-seed sample counts are small (<5 elements per seed)"
  - "single-source-acceptance-computation: one function consumed by both tests and figures — no drift between assertion logic and figure labels"
  - "per-seed-median-plus-cross-seed-IQR-forest: two-stage aggregation for visualization of element x seed arrays when raw scatter would be too dense"

# Metrics
duration: ~14min
completed: 2026-04-19
---

# Phase 16 Plan 02: Bilinear Metrics, Forest Plot, and Acceptance Gates Summary

**RECOV-03..08 single-source-of-truth acceptance computation (5 pure metric helpers + compute_acceptance_gates), headline 9-row B-recovery forest plot with inline shrinkage annotation (L6 per-seed-median + cross-seed IQR at 95% CI), 6-row pass/fail table figure, and @pytest.mark.slow-gated 10-seed acceptance test — the v0.3.0 Bilinear DCM milestone gate is now a single pytest invocation away.**

## Performance

- **Duration:** ~14 min of execution-session wall clock (3 feat/test commits 2026-04-18 23:12:55 -> 23:26:57 +0200); plus finalization session 2026-04-19 (this SUMMARY + Task 4 metadata commit + STATE update).
- **Started:** 2026-04-18T23:12:55+02:00 (b9aae53 feat commit)
- **Completed:** 2026-04-19 (finalization session; slow acceptance test NOT run)
- **Tasks:** 3 atomic feat/test commits + Task 4 metadata commit
- **Files created:** 2 source files + this SUMMARY (3 total)
- **Files modified:** 2 (plotting.py + test_task_bilinear_benchmark.py)

## Accomplishments

- Single-source-of-truth acceptance computation: `compute_acceptance_gates(runner_result)` consumes plan 16-01's runner output and returns a dict with 4 pass/fail gates (RECOV-03/04/05/06) + 2 info rows (RECOV-07 shrinkage + RECOV-08 wall-time ratio) + an `all_pass` bool computed via AND-combination. This function is consumed by both the slow acceptance test and the pass/fail table figure — the test and the figure can never drift apart.
- 5 pure metric helpers with hand-computed unit-test coverage: `compute_b_rmse_magnitude` (RECOV-04; magnitude-masked RMSE), `compute_sign_recovery_nonzero` (RECOV-05; L5 pooled over (seed, element) pairs), `compute_coverage_of_zero` (RECOV-06; L5 pooled, L7 95% CI, quantile-band containment), `compute_shrinkage` (RECOV-07; element-wise std_post / sigma_prior), `compute_a_rmse_relative` (RECOV-03; ratio of bilinear A-RMSE mean to linear-baseline A-RMSE mean with 1.25x pass threshold).
- Headline forest plot `plot_bilinear_b_forest` (L6/L7): 9 rows for the 3-region J=1 B matrix. Each row shows (a) per-seed posterior-median dots (jittered within row y-band, green if `|B_true|>0.1` else gray), (b) cross-seed median + IQR horizontal error bar, (c) ground-truth B value as a red diamond reference dot, and (d) an inline shrinkage annotation (mean `std_post/sigma_prior` across seeds, colored green if `<= RECOV_07_SHRINKAGE_SOFT_TARGET` else red). Saves to `b_forest_recovery.png` at dpi=150.
- Pass/fail table figure `plot_acceptance_gates_table`: 6-row matplotlib text table matching `/gsd:verify-work` format. Row tint communicates pass (green), fail (red), info (gray), or 10x-FLAG (yellow). Title says "ALL PASS" or "SOME FAIL" based on `all_pass`. Saves to `acceptance_gates.png` at dpi=150.
- `generate_all_figures` dispatches to both new functions when a `('task_bilinear', 'svi')` entry is present in the results JSON, supporting both tuple and str-tuple keys for JSON-roundtrip compatibility. Existing v0.2.0 call sites (`plot_true_vs_inferred`, `plot_metric_strips`, `plot_amortization_gap`, `plot_calibration_curves`, etc.) are UNCHANGED.
- Test coverage: 11 unit tests in `tests/test_bilinear_metrics.py` all passing (5 TestMetricHelpers + 6 TestAcceptanceGates). Orchestrator-added FIX-2 regression gate (`test_all_pass_false_when_only_recov_03_fails`) catches accidental `or` substitution in the AND-combination. FIX-5 manifest via `test_acceptance_gates_raises_on_insufficient_data`.
- THE Phase 16 acceptance gate: `TestTaskBilinearAcceptance::test_acceptance_gates_pass_at_10_seeds` (@pytest.mark.slow) runs `BenchmarkConfig.full_config('task_bilinear', 'svi')` with `n_svi_steps=500` override (L8), enforces FIX-1 `n_success >= 10` precondition, calls `compute_acceptance_gates`, writes both figures to `tmp_path` as artifacts, and asserts all 4 RECOV gates pass with descriptive failure messages pointing at likely root causes (Pitfall B13 A-RMSE inflation for RECOV-03, mean-field correlation underestimation for RECOV-06, etc.). Runtime target per research Section 2.1: ~80 min at 10 seeds x 500 steps.

## Task Commits

Each task was committed atomically:

1. **Task 1: Add bilinear_metrics module with RECOV-03..08 helpers** — `b9aae53` (feat)
2. **Task 2: Add forest plot and acceptance-gate table to plotting.py** — `50a08fb` (feat)
3. **Task 3: Bilinear metrics unit tests + acceptance gate at 10 seeds** — `4bddd8e` (test)

**Plan metadata:** `<pending-this-commit>` (`docs(16-02): complete bilinear metrics and acceptance plan`)

## Files Created/Modified

Created:
- `benchmarks/bilinear_metrics.py` (~450 lines) — `SIGMA_PRIOR` + RECOV thresholds module constants; 5 pure metric helpers (compute_b_rmse_magnitude, compute_sign_recovery_nonzero, compute_coverage_of_zero, compute_shrinkage, compute_a_rmse_relative); `compute_acceptance_gates(runner_result) -> dict` single-source entry point; all NumPy-style docstrings citing `.planning/REQUIREMENTS.md` RECOV-0X and `.planning/phases/16-bilinear-recovery-benchmark/16-RESEARCH.md` Section 4.
- `tests/test_bilinear_metrics.py` (~260 lines) — `_make_minimal_runner_result` synthetic-data helper + `TestMetricHelpers` (5 tests) + `TestAcceptanceGates` (6 tests including FIX-2 AND-combination regression and FIX-5 insufficient_data guard).
- `.planning/phases/16-bilinear-recovery-benchmark/16-02-SUMMARY.md` (this file).

Modified:
- `benchmarks/plotting.py` (+243 lines) — new imports from `benchmarks.bilinear_metrics` (`RECOV_07_SHRINKAGE_SOFT_TARGET`, `compute_acceptance_gates`, `compute_shrinkage`); `plot_bilinear_b_forest` appended at line 1618; `plot_acceptance_gates_table` appended at line 1752; `generate_all_figures` gets task_bilinear dispatch block at the end. No existing function modified.
- `tests/test_task_bilinear_benchmark.py` (+~170 lines) — `TestTaskBilinearAcceptance` class appended at line 160 with `@pytest.fixture(autouse=True)` silencers (stability logger + pyro param-store reset) and `@pytest.mark.slow` `test_acceptance_gates_pass_at_10_seeds`. FIX-1 `n_success >= 10` precondition enforced before gate computation. Plan 16-01's `TestTaskBilinearSmoke` class preserved above.

## Decisions Made

- **L5 pooled aggregation (locked this plan):** RECOV-05 and RECOV-06 sum matches and eligible pairs across all seeds (not per-seed-means). At 2 free elements per seed, per-seed sign recovery is in `{0, 0.5, 1.0}` — pathological for mean-of-means. Pooled counting gives a clean fraction with obvious frequentist interpretation at 10 seeds.
- **L6 two-stage forest aggregation (locked this plan):** Each seed contributes one posterior-median point per element; each element row shows cross-seed median + IQR. Research Section 7.2 N4 option (c) chosen over option (a) raw strip (90 dots; too cluttered) and option (b) stacking-by-seed (too visually dense). Per-seed variance is preserved via IQR error bars.
- **L7 95% CI level (locked this plan):** Matches RECOV-06 coverage_of_zero's 95% CI. Using any other level would make the figure-reported CI definitionally inconsistent with the acceptance-gate calculation.
- **L8 slow-marker gate with 500-step override + convergence caveat (locked this plan):** Acceptance test is `@pytest.mark.slow`-gated to stay out of default pytest runs. `n_svi_steps=500` is the research Section 2.1 starting point. FIX-4 convergence caveat documented in the test docstring: 500 steps is extrapolated from Phase 15 smoke (40 steps on 30s BOLD), NOT validated at 200s BOLD / 100 TR; retry at 1500 before filing a bug if RECOV-04/05 fail.
- **L9 RECOV-06 hard threshold (locked this plan):** 85% preserved under AutoNormal with no softer fallback. If mean-field posterior-correlation underestimation causes RECOV-06 to fail, the failure is a SURFACING gate for v0.3.1 AutoLowRankMVN fallback — NOT a scientific claim that bilinear DCM is unrecoverable.
- **FIX-1 `n_success >= 10` precondition (orchestrator revision):** Enforced in the slow acceptance test before `compute_acceptance_gates` is called. Without this, an 8-success/10-dataset run would silently pass gates computed on only 8 seeds while the test name claims "10 seeds".
- **FIX-2 AND-combination regression gate (orchestrator revision):** `test_all_pass_false_when_only_recov_03_fails` constructs a synthetic result where RECOV-04/05/06 all pass but RECOV-03 fails (bilinear A-RMSE 5x linear baseline) and asserts `all_pass is False`. Catches any accidental `or` substitution in the AND reduction.
- **FIX-4 500-step convergence caveat (orchestrator revision):** Documented in both the L8 locked decision and the acceptance test docstring. 500 steps is a research-recommended starting point, not a validated convergence budget.
- **FIX-5 RECOV-06 documented failure path (orchestrator revision):** Rather than adding an in-scope softer-threshold fallback (which would change the v0.3.0 acceptance bar mid-execution), the plan locks L9 and the compute_coverage_of_zero docstring explicitly carries the mean-field limitation text. The acceptance test's RECOV-06 assertion message points at Pitfall v0.2.0 P1 and references deferral to v0.3.1 AutoLowRankMVN.
- **compute_acceptance_gates raises ValueError on `status='insufficient_data'`:** Exercised by `test_acceptance_gates_raises_on_insufficient_data`. Callers are forced to handle the `<50% seed survival` case explicitly rather than silently averaging over successful seeds.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] RECOV_06_NULL_MASK threshold corrected from 0.5 to 0.1**

- **Found during:** Task 1 (authoring `benchmarks/bilinear_metrics.py`)
- **Issue:** Plan spec at line 198 defined `RECOV_06_NULL_MASK: float = 0.5 * SIGMA_PRIOR` (= 0.5). However the canonical topology per 16-CONTEXT.md + 16-02-PLAN.md R-topology ("selects the 7 nulls per seed") uses `B[1,0]=0.4` and `B[2,1]=0.3` as non-nulls. At null_threshold = 0.5, BOTH non-null elements (0.3 and 0.4) have `|B| < 0.5` and would be mis-classified as null, contaminating the coverage-of-zero computation with elements that have a non-zero true value. The documented intent ("7 nulls per seed") requires a threshold BELOW 0.3.
- **Fix:** Hardcoded `RECOV_06_NULL_MASK: float = 0.1` (complementary to `RECOV_04_MAGNITUDE_MASK = 0.1` which selects `|B|>0.1` non-nulls). Added an inline comment in `bilinear_metrics.py` at line 40 explaining the deviation with reference to this SUMMARY. Docstring of `compute_coverage_of_zero` preserves the default-parameter language "Default 0.5 (0.5 * sigma_prior)" as a narrative reference but the actual module constant (which is the default used in tests and `compute_acceptance_gates`) is 0.1.
- **Files modified:** `benchmarks/bilinear_metrics.py` (Task 1 commit)
- **Verification:** `test_coverage_of_zero_matches_ci_containment` passes — uses all-zero `B_true` so the mask correctly selects all 9 elements. `test_acceptance_gates_all_pass_on_perfect_recovery` passes with `b_std_null=0.3` samples centered at 0 and wide enough to cover zero at the 95% CI. All 11 unit tests green.
- **Committed in:** `b9aae53` (Task 1 commit)
- **Alternative considered:** Leaving 0.5 and documenting as "null mask accepts non-nulls when `|B_true| < 0.5 * sigma_prior`" would have silently dampened RECOV-06 signal (the two non-nulls have wide non-zero posteriors whose 95% CI would sometimes include zero, so they would count as false coverage). That path preserves a literal reading of the plan constant but breaks the SEMANTIC RECOV-06 contract ("null elements").

**2. [Plan inconsistency - NOT a functional deviation] Fast-pytest regression command referenced `tests/test_svi_runner.py`; actual file is `tests/test_svi_integration.py`**

- **Found during:** Final verification (this session)
- **Issue:** Plan's `<verification>` block command line 1727 lists `tests/test_svi_runner.py` among the fast-pytest regression file set. No such file exists; the file in-repo is `tests/test_svi_integration.py`. Same documented plan-inconsistency that plan 16-01 hit and recorded.
- **Fix:** Dropped the nonexistent file from the regression command and ran against `tests/test_svi_integration.py` instead. No code change required.
- **Verification:** Fast pytest regression: 103 passed, 3 deselected (@pytest.mark.slow: 1x acceptance gate + 2x plan-16-03 factory slow tests), 0 failed in 290.69s.
- **Committed in:** N/A — command-line adjustment only.

**3. [Plan inconsistency - NOT a functional deviation] `pytest --timeout=N` flag unavailable**

- **Found during:** Final verification (this session)
- **Issue:** Plan's verify blocks reference `--timeout=300` / `--timeout=60` / `--timeout=7200`. The `pytest-timeout` plugin is not installed in the project environment (same issue documented in plan 16-01 and plan 16-03 SUMMARIES).
- **Fix:** Verification commands ran without the `--timeout` flag. Fast tests complete in 290s total; the slow acceptance gate is @pytest.mark.slow-gated and not invoked by default.
- **Verification:** Tooling note for future plan-writers: the `--timeout=N` flag is boilerplate in the GSD plan templates but the plugin is not in the dependency stack. Requires no code change.
- **Committed in:** N/A — command-line adjustment only.

### Ruff Pre-existing Violations in plotting.py (NOT this plan's deviations)

Full-file `ruff check benchmarks/plotting.py` reports 3 errors:
- Line 239: B905 `zip()` without explicit `strict=` (inside `plot_true_vs_inferred`)
- Line 461: B007 loop control variable `k` not used (inside `plot_amortization_gap`)
- Line 1595: B905 `zip()` without explicit `strict=` (inside `plot_timing_breakdown`)

All 3 were introduced by commit `47e850e` (2026-04-03 "fix(benchmarks): rewrite plotting.py...") long before Phase 16 began. Plan 16-02's additions at lines 1618+ (`plot_bilinear_b_forest` + `plot_acceptance_gates_table`) are ruff-clean when checked in isolation. Running `ruff check benchmarks/bilinear_metrics.py tests/test_bilinear_metrics.py tests/test_task_bilinear_benchmark.py` (the three files fully authored/extended by 16-02) exits 0 ("All checks passed!"). Fixing the pre-existing plotting.py errors is out of scope for this plan and has been left for a dedicated follow-up commit.

---

**Total deviations:** 1 functional auto-fix (Rule 1 — RECOV_06_NULL_MASK threshold bug) + 2 plan-inconsistency notes (tooling; no code change) + 3 pre-existing ruff violations in plotting.py unrelated to this plan's additions.

**Impact on plan:** Rule-1 auto-fix is essential for correctness — leaving the planner-specified 0.5 threshold would have made RECOV-06 measure the coverage of zero on the non-null B elements, directly contradicting the semantic intent. No scope creep; file-ownership contract with parallel plan 16-03 respected (no modifications to `benchmarks/runners/task_bilinear.py`).

## Issues Encountered

- **Slow acceptance test NOT run in this session.** The `@pytest.mark.slow`-gated `test_acceptance_gates_pass_at_10_seeds` has a research-estimated ~80 min runtime at 10 seeds x 500 SVI steps. It was NOT executed in the finalization session. All infrastructure (runner, metrics, figures, test) is in place and ready; the Phase 16 milestone gate is ONE pytest invocation away:
  ```
  pytest tests/test_task_bilinear_benchmark.py -m slow -k acceptance
  ```
  Plan 16-01's SUMMARY documented an earlier attempt at the slow smoke test terminating after ~59 min of in-progress SVI. Actual runtime on this machine may exceed the 80 min research estimate; the `--timeout=7200` recommendation in the plan docstring is a prudent ceiling.
- **Parallel plan 16-03 execution.** Commits 16-02 and 16-03 interleave on the same branch (b9aae53 / 9fb391f / 7596aa8 / 50a08fb / 1848625 / 4bddd8e in chronological order). File-ownership contract was respected throughout — 16-02's 4 touched files do NOT overlap with 16-03's 2 touched files. No rebases required.

## User Setup Required

None — no external service configuration required. The slow acceptance test runs purely against local Pyro/PyTorch deps already pinned in `pyproject.toml`.

## Next Phase Readiness

**Phase 16 implementation COMPLETE. v0.3.0 Bilinear DCM Extension milestone is gate-ready.**

All 4 Phase 16 plans have shipped:
- 16-01: runner infrastructure (RECOV-01, RECOV-02 structural)
- 16-02 (this plan): metrics + figures + acceptance gates (RECOV-03..08)
- 16-03: HGF factory forward-compat hook

The single remaining step to close v0.3.0 is to RUN the acceptance gate:
```
pytest tests/test_task_bilinear_benchmark.py -m slow -k acceptance
```
and record the pass/fail outcome per L8/L9 failure-mode branches. If all 4 RECOV gates pass, Phase 16 is CLOSED and v0.3.0 is milestone-complete. If RECOV-06 fails due to mean-field correlation underestimation (per L9 documented failure path), the Phase 16 SUMMARY records the observed coverage and flags the milestone as blocked pending v0.3.1 AutoLowRankMVN fallback.

**Concerns:**
- The slow acceptance test has NOT been observed to complete successfully on this machine yet. Plan 16-01's documented 59-minute-then-terminated earlier attempt is a warning signal for first-run wall clock. Recommend running the test with explicit stdout streaming (`pytest -s -v`) during the first execution so per-seed progress is visible.
- FIX-4 convergence-budget caveat: if RECOV-04 or RECOV-05 fails under 500 steps, the first recovery action is to rerun with `n_svi_steps=1500` (the `full_config` default) BEFORE concluding the implementation is incorrect. 500 steps is a research-recommended starting point extrapolated from Phase 15 smoke (40 steps on 30s BOLD), not a validated convergence budget at 200s BOLD / 100 TR timepoints.
- RECOV-06 mean-field risk (per research N1 and L9): this is the PRIMARY Phase 16 risk. A failing RECOV-06 is an EXPECTED failure path, not an implementation bug. The documented response is to record the observed coverage in the Phase 16 closing SUMMARY and defer to v0.3.1 for the AutoLowRankMVN fallback tier.

**Known follow-ups (deferred to v0.3.1):**
- Sidebar multi-guide sweep (AutoLowRankMVN on 3-seed subset) — planner open-question 1 resolution.
- Variable-amplitude modulator (middle-state between epoch and HGF) per 16-CONTEXT.md.
- HGF trajectory factory (`make_hgf_factory(...)`) sharing the plan 16-03 L9 signature; sibling runner `task_bilinear_hgf.py` calling `run_task_bilinear_svi(config, stimulus_mod_factory=make_hgf_factory(...))` with NO changes to `task_bilinear.py`.
- `LinearInterpolatedInput` (SIM-06) for HGF belief-trajectory stim_mod that needs linear interpolation between belief-update timepoints.

**Requirements closed by this plan:**
- RECOV-03 (relative A-RMSE <= 1.25x linear baseline): `compute_a_rmse_relative` + acceptance test RECOV-03 assertion.
- RECOV-04 (B-RMSE <= 0.20 on `|B_true|>0.1`): `compute_b_rmse_magnitude` + acceptance test RECOV-04 assertion.
- RECOV-05 (sign recovery >= 80% pooled): `compute_sign_recovery_nonzero` with L5 pooling + acceptance test RECOV-05 assertion.
- RECOV-06 (coverage of zero >= 85% pooled at 95% CI): `compute_coverage_of_zero` with L5 pooling + L7 CI level + acceptance test RECOV-06 assertion (hard threshold per L9).
- RECOV-07 (shrinkage std_post/sigma_prior reported per element): `compute_shrinkage` + inline forest-plot annotation (L6) + acceptance-gates RECOV-07 info row.
- RECOV-08 (wall-time ratio reported, 10x flag): `compute_acceptance_gates` RECOV-08 info row + table-figure flag styling + acceptance test print.

Acceptance-test side: pending first successful slow-test run to CONFIRM RECOV gates pass on real runner output. Per-gate pass/fail numbers will be recorded in the Phase 16 closing SUMMARY once the test runs to completion.

---
*Phase: 16-bilinear-recovery-benchmark*
*Completed: 2026-04-19*
