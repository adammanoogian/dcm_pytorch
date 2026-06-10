---
phase: 30-recovery-matrix-sweep
plan: 03
subsystem: testing
tags: [recovery-matrix, variational-laplace, classifier, identifiability, vlrec-04, vlrobust-03]

# Dependency graph
requires:
  - phase: 30-02
    provides: "M3 array job 56346424 per-cell JSONs (10 cells, 120 fits) synced locally"
  - phase: 30-01
    provides: "assemble_cell_metrics contract + NEAR_BOUNDARY band constants"
provides:
  - "benchmarks/recovery_matrix_thresholds.py — documented per-cell thresholds + classify_cell (pass | identifiability_limit with evidence)"
  - "cluster/scripts/recovery_matrix_aggregate.py — harvest per-cell JSON -> matrix CSV/JSON + report + eig_clamp/boundary regime characterization"
  - "30-RECOVERY-MATRIX-REPORT.md — per-cell verdict table (5 PASS / 1 ident-limit / 4 errored surfaced)"
  - "benchmarks/results/recovery_matrix.{csv,json} — the harvested matrix over (variant, N, SNR)"
affects: [phase-31-tempering-calibration, vl-coverage-diagnostics]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Pass-or-documented-limit classifier (VLREC-04): a failing cell is a documented identifiability limit WITH evidence, never a silent drop or raised error"
    - "Parameterized report path so tests write to tmp_path, not the real .planning/ report"

key-files:
  created:
    - benchmarks/recovery_matrix_thresholds.py
    - cluster/scripts/recovery_matrix_aggregate.py
    - tests/test_recovery_matrix_aggregate.py
    - .planning/phases/30-recovery-matrix-sweep/30-RECOVERY-MATRIX-REPORT.md
    - benchmarks/results/recovery_matrix.csv
    - benchmarks/results/recovery_matrix.json
  modified:
    - .planning/ROADMAP.md
    - .planning/STATE.md

key-decisions:
  - "Boundary regime characterized from raw.max_real_eig_list (no explicit boundary_rejections field exists in the JSON) + shrinkage overconfidence signal"
  - "Task-DCM cells (4-7) errored at the simulator (torchdiffeq 'underflow in dt 0.0'); surfaced as errored, not retried — a real task-path fragility, not a classifier outcome"
  - "Cell 8 (latent_circuit N=4 SNR=1) is a marginal identifiability limit (RMSE 0.0501 vs provisional 0.05); the audit-outlier case the threshold-research note anticipated"

# Metrics
duration: 15min
completed: 2026-06-10
---

# Phase 30 Plan 03: Recovery Matrix Harvest + Verdict Report Summary

**Documented per-cell threshold classifier + matrix aggregator run over the complete M3 sweep (job 56346424): all 10 cells classified — 5 PASS, 1 identifiability-limit-with-evidence, 4 errored task cells surfaced (no silent failures) — with the eig_clamp/stability-boundary regime characterized.**

## Performance

- **Duration:** 15 min
- **Started:** 2026-06-10T22:01:24Z
- **Completed:** 2026-06-10T22:16:32Z
- **Tasks:** 4 (3 auto + 1 post-results verdict)
- **Files modified/created:** 8

## Accomplishments

- `classify_cell` enforces VLREC-04 structurally: each cell is PASS (meets documented RMSE/sign/coverage thresholds) or `identifiability_limit` WITH evidence (shrinkage, coverage, RMSE IQR, convergence). It never raises on a failing cell; a missing metric is skipped (`pass=None`), not auto-failed. It raises only on structurally malformed input.
- The aggregator globs every per-cell JSON (excluding local pre-checks — including cell 9's parent-job-id filename), classifies each, writes `recovery_matrix.csv` + `recovery_matrix.json`, surfaces error-status cells in the CSV (`status=error`) and report, and fails loud (expected-vs-actual) when zero result files match.
- The eig_clamp / stability-boundary regime is characterized (VLROBUST-03): 0 accepted ground-truth draws fell in the `[-0.05, 0]` band (exclusion held); low shrinkage at high SNR is documented as the expected Laplace-overconfidence regime, not a bug.
- 4 vl tests (laptop, ~5s) concretely enforce no-silent-failures and fail-loud-on-empty.

## Per-cell verdict summary (REAL results, job 56346424)

| Verdict | Count | Cells |
|---|---|---|
| PASS | 5 | spectral 0,1,2,3 (RMSE 0.0007–0.013, sign 1.0, cov 1.0); latent_circuit cell 9 (N=4 SNR=3, RMSE 0.0396, sign 1.0, R2/region 0.345) |
| IDENTIFIABILITY-LIMIT | 1 | latent_circuit cell 8 (N=4 SNR=1): median A-RMSE 0.0501 marginally over the provisional 0.05 threshold (sign 1.0, R2/region 0.076); sister cell 9 passes at higher SNR — the expected audit-outlier case |
| ERRORED (surfaced) | 4 | task cells 4,5,6,7: torchdiffeq `underflow in dt 0.0` in `simulate_task_dcm` (adaptive-step underflow) — listed in report + CSV, never dropped |

**All 10 cells received an explicit verdict; no cell silently skipped.**

## Task Commits

1. **Task 1: documented thresholds + classifier** — `29978dd` (feat)
2. **Task 2: harvest + aggregate + boundary characterization** — `32a93ed` (feat)
3. **Task 3: aggregator tests on synthetic fixtures** — `819201e` (test)
4. **Task 4: run on real results + report deliverables** — `eb30643` (feat)

**Plan metadata:** see `docs(30-03)` commit.

## Files Created/Modified

- `benchmarks/recovery_matrix_thresholds.py` — `RMSE_A_THRESHOLD`/`SIGN_RECOVERY_THRESHOLD`/`COVERAGE_95_FLOOR` (cited provenance) + `SHRINKAGE_SOFT_TARGET` (informational) + `classify_cell`
- `cluster/scripts/recovery_matrix_aggregate.py` — `aggregate()`/`main()`: glob → classify → CSV/JSON + markdown report + boundary regime
- `tests/test_recovery_matrix_aggregate.py` — 4 vl tests (classify, writes-outputs, no-silent-failures, fail-loud)
- `.planning/phases/30-recovery-matrix-sweep/30-RECOVERY-MATRIX-REPORT.md` — the human-readable matrix + verdicts
- `benchmarks/results/recovery_matrix.csv` / `.json` — the harvested matrix deliverable

## Decisions Made

- **Boundary regime from `max_real_eig_list`.** The per-cell JSON has no explicit `boundary_rejections` field; the regime is characterized from the accepted-draw eigenvalue distribution (proximity to the `[-0.05, 0]` band) plus the shrinkage overconfidence signal. All accepted draws sat outside the band (in-band count 0).
- **Provisional 0.05 RMSE threshold kept explicit.** Cell 8's 0.0501 marginal miss is exactly the "audit outlier" case the v0.7.0 research note flagged; the classifier surfaces it WITH evidence rather than nudging the threshold — a valid VLREC-04 documented-limit outcome.

## Deviations from Plan

None — plan executed as written. The plan's checkpoint was a post-results verdict review; per the execution prompt, the report was built and the verdict returned without pausing.

## Issues Encountered

- **Task-DCM cells (4-7) errored on the cluster** with `underflow in dt 0.0` from torchdiffeq inside `simulate_task_dcm`. This is a real task-simulation-path fragility (consistent with the pre-existing `dt_sim` signature-drift failures noted in 29-05's STATE entry for `test_vl_forward_model_protocol.py`), surfaced here as errored cells per VLREC-04. It is NOT a defect in the 30-03 classifier/aggregator — those correctly harvested, surfaced, and counted the failures. Re-running the task variant (fixing the simulator dt handling) is a follow-up, not in this plan's scope.

## Next Phase Readiness

- Phase 30 is COMPLETE (3/3). The recovery matrix (coverage + shrinkage per cell) is now available as `benchmarks/results/recovery_matrix.json` for Phase 31 tempering calibration.
- **Concern for Phase 31/follow-up:** the 4 task-DCM cells produced no coverage data (simulator underflow). If Phase 31 tempering calibration needs task-variant coverage, the task simulator's adaptive-step underflow must be fixed and those cells re-run on M3 first. Spectral + latent_circuit coverage is available.

---
*Phase: 30-recovery-matrix-sweep*
*Completed: 2026-06-10*
