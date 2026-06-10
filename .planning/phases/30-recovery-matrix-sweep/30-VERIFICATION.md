---
phase: 30
status: gaps_found
verified: 2026-06-11
score: "4/5 requirements fully met; VLREC-04 partial (task variant blocked)"
---

# Phase 30 Verification — Recovery Matrix Sweep

**Goal:** A per-cell recovery matrix over N × SNR × {spectral, task, latent-circuit} (≥10
seeds/cell) on M3 with the hardened metric suite, passing documented per-cell thresholds or
documenting identifiability limits with evidence (no silent failures).

**Sweep:** M3 array job 56346424, small validation grid (N∈{2,4} × SNR∈{1,3} × 3 models),
10 seeds/cell, 10 cells, all exit 0. Results: `benchmarks/results/recovery_matrix.{csv,json}`;
report: `30-RECOVERY-MATRIX-REPORT.md`.

## Per-criterion

| Req | Criterion | Status | Evidence |
|-----|-----------|--------|----------|
| VLREC-01 | Sweep on M3 cluster, ≥10 seeds/cell, per-cell JSON | ✅ Met | 10 cells ran on M3 (comp partition), 10 seeds each, 10 JSON files harvested |
| VLREC-02 | Hardened metrics (per-region R² not pooled, masked sign, coverage, RMSE, shrinkage) | ✅ Met | `recovery_matrix_metrics.assemble_cell_metrics`; CSV columns populated for spectral+latent |
| VLREC-03 | Exclude near-boundary A (maxRe eig∈[-0.05,0]); task dt≥0.1 | ✅ Met | 0 accepted draws in the band; task sbatch enforces dt≥0.1 |
| VLREC-04 | Every cell passes OR documents identifiability limit; no silent failures | ⚠️ **Partial** | 6/10 cells got recovery verdicts (5 PASS + 1 identifiability-limit); the 4 task cells **ERRORED** (surfaced explicitly in CSV/report, NOT silent) but produced no task-variant recovery verdict |
| VLROBUST-03 | eig_clamp/boundary regime characterized | ✅ Met | Boundary exclusion held (0 draws); low high-SNR shrinkage documented as the expected Laplace-overconfidence regime |

## Verdict summary

- **5 PASS**: spectral 0,1,2,3 (A-RMSE 0.0007–0.013, masked sign 1.0, coverage 1.0, 10/10);
  latent-circuit cell 9 (N=4 SNR=3, A-RMSE 0.040, sign 1.0).
- **1 identifiability-limit (with evidence)**: latent-circuit cell 8 (N=4 SNR=1, A-RMSE 0.0501
  marginally over the provisional 0.05 threshold; sign 1.0; sister cell 9 passes at higher SNR).
- **4 errored (surfaced, not silent)**: task cells 4–7 — `simulate_task_dcm` torchdiffeq
  "underflow in dt 0.0" during ground-truth generation at the sweep settings.

## Gap

**The task-DCM variant is unvalidated** — `simulate_task_dcm` hits an adaptive-step underflow
on the cluster at the sweep settings (task_vl ran fine locally at the N=2 smoke in 29-04, so it
is settings-dependent: a numerical-stiffness / dt_sim issue in the task simulator, consistent
with the pre-existing task-path fragility noted in 29-03/29-05). This blocks task-variant
coverage, which Phase 31 tempering calibration would want.

**Harness is complete and correct** (10/10 cells classified, no silent failures); spectral and
latent-circuit VL recovery are validated. The gap is a targeted simulator fix + 4-cell rerun.

## Recommended next

Either (a) diagnose + fix the `simulate_task_dcm` underflow and re-run task cells 4–7 on M3
(closes Phase 30 fully), or (b) document the task-variant gap and proceed — Phase 31 can use the
spectral + latent coverage now; revisit task before any task-variant claim.
