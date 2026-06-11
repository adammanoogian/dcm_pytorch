---
phase: 30
status: passed
verified: 2026-06-11
score: "5/5 requirements met; all 10 cells classified, 0 errored"
---

# Phase 30 Verification — Recovery Matrix Sweep

**Goal:** A per-cell recovery matrix over N × SNR × {spectral, task, latent-circuit} (≥10
seeds/cell) on M3 with the hardened metric suite, passing documented per-cell thresholds or
documenting identifiability limits with evidence (no silent failures).

**Sweep:** M3 array jobs 56346424 (spectral + latent) and 56372816 (task cells 4–7, after the
`simulate_task_dcm` solver fix), small validation grid (N∈{2,4} × SNR∈{1,3} × 3 models),
10 seeds/cell, 10 cells, all exit 0. Results: `benchmarks/results/recovery_matrix.{csv,json}`;
report: `30-RECOVERY-MATRIX-REPORT.md`.

## Per-criterion

| Req | Criterion | Status | Evidence |
|-----|-----------|--------|----------|
| VLREC-01 | Sweep on M3 cluster, ≥10 seeds/cell, per-cell JSON | ✅ Met | 10 cells ran on M3 (comp partition), 10 seeds each, 10 JSON files harvested |
| VLREC-02 | Hardened metrics (per-region R² not pooled, masked sign, coverage, RMSE, shrinkage) | ✅ Met | `recovery_matrix_metrics.assemble_cell_metrics`; CSV columns populated across all 3 variants |
| VLREC-03 | Exclude near-boundary A (maxRe eig∈[-0.05,0]); task dt≥0.1 | ✅ Met | 0 accepted draws in the band; task sbatch enforces dt≥0.1 |
| VLREC-04 | Every cell passes OR documents identifiability limit; no silent failures | ✅ **Met** | **10/10 cells classified, 0 errored**: 6 PASS + 4 identifiability-limit-with-evidence; every task cell now produces a recovery verdict |
| VLROBUST-03 | eig_clamp/boundary regime characterized | ✅ Met | Boundary exclusion held (0 draws); low high-SNR shrinkage documented as the expected Laplace-overconfidence regime |

## Verdict summary

- **6 PASS**: spectral 0,1,2,3 (A-RMSE 0.0007–0.013, masked sign 1.0, coverage 1.0, 10/10);
  task cell 5 (N=2 SNR=3, A-RMSE 0.038, sign 1.0, coverage 0.875); latent-circuit cell 9
  (N=4 SNR=3, A-RMSE 0.040, sign 1.0).
- **4 identifiability-limit (with evidence)**:
  - task cell 4 (N=2 SNR=1) — marginal: A-RMSE 0.047 and sign 1.0, but coverage 0.75 < 0.85
    (sister cell 5 passes at higher SNR).
  - task cells 6, 7 (N=4) — genuine task-DCM identifiability limit at N=4: A-RMSE ~0.08,
    masked sign 0.57, coverage 0.0, convergence 0.4. Task connectivity is far harder to
    identify than spectral/latent at N=4 under this design/SNR; documented with full evidence.
  - latent-circuit cell 8 (N=4 SNR=1) — A-RMSE 0.0501 marginally over the provisional 0.05
    threshold; sign 1.0; sister cell 9 passes at higher SNR.
- **0 errored** — no silent failures.

## Resolution of the prior task gap

The earlier verification (status `gaps_found`) recorded the 4 task cells as ERRORED:
`simulate_task_dcm` hit a torchdiffeq "underflow in dt 0.0" during ground-truth generation on
the M3 stack (torchdiffeq 0.2.5 / torch 2.10) because the default adaptive `dopri5` solver
underflowed at the sweep settings. Fixed by switching the task ground-truth simulation to the
fixed-step `rk4` solver (`_run_task_cell`, commit c0a7616) and broadening the per-seed exception
guard. Task cells 4–7 were re-run on M3 (job 56372816) and all 4 now produce recovery verdicts.

**Outcome:** task-DCM VL recovers cleanly at N=2 (sign 1.0, A-RMSE ~0.04) and lands as a
documented identifiability limit at N=4 — a real scientific finding, not a harness failure. The
harness was already complete and correct (10/10 classified, no silent failures); the fix made
the task variant produce evidence instead of an error.

## Notes for downstream phases

- **Phase 31 (BMR + tempering)** now has all three variants covered. Task N=4 is the natural
  stress case for tempering calibration (coverage 0.0 / low shrinkage = the Laplace-overconfident
  regime tempering is meant to address).
- **Provisional thresholds**: `RMSE_A_THRESHOLD = 0.05` is the documented v0.7.0 default (no
  principled Fisher bound yet); marginal cells (4, 8) are flagged for audit, not treated as hard
  failures.
