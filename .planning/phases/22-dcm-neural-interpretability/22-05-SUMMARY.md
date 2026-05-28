---
phase: 22-dcm-neural-interpretability
plan: 05
subsystem: interpretability-validation
tags: [perturbation, validation, spectral-dcm, experiment, analysis, cluster]
dependency-graph:
  requires: ["22-01", "22-02", "22-03", "22-04"]
  provides:
    - Perturbation experiment framework (9 conditions: strengthen/weaken/remove across strong/medium/weak connections)
    - Analysis script with 5 diagnostic figures and z-score detection metric
    - M3 cluster sbatch for full perturbation sweep (12h, 32GB)
    - Smoke tests verifying perturbation detection and baseline stability
  affects:
    - "22-06 (publication figures use perturbation results)"
tech-stack:
  added: []
  patterns:
    - "Perturbation validation: known A delta -> same autoencoder -> detectable posterior delta"
    - "Z-score detection metric: |delta_A[perturbed]| / std(delta_A[unperturbed])"
key-files:
  created:
    - scripts/22_perturbation_experiment.py
    - scripts/22_analyze_perturbation.py
    - cluster/sbatch/22_perturbation.sbatch
    - tests/test_perturbation_recovery.py
  modified: []
decisions: []
metrics:
  duration: ~8 minutes
  completed: 2026-05-27
---

# Phase 22 Plan 05: Perturbation Experiment Framework Summary

Perturbation experiment framework validating DCM as an interpretability tool: 9 conditions (strengthen/weaken/remove) across strong/medium/weak connections with z-score detection metric, analysis script producing 5 diagnostic figures, M3 cluster sbatch, and two smoke tests (perturbation detection + baseline stability).

## What Was Done

### Task 1: Perturbation experiment script and smoke tests (76ae269)

Created `scripts/22_perturbation_experiment.py` with the full perturbation pipeline:

1. **Baseline generation**: `make_sensorimotor_A(seed)` -> `simulate_meg_timeseries(n_train)` for autoencoder training + `simulate_meg_timeseries(n_eval, seed+1000)` for baseline evaluation
2. **Autoencoder training**: `MEGAutoencoder(n_roi=10, n_latent=20)` trained with `AutoencoderTrainer` for configurable epochs, checkpoint saved
3. **Baseline DCM fit**: `extract_latent_trajectories` -> `compute_latent_csd` -> `prepare_for_spectral_dcm` -> `spectral_dcm_model` with `prior_a_var=1/16`, `eig_clamp=-1.0`
4. **Perturbation sweep** (9 conditions):
   - M1_S1_strengthen/weaken/remove (strong connection: 0.15)
   - PMC_M1_strengthen/weaken/remove (feedforward: 0.10)
   - A1_M1_strengthen/remove (weak auditory-motor: 0.05)
   - bilateral_M1_strengthen (bilateral: 0.08)
   - Each condition: perturb A -> generate data -> same autoencoder -> latent CSD -> DCM -> delta_A
5. **Results saved** to `perturbation_results.npz` with all posterior statistics

CLI arguments: `--output-dir`, `--n-train`, `--n-eval`, `--n-latent-multiplier`, `--hidden-size`, `--ae-epochs`, `--svi-steps`, `--n-restarts`, `--seed`

Created `tests/test_perturbation_recovery.py` with 2 smoke tests:

| # | Test | What it validates |
|---|------|-------------------|
| 1 | test_perturbation_changes_posterior | 4-region net, A[0,1]*=2, verifies max delta > median delta and > 1e-4 |
| 2 | test_no_perturbation_stable_posterior | Baseline twice with different seeds, verifies max delta < 0.5 |

Both marked `@pytest.mark.slow`, run in ~25-29s total.

### Task 2: Analysis script and cluster sbatch (8ef614d)

Created `scripts/22_analyze_perturbation.py` with 5 diagnostic figures:

| # | Figure | Description |
|---|--------|-------------|
| 1 | detection_heatmap.png | Rows=conditions, cols=A elements, color=|delta_A|, red star marks perturbed element |
| 2 | effect_size_bar.png | Horizontal bar chart of z-scores, colored green/orange/red by detection threshold |
| 3 | sensitivity_vs_strength.png | Scatter of baseline |A[i,j]| vs detection z-score with condition labels |
| 4 | true_vs_recovered_delta.png | Scatter of true delta vs posterior delta for perturbed elements |
| 5 | baseline_A_heatmap.png | Ground-truth A matrix with ROI labels |

Also prints summary table: condition | ij | true_delta | post_delta | z_score | detected?

Created `cluster/sbatch/22_perturbation.sbatch`:
- SLURM: `--job-name=perturb_22`, `--time=12:00:00`, `--mem=32G`, `--cpus-per-task=4`, `--partition=comp`
- Uses `cluster/lib/cluster_env.sh` conventions (crlf_guard, setup_torch_threads, activate_env, verify_torch)
- No pip install in job (follows convention)
- Configurable via environment variables: `PERTURB_SVI_STEPS`, `PERTURB_N_RESTARTS`, `PERTURB_AE_EPOCHS`, `PERTURB_SEED`, `PERTURB_OUTPUT_DIR`
- Defaults: 1000 SVI steps, 10 restarts, 150 AE epochs (production settings)

## Test Results

- **New tests:** 2/2 pass (both `@pytest.mark.slow`, ~25s total)
- **Ruff:** Clean on all 4 new files

## Deviations from Plan

None -- plan executed exactly as written.

## Decisions Made

None -- no architectural decisions needed.

## Verification

- [x] `python scripts/22_perturbation_experiment.py --help` prints usage
- [x] `python scripts/22_analyze_perturbation.py --help` prints usage
- [x] `pytest tests/test_perturbation_recovery.py -v -m slow` -- 2/2 pass
- [x] `ruff check` clean on all new files
- [x] Sbatch has valid SLURM headers and follows cluster conventions
- [x] 9 perturbation conditions covering strengthen/weaken/remove across strong/medium/weak connections

## Next Phase Readiness

Plan 22-05 delivers the complete perturbation validation framework for Phase 22:
- Experiment script ready for M3 cluster submission via sbatch
- Analysis script generates all diagnostic figures needed for Plan 22-06 (publication figures)
- Smoke tests confirm the approach works (perturbation detection + baseline stability)
- No blockers for Plan 22-06
