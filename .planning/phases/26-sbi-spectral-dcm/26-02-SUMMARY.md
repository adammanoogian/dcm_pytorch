---
phase: 26-sbi-spectral-dcm
plan: 02
subsystem: inference
tags: [sbi, npe, sbc, spectral-dcm, amortized-inference, cluster, calibration]
backfilled: 2026-06-09
dependency-graph:
  requires: ["26-01"]
  provides: ["train_sbi_spectral", "compare_sbi_svi", "sbi_spectral_train_sbatch"]
  affects: []
tech-stack:
  added: []
  patterns: ["npe-posterior-nn", "manual-simulation-loop", "sbc-ks-uniformity", "amortized-speed-benchmark", "layernorm-embedding"]
key-files:
  created:
    - scripts/train_sbi_spectral.py
    - scripts/compare_sbi_svi.py
    - cluster/sbi_spectral_train.sh
  modified:
    - src/pyro_dcm/inference/sbi_spectral.py
    - src/pyro_dcm/inference/sbi_embedding.py
decisions:
  - id: "26-02-D1"
    description: "train_npe uses sbi 0.26 posterior_nn factory with a manual simulation loop rather than simulate_for_sbi; SNPE takes a prebuilt density_estimator (not an embedding_net kwarg) in this API version."
  - id: "26-02-D2"
    description: "CSD embedding net switched from BatchNorm1d to LayerNorm because sbi probes the network with batch_size=1 during init, which BatchNorm1d rejects (requires >1 sample)."
  - id: "26-02-D3"
    description: "simulate_for_sbi 1D simulator outputs concatenated with torch.cat (not torch.stack) to preserve the flattened (2*F*N*N) observation layout expected by the embedding net."
metrics:
  duration: "~5 minutes (two commits, same evening)"
  completed: "2026-05-28"
---

# Phase 26 Plan 02: NPE Training, SBC Validation, and SBI-vs-SVI Comparison Summary

End-to-end SBI pipeline for spectral DCM: an NPE training + SBC validation script,
a cluster sbatch wrapper for M3, and a quantitative SBI-vs-SVI posterior comparison
script. Includes sbi 0.26 API fixes to the underlying inference module.

## What Was Done

### Task 1: NPE training + SBC validation script + cluster sbatch

**1. Created `scripts/train_sbi_spectral.py`** (283 lines, commit 6b27161):

End-to-end NPE training and SBC calibration script with an argparse CLI
(`--n-regions`, `--n-sims`, `--n-sbc`, `--n-freqs`, `--tr`, `--embed-dim`,
`--hidden-features`, `--num-transforms`, `--max-epochs`, `--batch-size`,
`--output-dir`, `--seed`). Pipeline:

- Builds a fully-connected `a_mask`, a frequency grid, the spectral DCM simulator,
  prior, and a `CSDEmbeddingNet` embedding network.
- Trains the NPE density estimator via `train_npe`, timing the run.
- Builds the amortized posterior and runs SBC validation over `--n-sbc` trials.
- Saves three artifacts to `--output-dir`: `estimator.pt` (trained estimator),
  `sbc_ranks.pt` (SBC rank tensor), and `training_metadata.pt` (run config + timing).
- Reports SBC rank uniformity via a per-parameter Kolmogorov-Smirnov test (PASS/FAIL
  at p > 0.05) and an amortized inference speed benchmark (100 posterior samples,
  asserting mean < 1s/subject — Phase 26 success criterion 2).

**2. Created `cluster/sbi_spectral_train.sh`** (115 lines, commit 6b27161):

M3 sbatch script following project conventions for the full 50k-simulation training
run, writing stdout/stderr to `cluster/logs/sbi_spectral_${SLURM_JOB_ID}.{out,err}`
and saving results to `results/sbi_spectral_${SLURM_JOB_ID}`.

**3. sbi 0.26 API fixes** (commit 6b27161):

- `src/pyro_dcm/inference/sbi_spectral.py`: `train_npe` rewritten to use the
  `posterior_nn` factory plus a manual simulation loop. In sbi 0.26, `SNPE` takes a
  prebuilt `density_estimator` rather than an `embedding_net` kwarg, and
  `simulate_for_sbi` concatenates 1D simulator outputs with `torch.cat` (not
  `torch.stack`).
- `src/pyro_dcm/inference/sbi_embedding.py`: replaced `BatchNorm1d` with `LayerNorm`
  because sbi probes the network with `batch_size=1` during initialization, which
  `BatchNorm1d` rejects.

### Task 2: SBI vs SVI comparison script

**Created `scripts/compare_sbi_svi.py`** (322 lines, commit a2db418):

Quantitative comparison of SBI (NPE) and SVI posteriors on synthetic spectral DCM
ground truth. For each synthetic test subject it runs amortized SBI inference and
per-subject SVI inference, then computes per-subject metrics (RMSE vs ground truth,
95% CI overlap, inference timing). Aggregates into a summary table reporting the
speed ratio and the amortization gap, and saves results to `comparison_results.pt`.

## Cluster Validation

The full training run was executed on M3 as **job 55772094** (node m3s111,
sbi 0.26.1, Python 3.10.20) and **completed exit-0**. Configuration: N=3 regions,
32 frequencies, 50,000 simulations, 200 SBC trials, seed 42.

Outcome (from `cluster/logs/sbi_spectral_55772094.out`):

- **NPE trained**: converged after 285 epochs in 2862.1s. Estimator saved.
- **SBC ranks produced**: 200 trials in 10.1s; ranks saved.
- **Amortized inference speed**: 0.0239 +/- 0.0010 s/call (n=100) — **PASS** (<1s,
  Phase 26 success criterion 2).
- **Artifacts** saved to `results/sbi_spectral_55772094/`: `estimator.pt`,
  `sbc_ranks.pt`, `training_metadata.pt` (all present on disk).

**SBC calibration verdict: FAIL (2/9 parameters pass).** The KS uniformity test
reports only `param_6` (p=0.0589) and `param_8` (p=0.3892) above the p>0.05
threshold; the other 7 parameters fail (KS stats 0.12–0.29, p < 0.006). This does
not meet the plan's success criterion "SBC validates posterior calibration (KS test
p > 0.05 for majority of parameters)". The trained estimator amortizes correctly and
fast, but the posterior is not well-calibrated at the current 50k-simulation budget /
NSF architecture.

## Checkpoint Status

Task 2 is a `checkpoint:human-verify` gate (blocking). The automated work (Tasks 1
and the cluster run) is complete, but the SBC calibration check fails. Per the plan's
resume signal, this is the branch where the user decides whether to increase `n_sims`,
switch density estimator architecture (NSF -> MAF), or move to TSNPE (RESEARCH.md
Open Question 2). No "approved" sign-off is recorded.

## Next Steps

1. Address SBC miscalibration (more simulations, alternative density estimator, or
   TSNPE) and re-run the M3 training job.
2. Run `scripts/compare_sbi_svi.py` against the (re-)trained `estimator.pt` to
   quantify the amortization gap once calibration is acceptable.
3. Obtain human verification sign-off on the checkpoint gate.

## Anomalies

- Commit **a2db418** bundled four files: `scripts/compare_sbi_svi.py` (this plan)
  plus three unrelated Phase 24 files (`cluster/sbatch/24_tribe_extract.slurm`,
  `scripts/24_extract_tribe_latents.py`, `scripts/24_fit_dcm_tribe.py`). Only
  `compare_sbi_svi.py` belongs to 26-02.
