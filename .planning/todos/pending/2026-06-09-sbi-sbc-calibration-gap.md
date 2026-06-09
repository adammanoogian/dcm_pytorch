---
created: 2026-06-09T23:30
title: Phase 26 SBI-03 SBC calibration fails (2/9 params)
area: sbi-spectral-dcm
priority: medium
files:
  - cluster/logs/sbi_spectral_55772094.out
  - results/sbi_spectral_55772094/sbc_ranks.pt
  - src/pyro_dcm/inference/sbi_diagnostics.py
  - .planning/phases/26-sbi-spectral-dcm/26-02-SUMMARY.md
---

## Problem

The SBI (NPE) estimator for spectral DCM trained fine and is fast (<1s amortized), but
**simulation-based calibration FAILS: only 2/9 parameters pass the KS rank-uniformity test**
(p>0.05) on cluster job 55772094. SBI-03 requires nominal coverage. Posteriors are therefore
not calibrated -- the amortized SBI inference is not yet trustworthy.

## Diagnosis 2026-06-09 (failure mode characterised)

NOT under-training: the run used **50,000 simulations** (not 1000 — that was the prior-sampling
progress bar) and converged at 285 epochs. NOT global overconfidence either. The SBC rank
histograms (`results/sbi_spectral_55772094/sbc_ranks.pt`, 200×9) show **parameter-specific
bias**:

| param | A entry | mean-rank | KS | shape |
|-------|---------|-----------|-----|-------|
| 0 | A[0,0] (diag) | 0.32 | 0.29 | **biased high** (posterior over-shrinks self-inhibition) |
| 2 | A[0,2] | 0.65 | 0.28 | **biased low** |
| 1 | A[0,1] | 0.49 | 0.14 | underconfident (too wide) |
| 3–8 | — | 0.43–0.58 | 0.06–0.16 | mild biases (strict KS at 200 trials flags them) |

The 9 params are the full A_free (N=3, all-ones mask); **noise is fixed**, so SBC is purely over
connectivity.

### Root cause #1 (FIXED): dead flow-capacity args

`scripts/train_sbi_spectral.py` parsed `--num-transforms`, `--hidden-features`, `--max-epochs`
but **never passed them to `train_npe`** — `posterior_nn(model="nsf")` used the small nsf
defaults (5 transforms, 50 hidden). The diagonal parameterization `a_ii = -exp(A_free_ii)/2`
makes diagonal-param posteriors skewed, so flow capacity matters. **Fixed** (commit): plumbed
through; sbatch defaults now `NUM_TRANSFORMS=8`, `HIDDEN_FEATURES=128`, `MAX_EPOCHS=500`.
**Retrain experiment job 56274446 in flight** to test whether this restores calibration.

### Root cause #2 (hypothesis, if #1 insufficient): stability-boundary non-injectivity

Off-diagonals up to ±3σ≈±0.375 can push `A` past the stability boundary, where `eig_clamp`
(`-1/32`) clamps the dynamics → different `A_free` map to near-identical CSD → those params are
locally unidentifiable and NPE biases. If SBC still fails after #1, restrict the prior to the
stable region (or reparameterize), and/or compare against the VL posterior on the same CSD as a
calibrated oracle.

This blocks any SBI-based scientific claim; SVI/VL remains the trustworthy path meanwhile.
