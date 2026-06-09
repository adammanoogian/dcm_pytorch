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

## Solution (sketch / hypotheses)

1. Under-training / too few simulations: 1000-sim budget may be too small for the CSD
   dimensionality. Try a larger simulation budget and/or sequential (SNPE) rounds.
2. Embedding-network mismatch: the CSD embedding (LayerNorm variant) may discard
   phase/Hermitian structure; check the embedding against the spectral features that matter.
3. Prior/range mismatch: SBC fails systematically if the prior used for SBC differs from the
   training prior, or if the simulator occasionally diverges (NaN CSD) and those draws are
   silently dropped -- audit the reject/replace path.
4. Compare against the VL posterior on the same synthetic CSD as a calibrated reference.

This blocks any SBI-based scientific claim; SVI/VL remains the trustworthy path meanwhile.
