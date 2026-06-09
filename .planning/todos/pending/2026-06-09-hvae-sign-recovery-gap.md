---
created: 2026-06-09T23:30
title: Phase 25 HVAE-02 amortized sign recovery 0.44 < 0.6
area: hybrid-vae-dcm
priority: medium
files:
  - results/hybrid_vae_dcm/recovery_report.json
  - src/pyro_dcm/models/hybrid_vae_dcm.py
  - .planning/phases/25-hybrid-vae-dcm/25-04-SUMMARY.md
---

## Problem

Full HVAE-DCM cluster training (job 55774467, 200 epochs) met 3/4 HVAE acceptance criteria but
**A sign recovery = 0.4425 vs the >0.6 threshold** (A_free RMSE 0.076, KL 18.8 no-collapse,
inference 0.76ms all pass). This is the same connectivity-sign identifiability weakness that
the Phase 20-05 mean-field SVI showed -- and which **Variational Laplace fixed** there
(B-RMSE 0.31 -> 0.0048, sign 1.00).

## Solution (sketch / hypotheses)

The HVAE encoder is an amortized mean-field-style posterior; like SVI it likely under-constrains
the sign of weakly-identified connections. Options, in rough order:
1. Diagnose whether the 0.44 is a metric/ground-truth artifact (as the 20-05 R2 turned out to be)
   -- e.g. sign recovery on near-zero true entries is coin-flip; mask to |A_true| above a
   threshold before scoring, mirroring the B sign-recovery mask in latent_circuit_recovery.
2. If real: richer amortized posterior (full/low-rank covariance head on the encoder) or a
   physics-informed prior that breaks the sign symmetry.
3. Cross-check against a VL fit on the same synthetic examples as an oracle for what sign
   recovery is achievable at this SNR.

Start with (1) -- the 20-05 lesson is that "recovery failures" here have repeatedly been
metric/identifiability artifacts, not method failures.
