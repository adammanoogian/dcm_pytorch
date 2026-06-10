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

## Resolution 2026-06-09: METRIC ARTIFACT (same class as the 20-05 R2)

Confirmed it was the metric, not the model. `train_hybrid_vae_dcm.py` computed sign recovery
as `(sign(A_pred)==sign(A_true)).mean()` over ALL 16 A_free entries. `make_stable_latent_circuit_A`
produces ~10 non-zero of 16 (avg, verified locally) -> ~6 structural zeros, and `sign(0)=0` can
never match a non-zero prediction, so each is a guaranteed miss. The observed 0.4425 = 7.08/16:
the model gets ~7.08 of the ~10 non-zero signs right (= **~0.71 masked**, which PASSES the >0.6
gate); the 6 zeros drag the unmasked score to 0.44.

Fix (commit pending):
- Added `masked_sign_recovery(pred, true, magnitude_threshold=0.1)` to `hybrid_vae_dcm.py`
  (mirrors the B sign-recovery mask in latent_circuit_recovery), unit-tested.
- `train_hybrid_vae_dcm.py` now reports masked `A_sign_recovery` (gate metric) + keeps
  `A_sign_recovery_unmasked` for transparency.

**Remaining (small):** the ~0.71 is estimated from the aggregate (7.08/10). To get the EXACT
masked number on the *existing* trained encoder, add an eval-only path to the train script
(load `results/hybrid_vae_dcm/encoder_checkpoint.pt`, skip training, recompute) and re-run on
M3 -- a quick job, no retraining. Expected to confirm HVAE-02 passes.

## Audit disposition (2026-06-10) — IN v0.6.0 SCOPE; the one closing action

HVAE-02 is **synthetic recovery**, so it falls inside v0.6.0's scope-cut "delivered core"
(unlike the real-data gaps, which moved to v0.7.0). The milestone audit
(`.planning/v0.6.0-AUDIT.md`) rates it **indeterminate**: the masked-metric *fix* is committed
but the masked number (~0.71) was never recomputed on the actual checkpoint — `recovery_report.json`
still holds the unmasked 0.4425.

**This is the only gap that should ideally close before v0.6.0 completion.** Two acceptable
resolutions:
1. **Confirm** — run the eval-only path on M3 (load checkpoint, recompute masked sign recovery,
   no retraining; <5 min cluster job) and record the exact number. Converts indeterminate → met.
2. **Accept-with-caveat** — close v0.6.0 noting HVAE-02 as "met under the corrected masked metric
   (~0.71 estimated); exact re-eval deferred." Defensible since the unmasked failure is a proven
   metric artifact, but leaves a soft spot.

Recommendation: option 1 if M3 is unlocked (cheap, makes the milestone airtight); else option 2.
