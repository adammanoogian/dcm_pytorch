---
created: 2026-06-09T23:55
title: VL Laplace overconfidence suppresses absolute BMR pruning
area: model-selection / inference
priority: medium
files:
  - src/pyro_dcm/inference/variational_laplace.py
  - src/pyro_dcm/model_selection/bmr.py
  - cluster/scripts/lc_vl_bmr_selection.py
  - .planning/phases/20-latent-circuit-forward-model/20-05-SUMMARY.md
---

## Problem

Discovered closing SYNTH-03 (job 56270544): the VL (Laplace/ReML) posterior on the latent
circuit is **over-confident** at high SNR — posterior std/prior std ~0.001. Bayesian Model
Reduction then never prunes anything: every single-connection prune ΔF is strongly negative
(true chain ~ -1.1M, genuinely-absent edges -34 to -77k), so exhaustive best-model
`bmr_circuit_selection` keeps the FULL model (full_model_rank=1/4096).

The *relative* prune-cost ranking still recovers the true structure perfectly (essential
edges = true chain, 3/3 seeds), which is how SYNTH-03 was closed. But the **absolute** "prune
if ΔF > 0" rule is miscalibrated by the overconfident covariance. This affects **any absolute
BMR usage** (Phase 23), not just the latent circuit.

## Solution (sketch / options)

1. **Temper the VL posterior covariance** before BMR: ReML can underestimate parameter
   covariance when the observation precision is large. Inflate `sigma_post` by a calibrated
   factor, or cross-check against a posterior-predictive / held-out evidence.
2. **Use BMR relative ranking + a separation-gap criterion** as the standard structure-
   selection API (what SYNTH-03 now does) rather than the absolute ΔF>0 threshold. Consider
   adding a `rank_connections()` helper to `model_selection/bmr.py`.
3. Sanity-check whether the overconfidence is partly a *synthetic-data* artifact (clean,
   high-SNR, model exactly correct) that will be less severe on real M/EEG latents.
4. Validate against the Phase 23 brute-force ELBO comparison (which passed) to quantify how
   much the absolute ΔF is off.

Note: this does NOT affect the Phase 20-05 closure (structure recovered via ranking) or the
SYNTH-01/02 recovery numbers — it is a calibration caveat for absolute BMR evidence.

## Audit disposition (2026-06-10) — DEFER to v0.7.0 (Phase C)

Does **not** block v0.6.0: relative-ranking BMR is the standardized API and it recovers structure
(SYNTH-03 3/3). The absolute-ΔF calibration fix (posterior tempering + a `rank_connections()`
helper) is a **VL refinement** → maps to **v0.7.0 Phase C (VL + BMR model comparison)** in
`.planning/v0.7.0-VL-RECONCILIATION-DRAFT.md`. Carry forward as-is.
