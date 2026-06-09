---
phase: 20-latent-circuit-forward-model
plan: 05
subsystem: benchmarks
type: diagnostic
status: acceptance-failed-root-caused
tags: [latent-circuit, svi, recovery, b-identifiability, elbo-model-selection, diagnostic]
dependency-graph:
  requires: ["20-04"]
  provides: ["lc_prior_calibration_sweep", "lc_acceptance_run", "lc_elbo_model_selection", "failure-root-cause"]
  affects: ["16.1", "23", "25", "v0.7.0-VL"]
tech-stack:
  added: []
  patterns: ["slurm-array-acceptance", "prior-monkeypatch-context", "median-gate-aggregation"]
key-files:
  created:
    - cluster/scripts/lc_prior_calibration_sweep.py
    - cluster/scripts/lc_acceptance_run.py
    - cluster/scripts/lc_elbo_model_selection.py
    - cluster/sbatch/lc_calibration.sbatch
    - cluster/sbatch/lc_acceptance.sbatch
    - cluster/sbatch/lc_elbo.sbatch
  modified: []
decisions:
  - id: "20-05-D1"
    description: "Acceptance NOT achieved. A-RMSE passes; B-RMSE, trajectory R-squared, and ELBO model selection fail. Root causes identified (below); thresholds in latent_circuit_metrics.py left PROVISIONAL pending an inference-method decision (mean-field SVI vs Variational Laplace)."
  - id: "20-05-D2"
    description: "ELBO-based model order selection (compare best -ELBO across N in {2..6}) is methodologically invalid as implemented: candidates fit datasets of DIFFERENT observed dimensionality, so the summed likelihood (final_loss) scales with N and min-loss trivially selects N=2. BMR (Phase 23) is the principled replacement."
  - id: "20-05-D3"
    description: "The CPU-feasibility rework (b329e88: duration 100s->50s, 500 steps, 5 restarts) destroyed B identifiability: with the 80/20 split, training data (first 40s) contains only ONE 8s modulator window (t=10-18); the t=40 epoch falls in the test split and the t=70 epoch is cut. B is under-identified by construction, independent of inference quality."
metrics:
  duration: "diagnostic backfill"
  completed: "2026-06-09"
backfilled: 2026-06-09
---

# Phase 20 Plan 05: Prior Calibration / Acceptance / ELBO Selection — DIAGNOSTIC SUMMARY

**Status: acceptance FAILED, root-caused.** This plan's scripts were built and reworked
across 6 commits (26ba2e5 → 75eb308) but the 10-seed acceptance gate did not pass and no
`20-05-SUMMARY.md` was written at the time. This document backfills the write-up as a
**failure analysis**: A-RMSE passes, but B-RMSE, trajectory R², and ELBO model selection
fail. The root causes are identified below with code-level evidence. The committed result
JSONs from the failing run are not in the tree (cluster-only); the analysis is grounded in
the runner/model/script source and the ground-truth construction.

## What Was Built

- `cluster/scripts/lc_prior_calibration_sweep.py` — SLURM-array prior-var × init_scale sweep
  (`lc_a_prior_var ∈ {1/64,1/16,1/4,1}`, `lc_b_prior_var ∈ {0.25,1,4}`, `init_scale ∈
  {0.01,0.05,0.1,0.5}`), monkey-patching the module priors via `_patch_lc_priors`.
- `cluster/scripts/lc_acceptance_run.py` — 10-seed (SLURM array) acceptance runner.
  Effective config: **`duration=50s`, `500 SVI steps`, `5 restarts`, `init_scale=0.1`**,
  `auto_normal` guide, `lr=0.005` decaying ×0.01.
- `cluster/scripts/lc_elbo_model_selection.py` — fits candidates `N ∈ {2,3,4,5,6}` (3 seeds,
  300 steps, 3 restarts) and compares best `-ELBO`.
- Three matching sbatch files.

## Ground Truth (from `benchmarks/runners/latent_circuit_recovery.py`)

| Element | Value | Identifiability |
|---------|-------|-----------------|
| A diagonal (self-inhibition) | −0.5 Hz | strong — driving input active ~50% of all 50s |
| A off-diagonal chain `A[i+1,i]` | 0.15 | strong — linear, always-on |
| B chain `[B₁₀,B₂₁,B₃₂]` | **[0.4, 0.3, 0.2]** (scale=1.0, no stability halving per 20-04-D4) | **weak — multiplicative, gated by `u_mod`** |
| Driving stimulus | 5×(10s ON/10s OFF) over 50s | rich |
| Modulator `u_mod` epochs | t = [10, 40, 70], 8s each | sparse; **only t=10 survives in the training split** |

## Root-Cause Analysis

### 1. B-RMSE fails (gate 0.20) — weak multiplicative identifiability + a split artifact

B enters the dynamics only through `u_mod(t) · B @ x` ([REF-001] Eq. 1), so it is informed
**only while the modulator is ON**. The 50s CPU-feasibility rework (20-05-D3) is fatal here:

- The `t=70` epoch is **cut** (beyond 50s).
- The 80/20 split puts `T_train = first 40s`; the `t=40–48` epoch lands in the **test**
  segment. → **B is trained from a single 8-second window (t=10–18).**

With only one modulation window and a mean-field `AutoNormal` guide whose `B_free_j ~
N(0,1)` prior pulls toward zero, the B posterior collapses toward 0. A fully-collapsed B
gives `RMSE = √((0.4²+0.3²+0.2²)/3) = √0.0967 = 0.31`, which matches the **systematic ~0.34
B-RMSE Phase 16.1 documented for the bilinear BOLD model and never resolved.** This is the
same shrinkage pathology, inherited unchanged.

### 2. Trajectory R² fails (gate 0.95) — downstream of #1 + the same split artifact

`_predict_trajectories` integrates from t=0 and returns the held-out tail (t=40–50). That
window **contains the t=40–48 modulator epoch**, which a collapsed-B model cannot reproduce
→ large residual exactly where the test signal lives. ODE error from any A mismatch also
compounds over the 40s of integration before the test segment begins. The 0.95 gate is very
tight even before these effects. R² failure is therefore a *consequence* of the B failure
and the train/test boundary placement, not an independent problem.

### 3. ELBO model selection fails — methodological bug (20-05-D2), not tuning

`lc_elbo_model_selection.py::_prepare_data_for_n` changes the **observed dimensionality** per
candidate: `N=2 → (T,2)`, `N=6 → (T,6)` (augmenting with noise columns for N>4). The model
likelihood `dist.Normal(y_mean, noise_std).to_event(2)` sums log-probs over all `T×N`
elements, so `final_loss = −ELBO` **scales with N**. `compute_elbo_model_selection` picks
`min(final_loss)` → it will **systematically select N=2** (fewest likelihood terms), never the
true N=4. ELBOs computed on datasets of different dimensionality are not comparable. The
principled fix is **Bayesian Model Reduction (Phase 23)** — score nested/reduced models
against a single full-model fit on the *same* data — or, at minimum, a per-element-normalised
evidence with a fixed observation dimension.

### 4. A-RMSE passes — and explains everything else

A enters **linearly** and the driving input is active ~half of every trajectory, so A (and C)
receive strong, sustained gradient signal across the full 40s training segment. The contrast
— **A: linear, always-on, recovered; B: multiplicative, one-window, collapsed** — is the
crux of the whole failure pattern.

## Cross-Cutting Significance

This is not an isolated Phase 20 issue. The same weak-B / sign-identifiability mechanism
recurs across the milestone:

- **[[16.1-recov-04-b-rmse-diagnostic]]** (v0.3.0) — identical ~0.34 systematic B-RMSE, never executed/closed.
- **Phase 25 HVAE-02** — amortized A **sign recovery 0.4425 < 0.6** (cluster job 55774467): the same connectivity-sign information is unidentified under amortized inference.
- **Phase 23 BMR** exists precisely because brute-force ELBO model comparison (this plan's approach) is unreliable.
- The post-consolidation **Variational Laplace** pivot (SPM12 hyperpriors, ReML M-step, Q-based CSD precision) is the natural inference-side response to mean-field shrinkage.

## Recommended Next Actions (prioritised)

**Tier A — methodological bugs (fixed 2026-06-09, commit pending):**
1. ✅ **B-informative experiment fixed.** Modulator epochs are now placed at FRACTIONS of the
   effective duration (`_MOD_EVENT_FRACTIONS = [0.10, 0.35, 0.60]`) and `_build_ground_truth`
   takes a `duration` arg so the modulator grid matches the simulated grid. All three epochs
   now fall in the first 60% of the trajectory → inside the 80% training split for any
   duration (was: one window for the 50s run). Regression-tested in
   `tests/test_latent_circuit_metrics.py::TestModulatorInTrainingSplit`.
2. ✅ **ELBO selection hardened.** `compute_elbo_model_selection` gained an
   `observed_element_counts` guard that REFUSES cross-dimensional comparisons (the invalid
   min-`−ELBO` scan), citing 20-05-D2 and pointing to VL free energy + BMR.
   `lc_elbo_model_selection.py` marked deprecated. Guard tested in
   `TestElboSelectionGuard`. **Note:** this only prevents the silently-wrong answer; the
   actual model-comparison must be done via VL free energy + BMR (Tier B).

**Tier B — inference quality (decision 2026-06-09: Variational Laplace, NOT a structured SVI
guide).** VL already returns a **full posterior covariance** (`sigma_post`), so it IS the
structured posterior — `AutoLowRankMVN`/`AutoIAFNormal` are not needed. VL also yields the
SPM 3-term **free energy**, which feeds BMR directly and resolves the model-comparison
question. VL is validated for spectral (CSD) and task (BOLD) DCM.
3. ✅ **`LatentCircuitForward` adapter BUILT 2026-06-09** (`src/pyro_dcm/inference/forward_models.py`).
   Direct observation (`hemodynamic=False`, `y0=zeros(N)`), **bilinear B in the packed
   parameter vector** (`A_free + C_free + B_free`), real time-domain trajectory residual;
   plugs into the existing `_run_vl_generic`. Observation noise precision is the VL ReML
   hyperparameter (not a free parameter), per SPM. Validated by
   `tests/test_latent_circuit_vl.py` (slow, ~80s): N=3 chain, VL recovers A stability + 0→1
   sign + **B sign**, returns a **full (non-diagonal) posterior covariance** (confirming VL is
   the structured posterior), and reconstructs the trajectory (R² > 0.7). This unblocks the
   VL-based 20-05 re-run.
   **Next:** build the cluster acceptance script that fits `latent_circuit_dcm_model` ground
   truth via `LatentCircuitForward` + VL across 10 seeds (replacing the SVI runner), use VL
   free energy + BMR for model comparison, then recalibrate thresholds.
4. Only after #3, **recalibrate the provisional thresholds** in `latent_circuit_metrics.py`
   against the observed VL recovery distribution (still placeholder bilinear-BOLD values; the
   0.95 R² gate in particular may be unrealistic).

**Compute routing:** any re-run is a multi-seed VL sweep → **M3 cluster** (per project
policy), not laptop.

## Deviations from Plan

The plan's success criteria were not met (acceptance gate did not pass; ELBO did not select
N=4). The `checkpoint:human-verify` calibration gate was never signed off, and the calibrated
prior constants were never committed — `LC_A_PRIOR_VARIANCE` (1/16) and `LC_B_PRIOR_VARIANCE`
(1.0) remain at their pre-calibration values, and `latent_circuit_metrics.py` thresholds
remain PROVISIONAL.

## Commits

| Hash | Message |
|------|---------|
| 26ba2e5 | feat(20-05): add prior calibration sweep script and sbatch |
| 856c76c | fix(20-05): restructure calibration sweep as SLURM array job |
| 786902d | fix(20-05): increase calibration duration to 30s (first modulator at t=10) |
| a0c13a7 | fix(20-05): integrate ODE from t=0 for held-out trajectory R-squared |
| 7a4186a | feat(20-05): add 10-seed acceptance run script and sbatch |
| b329e88 | fix(20-05): rework acceptance run for CPU feasibility (50s, 500 steps, 5 restarts) |
| 75eb308 | feat(20-05): add ELBO model selection script and sbatch (SYNTH-03) |

## Next Phase Readiness

Phase 20 remains 4/5. SYNTH-01 (B), SYNTH-02 (R²), SYNTH-03 (ELBO selection) are **not**
satisfied. Closing 20-05 requires the Tier A redesign + a Tier B inference decision, most
naturally folded into the v0.7.0 Variational Laplace reconciliation.
