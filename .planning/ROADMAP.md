# Roadmap: Pyro-DCM

## Milestones

- **v0.1.0 Foundation** - Phases 1-8 (shipped 2026-04-03)
- **v0.2.0 Cross-Backend Inference Benchmarking** - Phases 9-12 (shipped 2026-04-13)
- **v0.3.0 Bilinear DCM Extension** - Phases 13-16 (in progress; started 2026-04-17)

<details>
<summary>v0.1.0 Foundation (Phases 1-8) - SHIPPED 2026-04-03</summary>

See `.planning/milestones/v0.1.0-ROADMAP.md` for details. 8 phases, 26 plans, 127 commits.

</details>

<details>
<summary>v0.2.0 Cross-Backend Inference Benchmarking (Phases 9-12) - SHIPPED 2026-04-13</summary>

See `.planning/milestones/v0.2.0-ROADMAP.md` for details. 4 phases, 11 plans, 47 commits.

- [x] Phase 9: Benchmark Foundation (3/3 plans) -- completed 2026-04-07
- [x] Phase 10: Guide Variants (3/3 plans) -- completed 2026-04-12
- [x] Phase 11: Calibration Analysis (3/3 plans) -- completed 2026-04-12
- [x] Phase 12: Documentation (2/2 plans) -- completed 2026-04-13

</details>

---

## Current Milestone: v0.3.0 Bilinear DCM Extension

**Status:** In progress (started 2026-04-17)
**Phases:** 13-16 (4 phases)
**Requirements covered:** 27/27 v0.3.0 requirements

### Overview

v0.3.0 extends the neural state equation from the shipping linear form `dx/dt = Ax + Cu`
to the full Friston 2003 bilinear form `dx/dt = Ax + Σ_j u_j(t)·B_j·x + Cu`, propagating
B-matrix modulatory inputs end-to-end through the forward model, simulator, Pyro
generative model + priors, and a 3-region recovery benchmark. Research (MEDIUM-HIGH
confidence) confirms the extension is a narrow, well-bounded mathematical superset of
linear DCM: no new runtime dependencies, no API churn for existing callers (None-default
kwargs on five existing functions), and spectral DCM / rDCM are architecturally
untouched. The critical path is strictly linear:
**Phase 13 (neural state + stability) -> Phase 14 (simulator + stimulus utilities) ->
Phase 15 (Pyro model) -> Phase 16 (recovery benchmark).**

**Milestone acceptance gate:** Phase 16 passes all four RECOV criteria (A RMSE <= 1.25x
linear baseline, B RMSE <= 0.20 on |B_true|>0.1, sign_recovery_nonzero >= 80%,
coverage_of_zero >= 85%) on >=10 seeds at SNR=3, with identifiability shrinkage metric
reported.

### Phases

#### Phase 13: Bilinear Neural State & Stability Monitor

**Goal:** The neural state equation computes the Friston 2003 bilinear form
`A_eff(t)·x + C·u` with a documented eigenvalue stability monitor, while preserving
bit-exact linear behavior when bilinear arguments are omitted.

**Branch:** `gsd/phase-13-bilinear-neural-state`
**Depends on:** v0.2.0 shipping infrastructure (linear `NeuralStateEquation`,
`CoupledDCMSystem`, `torchdiffeq` integrator).
**Requirements:** BILIN-01, BILIN-02, BILIN-03, BILIN-04, BILIN-05, BILIN-06, BILIN-07
**Success Criteria** (what must be TRUE):

  1. `test_linear_invariance.py` passes at `atol=1e-10`: with `B_list=None` and
     `u_mod=None`, `NeuralStateEquation.derivatives` produces bit-exact output matching
     the current linear form `A @ x + C @ u` (BILIN-03).
  2. All existing `test_neural_state.py` and `test_ode_integrator.py` tests pass
     unchanged; no existing caller of `CoupledDCMSystem` requires edits (BILIN-04).
  3. Worst-case 3-sigma B stability test passes: bilinear ODE at
     `B = 3 * sigma_prior`, sustained `u_mod = 1`, 500s integration, no NaN in output
     (BILIN-06, mitigates Pitfall B1).
  4. A_eff eigenvalue monitor logs a warning when `max(Re(eig(A_eff(t)))) > 0`
     (strict threshold, log-only, never raises during SVI per D4) at a subsample of
     ODE steps (BILIN-05).
  5. `NeuralStateEquation` class docstring and `neural_state.py` module header no
     longer describe the `A + Cu` form as "bilinear"; the true bilinear form is
     documented in the new code path (BILIN-07, mitigates Pitfall B4).

#### Phase 14: Stimulus Utilities & Bilinear Simulator

**Goal:** Users can construct variable-amplitude event and epoch stimuli and run the
simulator in bilinear mode to produce context-dependent BOLD ground truth, while the
existing linear simulator output is exactly preserved when bilinear arguments are
omitted.

**Branch:** `gsd/phase-14-stimulus-and-bilinear-simulator`
**Depends on:** Phase 13 (bilinear forward model and `compute_effective_A`).
**Requirements:** SIM-01, SIM-02, SIM-03, SIM-04, SIM-05
**Plans:** 2 plans (2 waves)
Plans:
- [ ] 14-01-PLAN.md — Stimulus utilities (`make_event_stimulus`, `make_epoch_stimulus`) + `merge_piecewise_inputs` helper + unit tests (SIM-01, SIM-02)
- [ ] 14-02-PLAN.md — `simulate_task_dcm` bilinear extension + return-dict update + linear bit-exactness regression + bilinear-vs-linear distinguishability + dt-invariance (linear & bilinear) (SIM-03, SIM-04, SIM-05)

**Success Criteria** (what must be TRUE):

  1. All existing `test_task_simulator.py` tests (40+) pass unchanged; calling
     `simulate_task_dcm(...)` with `B_list=None` produces output identical to the
     current linear simulator (SIM-03 regression test).
  2. `make_event_stimulus` (stick functions) and `make_epoch_stimulus` (boxcars,
     documented as preferred primitive per Pitfall B12) produce `(T, J)` tensors
     of variable-amplitude modulatory inputs via piecewise-constant interpolation
     (SIM-01, SIM-02).
  3. `simulate_task_dcm` in bilinear mode (non-zero `B_list` + non-trivial
     `stimulus_mod`) produces BOLD that is numerically distinguishable from the
     linear null (`B_list=None`) on the same seed and inputs (SIM-03).
  4. Simulator return dict contains `B_list` and `stimulus_mod` keys (set to `None`
     in linear mode for forward compatibility with Phase 15 / 16 consumers)
     (SIM-04).
  5. dt-invariance test passes: ODE integration at `dt=0.01` and `dt=0.005` produces
     equivalent BOLD within `atol=1e-4` under a fixed bilinear ground truth
     (SIM-05).

#### Phase 15: Pyro Generative Model with B Priors and Masks

**Goal:** The task-DCM Pyro model samples per-modulator `B_free_j ~ Normal(0, 1.0)`
with per-site masking and auto-discoverable sample sites, such that SVI converges on
bilinear simulated data across AutoNormal, AutoLowRankMVN, and AutoIAFNormal without
any guide-factory changes.

**Branch:** `gsd/phase-15-pyro-bilinear-model`
**Depends on:** Phase 13 (bilinear forward model) + Phase 14 (simulated ground truth).
**Requirements:** MODEL-01, MODEL-02, MODEL-03, MODEL-04, MODEL-05, MODEL-06, MODEL-07
**Success Criteria** (what must be TRUE):

  1. `task_dcm_model(..., b_masks=None, stim_mod=None, ...)` reduces to the current
     linear model when `b_masks=None` or `b_masks=[]`; SVI smoke test on 3-region
     bilinear simulated data converges with decreasing ELBO (MODEL-01, MODEL-04).
  2. Module-level constant `B_PRIOR_VARIANCE = 1.0` exists with a docstring citing
     the D1 decision (SPM12 one-state match); a unit test asserts the documented
     value and fails loudly if changed without review (MODEL-02, corrects Pitfall
     B8 / YAML error).
  3. `create_guide` auto-discovers the new `B_free_j` sample sites without factory
     changes; trace test confirms this on `AutoNormal`, `AutoLowRankMVN`, and
     `AutoIAFNormal` (MODEL-06).
  4. `parameterize_B` zeroes `b_mask` diagonal by default; explicit non-zero diagonal
     triggers a `DeprecationWarning` with rationale (MODEL-03, mitigates Pitfall
     B5); `extract_posterior_params` returns per-modulator `B_j` medians alongside
     existing `A`, `C`, `noise_prec` (MODEL-05).
  5. `amortized_wrappers.py` / `TaskDCMPacker` refuse bilinear sample sites with an
     explicit error message referencing v0.3.1 as the target milestone for
     amortized bilinear support (MODEL-07, mitigates Pitfall B3 per D5).

#### Phase 16: 3-Region Bilinear Recovery Benchmark

**Goal:** The bilinear DCM implementation recovers ground-truth parameters on a
3-region network (1 driving input, 1 modulatory input, 2 non-zero B elements) within
documented acceptance criteria on >=10 seeds at SNR=3, integrating with the v0.2.0
shared-fixture benchmark pipeline.

**Branch:** `gsd/phase-16-bilinear-recovery-benchmark`
**Depends on:** Phase 15 (working Pyro bilinear model) + v0.2.0 shared `.npz` fixture
infrastructure and `BenchmarkConfig` / figure pipeline.
**Requirements:** RECOV-01, RECOV-02, RECOV-03, RECOV-04, RECOV-05, RECOV-06, RECOV-07,
RECOV-08
**Success Criteria** (what must be TRUE):

  1. `benchmarks/runners/task_bilinear.py` runner executes end-to-end using v0.2.0
     shared `.npz` fixture infrastructure, `BenchmarkConfig`, and the existing
     figure pipeline; 3-region network with 1 block-design driving input + 1
     event-related variable-amplitude modulator and 2 non-zero B elements (RECOV-01,
     RECOV-02).
  2. All four RECOV acceptance criteria pass on >=10 seeds at SNR=3:
     - A RMSE <= 1.25 * linear-baseline RMSE (RECOV-03, mitigates Pitfall B13
       A-RMSE inflation)
     - B RMSE <= 0.20 on |B_true| > 0.1 elements (RECOV-04)
     - sign_recovery_nonzero >= 80% on |B_true| > 0.1 (RECOV-05, per D3)
     - coverage_of_zero >= 85% on |B_true| < 0.5 * prior_std (RECOV-06, per D3).
  3. Identifiability shrinkage metric `std_post / std_prior` is reported per free
     `B_ij`; documented with a soft target of <= 0.7 but does not block acceptance
     (RECOV-07, mitigates Pitfall B2 / Rowe 2015 identifiability).
  4. Wall-time benchmark reports bilinear (3-region, J=1) runtime vs linear
     3-region baseline (~235s/500 steps); expected 3-6x slowdown (Pitfall B10),
     flagged as a milestone risk if >10x (RECOV-08).

### Progress

**Execution Order:** 13 -> 14 -> 15 -> 16

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 13. Bilinear Neural State & Stability Monitor | 4/4 | Complete | 2026-04-17 |
| 14. Stimulus Utilities & Bilinear Simulator | 0/2 | Pending | -- |
| 15. Pyro Generative Model with B Priors and Masks | 0/TBD | Pending | -- |
| 16. 3-Region Bilinear Recovery Benchmark | 0/TBD | Pending | -- |

---

## Cumulative Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1-8 | v0.1.0 | 26/26 | Complete | 2026-04-03 |
| 9. Benchmark Foundation | v0.2.0 | 3/3 | Complete | 2026-04-07 |
| 10. Guide Variants | v0.2.0 | 3/3 | Complete | 2026-04-12 |
| 11. Calibration Analysis | v0.2.0 | 3/3 | Complete | 2026-04-12 |
| 12. Documentation | v0.2.0 | 2/2 | Complete | 2026-04-13 |
| 13. Bilinear Neural State & Stability Monitor | v0.3.0 | 4/4 | Complete | 2026-04-17 |
| 14. Stimulus Utilities & Bilinear Simulator | v0.3.0 | 0/2 | Pending | -- |
| 15. Pyro Generative Model with B Priors and Masks | v0.3.0 | 0/TBD | Pending | -- |
| 16. 3-Region Bilinear Recovery Benchmark | v0.3.0 | 0/TBD | Pending | -- |

---
*Roadmap created: 2026-04-07*
*Last updated: 2026-04-17 after Phase 14 plans created (2 plans, 2 waves)*
