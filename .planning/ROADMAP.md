# Roadmap: Pyro-DCM

## Milestones

- **v0.1.0 Foundation** - Phases 1-8 (shipped 2026-04-03)
- **v0.2.0 Cross-Backend Inference Benchmarking** - Phases 9-12 (shipped 2026-04-13)
- **v0.3.0 Bilinear DCM Extension** - Phases 13-16 (in progress; started 2026-04-17)
- **v0.4.0 Circuit Explorer** - Phase 17+ (defined 2026-04-24; not yet started)
- **v0.5.0 MNE-Python Integration** - Phases 18-19 (in progress; started 2026-05-21)
- **v0.6.0 Latent Circuit DCM** - Phases 20-25 (defined 2026-05-24)

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

**Status:** In progress (started 2026-04-17; Phase 16.1 inserted 2026-04-24 for RECOV-04 diagnostic)
**Phases:** 13-16 + 16.1 (4 phases + 1 inserted)
**Requirements covered:** 27/27 v0.3.0 requirements (Phase 16.1 may tighten or amend RECOV-04 / RECOV-07)

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
- [x] 14-01-PLAN.md — Stimulus utilities (`make_event_stimulus`, `make_epoch_stimulus`) + `merge_piecewise_inputs` helper + unit tests (SIM-01, SIM-02)
- [x] 14-02-PLAN.md — `simulate_task_dcm` bilinear extension + return-dict update + linear bit-exactness regression + bilinear-vs-linear distinguishability + dt-invariance (linear & bilinear) (SIM-03, SIM-04, SIM-05)

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
**Plans:** 3 plans (2 waves)
Plans:
- [x] 15-01-PLAN.md — `task_dcm_model` bilinear extension + `B_PRIOR_VARIANCE`=1.0 constant + `_validate_bilinear_args` helper + NaN-safe predicted_bold guard + bilinear SVI smoke (MODEL-01, MODEL-02, MODEL-03 model-side, MODEL-04)
- [x] 15-02-PLAN.md — `create_guide` auto-discovery verification across AutoNormal, AutoLowRankMVN, AutoIAFNormal (MODEL-06; test-only, zero src changes)
- [x] 15-03-PLAN.md — `TaskDCMPacker.pack` + `amortized_task_dcm_model` bilinear refusal (v0.3.1 deferral per D5) + `extract_posterior_params` docstring extension + bilinear posterior-extraction test (MODEL-05, MODEL-07)

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

#### Phase 16.1: RECOV-04 B-RMSE Shrinkage Diagnostic & Fix (INSERTED)

**Goal:** Diagnose and resolve the systematic RECOV-04 acceptance failure observed on
cluster job 54933838 (2026-04-24): B-RMSE = 0.3424 across all 10 seeds
(distribution 0.335-0.348, tightly clustered — systematic underfit, not outlier
noise) vs the <= 0.20 threshold on `|B_true| > 0.1` elements. RECOV-07 shrinkage
means (~0.008 on nonnull B entries) indicate the SVI guide is collapsing the B
posterior toward zero. Unblocks v0.3.0 milestone closure without renumbering the
roadmap.

**Branch:** `gsd/phase-16.1-recov-04-b-rmse-diagnostic` (proposed)
**Depends on:** Phase 16 (acceptance runner + ground truth fixtures + cluster harness)
**Requirements:** RECOV-04 (status flip pending cluster re-run); annotation on
RECOV-07 (shrinkage_nonnull means in [0.05, 0.6] under raised init_scale).
Plan 16.1 does NOT tighten or relax any RECOV threshold; the only REQUIREMENTS
edits are the RECOV-04 status flip on cluster pass and a citation note on
RECOV-07.

**Plans:** 2 plans (2 waves)
Plans:
- [ ] 16.1-01-PLAN.md — Single-seed init_scale sweep diagnostic on seed 42 across {0.005, 0.05, 0.1, 0.5} at 500 steps; produces machine + human diagnostic artifacts (autopushed to a `results/phase16_1-init-scale-sweep-*` branch) and a SUMMARY recording the chosen `_BILINEAR_INIT_SCALE` (or escalation if no winner). CLUSTER execution via sbatch on M3 (~30-40 min walltime); LOCAL harness-faithfulness pre-check (single fit at init_scale=0.005, &lt;5 min) BEFORE submission per cluster policy carve-out.
- [ ] 16.1-02-PLAN.md — Apply chosen init_scale to `benchmarks/runners/task_bilinear.py`, replace inverted `_BILINEAR_INIT_SCALE_RETRY = 0.001` with "halve once on NaN at step 0", reuse Phase 16 cluster sbatch scaffolding to re-run the 10-seed acceptance gate, then flip RECOV-04 in REQUIREMENTS.md on pass (or document escalation on RECOV-06 degradation / RECOV-04 still-failing). CLUSTER execution (~80-150 min).

**Hypotheses to investigate (planning input, not a plan):**
  1. **Prior-variance / init-scale interaction.** `B_PRIOR_VARIANCE = 1.0` (D1) + auto_normal
     `init_scale = 0.005` (Plan 16-01 L2) may start the B guide distribution so tight
     around zero that the ELBO prefers staying there over expanding — gradient signal to
     B is weaker than to A because B enters multiplicatively through `u_mod`.
  2. **Guide family insufficient.** AutoNormal may be too restrictive for the bilinear
     posterior geometry; AutoLowRankMVN or AutoIAFNormal (verified to auto-discover B
     sites in Plan 15-02) may recover better. Sidebar was explicitly deferred to v0.3.1
     per Plan 16-01 L2 decision, but may need to move forward.
  3. **B_true vs prior magnitude mismatch.** If the ground-truth |B_true| magnitudes at
     the nonnull elements are much larger than `sqrt(B_PRIOR_VARIANCE) = 1.0`, a
     Normal(0, 1) prior plus strong data likelihood could still pull the posterior
     partway to zero while a MAP under weak data simply shrinks. Worth verifying the
     ground-truth generator's B amplitudes against the prior scale.
  4. **Stim_mod magnitude / SNR interaction.** If the modulatory stimulus amplitude
     relative to driving input is too small, B is under-identified regardless of guide.
  5. **Step count / LR schedule.** 500 steps may not be enough for B to escape the
     near-zero init basin even if the other levers are right.

**Success Criteria** (what must be TRUE — provisional, finalized during planning):

  1. Root cause of the ~0.34 systematic B-RMSE identified with evidence (per-step B
     trajectory plot, posterior mean vs true-B scatter across seeds, or prior-
     sensitivity sweep showing which lever moves B-RMSE).
  2. Fix (parameter change, guide swap, step-count increase, or scope amendment)
     applied and cluster-re-run passes RECOV-04 (<= 0.20 on |B_true| > 0.1) on >= 10
     seeds at SNR=3. RECOV-03 / RECOV-05 / RECOV-06 must continue to pass (no
     regression on currently-passing gates).
  3. RECOV-07 shrinkage means land in the documented soft-target range
     (std_post / std_prior <= 0.7) OR the soft target is explicitly revised with
     citation to observed bilinear identifiability limits.
  4. Diagnostic findings captured in a SUMMARY document
     (`.planning/phases/16.1-recov-04-b-rmse-diagnostic/16.1-02-SUMMARY.md`) so that
     the v0.3.1 amortized-bilinear work and any future RECOV tuning inherit the
     lessons learned.
  5. If the acceptance threshold itself needs revision (rather than the
     implementation), the revision is justified against research or upstream
     reference (SPM12 or comparable) and the milestone acceptance-gate line in this
     ROADMAP is updated accordingly.

### Progress

**Execution Order:** 13 -> 14 -> 15 -> 16 -> 16.1 (INSERTED)

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 13. Bilinear Neural State & Stability Monitor | 4/4 | Complete | 2026-04-17 |
| 14. Stimulus Utilities & Bilinear Simulator | 2/2 | Complete | 2026-04-18 |
| 15. Pyro Generative Model with B Priors and Masks | 3/3 | Complete | 2026-04-18 |
| 16. 3-Region Bilinear Recovery Benchmark | 3/3 | Implementation complete; acceptance FAILED 2026-04-24 (RECOV-04) | -- |
| 16.1. RECOV-04 B-RMSE Shrinkage Diagnostic & Fix (INSERTED) | 0/2 | Planned | -- |

---

## Next Milestone: v0.4.0 Circuit Explorer

**Status:** Defined 2026-04-24 (not yet started; may run in parallel with v0.3.0 Phase 16 cluster re-run since Phase 17 depends only on Phase 15 APIs).
**Phases:** 17+
**Theme:** Interactive serialization + rendering tooling for DCM model configs and fitted posteriors. Distinct from v0.3.0's fitting/recovery scope — acceptance is structural (JSON schema validity, round-trip equality, planned↔fitted toggle semantics) rather than RECOV-style RMSE/coverage gates.

### Overview

v0.4.0 delivers a Python-side serializer (`CircuitViz` in `src/pyro_dcm/utils/circuit_viz.py`) that converts a Pyro-DCM model config and/or fitted SVI posterior into the `dcm_circuit_explorer/v1` JSON schema consumed by `docs/dcm_circuit_explorer_template.html`. The renderer is already fully specified and shipped; the handoff doc (`docs/HANDOFF_viz.md`) contains the complete class interface, including verbatim implementations of `from_posterior()` and `load()`. Only `from_model_config()` is a stub. Research (MEDIUM-HIGH confidence) confirms zero upstream API changes are required; Phase 17 is purely additive.

**Milestone acceptance gate:** `CircuitViz.from_model_config(...).to_dict()` and `CircuitViz.from_posterior(...).to_dict()` both produce dicts that serialize to valid `dcm_circuit_explorer/v1` JSON, round-trip through `load()` with equality, and set `_status` correctly (`"planned"` vs `"fitted"`). No fitting metrics gated.

### Phases

#### Phase 17: Circuit Visualization Module

**Goal:** Implement `src/pyro_dcm/utils/circuit_viz.py` — a `CircuitViz` class with `from_model_config`, `from_posterior`, `to_dict`, `save`, and `load` methods producing `dcm_circuit_explorer/v1` JSON from Pyro-DCM model configs and/or SVI posteriors, verified by structural unit tests and a Pyro smoke integration test.

**Branch:** `gsd/phase-17-circuit-visualization-module` (proposed)
**Depends on:** Phase 15 (`extract_posterior_params` from MODEL-05). Does NOT depend on Phase 16.
**Requirements:** VIZ-01, VIZ-02, VIZ-03, VIZ-04, VIZ-05, VIZ-06, VIZ-07, VIZ-08, VIZ-09, VIZ-10 (derived from `docs/HANDOFF_viz.md` during /gsd:plan-phase 17 on 2026-04-24; see `.planning/REQUIREMENTS.md` v0.4.0 Requirements section).
**Plans:** 1 plan (1 wave)
Plans:
- [x] 17-01-PLAN.md — CircuitViz core (`CircuitVizConfig` + `from_model_config` + `from_posterior` + `load` + `flatten_posterior_for_viz` helper) + 12 structural/integration tests (A-01..A-10 + B-01/B-02) + utils re-export + REQUIREMENTS.md VIZ-01..10 append (VIZ-01..10)

**Success Criteria** (what must be TRUE — provisional, finalized during planning):

  1. `CircuitViz.from_model_config(...)` produces a dict matching `dcm_circuit_explorer/v1` schema with `_status == "planned"` and `fitted_params is None`.
  2. `CircuitViz.from_posterior(extract_posterior_params(...))` produces a dict with `_status == "fitted"` and populated `fitted_params` (per-matrix means or medians, shape-matched to A/B_j/C).
  3. Round-trip test: `CircuitViz.load(viz.save(path))` reads back equal to the original on a reference 3-region bilinear fixture.
  4. Schema tolerance: `from_model_config` works for a bare bilinear DCM with no HEART2ADAPT metadata (empty `phenotypes`/`hypotheses`/`drugs`); renderer handles missing optional fields without JS errors (verified structurally, not via headless browser per research recommendation).
  5. Zero upstream API changes — no edits to `task_dcm_model`, `extract_posterior_params`, `parameterize_A`, `parameterize_B`, or any file outside `src/pyro_dcm/utils/` and `tests/`.

### Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 17. Circuit Visualization Module | 1/1 | Complete (verified 15/15 must-haves) | 2026-04-24 |

---

## Current Milestone: v0.5.0 MNE-Python Integration

**Status:** In progress (started 2026-05-21)
**Phases:** 18-19 (2 phases)
**Requirements covered:** 18/18 v0.5.0 requirements

### Overview

v0.5.0 validates the existing MNE-Python IO loaders (`src/pyro_dcm/io/mne_loader.py`,
`bids_loader.py`) which shipped in v0.4.0 but have zero test coverage, then demonstrates
end-to-end usage via pipeline scripts. The IO code already exists; this milestone proves
it works correctly by encoding critical scientific pitfalls (CSD frequency conventions,
Hermitian symmetry, channel picks inconsistency) as explicit test cases, and by building
two demo scripts showing the full path from synthetic MNE data through DCM fitting to
posterior A matrices. Research (HIGH confidence) confirms all test fixtures are synthetic
(MNE's `RawArray`, `EpochsArray`, `create_info`), no data downloads required, and the
`stc_to_roi_timeseries` path uses `unittest.mock.patch` on `mne.extract_label_time_course`.
The critical path is strictly linear: **Phase 18 (tests) -> Phase 19 (pipeline scripts).**

**Milestone acceptance gate:** All 18 requirements pass. Test suite runs cleanly with
`pytest -m mne` (when MNE installed) and is fully skipped with `pytest -m "not mne"`
(when MNE absent). Both pipeline scripts execute end-to-end on synthetic data and produce
fitted posterior A matrices.

### Phases

#### Phase 18: MNE/BIDS IO Test Suite

**Goal:** The MNE and BIDS IO loaders have a comprehensive test suite that validates
shape contracts, mathematical properties, error handling, and critical scientific
pitfalls, runnable via `pytest -m mne` and cleanly skipped when MNE is not installed.

**Branch:** `gsd/phase-18-mne-io-test-suite`
**Depends on:** v0.4.0 shipping IO code (`src/pyro_dcm/io/mne_loader.py`,
`src/pyro_dcm/io/bids_loader.py`).
**Requirements:** TEST-01..13, BIDS-01..03
**Plans:** 2 plans (2 waves)
Plans:
- [x] 18-01-PLAN.md -- Register mne pytest marker + MNE loader test suite (12 tests, TEST-01..13)
- [x] 18-02-PLAN.md -- BIDS loader round-trip test suite (3 tests, BIDS-01..03)

**Success Criteria** (what must be TRUE):

  1. `pytest tests/test_mne_loader.py` passes with all shape validations green:
     `epochs_to_csd` returns `(F, N, N)` complex tensor, `epochs_to_timeseries`
     returns `(T, N)` float tensor (both averaged and unaveraged), `raw_to_timeseries`
     returns `(T, N)` float tensor, and `stc_to_roi_timeseries` returns `(T, N)`
     float tensor via mocked `extract_label_time_course` (TEST-01 through TEST-04).
  2. Channel picks subsetting works correctly (TEST-05, TEST-06).
  3. CSD mathematical properties hold: Hermitian symmetry, non-negative auto-spectra,
     10 Hz sine injection peak detection (TEST-07, TEST-08, TEST-09).
  4. Error and skip paths work (TEST-10, TEST-11, TEST-12, TEST-13).
  5. `pytest tests/test_bids_loader.py` passes: BIDS round-trip and BAD_ACQ_SKIP
     annotation handling (BIDS-01, BIDS-02, BIDS-03).

#### Phase 19: End-to-End Pipeline Demos

**Goal:** Users can follow two self-contained demo scripts that show the complete path
from synthetic MNE data through Pyro-DCM model fitting to posterior connectivity
matrices, serving as copy-pasteable starting points for real neuroimaging workflows.

**Branch:** `gsd/phase-19-pipeline-demos`
**Depends on:** Phase 18 (confirmed IO contracts via passing tests).
**Requirements:** PIPE-01, PIPE-02
**Plans:** 2 plans (1 wave)
Plans:
- [x] 19-01-PLAN.md — Spectral DCM pipeline demo: synthetic MNE Epochs -> epochs_to_csd -> spectral_dcm_model -> SVI -> posterior A (PIPE-01)
- [x] 19-02-PLAN.md — Task DCM pipeline demo: synthetic MNE Epochs -> epochs_to_timeseries -> task_dcm_model (bilinear B) -> SVI -> posterior A + B (PIPE-02)

**Success Criteria** (what must be TRUE):

  1. `pytest tests/test_mne_loader.py` and `pytest tests/test_bids_loader.py` still pass (no regressions).
  2. Both pipeline scripts execute end-to-end on synthetic data and produce fitted posterior A matrices.

### Progress

**Execution Order:** 18 -> 19

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 18. MNE/BIDS IO Test Suite | 2/2 | Complete (verified 17/17 must-haves) | 2026-05-21 |
| 19. End-to-End Pipeline Demos | 2/2 | Complete (verified 10/10 must-haves) | 2026-05-24 |

---

## Upcoming Milestone: v0.6.0 Latent Circuit DCM

**Status:** Defined 2026-05-24; restructured 2026-05-26
**Phases:** 20-27 (8 phases)
**Requirements covered:** 38+ v0.6.0 requirements (new phases TBD during planning)

### Overview

v0.6.0 extends Pyro-DCM into a framework for **DCM-based interpretability of deep
learning models trained on neural data**. The milestone has four pillars:

1. **Synthetic validation** (Phase 20): Prove DCM recovers known bilinear parameters
   from a direct observation model, establishing the methodological foundation.
2. **DCM interpretability for neural data models** (Phase 22, core contribution):
   Train a temporal deep learning model on real M/EEG data (e.g., Cam-CAN MEG),
   extract latent dynamics, and fit spectral or bilinear DCM to characterize
   effective connectivity inside the model's learned representations.
3. **Foundation model use cases** (Phase 24): Apply DCM interpretability to
   pretrained brain foundation models (TRIBE for fMRI; LaBraM/MEG-GPT for M/EEG).
4. **Advanced inference** (Phases 25-26): Hybrid VAE-DCM (physics-informed
   generative model) and SBI for spectral DCM (amortized inference).

Phase 21 (CT-RNN training on neurogym) is complete and provides a methods-validation
baseline. The scientific focus has shifted from RNN circuit extraction to DCM as an
interpretability tool for models trained on real neuroimaging data.

**Critical path:** Phase 20 (synthetic validation) -> Phase 22 (main contribution)
-> Phase 27 (publication). Phase 21 (RNN) complete. Phase 23 (BMR) depends on
Phase 20. Phase 24 (foundation models) depends on Phase 20 + 22. Phases 25-26
(hybrid VAE-DCM, SBI) can run in parallel after Phase 20.

**Milestone acceptance gate:** (1) Parameter recovery on synthetic ground truth
passes RECOV-equivalent criteria; (2) DCM fit to neural-data-model latent dynamics
produces interpretable effective connectivity on real M/EEG; (3) ELBO or BMR
correctly selects circuit architecture; (4) Foundation model comparison across
modalities; (5) Publication figures and methods section complete.

### Phases

#### Phase 20: Direct Observation Forward Model, Simulator & Synthetic Validation

**Goal:** Users can fit bilinear DCM to N-dimensional neural-state trajectories via a
direct observation model (no hemodynamic convolution), with parameter recovery validated
on synthetic bilinear ground truth at the same standard as v0.3.0 RECOV benchmarks.

**Branch:** `gsd/phase-20-latent-circuit-forward-model`
**Depends on:** v0.3.0 shipping bilinear infrastructure (`NeuralStateEquation`,
`parameterize_A`, `parameterize_B`, `compute_effective_A`, `create_guide`, `run_svi`).
**Requirements:** OBS-01, OBS-02, OBS-03, OBS-04, SIM-01, SIM-02, MODEL-01, MODEL-02,
MODEL-03, MODEL-04, MODEL-05, SYNTH-01, SYNTH-02, SYNTH-03
**Plans:** 5 plans (4 waves)
Plans:
- [ ] 20-01-PLAN.md — CoupledDCMSystem hemodynamic toggle + latent circuit simulator + tests (OBS-01, OBS-03, OBS-04, SIM-01, SIM-02)
- [ ] 20-02-PLAN.md — Multi-start SVI extension to run_svi (MODEL-05)
- [ ] 20-03-PLAN.md — Pyro latent_circuit_dcm_model + guide auto-discovery verification (MODEL-01, MODEL-02, MODEL-03, MODEL-04)
- [ ] 20-04-PLAN.md — Recovery benchmark runner + metrics infrastructure (SYNTH-01, SYNTH-02 infrastructure)
- [ ] 20-05-PLAN.md — Prior calibration sweep + acceptance run + ELBO model selection on cluster (MODEL-02 finalization, SYNTH-01, SYNTH-02, SYNTH-03)

**Success Criteria** (what must be TRUE):

  1. `CoupledDCMSystem(hemodynamic=False)` integrates an N-dimensional state vector
     (no hemodynamic states) via `torchdiffeq.odeint`; `hemodynamic=True` preserves
     bit-exact existing behavior; zero edits to `neural_state.py`, `balloon_model.py`,
     `bold_signal.py` (OBS-01, OBS-03, OBS-04; `coupled_system.py` edited per CONTEXT
     decision to add toggle).
  2. `simulate_latent_circuit(...)` generates synthetic N-dimensional trajectories
     from known bilinear ground truth, and its output matches
     `CoupledDCMSystem(hemodynamic=False)` ODE integration at `atol=1e-6` given
     identical parameters (SIM-01, SIM-02).
  3. `latent_circuit_dcm_model` Pyro model samples `A_free`, `C`, `B_free_j`,
     `noise_prec` with priors recalibrated for RNN-scale dynamics
     (`LC_A_PRIOR_VARIANCE` documented separately from task DCM's
     `A_PRIOR_VARIANCE`); `C_obs` fixed at identity for v0.6.0; multi-start SVI
     (>=10 restarts, select by best ELBO) implemented as standard fitting procedure
     (MODEL-01, MODEL-02, MODEL-04, MODEL-05).
  4. `create_guide()` auto-discovers all sample sites in `latent_circuit_dcm_model`
     without factory changes; verified on AutoNormal, AutoLowRankMVN, AutoIAFNormal
     (MODEL-03).
  5. Parameter recovery on N=4 synthetic bilinear ground truth achieves: A RMSE
     within documented threshold, B RMSE within threshold, sign recovery >= 80%,
     95% CI coverage >= 85%; trajectory R-squared >= 0.95 on held-out trials;
     ELBO correctly selects true N=4 from candidates N={2,3,4,5,6}
     (SYNTH-01, SYNTH-02, SYNTH-03).

#### Phase 21: CT-RNN Training & Latent Extraction

**Goal:** Users can train a continuous-time RNN on a cognitive task, extract hidden
state trajectories, and reduce them to a low-dimensional space with quality diagnostics,
producing the observed data that bilinear DCM fits to.

**Branch:** `gsd/phase-21-ctrnn-training`
**Depends on:** Nothing in v0.6.0 (independent of Phase 20; requires only PyTorch +
neurogym + scikit-learn). Can run in parallel with Phase 20.
**Requirements:** RNN-01, RNN-02, RNN-03, RNN-04, DIM-01, DIM-02, DIM-03
**Plans:** 4 plans (4 waves)
Plans:
- [ ] 21-01-PLAN.md — CT-RNN module + rnn package + pyproject.toml deps + unit tests (RNN-01)
- [ ] 21-02-PLAN.md — RNN trainer + CDDM task integration + training tests (RNN-02)
- [ ] 21-03-PLAN.md — Fixed-point analysis + latent extraction (PCA, R2 gate, variance diagnostic) + tests (RNN-04, DIM-01, DIM-02, DIM-03)
- [ ] 21-04-PLAN.md — Cluster training script + sbatch for 20-seed ensemble + trajectory extraction (RNN-03)

**Success Criteria** (what must be TRUE):

  1. `ContinuousTimeRNN(nn.Module)` implements `tau * dh/dt = -h + f(W_rec @ h +
     W_in @ u + b)` with ReLU activation; trainable via BPTT with behavioral
     cross-entropy loss; at least 20 RNNs trained on CDDM with different seeds
     and saved weights + full h(t) trajectories per trial condition
     (RNN-01, RNN-02, RNN-03).
  2. Fixed-point analysis utilities find fixed points via `torch.optim`, compute
     Jacobians via `torch.autograd.functional.jacobian`, and decompose eigenvalues
     via `torch.linalg.eig`; used to report linearization quality at fixed points
     (RNN-04).
  3. PCA-based dimensionality reduction maps H-dimensional RNN hidden states to
     N-dimensional DCM state space with cumulative variance explained diagnostic;
     output reconstruction R-squared gate verifies PCA-projected states reconstruct
     RNN behavioral readout with R-squared >= 0.90 before fitting DCM
     (DIM-01, DIM-02, DIM-03).

#### Phase 22: DCM Interpretability for Neural Data Models

**Goal:** Train a temporal deep learning model on real M/EEG data, extract its
learned latent dynamics, and fit DCM (spectral or bilinear) to characterize
effective connectivity inside the model's representations -- demonstrating DCM
as an interpretability tool for neural data models.

**Branch:** `gsd/phase-22-dcm-neural-interpretability`
**Depends on:** Phase 20 (validated DCM model) + Phase 18 (MNE IO).
**Requirements:** INTERP-01 through INTERP-06 (TBD during planning)
**Success Criteria** (what must be TRUE — provisional, finalized during planning):

  1. A temporal model (VAE, transformer, or similar) trained on real M/EEG data
     (e.g., Cam-CAN MEG, sensory/motor task) achieves reconstruction quality
     sufficient for downstream analysis.
  2. Latent trajectories extracted from the trained model and mapped to
     source-localized ROI space (or ROI-aligned latent dimensions).
  3. Spectral or bilinear DCM fit to the model's latent dynamics produces
     posterior A matrices (and B_j if task-modulated) with credible intervals.
  4. Effective connectivity structure recovered by DCM relates to known
     neuroscience (e.g., sensory-to-motor pathways during a motor task),
     providing interpretability beyond what the upstream model offers alone.
  5. Comparison to DCM fit directly on the raw M/EEG ROI timeseries,
     demonstrating what the learned representation adds (denoising, compression,
     or structure that DCM alone misses).

#### Phase 23: Bayesian Model Reduction

**Goal:** Users can analytically score reduced DCM architectures (connection subsets)
from a single full-model SVI fit, enabling efficient circuit-size selection without
refitting.

**Branch:** `gsd/phase-23-bayesian-model-reduction`
**Depends on:** Phase 20 (latent circuit DCM posteriors).
**Requirements:** BMR-01, BMR-02, BMR-03
**Plans:** 3 plans
Plans:
- [ ] 23-01-PLAN.md -- Core BMR function + model_selection subpackage + unit tests (BMR-01)
- [ ] 23-02-PLAN.md -- Circuit-size selection: enumerate reduced architectures + score via BMR (BMR-02)
- [ ] 23-03-PLAN.md -- BMR vs brute-force ELBO validation on synthetic data + package finalization (BMR-03)
**Success Criteria** (what must be TRUE):

  1. `bayesian_model_reduction(posterior_mean, posterior_cov, prior_mean, prior_cov,
     reduced_prior_cov)` implements Friston & Penny (2011) analytic model evidence
     for reduced models, citing REF-070 (BMR-01).
  2. Circuit-size selection: fit full DCM (N=max), then analytically score all
     reduced architectures (connection subsets); log-evidence differences reported
     (BMR-02).
  3. BMR results compared to brute-force ELBO comparison (separate SVI fits per
     architecture) on synthetic data; agreement validates the analytic approximation
     (BMR-03).

#### Phase 24: Foundation Model Use Cases (TRIBE + M/EEG)

**Goal:** DCM distills interpretable circuit dynamics from pre-trained brain
foundation models, demonstrating generalization across modalities: fMRI (Meta
TRIBE v2) and M/EEG (LaBraM, MEG-GPT, or comparable).

**Branch:** `gsd/phase-24-foundation-model-use-cases`
**Depends on:** Phase 20 (validated DCM model) + Phase 22 (M/EEG pipeline).
**Requirements:** TRIBE-01, TRIBE-02, TRIBE-03, FMEEG-01, FMEEG-02, FMEEG-03
(TBD during planning)
**Success Criteria** (what must be TRUE — provisional, finalized during planning):

  1. Latent representations extracted from Meta TRIBE v2 (or comparable open-source
     brain encoding model) for a stimulus set; DCM fit produces posterior A/B
     matrices with credible intervals (TRIBE-01, TRIBE-02).
  2. Latent representations extracted from a pretrained M/EEG foundation model
     (LaBraM, MEG-GPT, or comparable) on source-localized ROI timeseries; DCM
     fit produces posterior connectivity with credible intervals (FMEEG-01,
     FMEEG-02).
  3. Comparison of DCM-derived effective connectivity across fMRI and M/EEG
     foundation models on overlapping tasks/regions, demonstrating modality-
     invariant circuit structure where expected (TRIBE-03, FMEEG-03).

#### Phase 25: Hybrid VAE-DCM

**Goal:** A sequential VAE where the decoder IS the DCM forward model (neural state
ODE + observation equation), providing a physics-informed generative model for
M/EEG data that jointly learns latent initial conditions and connectivity
parameters.

**Branch:** `gsd/phase-25-hybrid-vae-dcm`
**Depends on:** Phase 20 (DCM forward model) + Phase 22 (M/EEG data pipeline).
**Requirements:** HVAE-01 through HVAE-04 (TBD during planning)
**Success Criteria** (what must be TRUE — provisional, finalized during planning):

  1. VAE encoder maps M/EEG observations to approximate posterior over DCM
     parameters (A, B_j, C) and initial conditions; DCM forward model serves
     as the decoder generating predicted timeseries.
  2. Training via ELBO maximization recovers known connectivity on synthetic
     data with quality comparable to standalone SVI.
  3. On real M/EEG data, the hybrid model produces both reconstruction quality
     (VAE) and interpretable connectivity (DCM) from a single trained model.
  4. Amortized inference: once trained, parameter estimation for a new subject
     is a single encoder forward pass (no per-subject SVI).

#### Phase 26: SBI for Spectral DCM

**Goal:** Simulation-based inference (SBI) as an alternative to SVI for spectral
DCM, following VBI (eLife 2025) and spectral graph model (Comms Physics 2024)
approaches. Amortized Bayesian inference via neural density estimation on
simulated cross-spectral densities.

**Branch:** `gsd/phase-26-sbi-spectral-dcm`
**Depends on:** Phase 20 (spectral DCM forward model as simulator).
**Requirements:** SBI-01 through SBI-04 (TBD during planning)
**Success Criteria** (what must be TRUE — provisional, finalized during planning):

  1. Neural density estimator (SNPE or SNLE) trained on simulated CSD from the
     spectral DCM forward model recovers posterior parameter distributions that
     agree with SVI posteriors on synthetic data.
  2. Amortized inference: trained estimator performs posterior inference on new
     observations in a single forward pass (<1 second per subject).
  3. Coverage calibration: simulation-based calibration (SBC) confirms posterior
     credible intervals have nominal coverage.
  4. Demonstration on real M/EEG data (e.g., Cam-CAN) showing scalability to
     large N that would be intractable with per-subject SVI.

#### Phase 27: Publication Artifacts

**Goal:** All publication-quality figures, methods section, and reference updates are
complete, making the v0.6.0 results paper-ready.

**Branch:** `gsd/phase-27-publication-artifacts`
**Depends on:** Phase 22 (main results) + Phase 23 (BMR results) + Phase 24
(foundation model results).
**Requirements:** PUB-01, PUB-02, PUB-03
**Plans:** 3 plans (2 waves)
Plans:
- [ ] 27-01-PLAN.md -- REFERENCES.md update + methods section extension (Markdown + LaTeX) for all v0.6.0 methodology (PUB-02, PUB-03)
- [ ] 27-02-PLAN.md -- Publication figure generation script + pipeline schematic figure (PUB-01)
- [ ] 27-03-PLAN.md -- Equations quick-reference update + cross-consistency check + user verification checkpoint (PUB-01, PUB-02, PUB-03)

**Success Criteria** (what must be TRUE):

  1. Publication-quality figures covering: pipeline schematic, parameter recovery
     on synthetic, DCM-derived connectivity from M/EEG model latents, BMR circuit
     selection, foundation model comparison, hybrid VAE-DCM results (PUB-01).
  2. Methods section (Markdown + LaTeX) covers: DCM equations (bilinear + spectral),
     direct observation model, SVI and SBI inference, BMR, neural data model
     training, DCM interpretability framework (PUB-02).
  3. REFERENCES.md updated with all new references (BMR, SBI, VBI, foundation
     models, ODEBrain, Freeman et al. 2025, etc.) (PUB-03).

### Progress

**Execution Order:** 20 (+ 21 in parallel) -> 22 -> 23 (can overlap with 22) -> 24 (can overlap with 23) -> 25/26 (can overlap) -> 27

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 20. Direct Observation Forward Model, Simulator & Synthetic Validation | 4/5 | Plans 01-04 complete; Plan 05 acceptance needs rework | -- |
| 21. CT-RNN Training & Latent Extraction | 4/4 | Complete (20 seeds trained, trajectories extracted) | 2026-05-26 |
| 22. DCM Interpretability for Neural Data Models | 0/TBD | Not started | -- |
| 23. Bayesian Model Reduction | 0/3 | Planned | -- |
| 24. Foundation Model Use Cases (TRIBE + M/EEG) | 0/TBD | Not started | -- |
| 25. Hybrid VAE-DCM | 0/TBD | Not started | -- |
| 26. SBI for Spectral DCM | 0/TBD | Not started | -- |
| 27. Publication Artifacts | 0/3 | Planned | -- |

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
| 14. Stimulus Utilities & Bilinear Simulator | v0.3.0 | 2/2 | Complete | 2026-04-18 |
| 15. Pyro Generative Model with B Priors and Masks | v0.3.0 | 3/3 | Complete | 2026-04-18 |
| 16. 3-Region Bilinear Recovery Benchmark | v0.3.0 | 3/3 | Implementation complete; acceptance FAILED 2026-04-24 (RECOV-04) | -- |
| 16.1. RECOV-04 B-RMSE Shrinkage Diagnostic & Fix (INSERTED) | v0.3.0 | 0/2 | Planned | -- |
| 17. Circuit Visualization Module | v0.4.0 | 1/1 | Complete | 2026-04-24 |
| 18. MNE/BIDS IO Test Suite | v0.5.0 | 2/2 | Complete (verified 17/17 must-haves) | 2026-05-21 |
| 19. End-to-End Pipeline Demos | v0.5.0 | 2/2 | Complete (verified 10/10 must-haves) | 2026-05-24 |
| 20. Direct Observation Forward Model, Simulator & Synthetic Validation | v0.6.0 | 4/5 | Plans 01-04 complete; 05 needs rework | -- |
| 21. CT-RNN Training & Latent Extraction | v0.6.0 | 4/4 | Complete | 2026-05-26 |
| 22. DCM Interpretability for Neural Data Models | v0.6.0 | 0/TBD | Not started | -- |
| 23. Bayesian Model Reduction | v0.6.0 | 0/3 | Planned | -- |
| 24. Foundation Model Use Cases (TRIBE + M/EEG) | v0.6.0 | 0/TBD | Not started | -- |
| 25. Hybrid VAE-DCM | v0.6.0 | 0/TBD | Not started | -- |
| 26. SBI for Spectral DCM | v0.6.0 | 0/TBD | Not started | -- |
| 27. Publication Artifacts | v0.6.0 | 0/3 | Planned | -- |

---
*Roadmap created: 2026-04-07*
*Last updated: 2026-05-26 -- v0.6.0 restructured: dropped Langdon & Engel RNN-circuit-extraction focus. New core: DCM as interpretability tool for neural data models (Phase 22). Added Phases 25 (hybrid VAE-DCM), 26 (SBI for spectral DCM). Phase 24 expanded to include M/EEG foundation models alongside TRIBE. Phase 21 complete (20 RNN seeds trained). Phase 20-05 acceptance needs rework (24h timeout). Phase 27 planned: 3 plans (2 waves).*
