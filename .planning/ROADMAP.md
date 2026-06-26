# Roadmap: Pyro-DCM

## Milestones

- **v0.1.0 Foundation** - Phases 1-8 (shipped 2026-04-03)
- **v0.2.0 Cross-Backend Inference Benchmarking** - Phases 9-12 (shipped 2026-04-13)
- **v0.3.0 Bilinear DCM Extension** - Phases 13-16 (in progress; started 2026-04-17)
- **v0.4.0 Circuit Explorer** - Phase 17+ (defined 2026-04-24; not yet started)
- **v0.5.0 MNE-Python Integration** - Phases 18-19 (in progress; started 2026-05-21)
- ✅ **v0.6.0 Latent Circuit DCM** - Phases 20-28 (shipped 2026-06-10, scope-cut; real-data → v0.7.0)
- 📋 **v0.7.0 Variational Laplace Validation** - Phases 29-32 (defined 2026-06-10; not yet started)
- ✅ **v0.7.0 Variational Laplace Validation** - Phases 29-32 (complete; verified 2026-06-12)
- 📋 **v0.8.0 DCM for Evoked Responses (EEG/MEG ERP)** - Phases 33-36 (defined 2026-06-25; not yet started)

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

## v0.6.0 Latent Circuit DCM — SHIPPED 2026-06-10 (scope-cut)

<details>
<summary>✅ v0.6.0 Latent Circuit DCM (Phases 20-28) — SHIPPED 2026-06-10 (scope-cut; real-data deferred to v0.7.0)</summary>

Full archive: `.planning/milestones/v0.6.0-ROADMAP.md` · Audit: `.planning/milestones/v0.6.0-MILESTONE-AUDIT.md` · Forward scope: `.planning/v0.7.0-VL-RECONCILIATION-DRAFT.md`

- [x] Phase 20: Direct-Obs Forward Model + Synthetic Validation (5/5) — ✅ delivered via VL (A-RMSE 0.026, B-RMSE 0.0048, pooled-R² 0.961, BMR 3/3)
- [x] Phase 21: CT-RNN Training & Latent Extraction (4/4) — ✅
- [~] Phase 22: DCM Interpretability for Neural Data Models (6/6) — ⚠️ synthetic-only; real Cam-CAN M/EEG → v0.7.0
- [x] Phase 23: Bayesian Model Reduction (3/3) — ✅ (~93× faster than ELBO)
- [~] Phase 24: Foundation Model Use Cases (4/4) — ⚠️ infrastructure-only; real foundation-model runs → v0.7.0
- [x] Phase 25: Hybrid VAE-DCM (4/4) — ✅ (HVAE-02 confirmed, masked sign recovery 0.77)
- [~] Phase 26: SBI for Spectral DCM (2/2) — ❌ SBC 2/9 (structural); calibration → v0.7.0
- [x] Phase 27: Publication Artifacts (3/3) — ✅
- [x] Phase 28: Variational Laplace Inference Engine (retroactive) — ✅ SPM12 spm_nlsi_GN + ForwardModel protocol

**Delivered:** synthetic-validated DCM-recovery methodology + SPM12-grade VL inference engine + ready (un-run) real-data infrastructure.
**Deferred to v0.7.0 (not failed):** real Cam-CAN M/EEG (22), real foundation-model runs + cross-modal comparison (24), SBI SBC calibration + real demo (26).

</details>


---

## Current Milestone: v0.7.0 Variational Laplace Validation

**Status:** Defined 2026-06-10 (not yet started; ready to plan Phase 29).
**Phases:** 29-32 (4 phases)
**Requirements covered:** 19/19 v0.7.0 requirements (VLINFRA-01..05, VLREC-01..05,
VLBMR-01..03, VLSPM-01..03, VLROBUST-01..03) mapped to exactly one phase each.

### Overview

v0.7.0 is a **validation-breadth** milestone, not a new-capability one. The Variational
Laplace (VL) engine (SPM12 `spm_nlsi_GN` port, shipped retroactively as Phase 28), the BMR
model-selection module, and all three forward models (SpectralDCM, TaskDCM, LatentCircuit)
already exist and pass narrow smoke tests. v0.7.0 proves the engine is trustworthy across a
Cartesian product of network sizes, SNR levels, and forward models -- and that it agrees with
MATLAB SPM12 -- **before any real-data application** (real data and SBI calibration are
explicitly deferred to v0.8.0+). The work is almost entirely new glue code and test harnesses
over a frozen foundation: no new dependencies; the only `pyproject.toml` change is a new `vl`
pytest marker.

The single empirical finding that shapes the whole milestone is the **BMR overconfidence
failure** (cluster job 55772525): VL posterior std is ~0.001-0.01x prior std at high SNR, so
absolute-ΔF pruning is structurally broken. The only valid model-comparison mode is
**relative ranking + separation gap**; posterior tempering is exploratory and must be
calibrated from the Phase 30 coverage output before it can be trusted.

**Critical path:** Phase 29 (infra + BMR helpers, laptop, dependency root) -> Phase 30
(recovery-matrix sweep, M3 cluster) -> Phase 31 (BMR validation + tempering, depends on Phase
30 coverage). **Phase 32 (SPM12 cross-validation, local/MATLAB) runs in parallel with Phase
30** -- it is laptop-only and independent of the cluster sweep.

**Milestone acceptance gate:** (1) the recovery matrix passes documented per-cell thresholds
across N x SNR x {spectral, task, latent-circuit} -- or documents identifiability limits with
evidence (no silent failures); (2) BMR relative ranking recovers true circuit structure with a
reported separation gap and agrees with brute-force ELBO; (3) VL agrees with SPM12 `spm_nlsi_GN`
on >=1 spectral problem in free-parameter space (`Ep` within ~10%, F within ~5%, ranking
agreement); (4) VL convergence/determinism and numerical-robustness guards pass across all three
forward models. **Never** absolute-ΔF BMR, element-wise `Cp`, or absolute-F-across-models.

### Cross-cutting constraints (apply to multiple phases)

- **M3 cluster routing.** Every multi-seed recovery sweep (Phase 30) routes to the Monash M3
  cluster via `sbatch`; laptop is for smoke tests (N=2, 1 seed) only. No `pip install` inside
  SLURM array jobs.
- **Mutagen `models/` ignore is a PREREQUISITE for latent-circuit cluster runs.** The unanchored
  `models/` Mutagen ignore silently excludes `src/pyro_dcm/models/` from M3 sync; only
  `latent_circuit_vl` imports from that package (`LC_*_PRIOR_VARIANCE`). The session must be
  recreated with anchored ignores (todo `mutagen-models-ignore`) BEFORE any latent-circuit M3
  job in Phase 30. Spectral/task runners are unaffected.
- **SPM12 prior matching (Phase 32).** `hyperprior_mean = hE = 8.0`, `hyperprior_precision =
  1/128`, `prior_mean_a_offset = a_mask/128`; compare in **free-parameter space**, never
  parameterised A. **Never** compare element-wise `Cp` nor absolute F across engines -- model
  ranking + matched-problem F only.
- **Recovery metric hardening (Phases 30, 31).** Per-region R² (never variance-pooled),
  masked sign recovery (`|true| > threshold`, never `sign(0)`), CI coverage, RMSE, and
  identifiability shrinkage `std_post/std_prior`.
- **Stability-boundary exclusion (Phase 30).** Exclude near-boundary A (max Re eig in
  [-0.05, 0]) to avoid the `eig_clamp` non-injectivity confound; **task DCM enforces dt >= 0.1**.

### Phases

#### Phase 29: VL Validation Infrastructure & BMR Rank Functions

**Goal:** The benchmark and BMR layers expose everything downstream validation needs -- three
registered VL runners, VL-aware `BenchmarkConfig`, the corrected relative-ranking BMR API, a
PD-safe posterior-tempering primitive, and the precision-intractability / dt guards -- with VL
convergence + multi-restart determinism proven across all three forward models. Laptop-runnable
dependency root; no cluster requirement.

**Branch:** `gsd/phase-29-vl-validation-infra` (proposed)
**Depends on:** v0.6.0 frozen foundation (Phase 28 VL engine, `model_selection/bmr.py`, v0.2.0
`benchmarks/` runner registry + `.npz` fixtures + metrics).
**Requirements:** VLINFRA-01, VLINFRA-02, VLINFRA-03, VLINFRA-04, VLINFRA-05, VLREC-05,
VLROBUST-01, VLROBUST-02
**Plans:** 5 plans (3 waves)
Plans:
- [x] 29-01-PLAN.md — BenchmarkConfig VL fields + MATLAB_PATH in config.py + vl pytest marker (VLINFRA-01, VLINFRA-05) ✅ 2026-06-10
- [ ] 29-02-PLAN.md — rank_connections() relative ranking + temper_vl_posterior() PD guard + unit tests (VLINFRA-03, VLINFRA-04)
- [x] 29-03-PLAN.md — C-order CSD round-trip regression test + task-DCM precision intractability guard (VLREC-05, VLROBUST-02) ✅ 2026-06-10 (vl pytest marker registered here as prerequisite)
- [x] 29-04-PLAN.md — Three VL runners (spectral/task/latent_circuit) + RUNNER_REGISTRY + N=2 smoke tests (VLINFRA-02) ✅ 2026-06-10
- [x] 29-05-PLAN.md — VL convergence + multi-restart determinism tests across all 3 forward models + non-determinism docs (VLROBUST-01) ✅ 2026-06-10

**Success Criteria** (what must be TRUE):

  1. `BenchmarkConfig` gains optional VL fields (`max_iter`, `hyperprior_mean`,
     `hyperprior_precision`, `prior_mean_a_offset`) with `None` defaults; every existing
     benchmark caller and test passes unchanged (zero behavior change) (VLINFRA-01).
  2. Three VL runners (`spectral_vl`, `task_vl`, `latent_circuit_vl`) are registered under
     `method="vl"` in `RUNNER_REGISTRY`, reuse the v0.2.0 `.npz` fixture + metrics
     infrastructure, and smoke-test green at N=2, 1 seed on laptop. The `vl` pytest marker is
     registered in `pyproject.toml` and `MATLAB_PATH` is centralized in `config.py` (VLINFRA-02,
     VLINFRA-05).
  3. `rank_connections()` runs K single-connection BMR calls and returns connections ranked by
     prune cost with a separation-gap statistic; unit tests confirm the ranking and gap on a
     known ground-truth circuit (VLINFRA-03, builds the relative-ranking-from-the-start mode
     that avoids the absolute-ΔF pitfall C1).
  4. `temper_vl_posterior()` scales the VL posterior covariance by a temperature and asserts
     positive-definiteness (Cholesky) on the tempered covariance, raising loud on failure
     (VLINFRA-04).
  5. VL convergence + multi-restart **determinism** regression tests pass across all three
     forward models (fixed seed -> stable result; non-determinism sources documented), and the
     task-DCM precision-matrix **intractability guard** fails loud with expected-vs-actual matrix
     size when `dt x duration` exceeds a tractable T, with the `dt >= 0.1` floor documented
     (VLROBUST-01, VLROBUST-02).
  6. A C-order CSD round-trip regression test is added to the suite, guarding the
     column-major<->row-major / complex-CSD indexing bug class **before any SPM
     cross-validation run** (VLREC-05).

#### Phase 30: Recovery Matrix Sweep (M3 Cluster)

**Goal:** A per-cell recovery matrix over N x SNR x {spectral, task, latent-circuit} (>=10
seeds/cell) is computed on M3 with the hardened metric suite, producing the coverage and
identifiability output that every downstream diagnostic and the tempering calibration consume --
either passing documented per-cell thresholds or documenting identifiability limits with
evidence.

**Branch:** `gsd/phase-30-recovery-matrix-sweep` (proposed)
**Depends on:** Phase 29 (VL runners + VL-aware `BenchmarkConfig` + metric helpers).
**Requirements:** VLREC-01, VLREC-02, VLREC-03, VLREC-04, VLROBUST-03
**Plans:** 3 plans (3 waves)
Plans:
- [x] 30-01-PLAN.md — Hardened per-cell metric assembler (per-region R2, masked sign recovery, 95% coverage, RMSE, std_post/std_prior shrinkage) + near-boundary-A exclusion band [-0.05, 0] + per-model SNR injection + unit tests (laptop) (VLREC-02, VLREC-03) ✅ 2026-06-10
- [x] 30-02-PLAN.md — Recovery-matrix grid driver + env-driven single-cell entrypoint + SLURM array sbatch (no-pip, dt>=0.1) + local <3min faithfulness pre-check + SUBMIT small grid (N in {2,4} x SNR in {1,3} x 3 models, >=10 seeds = 120 fits) to M3 ✅ 2026-06-10 (job 56346424, all 10 tasks RUNNING)
- [x] 30-03-PLAN.md — POST-RESULTS harvest+aggregate: documented per-cell thresholds + classifier (pass | identifiability-limit-with-evidence, no silent failures) + matrix CSV/JSON + eig_clamp/boundary regime characterization + report (VLREC-04, VLROBUST-03) ✅ 2026-06-10; task underflow fixed (rk4) + cells 4-7 re-run on M3 (job 56372816) ✅ 2026-06-11 → **10/10 cells classified, 0 errored: 6 PASS, 4 ident-limit** (task N=2 recovers sign 1.0; task N=4 is a documented identifiability limit)

**Success Criteria** (what must be TRUE):

  1. The sweep executes on the **M3 cluster** over N x SNR x {spectral, task, latent-circuit}
     with >=10 seeds per cell, emitting per-cell JSON/CSV; latent-circuit jobs run only after the
     Mutagen `models/` sync fix is confirmed (VLREC-01).
  2. Per-cell metrics use **per-region R² (not variance-pooled)**, **masked** sign recovery
     (`|true| > threshold`), 95% CI coverage, RMSE, and identifiability shrinkage
     `std_post/std_prior` -- hardened against the `sign(0)` and pooled-R² artifacts (VLREC-02).
  3. The recovery design **excludes near-stability-boundary A** (max Re eig in [-0.05, 0]) to
     avoid the `eig_clamp` non-injectivity confound, and **task DCM enforces dt >= 0.1** (VLREC-03).
  4. Every cell either **passes its documented threshold** or its **identifiability limit is
     documented with evidence** (no silent failures); the eig_clamp / stability-boundary regime
     is characterized -- recovery degradation near the boundary is documented and the
     non-injective regime is flagged (VLREC-04, VLROBUST-03).

#### Phase 31: BMR Validation & Posterior Tempering (Exploratory)

**Goal:** Bayesian Model Reduction is validated as a defensible model-comparison tool -- relative
evidence ranking recovers the true circuit structure with a reported separation gap and agrees
with brute-force ELBO -- with posterior tempering offered only as an exploratory, PD-safe restore
of an absolute-ΔF regime calibrated against the Phase 30 coverage output.

**Branch:** `gsd/phase-31-bmr-validation-tempering` (proposed)
**Depends on:** Phase 29 (`rank_connections` + `temper_vl_posterior`) and Phase 30 (recovery
posteriors + coverage curves for tempering calibration).
**Requirements:** VLBMR-01, VLBMR-02, VLBMR-03

**Plans:** 3 plans (2 waves)

Plans:
- [x] 31-01-PLAN.md — VLBMR-01: relative-ranking recovery harness (spectral VL fit, top-K==true edges, separation gap) [Wave 1, laptop] ✅ 2026-06-11 (N=2/N=4 recover 5/5 seeds; reciprocal-edge ground truth — spectral DCM can't identify a lone off-diagonal A)
- [x] 31-02-PLAN.md — VLBMR-02: BMR vs brute-force VL-refit rank-agreement on a small model set [Wave 1, laptop] ✅ 2026-06-11 (present>absent prune cost on both methods, worst-model agreement, ρ=1.0)
- [x] 31-03-PLAN.md — VLBMR-03: exploratory tempering calibration vs Phase 30 coverage, PD-safe, side-by-side [Wave 2, laptop test + M3 task-N4 re-fit] ✅ 2026-06-11 (job 56397206; T=2.0 restores coverage 0.875→1.0, held-out C2c non-PD surfaced — no universal schedule)

**Success Criteria** (what must be TRUE):

  1. BMR **relative-evidence ranking** recovers the true circuit structure (top-K essential edges
     == true edges) on synthetic ground truth, with the separation gap reported -- the primary,
     defensible model-comparison result (VLBMR-01).
  2. BMR-on-VL agrees with **brute-force ELBO** model comparison on a small model set, validating
     the analytic approximation (VLBMR-02).
  3. Posterior tempering is reported as **exploratory only**: tempered and untempered rankings are
     shown side by side, the temperature is calibrated against Phase 30 coverage, and the tempered
     covariance is asserted PD (Cholesky); tempering is documented as NOT a headline claim and
     absolute-ΔF is never used as a pass/fail criterion (VLBMR-03, avoids pitfall C1/C2).

#### Phase 32: SPM12 Cross-Validation (Local / MATLAB)

**Goal:** VL output is cross-validated against MATLAB `spm_nlsi_GN` on a prior-matched spectral
DCM problem, agreeing on posterior means in free-parameter space, on matched-problem free energy,
and on model ranking -- reusing the existing `validation/` SPM bridge. Runs locally (MATLAB
required) and **in parallel with the Phase 30 cluster sweep**.

**Branch:** `gsd/phase-32-spm12-cross-validation` (proposed)
**Depends on:** Phase 29 (VL-aware `BenchmarkConfig` prior-matching fields + C-order CSD
round-trip test). Independent of Phase 30; runs concurrently.
**Requirements:** VLSPM-01, VLSPM-02, VLSPM-03

**Success Criteria** (what must be TRUE):

  1. VL output is cross-validated vs MATLAB `spm_nlsi_GN` on >=1 spectral DCM problem,
     **prior-matched** (`hE = 8.0`, `prior_mean_a_offset = a_mask/128`, comparison in
     **free-parameter space**) (VLSPM-01, avoids pitfalls S1+S2).
  2. `Ep` agrees within ~10% relative tolerance with model-ranking agreement, and free energy
     agrees (VL `free_energy` == SPM `DCM.F`, ~5%) on the **matched** problem; the suite
     **never** compares element-wise `Cp` nor absolute F across models (VLSPM-02, avoids pitfall
     S3).
  3. The existing `validation/` SPM pipeline (`export_to_mat`, MATLAB batch, `compare_results`)
     is reused, extended only by `run_vl_validation.py` + `compare_free_energies()`; the C-order
     CSD round-trip test (Phase 29) is green before any comparison run (VLSPM-03, avoids matrix
     layout pitfall S4).

### Progress

**Execution Order:** 29 -> 30 -> 31, with 32 in parallel with 30 (32 depends only on 29).

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 29. VL Validation Infrastructure & BMR Rank Functions | 5/5 | ✅ Complete (verified 6/6, 17/17 vl tests) | 2026-06-10 |
| 30. Recovery Matrix Sweep (M3 Cluster) | 3/3 | ✅ Complete (verified passed 5/5); 10/10 cells classified, 0 errored: 6 PASS, 4 ident-limit. Spectral+latent+task N=2 VALIDATED; task N=4 documented ID-limit. jobs 56346424 + 56372816 | 2026-06-11 |
| 31. BMR Validation & Posterior Tempering (Exploratory) | 3/3 | ✅ Complete (verified passed 3/3; 8 vl tests green) — VLBMR-01/02/03; tempering exploratory, never gates absolute ΔF | 2026-06-11 |
| 32. SPM12 Cross-Validation (Local / MATLAB) | 3/3 | ✅ Complete (verified passed 3/3; ran on M3, jobs 56407192+56407635). Model-ranking agreement 1.0 (all 5 seeds); VL≡SPM free energy up to a CONSTANT 269.895-nat offset (std=0); strict-5%-absolute-F + 10%-Ep not met (documented forward-model difference, VL tracks truth closer) — VLSPM-01/02/03 Complete with findings | 2026-06-12 |

---

## Current Milestone: v0.8.0 DCM for Evoked Responses (EEG/MEG ERP)

**Status:** Defined 2026-06-25 (not yet started; ready to plan Phase 33).
**Phases:** 33-36 (4 phases)
**Requirements covered:** 25/25 v0.8.0 requirements (CMC-01..07, EVOK-01..06, LEAD-01..06,
ERPDCM-01..06) mapped to exactly one phase each.

### Overview

v0.8.0 adds a complete **time-domain ERP forward stack** to Pyro-DCM: a canonical-microcircuit
(CMC) neural-mass model -> extrinsic laminar coupling + condition modulation + evoked integration
-> single-dipole lead-field -> scalp ERP, **SPM12-parity validated at every phase** and reusing the
v0.7.0 Variational Laplace + amortized inference with zero engine edits (`ERPDCMForward` implements
the existing `ForwardModel` protocol, the v0.6.0 `LatentCircuitForward` precedent). This is a
**forward + parity + synthetic** milestone: no empirical M/EEG data fitting. Research (HIGH
confidence, SPM12 source read line-by-line) identifies the single headline risk that shapes the
whole milestone: **SPM does NOT integrate ERPs with Runge-Kutta.** `spm_gen_erp` calls `spm_int_L`,
an exponential-Euler / frozen-Jacobian (Ozaki 1992) integrator; torchdiffeq rk4/dopri5 converges to
the true ODE but NOT to the SPM solution at the default dt=4 ms, producing smooth, plausible, WRONG
ERP traces that pass every NaN/shape test and fail parity by a growing drift insensitive to
step-size reduction. The fix -- a new pure-torch `utils/local_linearization.py` porting `spm_int_L`
(frozen Jacobian at `x0=0`, `Q=(matrix_exp(dt*D*J/N)-I)*inv(J)` via `torch.linalg.solve`
right-division, float64, `exp(-16)` regulariser) -- is the central new component and **must exist and
be fixture-verified before any other ERP work proceeds.** The stack needs **zero new dependencies**
(`torch.matrix_exp` + `torch.linalg.solve`, both present in the existing torch >= 2.0 pin) and is
**additive only** (new files or new symbols appended; existing fMRI/spectral/rDCM/latent paths stay
bit-exact).

The headline scientific deliverable is a **5-source auditory MMN network** (bilateral A1, bilateral
STG, rIFG) that reproduces the canonical deviant-minus-standard difference wave and produces a
quantitative **precision -> MMN-amplitude attenuation curve** by sweeping superficial-pyramidal
self-inhibition gain (`P.G[:,0] -> G[:,6]` via the intrinsic permutation remap) -- the Adams 2013 /
Ranlund 2016 aberrant-precision mechanism, handed off to the downstream `actinf_physics` Phase-133
forward-only adapter.

**Critical path is strictly linear: Phase 33 -> 34 -> 35 -> 36, with NO phase-level parallelism.**
Each phase contributes exactly one tier of the staged fixture ladder (`f(x,u,theta)` field -> `J0`
-> `Q_update` -> single-source trajectory -> `spm_gen_Q` Q -> multi-source trajectory ->
`spm_lx_erp` L -> scalp ERP -> difference wave); divergence at any tier compounds silently into all
downstream tiers (Phase-32 proved small forward differences compound). Unlike v0.7.0 (where Phase 32
ran in parallel with Phase 30), **no v0.8.0 phase can start before its predecessor's parity gate is
green.**

**Milestone acceptance gate:** all four phases pass their SPM12 **forward-parity** gates on frozen
MATLAB fixtures (R2022a + Carrick spm12 on M3) at the documented tolerance tiers (`J0` <= 1e-10,
`Q_update` <= 1e-9, single-source trajectory <= 1e-8, scalp ERP <= 1e-7); AND the 5-source MMN demo
reproduces a **monotone precision -> attenuation** transfer curve with a **frontal-dominant
negative-going** difference wave, **gated behind a green fixed-reference SPM forward-parity check
before any sweep output is produced.** **Forward element-wise parity only** -- never gate on
absolute free energy (Phase-32 proved a constant ~270-nat offset); any inference-vs-SPM comparison
uses deltaF / ranking, never absolute F or element-wise `Cp`.

### Cross-cutting constraints (apply to every phase)

- **Parity is vs-SPM, never vs-torch.** Every phase gate asserts against frozen MATLAB-exported
  arrays following the Phase-32 `validation/` bridge pattern (Python exports `.mat` ->
  MATLAB/SPM12 reference on M3 -> Python asserts in `tests/test_spm_erp_dcm_validation.py`).
  Self-referential torch-vs-torch tests are NOT parity gates.
- **Staged fixture ladder.** Assert at each boundary in order so divergence localises:
  `f(x,u,theta)` field -> `J0` -> `Q_update` -> single-source trajectory -> `spm_gen_Q` Q ->
  multi-source trajectory -> `spm_lx_erp` L -> scalp ERP -> difference wave.
- **Tolerance tiers (MEASURED, not assumed).** `J0` <= 1e-10, `Q_update` <= 1e-9, single-source
  trajectory <= 1e-8, scalp ERP <= 1e-7. Phase 33 MEASURES the `matrix_exp` vs `spm_expm` floor
  (the `Q_update` tier) and records it -- it sets the tolerance floor for all downstream phases.
- **Five mandatory guard tests (cannot be skipped):** (33) permutation guard -- perturb `P.G[:,0]`,
  assert `G[:,6]` (sp self-inhibition) changes, NOT `G[:,0]`; (34) `spm_gen_Q` fixture -- torch
  `Q.A{1..4}` and `Q.G[:,0]` match exported MATLAB element-wise; (35) `P.J` default guard -- assert
  observed state is index 2 (sp voltage), not index 6 (dp voltage), + kron column-major order vs
  exported `L_full`; (36) frozen-ref SPM forward parity green BEFORE the demo runs + monotone
  `gain -> |MMN|` attenuation assertion.
- **Additive-only / bit-exact.** Existing fMRI/spectral/rDCM/latent forward models, model classes,
  `ode_integrator.py`, and the VL engine core are NEVER edited; verifiable by `git diff` showing
  only insertions. **Zero new runtime dependencies** (`torch.matrix_exp` + `torch.linalg.solve`
  only). **float64** at the ForwardModel boundary. **Delays forced off (D=1)** for the first parity
  pass; full `spm_dcm_delay` deferred.
- **Compute routing.** Fast single-source / integrator unit tests run on laptop; all MATLAB
  fixture-generation jobs and any multi-source integration sweep projected > 3 min route to **M3**
  (R2022a + Carrick spm12, comp partition; no `pip` in array jobs).
- **Five research gaps carried forward into planning:** (1) `matrix_exp` vs `spm_expm` tolerance is
  MEASURED empirically in **Phase 33** (do not assume 1e-12); (2) 5-source **MNI coordinates**
  verified against the primary Garrido/Ranlund papers before hard-coding (Phase 34/36); (3) **delay
  off** confirmed (`D=1`) in the fixture-generation MATLAB script before the first fixture run
  (Phase 34); (4) **observation stacking layout** `(Cnd, ns, Nc)` locked in `predict` /
  `build_precision` / `.mat` output in **Phase 35** before writing scalp-ERP assertions; (5)
  **Zotero citations** REF-ERP-001..006 + REF-MMN-001..004 confirmed in Zotero before any `[REF-xxx]`
  is added to `REFERENCES.md` or docstrings (never fabricate Better BibTeX keys).

### Phases

#### Phase 33: CMC Core Dynamics, spm_int_L Integrator & Single-Source Parity

**Goal:** A verified single-source canonical-microcircuit forward (4 populations, 8 states) and a
verified `spm_int_L` exponential-Euler integrator exist, with single-source SPM12 parity proven on
frozen MATLAB fixtures -- catching the integration-scheme mismatch (the milestone headline risk) in
isolation before any extrinsic coupling can compound it.

**Branch:** `gsd/phase-33-cmc-core-and-integrator`
**Depends on:** v0.7.0 frozen foundation (Phase 28 VL engine + `ForwardModel` protocol; the Phase-32
`validation/` SPM bridge pattern: `export_to_mat` -> MATLAB batch on M3 -> Python assertion).
**Requirements:** CMC-01, CMC-02, CMC-03, CMC-04, CMC-05, CMC-06, CMC-07
**Plans:** 3 plans (3 waves)
Plans:
- [ ] 33-01-PLAN.md — Pure-torch CMC core: spm_int_L integrator (local_linearization.py), cmc_neural_mass.py (cmc_f/sigmoid/parameterize_cmc + permutation guard), cmc_priors.py, erp_input.py; C1-isolation + permutation-guard tests first (CMC-01..05, CMC-07; LAPTOP)
- [ ] 33-02-PLAN.md — MATLAB fixture generation on M3: export_erp_dcm append, run_spm_erp_dcm.m + spm_fx_cmc_nodelay.m (force D=1), cluster harness; produces erp_single_source_fixtures.mat (CMC-06 fixtures; M3/MATLAB, ssh-gated)
- [ ] 33-03-PLAN.md — SPM12 parity gate: V5 staged ladder (f_field→J0→matrix_exp measured→Q_update→y_states) + measured matrix_exp floor; real run on M3, auto-skip on laptop (CMC-06; M3/MATLAB, ssh-gated)
**Success Criteria** (what must be TRUE):

  1. **(SPM12 PARITY GATE -- written FIRST, before any extrinsic coupling exists.)** Single-source
     `spm_fx_cmc` + `spm_int_L` parity vs frozen MATLAB fixtures (D=1): `f(x,u,theta)` derivative
     field <= 1e-10, `J0` (frozen Jacobian at `x0=0`) <= 1e-10, `Q_update` <= 1e-9 (this MEASURES
     the `matrix_exp` vs `spm_expm` floor -- recorded, not assumed), full state trajectory
     `y_states (ns,8)` <= 1e-8. Fixture metadata header freezes SPM `$Id`, dt, ns, `M.ons`/dur, the
     `D=1` assertion, and the `x0==0` check (CMC-06).
  2. `utils/local_linearization.py` ports `spm_int_L`: frozen Jacobian at `x0=0`,
     `Q=(matrix_exp(dt*D*J/N)-I)*inv(J)` computed via `torch.linalg.solve` right-division
     (`solve(J.T,(E-I).T).T`, NEVER `torch.inverse`), float64 throughout, `exp(-16)` Jacobian
     regulariser applied BEFORE forming Q; CMC is NOT routed through `integrate_ode`/torchdiffeq
     (CMC-03).
  3. **Permutation guard (mandatory).** `parameterize_cmc` applies the SPM log/exp transforms with
     the intrinsic permutation `j=[7 2 3 4 1 5 6 8 9 10]` (only 4 free `G`, 4 free `T`) and the
     `+exp(P.A)` extrinsic convention (NOT the fMRI `-exp/2`); a unit test perturbs `P.G[:,0]` and
     asserts `G[:,6]` (sp self-inhibition = precision) changes, NOT `G[:,0]` (CMC-02).
  4. `forward_models/cmc_neural_mass.py` implements the single-source CMC state equations -- 4
     populations (ss/sp/ii/dp), 8 states, second-order synaptic kernel, sigmoid
     `S(V)=1/(1+exp(-Rx))-1/2` with `R=(2/3)*exp(P.S)` -- citing `spm_fx_cmc.m`;
     `forward_models/cmc_priors.py` provides prior means/variances + transform tables from
     `spm_cmc_priors.m`; `forward_models/erp_input.py` provides the Gaussian-bump evoked drive
     `u(t)` (onset, dispersion `P.R`, 32-scaling, ms timebase) porting `spm_erp_u.m` (CMC-01,
     CMC-04, CMC-05).
  5. float64 is enforced at the ForwardModel boundary; CMC steady state is asserted `x0 == zeros`
     (no Newton solve); the fMRI eigenvalue-clip rule is NOT applied to the CMC Jacobian (only the
     `exp(-16)` shift) -- guarded by a unit test (CMC-04, CMC-07).

#### Phase 34: Extrinsic Coupling, Condition B & Multi-Source Evoked Integration

**Goal:** The hierarchical CMC network -- extrinsic forward/backward/lateral coupling, condition-
specific `B` modulation (including the `diag(B)->G` precision path), and `C`-driven evoked
integration over the peristimulus window -- produces per-source per-condition LFPs that match
`spm_gen_erp` on frozen multi-source fixtures (delays off, D=1).

**Branch:** `gsd/phase-34-extrinsic-coupling-evoked`
**Depends on:** Phase 33 (verified single-source CMC forward + verified `spm_int_L` integrator).
**Requirements:** EVOK-01, EVOK-02, EVOK-03, EVOK-04, EVOK-05, EVOK-06
**Success Criteria** (what must be TRUE):

  1. **(SPM12 PARITY GATE.)** `spm_gen_Q` fixture: torch `Q.A{1..4}` and `Q.G[:,0]` match exported
     MATLAB element-wise (the critical B-wiring guard); the multi-source evoked trajectory matches
     `spm_gen_erp` for the 5-source MMN reference A/B/C (delays off, D=1) within <= 1e-8 (EVOK-05).
  2. `forward_models/erp_coupled_system.py` wires extrinsic `A` across a multi-source network --
     forward (sp->ss/dp, `+`), backward (dp->sp/ii, `-`), lateral (reciprocal `1/(1+4L)` reduction)
     -- citing `spm_fx_cmc.m` (EVOK-01).
  3. Condition-specific modulation `B` is applied additively in log-space to all `A{1..4}` AND via
     `diag(B)->G(:,1)->G[:,6]` (the precision path), porting `spm_gen_Q.m`; omitting the `diag->G`
     path is explicitly tested against (it destroys the MMN precision mechanism) (EVOK-02).
  4. Input `C` drives spiny-stellate only; evoked integration over the peristimulus window (default
     dt=4 ms, ns=128) via the Phase-33 integrator yields per-source per-condition LFP (the
     `spm_gen_erp` analog); `simulators/erp_simulator.py` provides `simulate_erp_dcm(...)` returning
     a per-condition source/scalp ERP dict + the deviant-minus-standard difference-wave hook
     (EVOK-03, EVOK-04).
  5. The delay operator is forced off (D=1) and asserted in the fixture-generation MATLAB script;
     the full `spm_dcm_delay` path is deferred to a later milestone (EVOK-06).

**Plans:** 3 plans

Plans:
- [ ] 34-01-PLAN.md — Pure-torch network core (cmc_network_f extrinsic A + apply_condition_modulation spm_gen_Q port + erp_simulator) + the C4 diag->G guard & negative test (Wave 1, laptop)
- [ ] 34-02-PLAN.md — Multi-source MATLAB fixtures on M3: locked 5-source A/B/C/X export + run_spm_erp_dcm_multisource.m (spm_gen_Q QA/QG, per-cond J0/Qupd, spm_gen_erp trajectory; D=1) (Wave 2, M3)
- [ ] 34-03-PLAN.md — Multi-source parity ladder vs frozen fixtures: spm_gen_Q algebra + diag->G negative + network J0/Q_update + trajectory (scheme/FD-Jacobian/measured-jacrev) (Wave 3, laptop)

#### Phase 35: Single-Dipole Lead-Field, Scalp Projection & ERPDCMForward

**Goal:** Per-source LFPs become the observed scalp ERP via a single-dipole lead-field
(`kron(P.J, L_spatial)`, LFP-first), the deviant-minus-standard difference wave is produced, and
`ERPDCMForward` (implementing the existing `ForwardModel` protocol) gives VL inference as free
reuse -- with scalp-ERP parity proven vs `spm_lx_erp` on frozen fixtures.

**Branch:** `gsd/phase-35-leadfield-scalp-projection`
**Depends on:** Phase 34 (verified multi-source evoked trajectories).
**Requirements:** LEAD-01, LEAD-02, LEAD-03, LEAD-04, LEAD-05, LEAD-06
**Success Criteria** (what must be TRUE):

  1. **(SPM12 PARITY GATE.)** Scalp ERP matches `spm_gen_erp` + `spm_lx_erp` (LFP mode) on frozen
     fixtures within <= 1e-7; the ECD gain (post `spm_cond_units`) is precomputed in MATLAB and
     exported via the `validation/` `.mat` bridge (Python reproduces only `kron(P.J,L)` +
     projection). The observation stacking layout `(Cnd, ns, Nc)` is locked across `predict` /
     `build_precision` / `.mat` output BEFORE these assertions are written (LEAD-05).
  2. `forward_models/erp_leadfield.py` builds `L_full = kron(P.J, L_spatial)` with a column-major
     state-blocked flatten; LFP diagonal spatial model first; projection `y = (x - x0) @ L_full.T`,
     citing `spm_lx_erp.m`/`spm_erp_L.m` (LEAD-01).
  3. **P.J default guard (mandatory).** `P.J` default = state index 2 (superficial-pyramidal
     voltage) is asserted (NOT index 6, deep-pyramidal); the kron column-major order is verified
     against an exported `L_full` fixture (LEAD-02).
  4. The deviant-minus-standard difference wave is computed, asserted non-zero and negative-going
     (LEAD-03).
  5. `class ERPDCMForward` appended to `inference/forward_models.py` implements the existing
     `ForwardModel` protocol (`residual_is_complex=False`; pack/unpack; `build_prior_cov` from
     `cmc_priors`; identity precision v1; `predict -> (Cnd, ns, Nc)`) with ERP-specific needs as
     constructor args/context -- zero edits to the protocol or the VL engine (LEAD-04).
  6. VL round-trip -- `run_variational_laplace_generic(ERPDCMForward(), ...)` recovers planted CMC
     params on synthetic ground truth (protocol confirmation; not a parity gate) (LEAD-06).

#### Phase 36: ERP-DCM Pyro Model, Amortized Wiring & MMN Precision-Sweep Demo

**Goal:** The milestone completes with a Pyro generative ERP-DCM model, amortized inference wiring,
and the headline 5-source MMN precision-sweep demo -- the `actinf_physics` hand-off artifact --
gated behind a green fixed-reference SPM forward-parity check.

**Branch:** `gsd/phase-36-erp-dcm-model-mmn-demo`
**Depends on:** Phase 35 (`ERPDCMForward` + scalp-ERP parity + difference wave).
**Requirements:** ERPDCM-01, ERPDCM-02, ERPDCM-03, ERPDCM-04, ERPDCM-05, ERPDCM-06
**Success Criteria** (what must be TRUE):

  1. **(SPM12 PARITY GATE.)** The MMN sweep figure is gated behind a GREEN fixed-reference SPM
     forward-parity check (LFP + difference wave reproduced from `spm_gen_erp` + `spm_lx_erp` at
     fixed reference params, within Phase-35 tolerances on frozen fixtures) BEFORE any sweep output
     is produced (ERPDCM-06).
  2. `models/erp_dcm_model.py` Pyro generative model -- log-normal priors (A,B,C,G,T,R), Gaussian
     likelihood on the scalp residual, B modulation -- consistent with the `spectral_dcm_model`
     idiom (ERPDCM-01).
  3. Amortized path -- `ERPDCMPacker` (`guides/parameter_packing.py`) + `amortized_erp_dcm_model`
     (`models/amortized_wrappers.py`) appended additively; the amortized flow guide trains on
     `erp_simulator` draws without error (ERPDCM-02).
  4. A 5-source auditory MMN network (A1 L/R, STG L/R, rIFG) with the forward/backward/lateral
     connection graph, `C` into bilateral A1, and deviant-vs-standard `B` modulation; MNI coords
     flagged for verification against the primary papers before hard-coding (ERPDCM-03).
  5. `scripts/demo_mmn_precision_sweep.py` sweeps superficial-pyramidal self-inhibition gain
     (`P.G[:,0] -> G[:,6]`) at rIFG + bilateral A1 and emits a `gain -> |MMN|` transfer curve;
     asserts monotone attenuation and a frontal-dominant negative-going difference wave (the
     Adams/Ranlund artifact), reusing the Phase-33 permutation guard (`P.G[:,0]` perturbation
     changes `G[:,6]`) (ERPDCM-04).
  6. A consumer-facing adapter API maps
     `(sp_inhibition_gain, a1_b_gain, rifg_b_gain, fwd_bwd_flag) -> CMC params` for the
     `actinf_physics` Phase-133 forward-only adapter (ERPDCM-05).

### Progress

**Execution Order:** 33 -> 34 -> 35 -> 36 (strictly linear; NO phase-level parallelism -- each phase
contributes one tier of the staged fixture ladder and cannot start before its predecessor's SPM12
parity gate is green).

| Phase | SPM12 reference | Tolerances | Plans Complete | Status | Completed |
|-------|-----------------|------------|----------------|--------|-----------|
| 33. CMC Core Dynamics, spm_int_L Integrator & Single-Source Parity | `spm_fx_cmc` + `spm_int_L` (single source, D=1) | J0 <=1e-10, Q <=1e-9, traj <=1e-8 | 3/3 | ✅ Complete (verified 7/7; parity ladder GREEN vs M3 fixtures job 57884677; measured matrix_exp floor 8.6e-11) | 2026-06-26 |
| 34. Extrinsic Coupling, Condition B & Multi-Source Evoked Integration | `spm_gen_Q` + `spm_gen_erp` (5-source, D=1) | Q.A/Q.G element-wise, traj <=1e-8 | 0/? | Not started | -- |
| 35. Single-Dipole Lead-Field, Scalp Projection & ERPDCMForward | `spm_lx_erp` (LFP mode) | scalp ERP <=1e-7 | 0/? | Not started | -- |
| 36. ERP-DCM Pyro Model, Amortized Wiring & MMN Precision-Sweep Demo | full pipeline at fixed-ref params (LFP) | same as Phase 35 + monotone curve | 0/? | Not started | -- |

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
| 20. Direct Observation Forward Model, Simulator & Synthetic Validation | v0.6.0 | 5/5 | ✅ Delivered (via VL) | 2026-06-09 |
| 21. CT-RNN Training & Latent Extraction | v0.6.0 | 4/4 | ✅ Delivered | 2026-05-26 |
| 22. DCM Interpretability for Neural Data Models | v0.6.0 | 6/6 | ⚠️ Synthetic-only; real-data → v0.7.0 | 2026-05-31 |
| 23. Bayesian Model Reduction | v0.6.0 | 3/3 | ✅ Delivered | 2026-05-31 |
| 24. Foundation Model Use Cases (TRIBE + M/EEG) | v0.6.0 | 4/4 | ⚠️ Infrastructure-only; real runs → v0.7.0 | 2026-05-31 |
| 25. Hybrid VAE-DCM | v0.6.0 | 4/4 | ✅ Delivered (HVAE-02 confirmed, masked 0.77) | 2026-06-10 |
| 26. SBI for Spectral DCM | v0.6.0 | 2/2 | ❌ SBC 2/9; calibration → v0.7.0 | 2026-05-31 |
| 27. Publication Artifacts | v0.6.0 | 3/3 | ✅ Delivered (synthetic results) | 2026-05-31 |
| 28. Variational Laplace Inference Engine (retroactive) | v0.6.0 | n/a | ✅ Delivered (SPM12 spm_nlsi_GN + ForwardModel) | 2026-06-09 |
| 29. VL Validation Infrastructure & BMR Rank Functions | v0.7.0 | 5/5 | ✅ Complete (verified 6/6) | 2026-06-10 |
| 30. Recovery Matrix Sweep (M3 Cluster) | v0.7.0 | 3/3 | ✅ Complete 2026-06-11 (10/10 classified, 0 errored; 6 PASS / 4 ident-limit; task underflow fixed via rk4) | 2026-06-11 |
| 31. BMR Validation & Posterior Tempering (Exploratory) | v0.7.0 | 3/3 | ✅ Complete 2026-06-11 (VLBMR-01/02/03; 8 vl tests; tempering exploratory/PD-safe, never gates absolute ΔF) | 2026-06-11 |
| 32. SPM12 Cross-Validation (Local / MATLAB) | v0.7.0 | 3/3 | ✅ Complete 2026-06-12 (VLSPM-01/02/03; ran on M3; ranking 1.0, constant 270-nat F offset, forward-model divergence documented) | 2026-06-12 |
| 33. CMC Core Dynamics, spm_int_L Integrator & Single-Source Parity | v0.8.0 | 3/3 | ✅ Complete 2026-06-26 (CMC-01..07; 25 tests; parity ladder GREEN vs SPM12 fixtures, measured matrix_exp floor 8.6e-11; spm_int_L≡rk4-risk retired in isolation) | 2026-06-26 |
| 34. Extrinsic Coupling, Condition B & Multi-Source Evoked Integration | v0.8.0 | 0/? | Not started | -- |
| 35. Single-Dipole Lead-Field, Scalp Projection & ERPDCMForward | v0.8.0 | 0/? | Not started | -- |
| 36. ERP-DCM Pyro Model, Amortized Wiring & MMN Precision-Sweep Demo | v0.8.0 | 0/? | Not started | -- |

---
*Roadmap created: 2026-04-07*
*Last updated: 2026-06-25 -- v0.8.0 DCM for Evoked Responses (EEG/MEG ERP) added (Phases 33-36; 25 reqs across CMC/EVOK/LEAD/ERPDCM). Strictly linear critical path 33->34->35->36 (NO phase parallelism); each phase carries an explicit SPM12 forward-parity gate on frozen MATLAB fixtures (J0 <=1e-10, Q <=1e-9, traj <=1e-8, scalp ERP <=1e-7). spm_int_L exp-Euler integrator (utils/local_linearization.py) is the central new component, fixture-verified first. CMC-only, single-dipole (LFP-first), VL+amortized reuse, 5-source MMN precision-sweep demo; forward/synthetic only. Research-grounded (.planning/research/v0.8.0/).*
*Last updated: 2026-06-10 -- v0.7.0 Variational Laplace Validation added (Phases 29-32; 19 reqs across VLINFRA/VLREC/VLBMR/VLSPM/VLROBUST). Critical path 29->30->31; 32 parallel to 30. Validation-led; real-data + SBI deferred to v0.8.0+.*
*Last updated: 2026-06-10 -- v0.6.0 audited + scope-cut. All 34 plans executed; goal-backward
audit (`.planning/v0.6.0-AUDIT.md`) found real-data claims (Phase 22/24/26) undelivered →
deferred to v0.7.0 (not failed). Added Phase 28 (retroactive VL/SPM12 inference-engine
consolidation). Progress tables reconciled from stale "0/N Planned" to audited status.*
*Prior update: 2026-05-26 -- Phase 24 planned (4 plans, 3 waves): foundation model extractor infrastructure, TRIBE v2 fMRI pipeline, M/EEG foundation model pipeline (LaBraM + BrainOmni), cross-modal comparison. Phase 26 planned (2 plans, 2 waves). Phase 25 planned (4 plans, 4 waves). Phase 27 planned (3 plans, 2 waves). v0.6.0 restructured: new core is DCM as interpretability tool for neural data models (Phase 22). Phase 21 complete. Phase 20-05 needs rework.*
