# Requirements: Pyro-DCM v0.3.0

**Defined:** 2026-04-17
**Core Value:** The A matrix (effective connectivity) remains an explicit, interpretable object with full posterior uncertainty throughout inference

## Milestone Decisions (finalized 2026-04-17)

These design decisions were resolved during milestone initialization and are baked into the
requirements below. Future re-opening requires explicit milestone revision.

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | `B_free` prior variance = **1.0** (SPM12 one-state match) | Required for future DCM.V2 cross-validation; auditably correct per `spm_dcm_fmri_priors.m`. Corrects the factually wrong YAML claim of "1/16 SPM12 convention." |
| D2 | Variable-amplitude semantics = **per-event piecewise-constant amplitudes** | Reuses existing `PiecewiseConstantInput`; matches standard SPM parametric-modulation convention. `LinearInterpolatedInput` deferred to v0.3.1 if continuous-ramp modulators are needed. |
| D3 | Recovery sign metric = **split by magnitude** | `sign_recovery_nonzero >= 80%` on `|B_true| > 0.1` AND `coverage_of_zero >= 85%` on `|B_true| < 0.5*prior_std`. Unambiguous Bayesian practice; avoids the "sign of zero" degeneracy. |
| D4 | Eigenvalue stability monitor = **strict `max Re > 0`, log-warn only** | Never raises during SVI (divergent draws are expected; hard-stop corrupts gradients). Logged for diagnostics. |
| D5 | Amortized-guide bilinear support = **deferred to v0.3.1** | `amortized_wrappers.py` and `TaskDCMPacker` remain linear-only in v0.3.0. DCM.V1 acceptance uses SVI paths. Isolates packer-versioning risk (Pitfall B3). |

## v0.3.0 Requirements

Requirements for Bilinear DCM Extension. Each maps to a roadmap phase.

### Bilinear Forward Model

- [ ] **BILIN-01**: `parameterize_B(B_free, b_mask)` utility returns a masked B matrix with safe diagonal default (diagonal = 0 unless explicitly set in `b_mask`).
- [ ] **BILIN-02**: `compute_effective_A(A, B_list, u_mod) -> A_eff` implements `A_eff = A + sum_j u_j * B_j` with documented tensor shapes `(N,N)`, `list[(N,N)]`, `(J,)` -> `(N,N)`.
- [ ] **BILIN-03**: `NeuralStateEquation.derivatives` accepts optional `B_list` and `u_mod` kwargs; when both are None, output is bit-exact equal to current linear form `A @ x + C @ u` (verified at `atol=1e-10`).
- [ ] **BILIN-04**: `CoupledDCMSystem` accepts optional `B_list` (stacked `(J,N,N)` buffer) and `input_mod_fn` (callable `t -> (J,)` modulator values); None defaults preserve exact linear behavior for all existing callers.
- [ ] **BILIN-05**: A_eff eigenvalue stability monitor logs a warning when `max(Re(eig(A_eff(t)))) > 0` at a subsample of ODE steps (default every 10 steps); never raises.
- [ ] **BILIN-06**: Worst-case stability test: bilinear ODE at `B = 3*sigma_prior`, sustained `u_mod = 1`, 500s integration, no NaN in output.
- [ ] **BILIN-07**: Docstring rename -- `NeuralStateEquation` class and `neural_state.py` module header stop calling the A+Cu form "bilinear" (it is linear); the true bilinear form is in the new branch.

### Simulator & Stimulus Utilities

- [ ] **SIM-01**: `make_event_stimulus(event_times, event_amplitudes, duration, dt) -> (T, J)` constructs variable-amplitude stick-function stimuli via piecewise-constant interpolation.
- [ ] **SIM-02**: `make_epoch_stimulus(event_times, event_durations, event_amplitudes, duration, dt) -> (T, J)` constructs boxcar-shaped modulatory inputs for sustained-amplitude regimes. Documented as preferred primitive for modulators (stick functions are blurred by rk4 mid-steps; see Pitfall B12).
- [ ] **SIM-03**: `simulate_task_dcm(..., B_list=None, stimulus_mod=None, ...)` accepts optional bilinear arguments. When `B_list=None`, output is exactly the current linear simulator output (regression test required).
- [ ] **SIM-04**: Simulator return dict gains `B_list` and `stimulus_mod` keys (set to `None` in linear mode for forward compatibility).
- [ ] **SIM-05**: `dt`-invariance test for stimulus utilities: ODE integration at `dt=0.01` and `dt=0.005` produce equivalent BOLD within `atol=1e-4` under a fixed bilinear ground truth.

### Pyro Generative Model

- [ ] **MODEL-01**: `task_dcm_model(..., b_masks=None, stim_mod=None, ...)` samples `B_free_j ~ Normal(0, 1.0)` per modulator via per-modulator loop (`pyro.sample(f"B_free_{j}", ...)`) with site-specific `b_mask` application. Rationale: matches rDCM precedent; preserves per-modulator model comparison.
- [ ] **MODEL-02**: B-prior variance is parameterized as a module-level constant `B_PRIOR_VARIANCE = 1.0` with docstring citing D1 decision; unit-tested to match documented value.
- [ ] **MODEL-03**: `b_masks[j]` default shape `(N,N)` with diagonal zeroed; explicit non-zero diagonal triggers a `DeprecationWarning` with rationale (Pitfall B5).
- [ ] **MODEL-04**: API edge cases handled: `b_masks=None` (reduces to linear), `b_masks=[]` (J=0; equivalent to None), `stim_mod` shape `(T_fine, J)` validated against `len(b_masks)`.
- [ ] **MODEL-05**: `extract_posterior_params` returns per-modulator `B_j` medians alongside existing `A`, `C`, `noise_prec`.
- [ ] **MODEL-06**: Pyro guide factory (`create_guide`) auto-discovers new `B_free_j` sample sites via `AutoGuide._setup_prototype` without factory changes; verified by trace test on `AutoNormal`, `AutoLowRankMVN`, and `AutoIAFNormal`.
- [ ] **MODEL-07**: Documentation note in `amortized_wrappers.py` and `TaskDCMPacker`: bilinear support is out of scope for v0.3.0; packer refuses bilinear sample sites with a clear error message referencing v0.3.1.

### Recovery Benchmark

- [ ] **RECOV-01**: `benchmarks/runners/task_bilinear.py` runner implements 3-region network, 1 driving input (block design), 1 modulatory input (event-related, variable amplitude), 2 non-zero B elements.
- [ ] **RECOV-02**: Benchmark integrates with v0.2.0 shared `.npz` fixture infrastructure and existing `BenchmarkConfig` / figure pipeline.
- [ ] **RECOV-03**: Acceptance criterion (A-matrix recovery): A RMSE <= 1.25 * linear-baseline RMSE (relative threshold; accounts for Bayesian parameter pricing per Pitfall B13), on >=10 seeds at SNR=3.
- [ ] **RECOV-04**: Acceptance criterion (B-matrix recovery magnitude): B RMSE <= 0.20 on `|B_true| > 0.1` elements, >=10 seeds, SNR=3.
- [ ] **RECOV-05**: Acceptance criterion (B sign recovery, non-null): sign_recovery_nonzero >= 80% on `|B_true| > 0.1` across seeds.
- [ ] **RECOV-06**: Acceptance criterion (B null coverage): coverage_of_zero >= 85% on `|B_true| < 0.5 * prior_std` across seeds.
- [ ] **RECOV-07**: Identifiability diagnostic: posterior-shrinkage metric `std_post / std_prior <= 0.7` for each free B_ij; reported alongside RMSE (does not block acceptance but documented per dataset).
- [ ] **RECOV-08**: Wall-time benchmark: bilinear DCM (3-region, J=1) runtime reported vs linear 3-region baseline (~235s/500 steps). Expected 3-6x slowdown (Pitfall B10); flagged if >10x.

## Future Requirements (deferred)

### v0.3.1 Candidates

- **AMORT-01**: `TaskBilinearDCMPacker` with packer-version tag and checkpoint compatibility assertion.
- **AMORT-02**: Amortized bilinear guide training pipeline with re-fit standardization.
- **AMORT-03**: Refusal of v0.2.0 linear-amortized warm-start with clear error message.
- **SIM-06**: `LinearInterpolatedInput` for smooth-ramp modulatory inputs (e.g., HGF belief-update trajectories).

### v0.4.0 Candidates

- **PEB-01..N**: PEB-lite group GLM on DCM parameters.
- **SPMVAL-01..N**: SPM12 cross-validation of bilinear DCM (requires MATLAB).
- **CIRCUIT-01..N**: HEART2ADAPT 4-node circuit benchmark (study-specific).
- **BILIN-08**: Two-state prior flag (variance 1/4) as alternative to D1.

## Out of Scope

Explicitly excluded from v0.3.0 (and often permanently).

| Feature | Reason |
|---------|--------|
| Nonlinear DCM (second-order terms x * x) | Anti-feature; PROJECT.md explicitly cites Nozari et al. 2024 -- bilinear suffices for macroscopic BOLD |
| Time-varying A(t) beyond modulatory form | Anti-feature; deferred per PROJECT.md Out-of-Scope list |
| Trial-by-trial Bayesian updating | Anti-feature; scope creep away from batch DCM |
| HRF-convolved stimulus pre-processing inside utilities | Would double-count hemodynamics (Balloon model already does neural->BOLD transform) |
| `pyro.plate` around B_j sampling | Breaks some AutoGuides; per-site loop is the right pattern |
| Amortized-guide bilinear support | Deferred to v0.3.1 per D5 |
| Group-level PEB analysis | Deferred to v0.4+; HEART2ADAPT-specific, not scoped to single-subject toolbox |
| SPM12 cross-validation | Deferred to v0.4+; requires MATLAB access |
| 4-node HEART2ADAPT circuit benchmark | Deferred; study-specific |
| NumPyro bilinear backend | v0.4+; multiplies scope |
| Real-time / clinical deployment | PROJECT.md permanent Out-of-Scope |
| GUI / web interface | PROJECT.md permanent Out-of-Scope |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| BILIN-01 | Phase 13 | Complete |
| BILIN-02 | Phase 13 | Complete |
| BILIN-03 | Phase 13 | Complete |
| BILIN-04 | Phase 13 | Complete |
| BILIN-05 | Phase 13 | Complete |
| BILIN-06 | Phase 13 | Complete |
| BILIN-07 | Phase 13 | Complete |
| SIM-01 | Phase 14 | Pending |
| SIM-02 | Phase 14 | Pending |
| SIM-03 | Phase 14 | Pending |
| SIM-04 | Phase 14 | Pending |
| SIM-05 | Phase 14 | Pending |
| MODEL-01 | Phase 15 | Pending |
| MODEL-02 | Phase 15 | Pending |
| MODEL-03 | Phase 15 | Pending |
| MODEL-04 | Phase 15 | Pending |
| MODEL-05 | Phase 15 | Pending |
| MODEL-06 | Phase 15 | Pending |
| MODEL-07 | Phase 15 | Pending |
| RECOV-01 | Phase 16 | Pending |
| RECOV-02 | Phase 16 | Pending |
| RECOV-03 | Phase 16 | Pending |
| RECOV-04 | Phase 16 | Pending |
| RECOV-05 | Phase 16 | Pending |
| RECOV-06 | Phase 16 | Pending |
| RECOV-07 | Phase 16 | Pending |
| RECOV-08 | Phase 16 | Pending |

**Coverage:**
- v0.3.0 requirements: 27 total
- Mapped to phases: 27/27 (all mapped)
- Unmapped: 0

**Per-phase distribution:**
- Phase 13 (Bilinear Neural State & Stability Monitor): 7 requirements (BILIN-01..07)
- Phase 14 (Stimulus Utilities & Bilinear Simulator): 5 requirements (SIM-01..05)
- Phase 15 (Pyro Generative Model): 7 requirements (MODEL-01..07)
- Phase 16 (Recovery Benchmark): 8 requirements (RECOV-01..08)

---
*Requirements defined: 2026-04-17*
*Last updated: 2026-04-17 after roadmap creation (Phases 13-16 mapped, coverage 27/27)*
