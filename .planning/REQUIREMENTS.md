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

- [x] **SIM-01**: `make_event_stimulus(event_times, event_amplitudes, duration, dt) -> (T, J)` constructs variable-amplitude stick-function stimuli via piecewise-constant interpolation.
- [x] **SIM-02**: `make_epoch_stimulus(event_times, event_durations, event_amplitudes, duration, dt) -> (T, J)` constructs boxcar-shaped modulatory inputs for sustained-amplitude regimes. Documented as preferred primitive for modulators (stick functions are blurred by rk4 mid-steps; see Pitfall B12).
- [x] **SIM-03**: `simulate_task_dcm(..., B_list=None, stimulus_mod=None, ...)` accepts optional bilinear arguments. When `B_list=None`, output is exactly the current linear simulator output (regression test required).
- [x] **SIM-04**: Simulator return dict gains `B_list` and `stimulus_mod` keys (set to `None` in linear mode for forward compatibility).
- [x] **SIM-05**: `dt`-invariance test for stimulus utilities: ODE integration at `dt=0.01` and `dt=0.005` produce equivalent BOLD within `atol=1e-4` under a fixed bilinear ground truth.

### Pyro Generative Model

- [x] **MODEL-01**: `task_dcm_model(..., b_masks=None, stim_mod=None, ...)` samples `B_free_j ~ Normal(0, 1.0)` per modulator via per-modulator loop (`pyro.sample(f"B_free_{j}", ...)`) with site-specific `b_mask` application. Rationale: matches rDCM precedent; preserves per-modulator model comparison.
- [x] **MODEL-02**: B-prior variance is parameterized as a module-level constant `B_PRIOR_VARIANCE = 1.0` with docstring citing D1 decision; unit-tested to match documented value.
- [x] **MODEL-03**: `b_masks[j]` default shape `(N,N)` with diagonal zeroed; explicit non-zero diagonal triggers a `DeprecationWarning` with rationale (Pitfall B5).
- [x] **MODEL-04**: API edge cases handled: `b_masks=None` (reduces to linear), `b_masks=[]` (J=0; equivalent to None), `stim_mod` shape `(T_fine, J)` validated against `len(b_masks)`.
- [x] **MODEL-05**: `extract_posterior_params` returns per-modulator `B_j` medians alongside existing `A`, `C`, `noise_prec`.
- [x] **MODEL-06**: Pyro guide factory (`create_guide`) auto-discovers new `B_free_j` sample sites via `AutoGuide._setup_prototype` without factory changes; verified by trace test on `AutoNormal`, `AutoLowRankMVN`, and `AutoIAFNormal`.
- [x] **MODEL-07**: Documentation note in `amortized_wrappers.py` and `TaskDCMPacker`: bilinear support is out of scope for v0.3.0; packer refuses bilinear sample sites with a clear error message referencing v0.3.1.

### Recovery Benchmark

- [ ] **RECOV-01**: `benchmarks/runners/task_bilinear.py` runner implements 3-region network, 1 driving input (block design), 1 modulatory input (event-related, variable amplitude), 2 non-zero B elements.
- [ ] **RECOV-02**: Benchmark integrates with v0.2.0 shared `.npz` fixture infrastructure and existing `BenchmarkConfig` / figure pipeline.
- [ ] **RECOV-03**: Acceptance criterion (A-matrix recovery): A RMSE <= 1.25 * linear-baseline RMSE (relative threshold; accounts for Bayesian parameter pricing per Pitfall B13), on >=10 seeds at SNR=3.
- [ ] **RECOV-04**: Acceptance criterion (B-matrix recovery magnitude): B RMSE <= 0.20 on `|B_true| > 0.1` elements, >=10 seeds, SNR=3.
- [ ] **RECOV-05**: Acceptance criterion (B sign recovery, non-null): sign_recovery_nonzero >= 80% on `|B_true| > 0.1` across seeds.
- [ ] **RECOV-06**: Acceptance criterion (B null coverage): coverage_of_zero >= 85% on `|B_true| < 0.5 * prior_std` across seeds.
- [ ] **RECOV-07**: Identifiability diagnostic: posterior-shrinkage metric `std_post / std_prior <= 0.7` for each free B_ij; reported alongside RMSE (does not block acceptance but documented per dataset).
- [ ] **RECOV-08**: Wall-time benchmark: bilinear DCM (3-region, J=1) runtime reported vs linear 3-region baseline (~235s/500 steps). Expected 3-6x slowdown (Pitfall B10); flagged if >10x.

## v0.4.0 Requirements

Requirements for Circuit Explorer (v0.4.0). Each maps to a phase in the v0.4.0 milestone.

### Circuit Visualization (v0.4.0)

- [ ] **VIZ-01**: `CircuitVizConfig` dataclass exists at `src/pyro_dcm/utils/circuit_viz.py` with the 13 handoff-spec fields (`schema`, `status`, `meta`, `palette`, `regions`, `region_colors`, `matrices`, `mat_order`, `phenotypes`, `hypotheses`, `drugs`, `peb`, `fitted_params`) PLUS a pass-through `extras: dict` field (V1 decision) and `to_dict()` / `export()` methods.
- [ ] **VIZ-02**: `CircuitViz.from_model_config(model_cfg, *, phenotypes=None, hypotheses=None, drugs=None, peb=None, palette=None)` produces a `CircuitVizConfig` with `status='planned'` and `fitted_params=None`. Requires `regions`, `region_colors` (length-matched to regions — V2 decision; raises `ValueError` otherwise), and `A_prior_mean` in `model_cfg`.
- [ ] **VIZ-03**: `CircuitViz.from_posterior(planned, posterior_means)` returns a NEW `CircuitVizConfig` (deepcopy; does not mutate `planned`) with `status='fitted'` and `fitted_params=posterior_means`. Validates `posterior_means` contains no NaN/Inf BEFORE deepcopy (V6 decision; raises `ValueError` with offending `(key, i, j)` tuple).
- [ ] **VIZ-04**: `CircuitViz.load(path)` reads `dcm_circuit_explorer/v1` JSON and reconstructs a `CircuitVizConfig`. All top-level JSON keys NOT in the 13-key first-class set are preserved in `cfg.extras` (V1; enables round-trip of `_study`, `_description`, `node_info`, `node_positions`, `svg_edges`, `b_overlays`).
- [ ] **VIZ-05**: `mat_order` is deterministic: `['A'] + sorted(B_matrices.keys()) + (['C'] if C_matrix present and non-empty else [])` (V7 decision). Never relies on caller dict insertion order.
- [ ] **VIZ-06**: `from_model_config` accepts `torch.Tensor`, `numpy.ndarray`, and `list[list[float]]` as matrix values via `_to_list_of_list` helper (V8). All emitted `matrices[key]['vals']` are `list[list[float]]` (never tensor, never ndarray).
- [ ] **VIZ-07**: Module-level helper `flatten_posterior_for_viz(posterior, mat_order, b_masks=None)` is exported (V4). Consumes `extract_posterior_params(...)` output and returns the `dict[str, list[list[float]]]` payload `from_posterior` expects. Handles A / C / B_j flattening with `B_free_{j}` + `parameterize_B` fallback when the `B` deterministic site is absent.
- [ ] **VIZ-08**: Round-trip equality: `CircuitViz.load(cfg.export(path)).to_dict() == cfg.to_dict()` for any `cfg` produced by `from_model_config` or `from_posterior`, including `configs/heart2adapt_dcm_config.json` via `extras` pass-through.
- [ ] **VIZ-09**: `pyro_dcm.utils` package re-exports `CircuitViz`, `CircuitVizConfig`, and `flatten_posterior_for_viz` (consistent with the existing `ode_integrator` re-export precedent in `src/pyro_dcm/utils/__init__.py`).
- [ ] **VIZ-10**: Zero upstream edits — no changes to `task_dcm_model`, `extract_posterior_params`, `parameterize_A`, `parameterize_B`, `create_guide`, `run_svi`, or any file outside `src/pyro_dcm/utils/`, `tests/`, and `.planning/`. Verified by `git diff --name-only main...HEAD` matching the allowed path set.

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
| SIM-01 | Phase 14 | Complete |
| SIM-02 | Phase 14 | Complete |
| SIM-03 | Phase 14 | Complete |
| SIM-04 | Phase 14 | Complete |
| SIM-05 | Phase 14 | Complete |
| MODEL-01 | Phase 15 | Complete |
| MODEL-02 | Phase 15 | Complete |
| MODEL-03 | Phase 15 | Complete |
| MODEL-04 | Phase 15 | Complete |
| MODEL-05 | Phase 15 | Complete |
| MODEL-06 | Phase 15 | Complete |
| MODEL-07 | Phase 15 | Complete |
| RECOV-01 | Phase 16 | Pending |
| RECOV-02 | Phase 16 | Pending |
| RECOV-03 | Phase 16 | Pending |
| RECOV-04 | Phase 16 | Pending |
| RECOV-05 | Phase 16 | Pending |
| RECOV-06 | Phase 16 | Pending |
| RECOV-07 | Phase 16 | Pending |
| RECOV-08 | Phase 16 | Pending |
| VIZ-01 | Phase 17 | Complete |
| VIZ-02 | Phase 17 | Complete |
| VIZ-03 | Phase 17 | Complete |
| VIZ-04 | Phase 17 | Complete |
| VIZ-05 | Phase 17 | Complete |
| VIZ-06 | Phase 17 | Complete |
| VIZ-07 | Phase 17 | Complete |
| VIZ-08 | Phase 17 | Complete |
| VIZ-09 | Phase 17 | Complete |
| VIZ-10 | Phase 17 | Complete |

**Coverage:**
- v0.3.0 requirements: 27 total
- Mapped to phases: 27/27 (all mapped)
- Unmapped: 0
- v0.4.0 requirements: 10 total
- Mapped to phases: 10/10 (all mapped)
- Unmapped: 0
- v0.5.0 requirements: 18 total
- Mapped to phases: 18/18 (all mapped)
- Unmapped: 0

**Per-phase distribution:**
- Phase 13 (Bilinear Neural State & Stability Monitor): 7 requirements (BILIN-01..07)
- Phase 14 (Stimulus Utilities & Bilinear Simulator): 5 requirements (SIM-01..05)
- Phase 15 (Pyro Generative Model): 7 requirements (MODEL-01..07)
- Phase 16 (Recovery Benchmark): 8 requirements (RECOV-01..08)
- Phase 17 (Circuit Visualization Module): 10 requirements (VIZ-01..10)
- Phase 18 (MNE/BIDS IO Test Suite): 16 requirements (TEST-01..13, BIDS-01..03)
- Phase 19 (End-to-End Pipeline Demos): 2 requirements (PIPE-01, PIPE-02)

## v0.5.0 Requirements

Requirements for MNE-Python Integration. Validates IO loaders and demonstrates
end-to-end usage with pipeline scripts. All test fixtures are synthetic -- no
data downloads. Critical scientific pitfalls (CSD frequency conventions, Hermitian
symmetry, channel picks inconsistency) are encoded as explicit test cases.

### IO Loader Tests

- [x] **TEST-01**: Shape validation for `epochs_to_csd` -- output `(F, N, N)` complex CSD tensor matches expected dimensions from synthetic Epochs
- [x] **TEST-02**: Shape validation for `epochs_to_timeseries` -- output `(T, N)` float tensor matches expected dimensions, both averaged and unaveraged paths
- [x] **TEST-03**: Shape validation for `raw_to_timeseries` -- output `(T, N)` float tensor matches expected dimensions from synthetic Raw
- [x] **TEST-04**: Shape validation for `stc_to_roi_timeseries` -- output `(T, N)` float tensor matches expected dimensions (mocked `extract_label_time_course`)
- [x] **TEST-05**: Channel picks subsetting -- loaders correctly subset channels by name list and by type string; output shape matches picks, not full channel count
- [x] **TEST-06**: Bad channel annotation handling -- documents behavior when channels marked as `info['bads']`; explicit picks required to exclude bads (pitfall P3)
- [x] **TEST-07**: CSD Hermitian symmetry -- `csd[f,i,j] == conj(csd[f,j,i])` for all frequency bins
- [x] **TEST-08**: CSD non-negative auto-spectra diagonal -- `csd[f,i,i].real >= 0` for all frequency bins and channels
- [x] **TEST-09**: CSD sine-injection round-trip -- inject 10 Hz sine into synthetic Epochs, verify CSD peak at 10 Hz bin (within 1 bin tolerance)
- [x] **TEST-10**: `_require_mne()` raises ImportError with install instructions when MNE not installed
- [x] **TEST-11**: `epochs_to_csd` raises ValueError for invalid `method` argument
- [x] **TEST-12**: `pytest.importorskip("mne")` at module level -- test file skips entirely when MNE absent
- [x] **TEST-13**: `@pytest.mark.mne` marker registered in pyproject.toml for `pytest -m "not mne"` exclusion

### BIDS Loader Tests

- [x] **BIDS-01**: `load_bids_raw` returns valid `mne.io.BaseRaw` from synthetic BIDS dataset written via `write_raw_bids` to `tmp_path`
- [x] **BIDS-02**: `load_bids_epochs` returns valid `mne.Epochs` from synthetic BIDS dataset
- [x] **BIDS-03**: BIDS annotation edge case -- handle `BAD_ACQ_SKIP` spans and non-trivial annotations without error

### Pipeline Scripts

- [x] **PIPE-01**: Spectral DCM demo script -- end-to-end: synthetic MNE Epochs -> `epochs_to_csd` -> SpectralDCMModel -> SVI -> posterior A matrix
- [x] **PIPE-02**: Task DCM demo script -- end-to-end: synthetic MNE Epochs -> `epochs_to_timeseries` -> TaskDCMModel -> SVI -> posterior A + B matrices

## v0.5.0 Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| TEST-01 | Phase 18 | Complete |
| TEST-02 | Phase 18 | Complete |
| TEST-03 | Phase 18 | Complete |
| TEST-04 | Phase 18 | Complete |
| TEST-05 | Phase 18 | Complete |
| TEST-06 | Phase 18 | Complete |
| TEST-07 | Phase 18 | Complete |
| TEST-08 | Phase 18 | Complete |
| TEST-09 | Phase 18 | Complete |
| TEST-10 | Phase 18 | Complete |
| TEST-11 | Phase 18 | Complete |
| TEST-12 | Phase 18 | Complete |
| TEST-13 | Phase 18 | Complete |
| BIDS-01 | Phase 18 | Complete |
| BIDS-02 | Phase 18 | Complete |
| BIDS-03 | Phase 18 | Complete |
| PIPE-01 | Phase 19 | Complete |
| PIPE-02 | Phase 19 | Complete |

## v0.7.0 Requirements

Requirements for **Variational Laplace Validation** (VL-validation-led; no real data; no SBI).
Prove the VL engine (SPM12 `spm_nlsi_GN` port, shipped v0.6.0) works completely on synthetic /
known ground truth and agrees with MATLAB SPM12, before any real-data application. Phase
numbering continues from 28; v0.7.0 phases are 29+. Mapped to phases during roadmap creation.

### Infrastructure & BMR Helpers (VLINFRA)

- [ ] **VLINFRA-01**: `BenchmarkConfig` gains optional VL fields (`max_iter`, `hyperprior_mean`, `hyperprior_precision`, `prior_mean_a_offset`) with `None` defaults; zero behavior change for existing callers.
- [ ] **VLINFRA-02**: Three VL benchmark runners (`spectral_vl`, `task_vl`, `latent_circuit_vl`) registered via `method="vl"` in `RUNNER_REGISTRY`, reusing the v0.2.0 `.npz` fixture + metrics infrastructure.
- [ ] **VLINFRA-03**: `rank_connections()` in `model_selection/bmr.py` runs K single-connection BMR calls and returns connections ranked by prune cost, with a separation-gap statistic.
- [ ] **VLINFRA-04**: `temper_vl_posterior()` in `model_selection/bmr.py` scales the VL posterior covariance by a calibrated temperature, with a positive-definiteness guard.
- [ ] **VLINFRA-05**: `vl` pytest marker registered in `pyproject.toml`; `MATLAB_PATH` centralized in `config.py`.

### Recovery Matrix (VLREC)

- [x] **VLREC-01**: Recovery-matrix sweep over N × SNR × {spectral, task, latent-circuit}, ≥10 seeds per cell, executed on the M3 cluster.
- [x] **VLREC-02**: Per-cell metrics use **per-region R² (not variance-pooled)**, **masked** sign recovery (`|true| > threshold`), CI coverage, RMSE, and identifiability shrinkage `std_post/std_prior` — hardened against the known metric artifacts (`sign(0)`, pooled R²).
- [x] **VLREC-03**: Recovery design excludes near-stability-boundary A matrices (max Re eig ∈ [−0.05, 0]) to avoid the `eig_clamp` non-injectivity confound; task DCM enforces dt ≥ 0.1.
- [x] **VLREC-04**: Recovery passes documented per-cell thresholds across the matrix, OR identifiability limits are documented with evidence (no silent failures).
- [ ] **VLREC-05**: C-order CSD round-trip regression test added to the suite (guards the column-major↔row-major / complex-CSD indexing class of bug) before any SPM cross-validation run.

### BMR Validation (VLBMR)

- [x] **VLBMR-01**: BMR **relative-evidence ranking** recovers the true circuit structure (top-K essential edges = true edges) on synthetic ground truth, with separation gap reported — the primary, defensible model-comparison result.
- [x] **VLBMR-02**: BMR-on-VL agreement with brute-force ELBO model comparison on a small model set, validating the analytic approximation.
- [x] **VLBMR-03**: *(exploratory)* Posterior tempering restores a usable absolute-ΔF regime, calibrated against VLREC coverage output; PD-safe; documented as exploratory, not a headline claim.

### SPM12 Cross-Validation (VLSPM)

- [ ] **VLSPM-01**: VL output cross-validated vs MATLAB `spm_nlsi_GN` on ≥1 spectral DCM problem, **prior-matched** (`hE=8.0`, `prior_mean_a_offset = a_mask/128`, comparison in free-parameter space).
- [ ] **VLSPM-02**: Compare `Ep` within ~10% relative tolerance + model-ranking agreement, and free energy (VL `free_energy` ≡ SPM `DCM.F`, ~5%); **never** element-wise `Cp` nor absolute-F across models.
- [ ] **VLSPM-03**: Reuse the existing `validation/` SPM pipeline (`export_to_mat`, MATLAB batch, `compare_results`) + new `run_vl_validation.py` + `compare_free_energies()`.

### Numerical Robustness (VLROBUST)

- [ ] **VLROBUST-01**: VL convergence + multi-restart determinism regression tests across the three forward models (fixed seed → stable result; non-determinism sources documented).
- [ ] **VLROBUST-02**: Precision-matrix intractability guard — task DCM VL fails loud (expected vs actual matrix size in the message) when dt × duration exceeds a tractable T; the dt ≥ 0.1 floor is documented.
- [ ] **VLROBUST-03**: Stability-boundary / `eig_clamp` behavior characterized — recovery degradation near the boundary documented; the non-injective regime flagged.

## v0.7.0 Out of Scope

Explicit anti-features for v0.7.0 (documented to prevent scope creep).

| Feature | Reason |
|---------|--------|
| Absolute-ΔF BMR pruning as a primary result | Structurally broken by Laplace overconfidence (job 55772525); relative ranking is the valid mode |
| Element-wise `Cp` comparison / absolute-F-across-models vs SPM12 | Not comparable across implementations; compare `Ep` + ranking + matched-problem F only |
| Standalone SBC / calibration dimension | Light CI-coverage rides inside VLREC-02; full SBC deferred |
| Real-data application (Cam-CAN M/EEG, foundation models) | Gated on VL being proven first → v0.8.0+ |
| SBI / SBC structural calibration fix | Separate (uncalibrated) inference path, not VL → later milestone |
| Rewriting PyTorch forward models in JAX | NumPyro NUTS only as a Gaussian-proxy secondary oracle, per the project NumPyro strategy |

## v0.7.0 Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| VLINFRA-01 | Phase 29 | Complete |
| VLINFRA-02 | Phase 29 | Complete |
| VLINFRA-03 | Phase 29 | Complete |
| VLINFRA-04 | Phase 29 | Complete |
| VLINFRA-05 | Phase 29 | Complete |
| VLREC-05 | Phase 29 | Complete |
| VLROBUST-01 | Phase 29 | Complete |
| VLROBUST-02 | Phase 29 | Complete |
| VLREC-01 | Phase 30 | Complete |
| VLREC-02 | Phase 30 | Complete |
| VLREC-03 | Phase 30 | Complete |
| VLREC-04 | Phase 30 | Complete (10/10 cells classified, 0 errored; task underflow fixed via rk4, job 56372816) |
| VLROBUST-03 | Phase 30 | Complete |
| VLBMR-01 | Phase 31 | Complete (spectral N=2/N=4 recover true structure 5/5 seeds, positive separation gap) |
| VLBMR-02 | Phase 31 | Complete (BMR vs brute-force VL refit agree on rank + worst model; ρ=1.0) |
| VLBMR-03 | Phase 31 | Complete (exploratory; T=2.0 restores task-N4 coverage 0.875→1.0, held-out C2c non-PD surfaced) |
| VLSPM-01 | Phase 32 | Pending |
| VLSPM-02 | Phase 32 | Pending |
| VLSPM-03 | Phase 32 | Pending |

**Coverage:** 19 v0.7.0 requirement IDs (VLINFRA-01..05, VLREC-01..05, VLBMR-01..03,
VLSPM-01..03, VLROBUST-01..03); 19/19 mapped to exactly one phase; 0 unmapped, 0 duplicated.

**Per-phase distribution:**
- Phase 29 (VL Validation Infrastructure & BMR Rank Functions): 8 reqs (VLINFRA-01..05,
  VLREC-05, VLROBUST-01, VLROBUST-02)
- Phase 30 (Recovery Matrix Sweep, M3 cluster): 5 reqs (VLREC-01..04, VLROBUST-03)
- Phase 31 (BMR Validation & Posterior Tempering): 3 reqs (VLBMR-01..03)
- Phase 32 (SPM12 Cross-Validation, local/MATLAB): 3 reqs (VLSPM-01..03)

---
*Requirements defined: 2026-04-17*
*Last updated: 2026-06-10 — v0.7.0 Variational Laplace Validation requirements added (17 across
VLINFRA/VLREC/VLBMR/VLSPM/VLROBUST). VL-validation-led; real-data + SBI deferred. v0.3.0 RECOV
(Phase 16.1) remains pending; v0.4.0/v0.5.0 complete.*
*Last updated: 2026-06-10 — v0.7.0 traceability mapped to Phases 29-32 by roadmapper (19 IDs; 100% coverage). 29=infra+BMR helpers+robustness guards, 30=recovery sweep (M3), 31=BMR validation+tempering, 32=SPM12 cross-validation (local).*
