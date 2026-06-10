# Research Summary: v0.7.0 Variational Laplace Validation

**Project:** Pyro-DCM
**Milestone:** v0.7.0 Variational Laplace Validation
**Domain:** Inference-engine validation -- systematic synthetic recovery, SPM12
cross-validation, BMR model comparison, and numerical robustness probing for the
Variational Laplace engine built in v0.6.0.
**Researched:** 2026-06-10
**Confidence:** HIGH

---

## Executive Summary

v0.7.0 is a validation-breadth milestone, not a new-capability milestone. The
Variational Laplace (VL) engine, BMR model selection, and all three forward models
(SpectralDCM, TaskDCM, LatentCircuit) are already implemented and pass narrow smoke
tests. The work of v0.7.0 is to prove systematically that the engine is trustworthy
across a Cartesian product of network sizes, SNR levels, and forward models before
the first real-data application. Research confirms that every necessary library
(torch 2.9.1, scipy 1.16.3, numpyro 0.19.0, h5py 3.15.1) and every infrastructure
component (benchmark runners, SPM12 subprocess bridge, cluster sbatch patterns) is
already in place. The milestone is almost entirely new glue code and test harnesses
over an existing foundation. No new dependencies are required; the only
pyproject.toml change is adding a vl pytest marker.

The single most important empirical finding that shapes all of v0.7.0 is the BMR
overconfidence failure (cluster job 55772525): VL posterior std is approximately
0.001-0.01x prior std at high SNR, making the absolute delta-F for pruning any
connection wildly negative (-116 to -182,293 nats). Absolute-threshold BMR pruning
is broken by design; the only valid operating mode is relative ranking plus
separation gap. This is a confirmed pitfall with direct cluster log evidence, not a
hypothesis. Posterior tempering is an exploratory fix requiring empirical calibration
from the Phase 30 recovery matrix before it can be trusted.

The SPM12 cross-validation requires careful handling of three confirmed
parameterisation traps: prior mean offset (a_mask/128 vs 0), hyperprior convention
(hE=8.0 vs data-dependent), and comparison space (free-parameter space, not
parameterised A). Absolute free energy values are never a valid cross-engine
comparison criterion; model ranking is the correct check. The Julia spectral-DCM
paper (Pfeffer et al. 2025, PMC12330849) is the closest precedent: it validates
posterior means and relative free energies, not Cp or absolute F, achieving O(1e-3)
mean agreement.

---

## Key Findings

### Recommended Stack

The full stack is already installed and requires no changes. scipy savemat/loadmat
plus a Python subprocess call to matlab -batch is the correct and proven MATLAB
interchange strategy; neither oct2py (Octave cannot run SPM MEX files) nor
matlab.engine (not installed, unnecessary overhead) should be added. NumPyro/JAX is
available as a secondary NUTS oracle for calibration comparison but must never be
used to rewrite PyTorch forward models in JAX (project memory constraint). All
torch.linalg primitives needed for robustness testing (cond, eigvals, matrix_exp,
cholesky, pinv) are confirmed available in torch 2.9.1.

**Core technologies:**

- scipy.io.savemat/loadmat + subprocess -batch: Python-MATLAB data bridge --
  already built in validation/, zero new work required
- torch.linalg (torch 2.9.1): Cholesky, SVD, cond, eigvals for robustness probing
  -- already used in variational_laplace.py
- numpyro 0.19.0 + JAX 0.4.31: NUTS oracle for posterior calibration comparison --
  installed; use only on Gaussian proxy model, not full DCM forward model
- benchmarks/ runner infrastructure: BenchmarkConfig, RUNNER_REGISTRY, .npz
  fixture format, compute_rmse_A, compute_coverage -- fully reusable; extend by
  adding method=vl and three new runner files
- pytest markers slow, spm (existing) + vl (new): gate expensive and
  MATLAB-dependent tests -- only pyproject.toml change needed

**What is genuinely new (minimal):**

- benchmarks/runners/spectral_vl.py, task_vl.py, latent_circuit_vl.py
- validation/run_vl_validation.py
- bmr.rank_connections() + bmr.temper_vl_posterior() in model_selection/bmr.py
- validation/compare_results.compare_free_energies()
- config.py MATLAB_PATH constant
- Two cluster sbatch scripts (spectral_vl_sweep.sh, task_vl_sweep.sh)

### Expected Features

**Must have (table stakes -- required for engine validated claim):**

- TS-1: Recovery matrix harness -- N in {2,3,5} x SNR in {2,5,10} x >=10 seeds x
  3 forward models; per-cell JSON/CSV output
- TS-2: Full metric suite per cell -- RMSE(A) off/diagonal, masked sign recovery
  (|A_true|>0.01), Pearson r, per-region R^2 mean+/-std, 95% CI coverage,
  identifiability shrinkage ratio, F-convergence flag
- TS-3: SPM12 cross-validation -- posterior mean agreement in free-parameter space
  (<10% relative error on |A_free|>0.01); model ranking agreement on >=2/3 pairs
- TS-4: BMR relative ranking harness -- separation gap > 3 nats; ranking-based,
  NOT absolute-threshold
- TS-5: Numerical robustness suite -- near-stability boundary, N=2 minimal network,
  max_iter reached, near-zero Jacobian columns, T=1000 stress test

**Should have (differentiators -- publication quality):**

- D-1: Coverage calibration curve at 50/75/90/95% nominal levels, split by
  connection presence/absence
- D-2: Identifiability shrinkage heatmap over (N, SNR)
- D-3: Free energy trajectory plots for typical and worst seeds per condition
- D-4: SPM12 marginal posterior variance diagonal comparison (factor-of-2 tolerance)

**Defer to later in v0.7.0 or v0.8.0:**

- D-5: Multi-restart analysis (oracle vs random vs default start)
- D-6: Connection density sweep (0.25/0.5/0.75) for latent-circuit model
- D-7: B-modulation strength sweep for BMR sensitivity threshold
- TS-8: Posterior tempering with calibrated schedule (exploratory; depends on D-1)

**Anti-features (explicitly forbidden):**

- Absolute delta-F threshold for BMR pruning (empirically broken, job 55772525)
- Element-wise Cp comparison as SPM cross-validation criterion (SVD-space mismatch)
- Absolute F value comparison across engines (normalisation constant mismatch)
- Full SBC with rank histograms (designed for samplers, not deterministic VL)
- Real data application before validation matrix is complete
- Single aggregate RMSE score across all forward models (Simpson paradox)

### Architecture Approach

v0.7.0 is an extension architecture, not a redesign. The VL engine
(variational_laplace.py), all three forward models (forward_models.py), and the
BMR module (bmr.py) are frozen. New work attaches at the benchmarks and validation
layers. The benchmark runner registry gains three VL runners following the existing
(BenchmarkConfig) -> dict signature. The SPM12 validation path gains a parallel
run_vl_validation.py reusing the existing subprocess bridge and mat export unchanged.
BMR gains rank_connections and temper_vl_posterior as additive functions that do not
modify existing interfaces.

**Major components:**

1. benchmarks/runners/ (spectral_vl, task_vl, latent_circuit_vl) -- VL recovery
   sweep drivers; registered in RUNNER_REGISTRY; run on M3 via new sbatch scripts
2. validation/run_vl_validation.py -- SPM12 cross-validation pipeline; reuses all
   existing mat export and subprocess infrastructure; runs locally (MATLAB required)
3. model_selection/bmr.py additions -- rank_connections() and temper_vl_posterior()
   for the corrected BMR harness
4. Test files: test_vl_robustness.py, test_vl_recovery_matrix.py, test_vl_vs_spm.py
   -- covering the three validation axes

**Mutagen pre-flight requirement:** The unanchored models/ Mutagen ignore silently
excludes src/pyro_dcm/models/ from M3 sync. The latent-circuit VL runner imports
LC_A_PRIOR_VARIANCE from that package. Verify M3 sync before submitting
latent-circuit cluster jobs.

### Critical Pitfalls

1. **Laplace overconfidence destroys absolute BMR thresholds (C1)** -- Posterior
   std is ~0.001-0.01x prior std at high SNR; all reduced-model delta-F values lie
   in the -100 to -200,000 nat range. Use relative ranking + separation gap
   exclusively; never assert delta-F > fixed threshold as a pass criterion.

2. **SPM12 parameterisation mismatch (S1+S2)** -- Comparing posterior A in the
   wrong space inflates self-connection errors 2x. Hyperprior mismatch biases F by
   10-50 nats. Always compare in free-parameter space; set hyperprior_mean=8.0,
   hyperprior_precision=128.0, prior_mean_a_offset=a_mask/128.

3. **Absolute F comparison across engines is invalid (S3)** -- Expected gap is
   50-200 nats even when posteriors agree. Model ranking only; require >=3 nats
   separation before counting a ranking disagreement.

4. **Recovery metric artifacts (R1+R2)** -- Pooled R^2 hides per-region failures;
   unmasked sign recovery on structural zeros is meaningless. Use per-region R^2 as
   mean+/-std; masked sign recovery with |A_true|>0.01.

5. **eig_clamp non-injectivity near stability boundary (N2)** -- A matrices with
   max real eigenvalue in [-0.05, 0] produce degenerate Jacobians; poor recovery
   looks like an engine bug. Exclude A_true with max eig > -0.1 from the recovery
   matrix; document this restriction explicitly.

---

## Implications for Roadmap

Phase numbering continues from 28; v0.7.0 starts at Phase 29.

### Phase 29: Infrastructure + BMR Rank Functions

**Rationale:** Dependency root. Every downstream phase requires either the VL
benchmark runner plumbing or the corrected BMR ranking API. Fast laptop-runnable
tasks; no cluster requirement.

**Delivers:** benchmarks/config.py VL fields (max_iter, hyperprior_mean,
hyperprior_precision, prior_mean_a_offset); RUNNER_REGISTRY with three VL runners
(smoke-tested at N=2, 1 seed); bmr.rank_connections() + bmr.temper_vl_posterior();
config.py MATLAB_PATH constant; new vl pytest marker; unit tests for rank_connections.

**Addresses:** TS-1 (harness plumbing), TS-4 (BMR ranking API)

**Avoids:** C1 (relative ranking built from the start); N1 (dt/T constraints
documented in BenchmarkConfig docstring)

**Research flag:** No deeper research needed -- all APIs confirmed from codebase
inspection. Standard extension patterns.

---

### Phase 30: Recovery Matrix Sweep (Cluster)

**Rationale:** Computational core of v0.7.0. All diagnostic analyses consume this
output. Must land before Phase 31 BMR tempering calibration. Per-seed wall times
(spectral N=3: 5-10 min; N=5: 15-30 min; task N=3: 20-40 min) require M3.

**Delivers:** Per-cell JSON results across N in {2,3,5} x SNR in {2,5,10} x 3
forward models x 10 seeds; aggregated CSV/heatmap; full TS-2 metric suite per cell;
D-1 coverage calibration curves; D-2 shrinkage heatmap; D-3 F trajectory plots.

**Addresses:** TS-1, TS-2, TS-5 (numerical edge cases), D-1, D-2, D-3

**Avoids:** R1 (per-cell matrix only); R2 (metric suite enforces masked sign
recovery and per-region R^2); R3 (>=10 seeds, fixed schedule 0-9); R4 (exclude
near-boundary A matrices; stratify by eigenvalue regime); R5 (max_iter=256;
report iteration at convergence)

**Cluster routing:** All multi-seed sweeps route to M3. Laptop for smoke tests
only. Mutagen pre-flight check required before latent-circuit jobs.

**Research flag:** Fisher-information-derived per-cell RMSE acceptance thresholds
(PITFALLS.md gap 4) are principled but require analytical work. If deferred, use
RMSE < 0.05 universally and audit outliers manually.

---

### Phase 31: BMR Validation + Posterior Tempering (Exploratory)

**Rationale:** Depends on Phase 30 coverage results for tempering calibration.
Primary deliverable (relative ranking, separation gap) is the known-working mode
from v0.6.0. Tempering is exploratory only.

**Delivers:** BMR separation-gap validation harness across recovery matrix
conditions; D-7 B-modulation strength sweep; exploratory tempered-BMR results using
temperature calibrated from Phase 30 D-1 coverage curves (if pursued).

**Addresses:** TS-4 (full BMR ranking validation), D-7

**Avoids:** C1 (relative ranking is primary; absolute threshold never pass/fail);
C2 (tempering validated on held-out SNR conditions; Cholesky post-tempering asserted)

**Research flag:** Temperature schedule has no published DCM precedent (PITFALLS.md
gap 2). Treat tempering as exploratory; report tempered and untempered rankings
side by side.

---

### Phase 32: SPM12 Cross-Validation

**Rationale:** Runs locally (requires MATLAB). Can begin in parallel with Phase 30
cluster jobs. Pipeline already built in validation/; new work is run_vl_validation.py
plus prior-matching parameter alignment.

**Delivers:** VL vs SPM12 spm_nlsi_GN comparison on spectral N=3 SNR=5, task N=3
SNR=5, latent-circuit N=3 SNR=10; posterior mean agreement in free-parameter space
(<10% relative error); model ranking agreement on >=3 pairs; optional D-4 marginal
Cp diagonal comparison.

**Addresses:** TS-3, D-4

**Avoids:** S1 (compare in free-parameter space); S2 (SPM-matched hyperpriors:
hE=8.0, hC=1/128, prior_mean_a_offset=a_mask/128); S3 (model ranking only, never
absolute F); S4 (round-trip test for matrix layout before any comparison); S5
(apply SPM BOLD scaling to exported data); S6 (upsample stimulus to microtime grid
with 32-sample prepend); N3 (explicit float64 cast on all exported arrays)

**Research flag:** CSD computation path (PITFALLS.md gap 3): export BOLD to SPM vs
export Python CSD directly. Recommend exporting BOLD (apples-to-apples with
spm_dcm_fmri_csd internal pipeline); decide before writing run_vl_validation.py.

---

### Phase Ordering Rationale

- Phase 29 before Phase 30: Runner infrastructure and BMR API must exist before any
  cluster sweep can be submitted.
- Phase 30 before Phase 31: Tempering calibration requires coverage curves from
  Phase 30; BMR separation-gap validation requires Phase 30 posteriors.
- Phase 32 parallel to Phase 30: SPM12 pipeline is laptop-only and independent of
  the cluster sweep.
- No real data until Phase 33+: Explicitly deferred. Phases 29-32 validate the
  engine on synthetic ground truth before any real-data application.

### Research Flags

Phases likely needing deeper research during planning:

- **Phase 30 (Recovery Matrix):** Optimal per-cell RMSE acceptance thresholds
  (PITFALLS.md gap 4). Fisher-information bounds are principled but require
  analytical work. Decision needed before writing recovery assertion logic.
- **Phase 31 (Tempering):** Temperature schedule has no DCM precedent (PITFALLS.md
  gap 2). A calibration sweep on held-out conditions is required before Phase 31
  can report tempered results as primary.
- **Phase 32 (SPM12):** CSD computation path (PITFALLS.md gap 3): export BOLD vs
  export Python CSD. Must be decided before run_vl_validation.py is written.

Phases with standard patterns (skip research-phase):

- **Phase 29 (Infrastructure):** All APIs confirmed from codebase inspection.
  BenchmarkConfig extension, RUNNER_REGISTRY, BMR API additions are well-documented
  internal patterns.
- **Phase 32 SPM bridge:** Existing validation/ code confirmed working; prior-
  matching parameters (hE=8, hC=1/128, a_mask/128) already in VL API (commit e1934e1).

---

## Confidence Assessment

| Area         | Confidence | Notes |
|--------------|------------|-------|
| Stack        | HIGH       | All packages version-verified; interchange code confirmed from source; no new dependencies |
| Features     | HIGH       | Table stakes grounded in codebase audit + cluster log (job 55772525) + published validation literature |
| Architecture | HIGH       | All component relationships verified from source code; build order from actual import dependencies |
| Pitfalls     | HIGH       | C1 has direct cluster log evidence; S1-S6 confirmed from SPM12 source; N1-N5 from codebase and commits |

**Overall confidence: HIGH**

### Gaps to Address

1. **Per-cell RMSE acceptance thresholds (gap 4):** No principled threshold exists.
   During Phase 30 planning, either derive Fisher-information bounds per (N, SNR,
   forward model), or use RMSE < 0.05 universally and flag per-cell outliers.

2. **Posterior tempering temperature schedule (gap 2):** No validated schedule for
   DCM. Phase 31 must treat tempering as exploratory until Phase 30 D-1 coverage
   curves provide empirical calibration data.

3. **SPM12 CSD computation path (gap 3):** Must be decided before writing
   run_vl_validation.py. Recommendation: export BOLD (cleanest, apples-to-apples
   with spm_dcm_fmri_csd internal pipeline).

4. **eig_clamp regime (gap 1):** Two separate tables needed -- one with eig_clamp
   disabled (pure engine test) and one with eig_clamp enabled (SPM-compatible).
   Decide primary reporting table in Phase 30 planning.

---

## Sources

### Primary (HIGH confidence)

- src/pyro_dcm/inference/variational_laplace.py -- VL engine as implemented
- src/pyro_dcm/model_selection/bmr.py -- BMR as implemented
- benchmarks/config.py, benchmarks/runners/__init__.py, benchmarks/fixtures.py,
  benchmarks/metrics.py -- benchmark runner infrastructure
- validation/run_validation.py, validation/export_to_mat.py,
  validation/compare_results.py, validation/matlab_scripts/*.m -- SPM12 bridge
- cluster/logs/bmr_vs_elbo_55772525.out -- empirical BMR overconfidence evidence
- .planning/milestones/v0.6.0-MILESTONE-AUDIT.md -- v0.6.0 post-hoc findings
- .planning/todos/pending/2026-06-10-mutagen-models-ignore.md -- Mutagen sync bug
- .planning/todos/pending/2026-06-09-vl-overconfidence-for-bmr.md -- BMR open todo
- SPM12 local source (R2022a): spm_nlsi_GN.m, spm_dcm_estimate.m,
  spm_dcm_fmri_csd.m -- prior conventions, BOLD scaling, microtime resolution
- Runtime package version verification -- all installed versions confirmed

### Secondary (MEDIUM confidence)

- Pfeffer et al. (2025). Increasing spectral DCM flexibility. PMC12330849.
  O(1e-3) mean agreement; model ranking over absolute F; Cp not compared.
- Zeidman, Friston and Parr (2024). A primer on Variational Laplace. PMC10951963.
  VL overconfidence documented.
- Frässle et al. (2015). Assessing DCM identifiability. PMC4335185.
  Dense networks + low SNR implies non-identifiability; SNR sweep methodology.
- Friston and Penny (2011). Post hoc Bayesian model selection. REF-070.
  BMR equations; 3-nats Jeffreys threshold.

### Tertiary (LOW confidence)

- Cold-posterior / temperature-scaling literature: tempering as fix for Laplace
  overconfidence -- no validated DCM schedule; use only after empirical calibration.
- General Bayesian calibration: coverage at multiple nominal levels as standard
  check for deterministic posterior approximations.

---
*Research completed: 2026-06-10*
*Ready for roadmap: yes*
