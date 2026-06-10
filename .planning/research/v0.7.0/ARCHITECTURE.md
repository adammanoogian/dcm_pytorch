# Architecture: v0.7.0 VL Validation Harness Integration

**Project:** Pyro-DCM
**Milestone:** v0.7.0 "Variational Laplace Validation"
**Researched:** 2026-06-10
**Scope:** Integration-architecture only (NOT new-project survey).

---

## (a) Recovery Matrix Sweep: benchmarks/ extension vs new validation/ harness

**Recommendation: Extend `benchmarks/` — but with a dedicated VL runner family, not SVI runner modifications.**

The v0.2.0 `.npz` fixture infrastructure (`benchmarks/fixtures.py`, `load_fixture`, `get_fixture_count`), the `BenchmarkConfig` dataclass, the `RUNNER_REGISTRY` pattern, and `benchmarks/metrics.py` are all cleanly model-agnostic. They encode the outer loop (load fixture → run inference → compute metrics) without any SVI-specific coupling. The existing VL cluster script `cluster/scripts/lc_vl_acceptance_run.py` already demonstrates VL fitting in exactly the same "load ground truth → fit → measure RMSE/coverage" pattern that the benchmark runners use — it just bypasses the `BenchmarkConfig`/`RUNNER_REGISTRY` plumbing.

The critical distinction from SVI runners: VL returns a full posterior covariance (`sigma_post`, shape `(np_full, np_full)`), which changes how coverage is computed. The existing `compute_coverage_from_samples` (sample-based) and `compute_coverage_from_ci` (CI-bounds) in `benchmarks/metrics.py` are both compatible because `extract_vl_posterior_generic` already draws Cholesky samples from `sigma_post`. No changes needed in `metrics.py`.

**What extends `benchmarks/`:**

- `benchmarks/runners/spectral_vl.py` — new file. Wraps `SpectralDCMForward` + `run_variational_laplace_generic` in the standard `(BenchmarkConfig) -> dict` runner signature. Loads `.npz` fixtures from `benchmarks/fixtures/spectral_{N}region/`. Computes A-RMSE, coverage (Cholesky samples), and convergence flag.
- `benchmarks/runners/task_vl.py` — new file. Wraps `TaskDCMForward`. Requires `BenchmarkConfig` to carry `stimulus_fn` and `c_mask` or a fixture-derived equivalent (see integration point below).
- `benchmarks/runners/latent_circuit_vl.py` — new file. Thin wrapper over `lc_vl_acceptance_run.py` logic, made `BenchmarkConfig`-compatible. The ground-truth builder `_build_ground_truth` lives in `benchmarks/runners/latent_circuit_recovery.py` and is already importable (the cluster script imports it today).

**What does NOT extend `benchmarks/` (stays in `validation/`):**

The SPM12 cross-validation loop (question b) is `validation/` territory. It requires a live MATLAB process, environment-specific paths, and must remain runnable independently of the benchmark dashboard. The existing `validation/run_validation.py` + `validation/export_to_mat.py` + `validation/compare_results.py` + `validation/matlab_scripts/*.m` already form a coherent SPM12 bridge. The v0.7.0 work updates and extends these, not replaces them.

**RUNNER_REGISTRY additions:**

```python
("spectral", "vl"):  run_spectral_vl,
("task", "vl"):      run_task_vl,
("latent_circuit", "vl"):  run_latent_circuit_vl,   # already has ("latent_circuit", "svi")
```

**BenchmarkConfig additions required:**

`BenchmarkConfig` currently lacks `inference_method` as a first-class field (it routes by `method` string already). The `method="vl"` string maps cleanly. However, VL runners need additional configuration that SVI runners do not: `max_iter` (VL Gauss-Newton iterations, default 128), `hyperprior_mean`, `hyperprior_precision`, and `prior_mean_a_offset`. These should be added as optional fields with `None` defaults so existing callers are unaffected.

VL runners are NOT expected to reuse existing SVI fixtures without modification: the spectral and task `.npz` fixtures were generated for SVI and store the same data that VL needs (observed CSD, observed BOLD, ground-truth A, a_mask). The fixture format is unchanged.

**The "VL vs SVI" harness seam:**

The recovery matrix (N × SNR grid over 3 forward models) should live in `benchmarks/recovery_validation.py` (already exists as a module). The existing `recovery_validation.py` drives benchmarks via the `RUNNER_REGISTRY`; extending it to accept a `method="vl"` sweep requires adding the VL runner tuple keys and a sweep loop. This is a modification, not a new file.

---

## (b) SPM12 Cross-Validation Loop Architecture

**Build on existing `validation/` files; extend, do not rewrite.**

The existing pipeline (`validation/run_validation.py` → `subprocess.run(matlab_cmd)` → `validation/compare_results.load_spm_results`) works for task DCM (SVI inference) vs SPM12. The v0.7.0 change is to run VL on the Python side instead of SVI.

**Pipeline for VL cross-validation:**

```
validation/run_vl_validation.py  (NEW — mirrors run_validation.py)
  Step 1: generate synthetic data (same simulators as today)
  Step 2: export .mat — validation/export_to_mat.py (UNCHANGED)
  Step 3: run SPM12 spm_nlsi_GN via subprocess — existing matlab scripts (UNCHANGED)
  Step 4: run Python VL on same data:
      SpectralDCMForward + run_variational_laplace_generic (already in src/)
      TaskDCMForward + run_variational_laplace_generic
  Step 5: load SPM results — validation/compare_results.load_spm_results (UNCHANGED)
  Step 6: compare — validation/compare_results.compare_posterior_means (UNCHANGED)
  Step 7: compare free energies (NEW):
      pyro_dcm VL free_energy[-1]  vs  SPM12 DCM.F
      Both are the same quantity (VB free energy = log evidence bound under Laplace).
      Tolerance: |F_vl - F_spm| / |F_spm| < 0.05 (SPM12 and dcm_pytorch use the same
      spm_nlsi_GN algorithm; disagreement indicates a numerical divergence, not just
      inference method difference as with SVI vs SPM12).
```

**New files in `validation/`:**

- `validation/run_vl_validation.py` — analogous to `run_validation.py` but Step 4 uses VL not SVI. Calls `run_variational_laplace` / `run_variational_laplace_generic` directly. Accepts `hyperprior_mean` and `hyperprior_precision` as kwargs to match the SPM12 `spm_dcm_fmri_csd` defaults (`hE=8`, `hC=1/128`).

**Existing files modified:**

- `validation/compare_results.py` — add `compare_free_energies(vl_F, spm_F, rtol=0.05) -> dict` helper. Minimal addition.

**Existing MATLAB scripts are reused without change.** The three `validation/matlab_scripts/*.m` scripts are already correct for SPM12 CSD and task DCM. The v0.7.0 cross-validation uses the same exported `.mat` format.

**Path configuration note:** `validation/run_validation.py` hardcodes `MATLAB_PATH = "C:/Program Files/MATLAB/R2022a/bin/matlab"`. The new `run_vl_validation.py` should read this from `config.py` instead. A `MATLAB_PATH` constant should be added to the project-root `config.py`.

---

## (c) BMR Tempering and rank_connections() Location

**All BMR additions belong in `src/pyro_dcm/model_selection/bmr.py`.**

The rationale: `model_selection/bmr.py` already owns `bayesian_model_reduction`, `bmr_circuit_selection`, `enumerate_reduced_models`, and `make_reduced_prior_zero_connection`. The overconfidence problem is a property of the VL posterior passed into BMR, not of BMR itself, so the fix is a pre-processing step applied before calling `bayesian_model_reduction`.

**New public functions to add to `bmr.py`:**

`rank_connections(posterior_mean, posterior_cov, prior_mean, prior_cov, prunable_indices, shrinkage_variance) -> list[dict]`

- Computes single-connection prune cost (delta_F) for each index in `prunable_indices`.
- Returns the list sorted ascending by delta_F (most-essential first).
- Exposes the separation-gap metric: `prune_dF[k-1] / prune_dF[k]` where the gap separates the K most-essential edges from the next.
- Does NOT call `enumerate_reduced_models` (quadratic in K); runs exactly K BMR calls.

`temper_vl_posterior(sigma_post, tempering_factor) -> torch.Tensor`

- Scales `sigma_post` by `tempering_factor` (scalar > 1 to inflate). For documentation: the motivation is that VL's ReML M-step underestimates posterior covariance when observation precision is large (SNR >> 1). The calibrated factor should be determined empirically in Phase C (comparison to held-out predictive uncertainty). Default `tempering_factor=1.0` (identity, backwards-compatible).

These functions are added to `model_selection/bmr.py` and exported from `model_selection/__init__.py`. No changes to the existing four functions.

**The cluster script `cluster/scripts/lc_vl_bmr_selection.py`** will be updated to call `rank_connections()` instead of the inline loop currently on lines 119–127. This keeps the cluster script thin.

---

## (d) New vs Modified Components and Build Order

### New Files

| File | Type | Description |
|------|------|-------------|
| `benchmarks/runners/spectral_vl.py` | New | VL runner for spectral DCM; `BenchmarkConfig` compatible |
| `benchmarks/runners/task_vl.py` | New | VL runner for task DCM |
| `benchmarks/runners/latent_circuit_vl.py` | New | VL runner wrapping `lc_vl_acceptance_run` logic |
| `validation/run_vl_validation.py` | New | SPM12 cross-validation for VL (mirrors `run_validation.py`) |

### Modified Files

| File | Modification | Scope |
|------|-------------|-------|
| `benchmarks/runners/__init__.py` | Add 3 VL entries to `RUNNER_REGISTRY` | 3 lines |
| `benchmarks/config.py` | Add `max_iter`, `hyperprior_mean`, `hyperprior_precision`, `prior_mean_a_offset` optional fields | ~8 lines |
| `benchmarks/recovery_validation.py` | Add `method="vl"` sweep logic using existing registry | Additive |
| `src/pyro_dcm/model_selection/bmr.py` | Add `rank_connections()` + `temper_vl_posterior()` | ~60 lines |
| `src/pyro_dcm/model_selection/__init__.py` | Export new BMR helpers | 2 lines |
| `validation/compare_results.py` | Add `compare_free_energies()` | ~20 lines |
| `config.py` (project root) | Add `MATLAB_PATH` constant | 1 line |
| `cluster/scripts/lc_vl_bmr_selection.py` | Refactor inline BMR loop → `rank_connections()` call | ~10 lines |

### Unchanged Components (confirmed)

- `src/pyro_dcm/inference/variational_laplace.py` — the VL engine itself is not modified.
- `src/pyro_dcm/inference/forward_models.py` — `ForwardModel`, `SpectralDCMForward`, `TaskDCMForward`, `LatentCircuitForward` are stable.
- `src/pyro_dcm/inference/csd_precision.py` — unchanged.
- `validation/export_to_mat.py` — unchanged.
- `validation/compare_results.py` (existing functions) — unchanged; only additions.
- `validation/matlab_scripts/*.m` — unchanged.
- `benchmarks/fixtures.py`, `benchmarks/metrics.py`, `benchmarks/plotting.py` — unchanged.

### Build Order (dependency-respecting)

```
Phase A (infra + BMR):
  1. config.py add MATLAB_PATH
  2. benchmarks/config.py VL fields
  3. model_selection/bmr.py: rank_connections() + temper_vl_posterior()
  4. model_selection/__init__.py re-export
  5. Tests: test_bmr_rank_connections.py (new, unit; fast)

Phase B (recovery runners):
  6. benchmarks/runners/spectral_vl.py
  7. benchmarks/runners/task_vl.py
  8. benchmarks/runners/latent_circuit_vl.py (thin, reuses lc_ cluster logic)
  9. benchmarks/runners/__init__.py: 3 RUNNER_REGISTRY entries
 10. benchmarks/recovery_validation.py: vl sweep
 11. Tests: smoke tests for each VL runner (fast, N=2, 1 seed)
     → cluster: full N×SNR sweep via sbatch array

Phase C (SPM12 cross-validation):
 12. validation/compare_results.py: compare_free_energies()
 13. validation/run_vl_validation.py
 14. Tests: test_vl_spm_validation.py (skipped unless MATLAB env var set)
     → requires MATLAB+SPM12; run locally on laptop (not M3)

Phase D (BMR tempering + overconfidence):
 15. Empirically determine tempering_factor from Phase B coverage results
 16. temper_vl_posterior() default set from empirical value
 17. cluster/scripts/lc_vl_bmr_selection.py refactor → rank_connections()
 18. Tests: integration test BMR ranking recovers synthetic structure

Phase E (numerical robustness):
 19. NaN/Inf guards already in VL engine; add regression tests covering
     ill-conditioned CSD (near-singular iS), zero-variance dimensions,
     max_iter=1 edge case.
     → unit tests only, no cluster needed
```

Phases A–C can proceed in a single phase (they share no intra-phase circular dependencies); D and E are sequential after B's cluster results land.

---

## (e) Cluster Execution and Mutagen-Ignore Fix

### Mutagen Fix — Must Precede Any M3 Run Touching `src/pyro_dcm/models/`

The unanchored `models/` Mutagen ignore (tracked in `.planning/todos/pending/2026-06-10-mutagen-models-ignore.md`) silently excludes `src/pyro_dcm/models/` from M3 sync. The VL runners in Phase B (`benchmarks/runners/spectral_vl.py`, etc.) do NOT import from `src/pyro_dcm/models/`; they import only from `src/pyro_dcm/inference/` (VL engine) and `src/pyro_dcm/forward_models/`. So the Mutagen bug does NOT block Phase B VL runners on M3. However:

- The latent-circuit VL runner imports `LC_A_PRIOR_VARIANCE`, `LC_B_PRIOR_VARIANCE` from `pyro_dcm.models.latent_circuit_dcm_model`, which IS in the affected package. This means `benchmarks/runners/latent_circuit_vl.py` will fail on M3 until the Mutagen session is recreated or the per-file SCP stopgap is repeated.
- Fix strategy: the Phase B plan should include a pre-flight check: `ssh m3 python -c "from pyro_dcm.models.latent_circuit_dcm_model import LC_A_PRIOR_VARIANCE"` before submitting any latent-circuit sweep. If it fails, apply the SCP stopgap (documented in the todo) before submitting.
- The real fix (recreate Mutagen session with anchored ignores) is user-owned and should be done before Phase B cluster submission.

### Cluster Execution Pattern

Follow the established pattern from `lc_vl_acceptance_run.py`:

- Each sbatch array task = one seed.
- Results written as per-seed JSON to `cluster/results/`.
- Aggregate step (separate Python script in `cluster/scripts/`) reads all JSONs and produces summary statistics.
- sbatch script sources `cluster/lib/cluster_env.sh` for environment setup.

New sbatch scripts needed:

- `cluster/spectral_vl_sweep.sh` — SLURM array over `N_REGIONS × SNR × SEED` combinations. Passes config via environment variables (matching the `LC_VL_*` pattern already established).
- `cluster/task_vl_sweep.sh` — analogous for task DCM VL sweep.

These are thin wrappers following the same structure as `cluster/submit_phase16.sh` and the existing LC scripts.

### Timing estimates (for cluster routing decisions)

| Sweep | Per-seed estimate | Decision |
|-------|-------------------|----------|
| Spectral VL, N=3, 128 iter | ~5–10 min (128 FD Jacobians, each 9+6+9+9+1+1=N²+2N+6=35 param evals at N=3) | M3 |
| Spectral VL, N=5 | ~15–30 min (38 reduced params, 25 FD evals each) | M3 |
| Task VL, N=3 | ~20–40 min (ODE integration per FD step, 15 params) | M3 |
| Latent circuit VL | ~5–15 min (fast ODE at dt=0.1) | M3 |
| SPM12 cross-validation | ~5–10 min/seed (MATLAB subprocess) | Laptop (requires MATLAB) |
| BMR rank_connections() | <1 min | Laptop |

All multi-seed sweeps (>3 min total) route to M3 per project policy.

---

## Component Boundaries After v0.7.0

```
src/pyro_dcm/inference/          ← VL engine + ForwardModel protocol (STABLE, no changes)
  variational_laplace.py
  forward_models.py              ← ForwardModel, SpectralDCMForward, TaskDCMForward, LatentCircuitForward
  csd_precision.py

src/pyro_dcm/model_selection/    ← BMR + NEW: rank_connections, temper_vl_posterior
  bmr.py
  __init__.py

benchmarks/
  config.py                      ← MODIFIED: add VL config fields
  runners/
    spectral_vl.py               ← NEW
    task_vl.py                   ← NEW
    latent_circuit_vl.py         ← NEW
    __init__.py                  ← MODIFIED: 3 new RUNNER_REGISTRY entries
  recovery_validation.py         ← MODIFIED: vl sweep
  fixtures.py, metrics.py        ← UNCHANGED

validation/
  run_vl_validation.py           ← NEW (VL vs SPM12 cross-val)
  compare_results.py             ← MODIFIED: +compare_free_energies()
  export_to_mat.py               ← UNCHANGED
  run_validation.py              ← UNCHANGED (SVI vs SPM12, historical)
  matlab_scripts/*.m             ← UNCHANGED

cluster/
  spectral_vl_sweep.sh           ← NEW
  task_vl_sweep.sh               ← NEW
  scripts/lc_vl_bmr_selection.py ← MODIFIED: use rank_connections()
  lib/cluster_env.sh             ← UNCHANGED

config.py                        ← MODIFIED: add MATLAB_PATH constant
```

---

## Integration Points (Explicit)

| Integration Point | Existing API | How VL hooks in |
|------------------|--------------|-----------------|
| `BenchmarkConfig` | `variant`, `method` strings | `method="vl"` routes to new VL runners; add `max_iter`, `hyperprior_*` fields |
| `RUNNER_REGISTRY[(variant, method)]` | `(config) -> dict` signature | VL runners follow identical signature |
| `load_fixture(variant, n_regions, index)` | Returns `dict[str, torch.Tensor]` | VL runners call `load_fixture` identically to SVI runners |
| `run_variational_laplace_generic(forward, observed, a_mask, ...)` | Returns `VariationalLaplaceResult` | Primary entry point for all VL runners |
| `extract_vl_posterior_generic(result, forward, N)` | Returns `{param: {mean, std, samples}}` | Samples from Laplace covariance; feeds `compute_coverage_from_samples` |
| `bayesian_model_reduction(...)` | Returns `(delta_F, post_mean, post_cov)` | `rank_connections()` calls this K times |
| `validation/export_to_mat` | `export_spectral_dcm_for_spm`, `export_task_dcm_for_spm` | Called unchanged from `run_vl_validation.py` |
| `validation/compare_results` | `load_spm_results`, `compare_posterior_means` | Called unchanged; new `compare_free_energies` added |
| `cluster/lib/cluster_env.sh` | `activate_env`, `verify_torch` | New cluster sbatch scripts source this unchanged |

---

## Sources

- Verified by direct inspection of: `src/pyro_dcm/inference/variational_laplace.py`, `forward_models.py`, `csd_precision.py`, `src/pyro_dcm/model_selection/bmr.py`, `benchmarks/config.py`, `benchmarks/runners/__init__.py`, `benchmarks/fixtures.py`, `benchmarks/metrics.py`, `validation/run_validation.py`, `validation/export_to_mat.py`, `validation/compare_results.py`, `validation/matlab_scripts/run_spm_spectral_dcm.m`, `cluster/scripts/lc_vl_acceptance_run.py`, `cluster/scripts/lc_vl_bmr_selection.py`, `cluster/lib/cluster_env.sh`, `.planning/todos/pending/2026-06-10-mutagen-models-ignore.md`, `.planning/todos/pending/2026-06-09-vl-overconfidence-for-bmr.md`, `.planning/v0.7.0-VL-RECONCILIATION-DRAFT.md`, `ROADMAP.md`.
- All component relationships verified from source code, not from documentation.
