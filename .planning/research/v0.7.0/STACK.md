# Stack Research: v0.7.0 VL-Validation Harness

**Domain:** Variational Laplace validation for DCM — synthetic recovery, SPM12
cross-validation, BMR model comparison, numerical robustness probing
**Researched:** 2026-06-10
**Confidence:** HIGH (all claims verified against installed packages or existing
source code)

---

## Installed Package Versions (Verified 2026-06-10)

These were read directly from the running environment — treat as ground truth.

| Package | Installed Version | Notes |
|---------|------------------|-------|
| torch | 2.9.1+cpu | PyTorch with autograd and `torch.linalg` full suite |
| pyro-ppl | 1.9.1 | SVI, AutoNormal, ELBO variants |
| numpyro | 0.19.0 | NUTS oracle; `numpyro.diagnostics` (hpdi, gelman_rubin, print_summary) available |
| jax | 0.4.31 | JAX backend for NumPyro |
| scipy | 1.16.3 | `scipy.io.savemat/loadmat`, `scipy.linalg.expm`, signal processing |
| numpy | 2.2.6 | Array ops; `.npz` fixture I/O |
| torchdiffeq | 0.2.5 | ODE integration for task/latent-circuit forward models |
| zuko | 1.6.0 | Normalizing flows (not used in v0.7.0) |
| h5py | 3.15.1 | MAT-v7.3 read-back if SPM writes large covariance files |
| matplotlib | 3.10.7 | Diagnostic plots, free-energy traces |
| tabulate | 0.10.0 | Recovery matrix tables |
| pytest | 9.0.1 | Test runner; markers `slow`, `spm`, `tapas` already registered |
| sbi | 0.26.1 | Present but OUT OF SCOPE for v0.7.0 |

**Not installed:**

| Package | Status | Decision |
|---------|--------|---------|
| oct2py | Not installed | Do NOT install — see rationale below |
| matlab.engine (MATLAB Engine API for Python) | Not installed | Do NOT install — see rationale below |

---

## Recommended Stack

### Core Technologies — all REUSED from existing foundation

| Technology | Version | Purpose | Status |
|------------|---------|---------|--------|
| `scipy.io.savemat/loadmat` | scipy 1.16.3 | Python ↔ MATLAB `.mat` interchange | ALREADY BUILT in `validation/export_to_mat.py` and `compare_results.py` |
| `subprocess` (stdlib) | Python 3.10+ | Invoke MATLAB via `-batch` flag | ALREADY BUILT in `validation/run_validation.py` |
| MATLAB batch scripts | SPM12 local install | Run `spm_nlsi_GN` / `spm_dcm_fmri_csd` | ALREADY BUILT in `validation/matlab_scripts/` |
| `torch.linalg` | torch 2.9.1 | Cholesky, SVD, eigvals, cond, matrix_exp | ALREADY PRESENT — used extensively in `variational_laplace.py` |
| NumPyro NUTS | numpyro 0.19.0 | Secondary reference oracle; posterior R-hat, hpdi | ALREADY INSTALLED — confirm existing `sbi_spectral.py` JAX setup as template |
| `benchmarks/fixtures.py` + `.npz` files | numpy 2.2.6 | Sweep fixture infrastructure | ALREADY BUILT — `load_fixture`, `BenchmarkConfig`, `generate_fixtures.py` |
| pytest `@pytest.mark.slow/spm` | pytest 9.0.1 | Mark and skip expensive / MATLAB-dependent tests | ALREADY REGISTERED in `pyproject.toml` |
| `h5py` | 3.15.1 | Read MAT-v7.3 files (SPM may produce these for large `Cp`) | ALREADY INSTALLED — no new code needed, fallback only |

### New Capabilities Needed for v0.7.0

Only four genuinely new pieces of plumbing are required.

#### 1. VL-specific validation runner in `validation/`

**What:** A `run_vl_validation.py` parallel to the existing `run_validation.py` that compares
`run_variational_laplace` (Python VL) against MATLAB `spm_nlsi_GN` output. Distinct from
the existing SVI-vs-SPM runners because it compares VL↔VL, not SVI↔VL.

**Reuses:**
- `export_to_mat.py` (unchanged) for `.mat` serialisation
- `compare_results.py` `load_spm_results` / `compare_posterior_means` (unchanged)
- MATLAB scripts `run_spm_spectral_dcm.m` and `run_spm_task_dcm.m` (may need `Cp` extraction
  added for covariance comparison — small diff)
- `subprocess` + `check_matlab_available()` pattern (copy exactly)

**New content:** `compare_free_energy(python_F, spm_F, rtol=0.01)` and
`compare_posterior_covariance(python_Cp, spm_Cp, rtol=0.10)` functions in `compare_results.py`.
The MATLAB scripts need one small addition: save `DCM.Cp` (full posterior covariance) to
`results.mat`. Currently `run_spm_spectral_dcm.m` already does `results.Cp = full(DCM.Cp)`.
Task DCM script (`run_spm_task_dcm.m`) — verify same.

#### 2. Recovery matrix sweep runner

**What:** A `benchmarks/runners/vl_recovery.py` following the existing `spectral_svi.py` /
`task_svi.py` runner pattern. Iterates over N ∈ {3, 5} × SNR ∈ {2, 5, 10, 20} for
spectral/task/latent-circuit forward models, stores results as `.csv`/`.json` in
`benchmarks/results/`.

**Reuses:**
- `BenchmarkConfig` dataclass (unchanged — add `method="vl"` string key)
- `load_fixture` / `generate_fixtures.py` (unchanged — same `.npz` fixture format)
- `benchmarks/metrics.py` `compute_rmse_A`, `compute_coverage` (unchanged)
- `VariationalLaplaceResult` from `variational_laplace.py` (already has `sigma_post` for coverage)

**New content:** `run_vl_recovery_sweep(config)` function analogous to `run_spectral_svi_benchmark`.
~100 lines. No new library dependencies.

#### 3. Posterior tempering utility

**What:** A small utility function `temper_posterior(sigma_post, alpha)` that inflates
the Laplace posterior covariance by scalar `alpha > 1` to correct Laplace overconfidence
before passing to BMR. Lives in `src/pyro_dcm/inference/variational_laplace.py` or a new
`src/pyro_dcm/inference/posterior_tempering.py`.

**Reuses:** Pure `torch.Tensor` ops. No new libraries.

#### 4. Numerical stress-test fixtures

**What:** A `tests/test_vl_robustness.py` that probes: ill-conditioned precision matrices
(near-singular CSD at stability boundary), dt edge cases, non-convergence detection.

**Reuses:** `torch.linalg.cond` (already used in `variational_laplace.py`), `torch.linalg.eigvals`,
existing `@pytest.mark.slow` decorator.

---

## Python ↔ MATLAB Interchange: Definitive Approach

### Method: `scipy.io.savemat/loadmat` + `subprocess` (EXISTING, keep exactly)

This is already implemented and working. Do not change it.

**Confirmed conventions (from `export_to_mat.py`):**

- All scalars: `np.array([[value]])` (2-D, MATLAB row-vector convention)
- String fields: `np.array([['text']], dtype=object)`
- Empty 3-D fields: `np.zeros((N, N, 0))` for unused B/D modulatory matrices
- Stimulus: TR/16 microtime resolution with 32-row zero padding
- `savemat` uses default MAT-v5 format (sufficient for all DCM sizes we test)

**Confirmed load convention (from `compare_results.py`):**

- `loadmat(path, squeeze_me=False)` — critical; `squeeze_me=True` breaks
  `[0, 0]` indexing on nested structs
- Access pattern: `data['results']['Ep_A'][0, 0]` — one `[0, 0]` de-nesting
  per struct level

**h5py fallback for MAT-v7.3:** SPM may write v7.3 (HDF5) if `Cp` is very large
(N > 15 regions, full parameter space). `h5py 3.15.1` is installed. No new code needed
for v0.7.0 because validation targets N = 3-5 regions; MAT-v5 is sufficient.

### Why NOT oct2py

oct2py routes through GNU Octave, not MATLAB. SPM12 is MATLAB-only (uses MEX files,
MATLAB-specific toolbox functions). oct2py cannot run `spm_nlsi_GN`. Do not install it.

### Why NOT MATLAB Engine API for Python

The MATLAB Engine API (`matlab.engine`) is useful for tight Python↔MATLAB integration but:
1. Not installed in the current environment
2. Installation requires running `python setup.py install` from
   `matlabroot/extern/engines/python` — user would need to do this manually
3. Adds a heavyweight dependency for functionality already covered by `subprocess`
4. The existing `subprocess` approach with `-batch` is sufficient: no interactive
   session needed, output captured, timeout handled

The `subprocess` approach is the correct choice for this project. Maintain it.

---

## NumPyro NUTS as Reference Oracle

**Status:** INSTALLED and ready. NumPyro 0.19.0, JAX 0.4.31.

**Usage pattern for v0.7.0:** Follow the NumPyro strategy from project memory —
do NOT rewrite PyTorch forward models in JAX. Instead use NumPyro for what it does
natively: run NUTS on a simple Gaussian likelihood proxy model for small N where
a closed-form likelihood exists (spectral DCM with fixed noise).

**Confirmed available:**
- `numpyro.infer.NUTS`, `numpyro.infer.MCMC`
- `numpyro.diagnostics.hpdi` — for credible interval construction
- `numpyro.diagnostics.gelman_rubin` — for chain convergence
- `numpyro.diagnostics.print_summary` — for R-hat reporting

**Integration point:** A `tests/test_vl_vs_nuts.py` can parametrize the spectral
DCM likelihood as a NumPyro model with fixed noise (eliminating the latent
hyperparameter M-step) and compare VL posterior means/variances to NUTS posteriors.
This provides the gold-standard check that VL is not systematically biased.

**Existing JAX infrastructure:** `src/pyro_dcm/inference/sbi_spectral.py` and
`sbi_embedding.py` are already JAX-aware — use the same JAX/NumPyro import pattern
there as a template.

---

## Systematic Recovery Sweeps: What to Reuse vs What is New

### Reuse as-is (no changes needed)

| Component | File | Role in v0.7.0 |
|-----------|------|---------------|
| `BenchmarkConfig` | `benchmarks/config.py` | Sweep configuration — add `method="vl"` to valid methods |
| `.npz` fixture format | `benchmarks/fixtures.py` + `generate_fixtures.py` | Same synthetic datasets, just run VL instead of SVI on them |
| `compute_rmse_A`, `compute_coverage` | `benchmarks/metrics.py` | Recovery matrix metrics |
| `@pytest.mark.slow` | `pyproject.toml` | Gate sweep tests |
| `submit_recovery.sh` | `benchmarks/submit_recovery.sh` | Cluster submission template |

### New (minimal additions)

| Component | File | What's new |
|-----------|------|-----------|
| `run_vl_recovery_sweep` | `benchmarks/runners/vl_recovery.py` | VL-specific runner iterating over N × SNR |
| `compare_free_energy` | `validation/compare_results.py` | F-value agreement check (|ΔF| < tol) |
| `compare_posterior_covariance` | `validation/compare_results.py` | Cp diagonal agreement check |
| `temper_posterior` | `src/pyro_dcm/inference/variational_laplace.py` | Covariance tempering for BMR overconfidence fix |
| `test_vl_robustness.py` | `tests/` | Stress-test fixtures (ill-cond, boundary, multi-restart) |
| `test_vl_recovery_matrix.py` | `tests/` | N × SNR recovery assertion |
| `test_vl_vs_spm.py` | `tests/` | VL-vs-MATLAB numeric agreement (extends existing `test_spm_*`) |

---

## Numerical Robustness Probing: Confirmed Available Primitives

All needed primitives are in `torch.linalg` (torch 2.9.1):

| Primitive | Use in Robustness Tests |
|-----------|------------------------|
| `torch.linalg.cond(M)` | Detect ill-conditioned precision matrices before inversion |
| `torch.linalg.eigvals(A)` | Verify stability boundary: max(real(λ)) < 0 |
| `torch.linalg.matrix_exp(J)` | Already used in `_spm_dx` for Gauss-Newton update |
| `torch.linalg.cholesky` with `LinAlgError` catch | Already used in posterior sampling |
| `torch.linalg.pinv` | Already used as fallback in `_spm_dx` |

`scipy.linalg.expm` (verified available) is an alternative for the matrix exponential
if torch's implementation produces NaN for pathological inputs — useful as a reference
comparison in robustness tests.

---

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| MATLAB invocation | `subprocess -batch` | MATLAB Engine API for Python | Not installed; subprocess already works; Engine API needs manual install from MATLAB root |
| MATLAB invocation | `subprocess -batch` | oct2py via Octave | SPM12 is MATLAB-only; Octave cannot run SPM MEX files |
| NumPyro use | NUTS on Gaussian proxy model only | Full JAX port of DCM forward model | Against project strategy: never rewrite PyTorch forward models in JAX |
| MAT file format | `scipy.io.savemat` (MAT-v5) | `h5py` direct HDF5 write | MAT-v5 sufficient for N ≤ 10; h5py available as fallback if SPM produces v7.3 |
| Recovery sweep config | Extend existing `BenchmarkConfig` | New dataclass | BenchmarkConfig already handles all needed fields |
| Coverage stats | Manual CI from Laplace covariance | `arviz` | arviz not installed; `numpyro.diagnostics.hpdi` provides the same for the NUTS oracle path; Laplace coverage computed analytically from `sigma_post` |

---

## What NOT to Add

| Avoid | Why | Use Instead |
|-------|-----|------------|
| `oct2py` | Does not support MATLAB/SPM MEX files | `subprocess` + MATLAB binary |
| `matlab.engine` | Not installed; subprocess already sufficient | `subprocess -batch` |
| `arviz` | Not installed; not needed — Laplace covariance is analytic; NUTS oracle uses `numpyro.diagnostics` | `numpyro.diagnostics.hpdi` for NUTS; analytic Laplace CI for VL |
| `corner` | Not installed; overkill for a validation harness | `matplotlib` already installed |
| `sbi` in v0.7.0 | Explicitly out of scope | Deferred to future milestones |

---

## No `pyproject.toml` Changes Required

The existing dependencies cover all v0.7.0 needs:

```toml
# Already in [project.dependencies]:
torch>=2.0       # 2.9.1 installed
pyro-ppl>=1.9    # 1.9.1 installed
scipy            # 1.16.3 installed
numpy            # 2.2.6 installed
torchdiffeq      # 0.2.5 installed

# Already in [project.optional-dependencies.dev]:
pytest           # 9.0.1 installed

# numpyro/jax/h5py/matplotlib/tabulate all installed; not in pyproject.toml
# but present in environment — no change needed
```

**New pytest marker needed** in `pyproject.toml`:

```toml
"vl: marks VL-specific validation tests (skip with '-m \"not vl\"')"
```

This is the only `pyproject.toml` addition.

---

## Version Compatibility

| Component | Compatibility Note |
|-----------|--------------------|
| torch 2.9.1 + numpyro 0.19.0 | No shared tensor state; JAX/NumPyro operates on its own arrays. No conflict. |
| scipy 1.16.3 `loadmat` + numpy 2.2.6 | Both work with the existing `[0, 0]` struct access pattern (verified). |
| MAT-v5 (`scipy.io.savemat` default) + SPM12 | SPM12 can read both v5 and v7.3. No compatibility issue loading v5 files in MATLAB. |
| h5py 3.15.1 + MAT-v7.3 | Only needed if SPM outputs exceed 2 GB. N ≤ 5 regions produces tiny files. Contingency only. |

---

## Sources

- Direct inspection: `src/pyro_dcm/inference/variational_laplace.py` — full VL engine
- Direct inspection: `validation/export_to_mat.py`, `validation/compare_results.py`,
  `validation/run_validation.py` — existing SPM interchange infrastructure
- Direct inspection: `validation/matlab_scripts/run_spm_spectral_dcm.m` — confirms `DCM.Cp`
  already saved
- Direct inspection: `benchmarks/config.py`, `benchmarks/fixtures.py` — sweep infrastructure
- Direct inspection: `tests/test_spm_task_dcm_validation.py`,
  `tests/test_spm_spectral_dcm_validation.py` — existing `@pytest.mark.spm` pattern
- Runtime verification: all package versions from `python -c "import X; print(X.__version__)"` — HIGH confidence
- Runtime verification: `scipy.io.savemat/loadmat` nested struct round-trip — confirmed working
- Runtime verification: `torch.linalg.cond/eigvals/matrix_exp`, `numpyro.diagnostics` — confirmed available
- Project memory: `feedback_numpyro_strategy.md` — do not rewrite PyTorch forward models in JAX

---

*Stack research for: v0.7.0 VL-Validation Harness*
*Researched: 2026-06-10*
