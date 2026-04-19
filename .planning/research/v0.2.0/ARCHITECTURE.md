# Architecture: Cross-Backend Inference Benchmarking

**Project:** Pyro-DCM v0.2.0
**Dimension:** Architecture integration for cross-backend benchmarking
**Researched:** 2026-04-06
**Confidence:** HIGH for Pyro-side integration, MEDIUM for NumPyro/DCM_PPLs integration

## Executive Summary

The v0.2.0 benchmark extension adds three architectural concerns to the existing
v0.1.0 benchmark infrastructure: (1) new Pyro guide variants that plug into the
existing runner pattern, (2) NumPyro backends that require a cross-framework data
exchange format and optional dependency handling, and (3) a results storage layer
that scales from 7 runners to 20+ while supporting multi-dimensional comparison
(method x variant x network_size). The existing architecture is well-suited for
extension -- the RUNNER_REGISTRY pattern, BenchmarkConfig dataclass, and metrics
module all accommodate new runners without modification. The main architectural
decisions concern data sharing across PyTorch/JAX, the DCM_PPLs integration
strategy, and regularization parameterization.

---

## 1. Shared Data Format for Cross-Backend Reproducibility

### Problem

Both Pyro (PyTorch) and NumPyro (JAX) runners must operate on identical synthetic
datasets for fair comparison. Currently, each runner generates its own data inline
(e.g., `make_random_stable_A` + `simulate_task_dcm` per dataset). This means:

- Runners use the same seeds but different RNG implementations (torch vs JAX)
- No guarantee of bit-identical data across backends
- Wasteful: each runner re-simulates the same ground truth

### Recommendation: NumPy .npz as interchange format

**Use `.npz` files as the cross-backend data exchange format.**

Rationale:
- Both `torch.from_numpy()` and `jax.numpy.load()` natively read `.npz` files
- NumPy is the common denominator -- already a dependency of both frameworks
- `.npz` supports named arrays (A_true, bold, stimulus, csd, etc.)
- Compact, fast, well-understood; no new dependencies
- The existing simulators already produce torch tensors that trivially convert
  to numpy via `.numpy()`

Alternatives considered and rejected:
- **HDF5 (h5py)**: Overkill for small arrays; adds a dependency
- **torch .pt files**: JAX cannot read these without torch installed
- **Pickle**: Not cross-framework safe; security concerns
- **JSON (current)**: Only for results, not efficient for array data

### Directory structure

```
benchmarks/
  fixtures/                     # NEW: pre-generated datasets
    task_3region/               # variant_NregionSize
      dataset_000.npz           # A_true, C, bold, stimulus, ...
      dataset_001.npz
      ...
      manifest.json             # metadata: n_datasets, seed, params
    task_5region/
    task_10region/
    spectral_3region/
    spectral_5region/
    spectral_10region/
    rdcm_3region/
    rdcm_5region/
    rdcm_10region/
```

### Data generation script

A new script `benchmarks/generate_fixtures.py` pre-generates all datasets:

```python
def generate_fixtures(
    variant: str,       # "task", "spectral", "rdcm"
    n_regions: int,     # 3, 5, 10
    n_datasets: int,    # 20 for quick, 50 for full
    seed: int,
    output_dir: str,
) -> None:
    """Generate and save synthetic datasets as .npz files.

    Uses torch-based simulators, converts to numpy for storage.
    Each .npz contains: A_true, C, bold/csd, stimulus, metadata.
    """
```

This script runs once before benchmarks. Runners load from fixtures instead
of generating data inline. The manifest.json records generation parameters
for reproducibility.

### Loading in runners

```python
# PyTorch runner (existing pattern)
data = np.load(path)
A_true = torch.from_numpy(data["A_true"])
bold = torch.from_numpy(data["bold"])

# JAX/NumPyro runner (new pattern)
data = np.load(path)
A_true = jnp.array(data["A_true"])
bold = jnp.array(data["bold"])
```

### Integration point

The `BenchmarkConfig` dataclass gains a `fixtures_dir` field:

```python
@dataclass
class BenchmarkConfig:
    # ... existing fields ...
    fixtures_dir: str = "benchmarks/fixtures"
    n_regions_list: list[int] = field(default_factory=lambda: [3, 5, 10])
```

Runners gain a helper `load_fixture(config, variant, n_regions, index)` that
returns a dict of tensors in the appropriate framework (torch or jax).

---

## 2. New Runner Architecture

### Problem

v0.2.0 adds ~13 new runners to the existing 7. The new runners fall into
three categories:

1. **Pyro guide variants** (6): low-rank MVN, full-rank MVN, IAF, structured,
   Laplace, amortized-refinement
2. **NumPyro backends** (3): NUTS, ADVI (AutoNormal), Laplace
3. **Regularization variants** (4+): non-centered, tighter priors, wider priors,
   prior ablations

### Recommendation: Parameterized runners, not separate files

**Extend existing runner files with a `guide_type` parameter rather than
creating one file per guide variant.**

Current pattern (v0.1.0):
```
runners/
  task_svi.py          -> run_task_svi(config)
  task_amortized.py    -> run_task_amortized(config)
  spectral_svi.py      -> run_spectral_svi(config)
  ...
```

New pattern (v0.2.0):
```
runners/
  task_svi.py          -> run_task_svi(config)           # MODIFIED: accepts guide_type
  task_amortized.py    -> run_task_amortized(config)     # UNCHANGED
  spectral_svi.py      -> run_spectral_svi(config)       # MODIFIED: accepts guide_type
  spectral_amortized.py                                   # UNCHANGED
  rdcm_vb.py                                              # UNCHANGED
  numpyro_nuts.py      -> run_numpyro_nuts(config)       # NEW: all variants
  numpyro_advi.py      -> run_numpyro_advi(config)       # NEW: all variants
  numpyro_laplace.py   -> run_numpyro_laplace(config)    # NEW: all variants
  spm_reference.py                                        # UNCHANGED
```

Rationale for this split:
- **Pyro guide variants** share 90% of code with existing SVI runners (only the
  guide creation differs). Parameterizing `guide_type` in `task_svi.py` avoids
  massive duplication.
- **NumPyro runners** are fundamentally different (JAX arrays, different
  inference API, different model definition). They deserve separate files.
- **Regularization variants** are model-level changes, not guide-level. They
  modify `BenchmarkConfig` (e.g., `prior_scale`, `parameterization`), not the
  runner file.

### Guide factory pattern

Replace the current `create_guide(model, init_scale=0.01)` with an extended
factory:

```python
def create_guide(
    model: Callable,
    guide_type: str = "mean_field",
    init_scale: float = 0.01,
    rank: int | None = None,
) -> AutoGuide:
    """Create a Pyro AutoGuide by type.

    Parameters
    ----------
    model : callable
        Pyro model function.
    guide_type : str
        One of: "mean_field", "low_rank", "full_rank",
        "iaf", "laplace", "structured".
    init_scale : float
        Initial scale for guide distributions.
    rank : int or None
        Rank for low-rank guide (default: min(latent_dim, 10)).
    """
    if guide_type == "mean_field":
        return AutoNormal(model, init_scale=init_scale)
    elif guide_type == "low_rank":
        return AutoLowRankMultivariateNormal(
            model, init_scale=init_scale, rank=rank or 10,
        )
    elif guide_type == "full_rank":
        return AutoMultivariateNormal(
            model, init_scale=init_scale,
        )
    elif guide_type == "iaf":
        return AutoIAFNormal(model, init_scale=init_scale)
    elif guide_type == "laplace":
        return AutoLaplaceApproximation(model)
    elif guide_type == "structured":
        return AutoStructured(model, init_scale=init_scale)
    else:
        raise ValueError(f"Unknown guide type: {guide_type}")
```

### Extended RUNNER_REGISTRY

```python
RUNNER_REGISTRY: dict[tuple[str, str, str], Callable] = {
    # Existing (variant, method, guide_type)
    ("task", "svi", "mean_field"): run_task_svi,
    ("task", "svi", "low_rank"): run_task_svi,
    ("task", "svi", "full_rank"): run_task_svi,
    ("task", "svi", "iaf"): run_task_svi,
    ("task", "svi", "laplace"): run_task_svi,
    ("task", "svi", "structured"): run_task_svi,
    ("task", "amortized", "nsf"): run_task_amortized,
    # NumPyro
    ("task", "nuts", "numpyro"): run_numpyro_nuts,
    ("task", "advi", "numpyro"): run_numpyro_advi,
    ("task", "laplace", "numpyro"): run_numpyro_laplace,
    # ... spectral, rdcm variants ...
}
```

**Alternative considered:** Keep the 2-tuple `(variant, method)` registry key and
encode guide type in the `method` string (e.g., `"svi_lowrank"`). This is simpler
but makes the string parsing fragile. The 3-tuple is cleaner.

**Migration note:** The existing 2-tuple registry must be preserved for backward
compatibility. Introduce the 3-tuple as an extension; the CLI `--guide` flag
selects it. Existing `--method svi` defaults to `guide_type="mean_field"`.

### BenchmarkConfig extension

```python
@dataclass
class BenchmarkConfig:
    # ... existing fields ...
    guide_type: str = "mean_field"        # NEW
    n_regions_list: list[int] = field(    # NEW: multi-size sweeps
        default_factory=lambda: [3],
    )
    prior_scale: float = 1.0 / 64.0      # NEW: for prior sensitivity
    parameterization: str = "centered"    # NEW: "centered" or "non_centered"
    elbo_type: str = "trace"              # NEW: "trace", "mean_field", "renyi"
    fixtures_dir: str = "benchmarks/fixtures"  # NEW
```

---

## 3. DCM_PPLs Integration Strategy

### Problem

The ins-amu/DCM_PPLs repository contains NumPyro DCM implementations for ERP
(not fMRI), structured as Jupyter notebooks (99.8% notebooks), with no
`pyproject.toml` or pip package. It is not directly usable as a dependency.

### Assessment

After examining DCM_PPLs and the associated paper (Baldy et al. 2024, J. Royal
Soc. Interface), the key finding is:

**DCM_PPLs implements ERP-DCM (EEG/MEG), not fMRI-DCM.** The forward model is
a neural mass model generating event-related potentials, not the
Balloon-Windkessel model generating BOLD. The model equations are fundamentally
different from Pyro-DCM's three variants.

What IS directly useful from DCM_PPLs/the paper:
- The inference strategy comparison (NUTS vs ADVI vs Laplace findings)
- Practical wisdom about multimodality in DCM posteriors
- NumPyro model-writing patterns for ODE-based models
- Chain initialization strategies

What is NOT directly reusable:
- The forward model code (ERP, not BOLD/CSD/frequency-domain)
- The specific model definitions

### Recommendation: Option (d) -- Write our own NumPyro models, cite DCM_PPLs

**Do not depend on DCM_PPLs. Instead, write thin NumPyro equivalents of our
existing Pyro models, guided by DCM_PPLs patterns and findings.**

Rationale:
1. DCM_PPLs is ERP-DCM, not fMRI-DCM -- the models don't transfer
2. DCM_PPLs is notebooks, not an installable package
3. Our forward model code is already correct and validated against SPM12
4. What we need is NumPyro model definitions that call the same math but
   use `numpyro.sample` instead of `pyro.sample`

### NumPyro model structure

New module: `src/pyro_dcm/numpyro_models/`

```
src/pyro_dcm/numpyro_models/
  __init__.py
  task_dcm_numpyro.py       # NumPyro version of task_dcm_model
  spectral_dcm_numpyro.py   # NumPyro version of spectral_dcm_model
  rdcm_numpyro.py           # NumPyro version of rdcm_model
  _forward_jax.py           # JAX reimpl of forward model math
```

The key challenge: the existing forward models use PyTorch (torchdiffeq for
ODE integration). NumPyro models need JAX equivalents. Two sub-options:

**(a) Rewrite forward models in JAX (diffrax for ODE)**
- Pro: Full JAX JIT compilation, maximum NumPyro performance
- Con: Duplicating validated math code; maintenance burden
- Effort: HIGH (rewrite Balloon-Windkessel, spectral transfer, rDCM in JAX)

**(b) Use numpy bridging -- convert JAX arrays to numpy, call scipy, convert back**
- Pro: Reuse existing scipy-based CSD code; minimal new code
- Con: Breaks JIT; NUTS will be slow without JIT compilation
- Effort: LOW but performance-limited

**(c) Hybrid: JAX forward models for task/spectral, numpy bridge for rDCM**
- Pro: NUTS/ADVI benefit from JIT on ODE models; rDCM already uses analytic VB
- Con: Still partial duplication
- Effort: MEDIUM

**Recommendation: Option (a) for task/spectral, skip NumPyro rDCM.**

The purpose of NumPyro backends is primarily NUTS sampling (gold standard
posterior) and ADVI/Laplace comparison. For rDCM, the analytic VB is already
the reference -- there's no benefit to running NUTS on a model designed for
closed-form inference.

For task and spectral DCM, writing JAX forward models is necessary for NUTS
performance. The math is well-specified (validated against SPM12), so
reimplementation is mechanical, not research.

### Optional dependency handling

NumPyro + JAX are optional dependencies. The pattern:

```python
# In pyproject.toml
[project.optional-dependencies]
numpyro = [
    "numpyro>=0.15",
    "jax[cpu]",
    "diffrax",     # JAX ODE solver
]
benchmark = [
    "matplotlib",
    "tabulate",
]

# In numpyro_models/__init__.py
from __future__ import annotations

try:
    import numpyro  # noqa: F401
    import jax      # noqa: F401
    HAS_NUMPYRO = True
except ImportError:
    HAS_NUMPYRO = False

# In numpyro runner
def run_numpyro_nuts(config: BenchmarkConfig) -> dict[str, Any]:
    if not HAS_NUMPYRO:
        return {
            "status": "skipped",
            "reason": "numpyro not installed (pip install pyro-dcm[numpyro])",
        }
    # ... proceed with inference ...
```

This follows the existing pattern from `spm_reference.py` runner, which
gracefully skips when MATLAB/SPM12 is unavailable.

---

## 4. Results Storage for Multi-Dimensional Comparison

### Problem

v0.1.0 stores results as a flat JSON with one key per variant. v0.2.0 needs
to store results for 9+ methods x 3 variants x 3 network sizes = 80+ cells,
plus per-element recovery data for scatter plots.

### Recommendation: Nested JSON with structured keys, flat CSV for tables

**Keep JSON as primary format** (backward compatible, human-readable), but
add a **flat CSV export** for the comparison table.

JSON structure:

```json
{
  "metadata": { ... },
  "results": {
    "task/svi/mean_field/3region": {
      "rmse_list": [...],
      "coverage_list": [...],
      "correlation_list": [...],
      "wall_time_list": [...],
      "a_true_list": [...],
      "a_inferred_list": [...],
      "summary": {
        "mean_rmse": 0.018,
        "mean_coverage": 0.71,
        "mean_correlation": 0.998,
        "mean_wall_time": 9.7
      },
      "metadata": { "variant": "task", "method": "svi", ... }
    },
    "task/svi/low_rank/3region": { ... },
    ...
  }
}
```

CSV export (for the comparison table figure):

```csv
variant,method,guide_type,n_regions,mean_rmse,std_rmse,mean_coverage,std_coverage,mean_corr,std_corr,mean_time,n_success,n_failed
task,svi,mean_field,3,0.018,0.004,0.71,0.23,0.998,0.001,9.7,5,0
task,svi,low_rank,3,0.015,0.003,0.85,0.12,0.999,0.001,12.1,5,0
...
```

### Alternatives considered

- **SQLite database**: Overkill; 80 rows don't need SQL
- **xarray + netCDF**: Good for N-dimensional scientific data, but adds heavy
  dependency; our data is tabular, not gridded
- **Parquet**: Good for large datasets; unnecessary for 80 rows
- **Pandas DataFrame pickled**: Not human-readable; version-sensitive

### Why not xarray?

The benchmark results are fundamentally tabular (each cell is a method applied
to a dataset configuration), not N-dimensional gridded data. xarray excels at
(time, latitude, longitude, level) arrays; it would be forced and unnatural for
(method, variant, network_size) with mixed-type value columns. A CSV with pandas
is simpler and more accessible.

### Result aggregation

New module `benchmarks/comparison.py`:

```python
def aggregate_results(results_dir: str) -> pd.DataFrame:
    """Load all benchmark JSON files and produce comparison DataFrame."""

def export_comparison_csv(df: pd.DataFrame, output: str) -> None:
    """Write flat CSV comparison table."""

def rank_methods(df: pd.DataFrame, metric: str = "coverage") -> pd.DataFrame:
    """Rank methods by metric, per variant and network size."""
```

---

## 5. Regularization Study Integration

### Problem

Non-centered parameterization, prior scale sensitivity, and ELBO variants
modify the generative model or inference loss, not just the guide. How to
integrate without duplicating model code?

### Recommendation: Composition via Pyro handlers, not model duplication

**Use Pyro's `poutine.reparam` handler for non-centered parameterization,
and config-driven prior scales for sensitivity analysis.**

### Non-centered parameterization

Pyro provides `LocScaleReparam` which transforms:
```
x ~ Normal(loc, scale)
```
into:
```
x_decentered ~ Normal(0, 1)
x = loc + scale * x_decentered
```

This is applied as a handler wrapping the existing model, NOT by modifying
the model code:

```python
from pyro.infer.reparam import LocScaleReparam
import pyro.poutine as poutine

# Wrap existing model with non-centered parameterization
reparam_config = {
    "A_free": LocScaleReparam(centered=0),  # fully decentered
    "C": LocScaleReparam(centered=0),
}
ncp_model = poutine.reparam(task_dcm_model, config=reparam_config)

# Use with any guide
guide = create_guide(ncp_model, guide_type="mean_field")
result = run_svi(ncp_model, guide, model_args, ...)
```

This is the cleanest integration because:
- Zero modification to existing model code
- Any guide works with any parameterization
- Can test centered=0, centered=0.5, centered=1 (original) as a sweep
- Pyro's reparam handler is well-tested and handles gradient plumbing

### Prior scale sensitivity

Add `prior_scale` to BenchmarkConfig. The model reads it:

```python
def task_dcm_model(
    observed_bold, stimulus, a_mask, c_mask, t_eval, TR, dt=0.5,
    prior_scale: float = 1.0 / 64.0,  # NEW parameter
) -> None:
    A_free_prior = dist.Normal(
        torch.zeros(N, N),
        prior_scale ** 0.5 * torch.ones(N, N),
    ).to_event(2)
```

**However**, this modifies the model signature, which is cleaner than a
global config but breaks the existing runner pattern where model_args is
a fixed tuple.

**Better alternative: Use `functools.partial`:**

```python
from functools import partial

model_with_prior = partial(task_dcm_model, prior_scale=1.0 / 32.0)
guide = create_guide(model_with_prior)
```

This keeps the existing model signature unchanged by using keyword arguments,
and the partial binds the extra parameter.

**Recommended approach:**

1. Add `prior_scale` as an optional keyword argument to the three Pyro models
   (default value preserves backward compatibility)
2. Runners pass `prior_scale` from BenchmarkConfig via functools.partial
3. This is a minor, backward-compatible change to existing models

### ELBO variants

Current: `Trace_ELBO(num_particles=1)` hardcoded in `run_svi`.

Extension: Add `elbo_type` to run_svi:

```python
def run_svi(
    model, guide, model_args,
    # ... existing params ...
    elbo_type: str = "trace",       # NEW
    num_particles: int = 1,
) -> dict:
    if elbo_type == "trace":
        elbo = Trace_ELBO(num_particles=num_particles)
    elif elbo_type == "mean_field":
        elbo = TraceMeanField_ELBO(num_particles=num_particles)
    elif elbo_type == "renyi":
        from pyro.infer import RenyiELBO
        elbo = RenyiELBO(alpha=0.5, num_particles=max(2, num_particles))
    ...
```

This is a clean extension to the existing `run_svi` function.

---

## 6. Component Map: New vs Modified

### New components

| Component | Location | Purpose |
|-----------|----------|---------|
| `benchmarks/generate_fixtures.py` | Script | Pre-generate synthetic datasets as .npz |
| `benchmarks/fixtures/` | Data dir | Stored .npz datasets |
| `benchmarks/runners/numpyro_nuts.py` | Runner | NumPyro NUTS inference |
| `benchmarks/runners/numpyro_advi.py` | Runner | NumPyro ADVI inference |
| `benchmarks/runners/numpyro_laplace.py` | Runner | NumPyro Laplace approximation |
| `benchmarks/comparison.py` | Module | Cross-backend result aggregation |
| `src/pyro_dcm/numpyro_models/` | Package | JAX/NumPyro model definitions |
| `src/pyro_dcm/numpyro_models/_forward_jax.py` | Module | JAX forward model math |
| `src/pyro_dcm/numpyro_models/task_dcm_numpyro.py` | Module | NumPyro task DCM model |
| `src/pyro_dcm/numpyro_models/spectral_dcm_numpyro.py` | Module | NumPyro spectral DCM |

### Modified components

| Component | Change | Risk |
|-----------|--------|------|
| `benchmarks/config.py` | Add guide_type, n_regions_list, prior_scale, parameterization, elbo_type, fixtures_dir | LOW: additive, defaults preserve behavior |
| `benchmarks/runners/__init__.py` | Extend RUNNER_REGISTRY with new entries | LOW: additive |
| `benchmarks/run_all_benchmarks.py` | Add --guide, --network-size, --parameterization CLI flags | LOW: new flags, old flags unchanged |
| `benchmarks/runners/task_svi.py` | Load from fixtures; accept guide_type via config | MEDIUM: must not break existing behavior |
| `benchmarks/runners/spectral_svi.py` | Same as task_svi | MEDIUM: same |
| `benchmarks/plotting.py` | Add comparison table and coverage calibration figures | LOW: additive functions |
| `benchmarks/metrics.py` | Possibly add ESS, R-hat for MCMC results | LOW: additive |
| `src/pyro_dcm/models/guides.py` | Extend create_guide with guide_type parameter | MEDIUM: must default to current behavior |
| `src/pyro_dcm/models/task_dcm_model.py` | Add optional prior_scale kwarg | LOW: default preserves behavior |
| `src/pyro_dcm/models/spectral_dcm_model.py` | Add optional prior_scale kwarg | LOW: same |
| `pyproject.toml` | Add [project.optional-dependencies] numpyro | LOW: additive |

### Unchanged components

- All forward_models/ (validated math, do not touch)
- All simulators/ (used by fixture generation, not modified)
- All existing tests/ (add new tests, don't modify old ones)
- guides/amortized_flow.py (existing NSF guide)
- guides/parameter_packing.py (existing packers)
- guides/summary_networks.py (existing summary nets)
- models/amortized_wrappers.py (existing wrapper models)
- models/rdcm_model.py (rDCM uses analytic VB, not SVI guides)

---

## 7. Data Flow Diagram

### v0.1.0 (current)

```
Runner(config)
  |
  +-> Simulator (torch) -> synthetic data (torch tensors)
  +-> Model + Guide (Pyro) -> SVI -> posterior
  +-> Metrics (torch) -> dict
  |
  v
JSON results -> Plotting -> figures
```

### v0.2.0 (proposed)

```
generate_fixtures.py
  |
  +-> Simulators (torch) -> .npz files (numpy)
  v

Runner(config)
  |
  +-> Load fixture (.npz) -> framework tensors (torch OR jax)
  |
  +--[Pyro path]--> Model + Guide(guide_type) + ELBO(elbo_type)
  |                  optionally wrapped with poutine.reparam
  |                  -> SVI -> posterior
  |
  +--[NumPyro path]--> NumPyro model + NUTS/ADVI/Laplace
  |                    -> inference result
  |
  +-> Metrics (framework-agnostic via numpy) -> dict
  v

JSON results (nested) -> comparison.py -> CSV table
                       -> plotting.py -> figures (scatter, coverage, table)
```

### Key change: Metric computation

Current metrics use torch tensors. NumPyro runners produce JAX arrays.
Solution: Convert to numpy in each runner before calling metrics.

```python
# In NumPyro runners:
A_inferred_np = np.asarray(A_inferred_jax)
A_true_np = np.asarray(A_true_jax)
rmse = compute_rmse(
    torch.from_numpy(A_true_np),
    torch.from_numpy(A_inferred_np),
)
```

Or better: make metrics accept numpy arrays directly (minor refactor to
metrics.py -- replace `torch.sqrt(torch.mean(...))` with `float(np.sqrt(np.mean(...)))`
or keep torch and convert at the boundary).

**Recommendation:** Keep metrics in torch (existing, tested). Convert at the
runner boundary. NumPyro runners convert jax -> numpy -> torch before metrics.

---

## 8. Suggested Build Order

Build phases are ordered by dependency chain and risk:

### Phase 1: Fixture Generation and Config Extension
**Dependencies:** None (uses existing simulators)
**Deliverables:**
- `benchmarks/generate_fixtures.py` script
- `BenchmarkConfig` extension (guide_type, n_regions_list, etc.)
- Fixtures for 3 variants x 3 sizes
- Update existing runners to optionally load from fixtures

**Rationale:** Foundation for everything else. Low risk, high value. All
subsequent runners depend on shared fixtures for fair comparison.

### Phase 2: Pyro Guide Variants
**Dependencies:** Phase 1 (fixtures, config)
**Deliverables:**
- Extended `create_guide` factory
- Benchmark runs: mean_field, low_rank, full_rank, IAF, Laplace, structured
- For task and spectral variants (6 guides x 2 variants = 12 new registry entries)
- Coverage calibration analysis

**Rationale:** Pure Pyro, no new dependencies. The existing infrastructure
handles these naturally. This phase answers "does a richer guide fix the
0.44-0.78 coverage problem?" before investing in NumPyro.

### Phase 3: Regularization Study
**Dependencies:** Phase 2 (guide variants for comparison)
**Deliverables:**
- `poutine.reparam` integration (non-centered parameterization)
- Prior scale sensitivity sweep (1/128, 1/64, 1/32, 1/16)
- ELBO variant comparison (Trace, MeanField, Renyi)
- Centered vs non-centered x guide type interaction analysis

**Rationale:** Still pure Pyro. Tests whether the coverage problem is a
parameterization/prior issue or a fundamental mean-field limitation.
Must follow Phase 2 so we can compare NCP x guide type combinations.

### Phase 4: NumPyro Backend Integration
**Dependencies:** Phase 1 (fixtures), Phase 2-3 results inform expectations
**Deliverables:**
- `src/pyro_dcm/numpyro_models/` package
- JAX forward models (Balloon-Windkessel, spectral transfer function)
- NumPyro model definitions (task, spectral)
- NUTS runner (gold standard posterior)
- ADVI runner (NumPyro's variational inference)
- Laplace runner
- Optional dependency handling in pyproject.toml

**Rationale:** Highest effort phase. Requires JAX reimplementation of forward
models. NUTS provides the gold standard against which all VI methods are
compared. Place after Pyro phases so the Pyro-side analysis is complete
and can immediately be compared once NumPyro results arrive.

### Phase 5: Cross-Backend Comparison and Analysis
**Dependencies:** Phases 2-4 (all results available)
**Deliverables:**
- `benchmarks/comparison.py` aggregation module
- CSV comparison table export
- Coverage calibration figure (key paper figure)
- Wall time vs accuracy Pareto frontier
- Method recommendation guide
- Benchmark narrative report update

**Rationale:** Pure analysis, no new infrastructure. Combines all prior
phase results into the final deliverable.

### Phase 6: Documentation and Polish
**Dependencies:** Phase 5 (analysis complete)
**Deliverables:**
- Updated methods.md/methods.tex with new methods
- User recommendation guide
- Updated quickstart with guide selection advice
- v0.2.0 release notes

---

## 9. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| JAX forward model diverges from torch version | MEDIUM | HIGH | Validate both against SPM12; shared .npz test data |
| NUTS fails to converge on ODE model | MEDIUM | MEDIUM | DCM_PPLs paper documents this; use their initialization strategies |
| Full-rank MVN guide doesn't fit (9x9 = 81 params for 5-region) | LOW | LOW | Fall back to low-rank; document when it fails |
| Pyro reparam handler incompatible with ODE model | LOW | MEDIUM | Test early in Phase 3; fallback to manual NCP |
| NumPyro optional dependency causes import errors | LOW | LOW | Robust try/except pattern; test both with and without |
| Too many benchmark cells to run in CI | HIGH | LOW | Quick mode runs 3-region only; full run is manual |

---

## 10. Anti-Patterns to Avoid

### Anti-Pattern 1: Duplicating Forward Model Code
**What:** Copy-pasting Balloon-Windkessel from torch to JAX
**Why bad:** Two implementations drift; bugs fixed in one, not other
**Instead:** Write JAX version from the SAME reference equations. Validate both
against the same .npz test fixtures. Shared validation, not shared code.

### Anti-Pattern 2: God Config
**What:** BenchmarkConfig grows to 30+ fields
**Why bad:** Every runner must handle every field; testing surface explodes
**Instead:** Use `BenchmarkConfig` for common fields; runner-specific config
in `metadata` dict. Or use dataclass inheritance for variant-specific configs.

### Anti-Pattern 3: Framework Abstraction Layer
**What:** Building a `Backend` abstraction that wraps both Pyro and NumPyro
**Why bad:** The APIs are fundamentally different (trace-based vs functional);
abstraction would be leaky and unmaintainable
**Instead:** Separate runner files per framework. Share data format (.npz) and
metrics (torch), not inference code.

### Anti-Pattern 4: Running NUTS in CI
**What:** Adding NUTS benchmark runs to the pytest CI suite
**Why bad:** NUTS takes minutes per dataset, even with JIT; CI would be 30+ min
**Instead:** NUTS benchmarks are manual/scheduled runs. CI only validates that
the NumPyro model can execute a few NUTS steps without error.

---

## Sources

- [Pyro Automatic Guide Generation](https://docs.pyro.ai/en/stable/infer.autoguide.html) -- HIGH confidence (official docs)
- [NumPyro Automatic Guide Generation](https://num.pyro.ai/en/stable/autoguide.html) -- HIGH confidence (official docs)
- [Pyro SVI documentation](https://docs.pyro.ai/en/stable/inference_algos.html) -- HIGH confidence (official docs)
- [NumPyro Reparameterizers](https://num.pyro.ai/en/stable/reparam.html) -- HIGH confidence (official docs)
- [DCM_PPLs GitHub repository](https://github.com/ins-amu/DCM_PPLs) -- HIGH confidence (primary source)
- [Baldy et al. 2024, J. Royal Soc. Interface](https://royalsocietypublishing.org/doi/10.1098/rsif.2024.0880) -- HIGH confidence (peer-reviewed paper)
- [Baldy et al. 2024, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12133347/) -- HIGH confidence (same paper, open access)
- [NumPyro GitHub](https://github.com/pyro-ppl/numpyro) -- HIGH confidence (official repo)
- [Pyro SVI Tips and Tricks](https://pyro.ai/examples/svi_part_iv.html) -- MEDIUM confidence (tutorial)
- [NumPyro VIP tutorial](https://num.pyro.ai/en/stable/tutorials/variationally_inferred_parameterization.html) -- MEDIUM confidence (tutorial)
