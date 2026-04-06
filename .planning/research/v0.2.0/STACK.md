# Technology Stack: v0.2.0 Cross-Backend Inference Benchmarking

**Project:** Pyro-DCM
**Researched:** 2026-04-06
**Scope:** Stack additions/changes for cross-backend inference benchmarking
**Overall confidence:** HIGH (all components are established libraries with stable APIs)

---

## Executive Summary

v0.2.0 adds **no new core dependencies** to the PyTorch/Pyro stack. The existing
pyro-ppl 1.9.1 already ships with all needed guide variants (AutoLowRankMultivariateNormal,
AutoStructured, AutoGaussian) and ELBO estimators (TraceMeanField_ELBO, RenyiELBO).
The main stack additions are: (1) **numpyro + jax + diffrax** for the NUTS/ADVI/Laplace
backends, (2) **arviz 0.22.x** for cross-backend MCMC diagnostics, and (3) **.npz** as
the interchange format for shared synthetic datasets.

**Critical finding:** The ins-amu/DCM_PPLs repo implements **ERP/EEG DCM** (Jansen-Rit
neural mass model), NOT fMRI DCM. It cannot be used as a drop-in NUTS gold standard for
our bilinear + Balloon-Windkessel models. We must write our own NumPyro generative models
that mirror the existing Pyro models, using diffrax for ODE integration in JAX.

---

## 1. Pyro Guide Variants (Already Available)

**No new dependencies.** All needed guides ship with pyro-ppl >= 1.9.

### 1.1 AutoLowRankMultivariateNormal

**Confidence:** HIGH (verified from Pyro source code)

```python
from pyro.infer.autoguide import AutoLowRankMultivariateNormal

guide = AutoLowRankMultivariateNormal(
    model,
    init_loc_fn=init_to_median,  # default
    init_scale=0.1,              # default; use 0.01 for DCM (ODE stability)
    rank=None,                   # default: auto-sets to ~sqrt(latent_dim)
)
```

**Key properties:**
- Captures posterior correlations via low-rank + diagonal covariance
- Covariance: `cov = cov_factor @ cov_factor.T + diag(cov_diag)`
- Rank auto-selects to `ceil(sqrt(latent_dim))` if None
- For our 3-region task DCM: latent_dim ~ 10-12, so rank ~ 3-4
- For our 5-region rDCM: latent_dim ~ 30, so rank ~ 6

**Integration note:** Drop-in replacement for existing `create_guide()`. Change
`AutoNormal(model, init_scale=0.01)` to
`AutoLowRankMultivariateNormal(model, init_scale=0.01)`.

### 1.2 AutoStructured (Block-Diagonal)

**Confidence:** HIGH (verified from Pyro source code)

```python
from pyro.infer.autoguide import AutoStructured

guide = AutoStructured(
    model,
    conditionals="mvn",    # or "normal", "delta", or dict per site
    dependencies="linear", # or dict specifying site->site dependencies
    init_loc_fn=init_to_feasible,  # default
    init_scale=0.1,                # default; use 0.01 for DCM
)
```

**Key properties:**
- Allows specifying which latent variables covary and which are independent
- `conditionals` controls per-site distribution type (delta, normal, mvn)
- `dependencies` controls inter-site linear dependencies (can be sparse)
- No cycles or self-loops allowed in dependency graph

**DCM-specific usage (recommended structure):**
```python
guide = AutoStructured(
    task_dcm_model,
    conditionals={
        "A_free": "mvn",       # Full covariance within A
        "C": "mvn",            # Full covariance within C
        "noise_prec": "normal", # Independent
    },
    dependencies={
        "C": {"A_free": "linear"},  # C can depend on A
    },
    init_scale=0.01,
)
```

This creates block-diagonal covariance (A_free block, C block) with optional
cross-block linear dependencies, which is the natural structure for DCM where
A parameters are correlated with each other but less so with noise parameters.

### 1.3 AutoGaussian (Optimal Conditional Independence)

**Confidence:** HIGH (verified from Pyro source code)

```python
from pyro.infer.autoguide import AutoGaussian

guide = AutoGaussian(
    model,
    init_loc_fn=init_to_feasible,  # default
    init_scale=0.1,                # default
    backend=None,                  # "dense" (default) or "funsor"
)
```

**Key properties:**
- Automatically determines optimal covariance structure from model's plate/dependency structure
- Equivalent to full-rank MultivariateNormal but with sparse precision matrix
- "dense" backend: same computational cost as AutoMultivariateNormal
- "funsor" backend: asymptotically cheaper but high constant overhead (needs `pip install pyro-ppl[funsor]`)

**Recommendation:** Try AutoGaussian with "dense" backend as a "smart full-rank"
option. However, DCM models do not use `pyro.plate` extensively, so the
conditional independence structure may not be exploitable. In that case,
AutoGaussian degenerates to AutoMultivariateNormal.

### 1.4 ELBO Estimators

**Confidence:** HIGH (verified from Pyro docs and source)

All already in pyro-ppl 1.9.1:

| ELBO Variant | Import | Key Parameters | When to Use |
|---|---|---|---|
| `Trace_ELBO` | `pyro.infer.Trace_ELBO` | `num_particles=1, vectorize_particles=False` | Default. Already in use. |
| `TraceMeanField_ELBO` | `pyro.infer.TraceMeanField_ELBO` | `num_particles=1` | Analytic KL when available. Requires mean-field guide (AutoNormal). Tighter gradients. |
| `RenyiELBO` | `pyro.infer.RenyiELBO` | `alpha=0, num_particles=2` | alpha < 1 gives tighter bound than ELBO. alpha != 1 required. |
| `TraceGraph_ELBO` | `pyro.infer.TraceGraph_ELBO` | `num_particles=1` | For non-reparameterizable distributions. Not needed for DCM (all Normal priors). |

**Recommended benchmark variants:**
1. `Trace_ELBO(num_particles=1)` -- baseline (current)
2. `TraceMeanField_ELBO(num_particles=1)` -- with AutoNormal only
3. `RenyiELBO(alpha=0.5, num_particles=4)` -- tighter bound, more compute
4. `Trace_ELBO(num_particles=4, vectorize_particles=True)` -- multi-particle baseline

**TraceMeanField_ELBO restrictions:**
- Requires mean-field guide (factorized)
- All latent variables must be reparameterizable
- Guide must not have dependencies between sites
- Compatible with AutoNormal but NOT with AutoLowRankMultivariateNormal

### 1.5 Prior Predictive Checks

**Confidence:** HIGH (verified from Pyro tutorial and source)

```python
from pyro.infer import Predictive

# Prior predictive (no guide, no data)
prior_predictive = Predictive(
    model,
    num_samples=1000,
    return_sites=("A", "predicted_bold", "obs"),
    parallel=False,  # True for vectorized (more memory)
)
samples = prior_predictive(*model_args)
# samples["A"].shape == (1000, N, N)
# samples["predicted_bold"].shape == (1000, T, N)

# Posterior predictive (with trained guide)
posterior_predictive = Predictive(
    model,
    guide=trained_guide,
    num_samples=500,
    return_sites=("A", "predicted_bold"),
)
pp_samples = posterior_predictive(*model_args)
```

**Key parameters:**
- `model`: Pyro model function
- `posterior_samples`: dict of samples (alternative to guide)
- `guide`: trained guide (alternative to posterior_samples)
- `num_samples`: number of draws
- `return_sites`: tuple of site names to return (including deterministic sites)
- `parallel`: vectorize sample draws (faster but more memory)

**Integration:** Works with all existing DCM models. Deterministic sites
(`A`, `predicted_bold`, `predicted_csd`) are automatically available through
Predictive, unlike raw guide sampling.

---

## 2. NumPyro Backends (NEW Dependencies)

### 2.1 Critical Finding: DCM_PPLs Is Not Usable

**Confidence:** HIGH (verified by reading the repo)

The [ins-amu/DCM_PPLs](https://github.com/ins-amu/DCM_PPLs) repository implements
**ERP/EEG DCM** using the Jansen-Rit neural mass model, not fMRI DCM with the
bilinear neural state equation and Balloon-Windkessel hemodynamics. The forward
models are fundamentally different:

| Aspect | DCM_PPLs (ERP) | Pyro-DCM (fMRI) |
|---|---|---|
| Neural model | Jansen-Rit neural mass | Bilinear dx/dt = Ax + Cu |
| Observation | ERP waveform (time domain) | BOLD (time domain) or CSD (frequency domain) |
| Hemodynamics | None (direct EEG observation) | Balloon-Windkessel ODE |
| Data type | EEG/MEG | fMRI |

**Consequence:** We must write our own NumPyro DCM models from scratch. However,
the forward model mathematics is identical -- we just need to reimplement the ODE
integration in JAX (using diffrax instead of torchdiffeq) and express the priors
using numpyro.distributions instead of pyro.distributions.

The Baldy et al. (2025) paper from the same group ("Dynamic causal modelling in
probabilistic programming languages", J. R. Soc. Interface) provides the
methodological template for cross-backend DCM benchmarking, though for ERP not fMRI.

### 2.2 NumPyro

**Version:** 0.16.x (last version supporting Python 3.11 natively)
**Confidence:** MEDIUM (version pinning needs verification at install time)

```
pip install numpyro[cpu]==0.16.1
```

NumPyro 0.20.0+ requires Python >= 3.11 (which we have), but also requires
JAX >= 0.4.x which can have breaking changes. Pin to a known-good version.

**UPDATE:** NumPyro 0.20.1 (released 2026-03-25) supports Python 3.11-3.14.
Use the latest if JAX compatibility is confirmed on Windows.

**NumPyro autoguide variants for benchmarking:**

| Guide | NumPyro Class | Description |
|---|---|---|
| Mean-field | `numpyro.infer.autoguide.AutoNormal` | Diagonal Gaussian (matches our Pyro AutoNormal) |
| Laplace | `numpyro.infer.autoguide.AutoLaplaceApproximation` | MAP + inverse Hessian covariance |
| Low-rank | `numpyro.infer.autoguide.AutoLowRankMultivariateNormal` | Matches Pyro variant |
| Full MVN | `numpyro.infer.autoguide.AutoMultivariateNormal` | Full covariance matrix |
| NUTS | `numpyro.infer.NUTS` | Gold standard MCMC |
| HMC | `numpyro.infer.HMC` | Tunable alternative to NUTS |

**NumPyro ELBO variants:**
- `numpyro.infer.Trace_ELBO` -- standard
- `numpyro.infer.TraceMeanField_ELBO` -- analytic KL
- `numpyro.infer.TraceGraph_ELBO` -- for non-reparameterizable

**Practical NumPyro pattern for DCM:**
```python
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, Predictive
from numpyro.infer.autoguide import AutoNormal, AutoLaplaceApproximation
import jax
import jax.numpy as jnp

def task_dcm_model_numpyro(observed_bold, stimulus_fn, a_mask, c_mask, ...):
    N = a_mask.shape[0]
    A_free = numpyro.sample("A_free", dist.Normal(jnp.zeros((N, N)), ...))
    # ... same structure as Pyro model but with JAX ops
    # ODE integration via diffrax instead of torchdiffeq

# NUTS gold standard
kernel = NUTS(task_dcm_model_numpyro)
mcmc = MCMC(kernel, num_warmup=500, num_samples=1000, num_chains=4)
mcmc.run(jax.random.PRNGKey(0), observed_bold, ...)
samples = mcmc.get_samples()
```

### 2.3 JAX

**Version:** >= 0.4.35 (whatever numpyro pins)
**Confidence:** HIGH

```
pip install jax jaxlib
```

On Windows, JAX CPU works. GPU support on Windows requires WSL2 or specific
jaxlib builds.

**Integration concern:** JAX and PyTorch can coexist in the same Python
environment. They use separate memory allocators and do not conflict. However,
tensors must be explicitly converted:
- `jnp.array(tensor.numpy())` -- PyTorch to JAX
- `torch.from_numpy(np.asarray(jax_array))` -- JAX to PyTorch

### 2.4 Diffrax

**Version:** 0.7.2 (latest, Python 3.10+)
**Confidence:** HIGH (verified on PyPI)

```
pip install diffrax
```

JAX-native ODE solver, equivalent to torchdiffeq but for the JAX ecosystem.

**Key API mapping (torchdiffeq -> diffrax):**

| torchdiffeq | diffrax |
|---|---|
| `odeint(func, y0, t, method='rk4')` | `diffeqsolve(ODETerm(func), Tsit5(), t0, t1, dt0, y0)` |
| `method='dopri5'` | `Tsit5()` or `Dopri5()` |
| `method='rk4'` | `Euler()` with small dt or `Midpoint()` |
| adjoint method | `RecursiveCheckpointAdjoint()` |

**Diffrax solvers for DCM:**
- `Tsit5()` -- adaptive 5th order (default, good for most cases)
- `Euler()` -- fixed-step (matches our rk4 pattern for predictable NUTS runtime)
- `Kvaerno3()` / `Kvaerno5()` -- implicit (for stiff Balloon model if needed)

**Performance note:** NUTS with diffrax ODE integration can be very slow
(reported ~28 minutes for 50 warmup + 100 samples on an M3 Pro for a PK-PD
model). For our Balloon-Windkessel ODE (5 states per region, 3-5 regions),
expect NUTS to be 100-1000x slower than SVI. This is expected and is the point
of the benchmark -- quantifying the cost of the variational approximation.

### 2.5 NUTS Performance Expectations for DCM

**Confidence:** MEDIUM (extrapolated from forum reports and model complexity)

| DCM Variant | Latent Dim | NUTS Expected Time (1000 samples) | SVI Time (3000 steps) |
|---|---|---|---|
| Task (3 regions) | ~12 | 30-120 min | ~30s |
| Task (5 regions) | ~30 | 2-8 hours | ~60s |
| Spectral (3 regions) | ~20 | 10-60 min (no ODE) | ~15s |
| Spectral (5 regions) | ~45 | 1-4 hours | ~30s |
| rDCM (3 regions) | ~15 | 5-20 min (no ODE) | ~5s (VB is instant) |

Task DCM NUTS will be the bottleneck because each NUTS step requires multiple
ODE integrations (one per leapfrog step, plus gradients). Spectral DCM avoids
ODE integration (frequency-domain), so NUTS is more feasible there.

**Recommendation:** Start benchmarking with spectral DCM NUTS (fastest), then
task DCM NUTS. For task DCM, consider limiting to `num_warmup=200, num_samples=500`
initially and scaling up only for the final paper run.

---

## 3. Data Interchange Format

**Recommendation:** `.npz` (NumPy compressed archive)
**Confidence:** HIGH

### Why .npz

| Format | PyTorch Load | JAX Load | Human-Readable | Size | Verdict |
|---|---|---|---|---|---|
| `.npz` | `np.load()` -> `torch.from_numpy()` | `np.load()` -> `jnp.array()` | No (binary) | Small | **Best** |
| `.pt` | `torch.load()` | Needs PyTorch installed | No | Small | PyTorch-only |
| HDF5 | h5py | h5py | No | Small | Overkill for arrays |
| JSON | Custom | Custom | Yes | Large | Too slow for tensors |

**.npz is the universal interchange format** because both PyTorch and JAX can
load NumPy arrays natively. No extra dependencies (numpy is already required by
both). The `.pt` format requires PyTorch to load, which defeats the purpose of
cross-backend benchmarking.

### Shared Dataset Schema

```python
# Save (from PyTorch simulator)
import numpy as np

np.savez_compressed(
    "benchmarks/datasets/task_dcm_3regions_seed42.npz",
    # Ground truth
    A_true=A_true.numpy(),           # (N, N)
    C_true=C_true.numpy(),           # (N, M)
    # Observed data
    bold=bold.numpy(),               # (T, N)
    stimulus_onsets=onsets,           # (n_blocks,)
    stimulus_durations=durations,    # (n_blocks,)
    # Metadata
    n_regions=np.array(N),
    n_inputs=np.array(M),
    TR=np.array(2.0),
    dt_sim=np.array(0.01),
    SNR=np.array(5.0),
    seed=np.array(42),
    variant=np.array("task"),
)

# Load (in NumPyro runner)
import jax.numpy as jnp

data = np.load("benchmarks/datasets/task_dcm_3regions_seed42.npz")
bold_jax = jnp.array(data["bold"])
A_true_jax = jnp.array(data["A_true"])
```

### Why NOT HDF5

HDF5 (via h5py) adds a dependency and is designed for large, hierarchical
datasets with partial I/O. Our datasets are small (< 1MB each). The added
complexity is not justified.

---

## 4. Diagnostics Infrastructure

### 4.1 ArviZ

**Version:** 0.22.0 (requires Python >= 3.10, supports 3.11)
**Confidence:** HIGH (verified Python 3.11 support)

```
pip install arviz==0.22.0
```

**IMPORTANT:** ArviZ 1.0.0 (released 2026-03-02) requires Python >= 3.12 and has a
completely restructured API (split into arviz-base, arviz-stats, arviz-plots). Since
our project targets Python 3.11, **pin to arviz 0.22.0** which is the last version
supporting Python 3.10+.

**Core diagnostic functions needed:**

| Function | Purpose | Works With |
|---|---|---|
| `az.from_numpyro(mcmc)` | Convert NumPyro MCMC to InferenceData | NumPyro MCMC |
| `az.from_pyro(mcmc)` | Convert Pyro MCMC to InferenceData | Pyro MCMC |
| `az.from_dict(posterior=...)` | Create InferenceData from raw arrays | SVI posteriors |
| `az.ess(idata, method="bulk")` | Bulk effective sample size | MCMC |
| `az.ess(idata, method="tail")` | Tail effective sample size | MCMC |
| `az.rhat(idata)` | Split R-hat convergence diagnostic | MCMC |
| `az.summary(idata)` | Summary table (mean, sd, hdi, ess, rhat) | All |
| `az.plot_trace(idata)` | Trace plots | MCMC |
| `az.plot_posterior(idata)` | Posterior distribution plots | All |

**Creating InferenceData from SVI posteriors:**
```python
import arviz as az
import numpy as np

# After SVI, draw samples from guide
guide_samples = {
    "A_free": samples_A_free.numpy(),  # (n_samples, N, N)
    "C": samples_C.numpy(),            # (n_samples, N, M)
}
idata = az.from_dict(
    posterior=guide_samples,
    observed_data={"bold": observed_bold.numpy()},
)
```

This enables unified diagnostic plots and comparison tables across all backends.

### 4.2 ESS and R-hat for SVI

ArviZ's `az.ess()` and `az.rhat()` are designed for MCMC chains. For SVI
posteriors (which are parametric, not chain-based), these diagnostics do not
directly apply. Instead:

**For SVI quality assessment, use:**
1. ELBO convergence (already implemented in `run_svi`)
2. Coverage calibration (already in `benchmarks/metrics.py`)
3. Posterior predictive checks (via `Predictive`)
4. Comparison against NUTS gold standard (the point of v0.2.0)

**For MCMC quality assessment, use:**
1. `az.ess(method="bulk")` > 400 (100 per chain x 4 chains)
2. `az.ess(method="tail")` > 400
3. `az.rhat()` < 1.01 (strict) or < 1.05 (lenient)
4. Visual trace plot inspection

### 4.3 posteriordb

**Confidence:** MEDIUM (useful reference but not directly applicable)

[posteriordb](https://github.com/stan-dev/posteriordb) is a database of benchmark
posteriors with reference draws for testing inference algorithms. Published at
AISTATS 2025. Contains 120 models with Stan/PyMC/Pyro implementations.

**NOT recommended for direct use** because:
- DCM models are not in posteriordb
- Our benchmark uses domain-specific synthetic data with known ground truth
- posteriordb's value is for generic inference algorithm testing

**Useful as a reference** for:
- Benchmark methodology (how they report ESS, R-hat, RMSE)
- Comparison table format
- Statistical testing of inference quality

---

## 5. Recommended Stack Additions

### New Dependencies (pyproject.toml additions)

```toml
[project.optional-dependencies]
benchmark = [
    "matplotlib",
    "tabulate",
    "arviz>=0.20,<1.0",   # MCMC diagnostics; pin below 1.0 for Py3.11
]
numpyro = [
    "numpyro[cpu]>=0.16",  # NumPyro + JAX CPU
    "diffrax>=0.6",        # JAX ODE solver
]
dev = [
    "pytest",
    "ruff",
    "mypy",
]
all = [
    "pyro-dcm[benchmark,numpyro,dev]",
]
```

**Rationale for optional dependency groups:**
- `numpyro` is a separate optional group because JAX installation can be
  platform-specific (CPU vs CUDA vs TPU) and adds significant install weight
- Users who only want Pyro-based inference should not need JAX
- `arviz` goes in `benchmark` because it is only needed for cross-backend
  comparison, not for core inference

### Version Pinning Summary

| Package | Version | Why This Version | Confidence |
|---|---|---|---|
| pyro-ppl | >= 1.9 (already installed) | Has all guide variants and ELBO estimators | HIGH |
| numpyro | >= 0.16, preferably 0.20.1 | Latest with Python 3.11 support | MEDIUM |
| jax + jaxlib | (pulled by numpyro) | CPU backend for Windows | HIGH |
| diffrax | >= 0.6 | JAX ODE solver, stable API | HIGH |
| arviz | >= 0.20, < 1.0 | Last major version supporting Python 3.11 | HIGH |
| zuko | >= 1.2 (already installed) | No changes needed | HIGH |
| torchdiffeq | (already installed) | No changes needed | HIGH |

### What NOT to Add

| Library | Why Not |
|---|---|
| **posteriordb-python** | DCM models not in database; our benchmarks use synthetic ground truth |
| **h5py** | Overkill for small datasets; .npz is simpler and universal |
| **xarray** | ArviZ handles xarray internally; no need for direct dependency |
| **blackjax** | Third MCMC backend adds complexity without clear benefit over NumPyro NUTS |
| **emcee** | Ensemble MCMC not competitive with NUTS for smooth posteriors |
| **cmdstanpy** | Stan adds too much installation complexity (needs C++ compiler) |
| **pymc** | Different PPL ecosystem; adds Theano/PyTensor dependency for no clear gain |
| **Click/Typer** | argparse already used; consistency over features |

---

## 6. Integration Architecture

### How New Components Fit Existing Code

```
Existing (v0.1.0)                    New (v0.2.0)
-----------------                    ------------
pyro_dcm/models/
  task_dcm_model.py          -->     pyro_dcm/models/numpyro/
  spectral_dcm_model.py               task_dcm_numpyro.py
  rdcm_model.py                        spectral_dcm_numpyro.py
  guides.py (AutoNormal)               rdcm_numpyro.py

benchmarks/runners/
  task_svi.py                -->     benchmarks/runners/
  task_amortized.py                    task_nuts.py (NumPyro NUTS)
  spectral_svi.py                      task_lowrank.py (Pyro LowRank guide)
  spectral_amortized.py                spectral_nuts.py
  rdcm_vb.py                           ...etc

benchmarks/config.py         -->     benchmarks/datasets/
  BenchmarkConfig                      task_dcm_3r_seed42.npz (shared data)
                                       spectral_dcm_3r_seed42.npz

                             -->     benchmarks/diagnostics.py
                                       arviz-based MCMC diagnostics
                                       coverage calibration computation
```

### Data Flow for Cross-Backend Comparison

```
1. GENERATE: PyTorch simulators create synthetic datasets
   -> Save as .npz to benchmarks/datasets/

2. INFER (Pyro backends): Load .npz, convert to torch.Tensor, run SVI
   -> Save posterior samples as .npz

3. INFER (NumPyro backends): Load .npz, convert to jnp.array, run NUTS/SVI
   -> Save posterior samples as .npz (or arviz InferenceData)

4. COMPARE: Load all posterior samples, compute metrics
   -> Coverage calibration, RMSE, wall time
   -> ArviZ diagnostics for MCMC backends
   -> Unified comparison table
```

---

## 7. Alternatives Considered

| Category | Recommended | Alternative | Why Not Alternative |
|---|---|---|---|
| MCMC backend | NumPyro NUTS | Pyro NUTS | NumPyro NUTS is 10-100x faster (JAX JIT) |
| ODE in JAX | diffrax | jax.experimental.ode | diffrax is mature, well-maintained, more solvers |
| Data format | .npz | .pt / HDF5 | .npz is framework-neutral; .pt needs PyTorch |
| Diagnostics | arviz 0.22 | custom | arviz is the standard; reinventing it wastes time |
| Cross-PPL reference | Write own NumPyro models | DCM_PPLs | DCM_PPLs is ERP, not fMRI |
| Structured guide | AutoStructured | Custom Pyro guide | AutoStructured covers block-diagonal natively |
| Tighter ELBO | RenyiELBO(alpha=0.5) | IWAE | RenyiELBO generalizes IWAE (alpha=0) |

---

## 8. Risk Assessment

### Low Risk
- **Pyro guide variants:** Already shipped, well-documented, drop-in replacements
- **ArviZ integration:** Mature library, clear API for from_dict/from_numpyro
- **Predictive API:** Already available, straightforward usage
- **.npz interchange:** Zero-dependency, universal format

### Medium Risk
- **NumPyro on Windows:** JAX CPU works on Windows but GPU requires WSL2.
  CPU-only is sufficient for benchmarking (NUTS is sequential anyway).
- **Diffrax + NumPyro NUTS performance:** ODE-based models can be very slow
  under NUTS. Mitigation: start with spectral DCM (no ODE), use fixed-step
  solvers, limit warmup/samples for initial runs.
- **NumPyro model parity:** Rewriting Pyro models in NumPyro requires careful
  verification that priors, transforms, and forward models match exactly.

### High Risk
- **None identified.** All components are established, stable libraries. The
  main effort is engineering (writing NumPyro model wrappers and benchmark
  runners), not fighting library limitations.

---

## Sources

### Pyro (HIGH confidence)
- [Pyro Autoguide documentation](https://docs.pyro.ai/en/stable/infer.autoguide.html)
- [Pyro SVI / ELBO documentation](https://docs.pyro.ai/en/stable/inference_algos.html)
- [Pyro AutoStructured source](https://github.com/pyro-ppl/pyro/blob/dev/pyro/infer/autoguide/structured.py)
- [Pyro AutoGaussian source](https://github.com/pyro-ppl/pyro/blob/dev/pyro/infer/autoguide/gaussian.py)
- [Pyro RenyiELBO source](https://github.com/pyro-ppl/pyro/blob/dev/pyro/infer/renyi_elbo.py)
- [Pyro Predictive tutorial](https://pyro.ai/examples/predictive_deterministic.html)
- [Pyro prior predictive checks](https://pyro.ai/examples/prior_predictive.html)
- [pyro-ppl 1.9.1 on PyPI](https://pypi.org/project/pyro-ppl/) (released 2024-06-02)

### NumPyro (HIGH confidence)
- [NumPyro autoguide documentation](https://num.pyro.ai/en/stable/autoguide.html)
- [NumPyro SVI documentation](https://num.pyro.ai/en/stable/svi.html)
- [NumPyro MCMC documentation](https://num.pyro.ai/en/stable/mcmc.html)
- [numpyro 0.20.1 on PyPI](https://pypi.org/project/numpyro/) (released 2026-03-25)

### DCM_PPLs (HIGH confidence -- verified it's ERP, not fMRI)
- [ins-amu/DCM_PPLs GitHub](https://github.com/ins-amu/DCM_PPLs)
- Baldy et al. (2025). Dynamic causal modelling in probabilistic programming languages. J. R. Soc. Interface.

### ArviZ (HIGH confidence)
- [ArviZ from_numpyro](https://python.arviz.org/en/stable/api/generated/arviz.from_numpyro.html)
- [ArviZ ESS documentation](https://python.arviz.org/en/stable/api/generated/arviz.ess.html)
- [ArviZ creating InferenceData](https://python.arviz.org/en/stable/getting_started/CreatingInferenceData.html)
- [arviz 0.22.0 on PyPI](https://pypi.org/project/arviz/) (Python >= 3.10)

### Diffrax (HIGH confidence)
- [Diffrax documentation](https://docs.kidger.site/diffrax/)
- [diffrax 0.7.2 on PyPI](https://pypi.org/project/diffrax/)

### NUTS + ODE performance (MEDIUM confidence)
- [Pyro forum: NUTS with Diffrax ODE models](https://forum.pyro.ai/t/seeking-advice-on-mcmc-sampling-for-an-ode-model-solved-with-diffrax/8646)
- [Diffrax issue #338: slow NUTS on GPU](https://github.com/patrick-kidger/diffrax/issues/338)

### posteriordb (MEDIUM confidence)
- [posteriordb GitHub](https://github.com/stan-dev/posteriordb)
- Magnusson et al. (2025). posteriordb: Testing, Benchmarking and Developing Bayesian Inference Algorithms. AISTATS.
