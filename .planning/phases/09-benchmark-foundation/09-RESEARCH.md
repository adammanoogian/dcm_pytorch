# Phase 9: Benchmark Foundation - Research

**Researched:** 2026-04-07
**Domain:** Benchmark infrastructure extension (shared fixtures, config, metrics)
**Confidence:** HIGH

## Summary

Phase 9 extends the existing v0.1.0 benchmark infrastructure (7 runners, BenchmarkConfig,
consolidated metrics) with three capabilities: (1) shared `.npz` fixture generation so all
runners operate on bit-identical synthetic data, (2) extended BenchmarkConfig with
`guide_type`, `n_regions_list`, `elbo_type`, and `fixtures_dir` fields, and (3) a
scientifically correct amortization gap metric using real ELBO evaluation instead of the
current RMSE-ratio proxy.

The existing codebase is well-structured for extension. All three simulators
(`simulate_task_dcm`, `simulate_spectral_dcm`, `simulate_rdcm`) produce dict outputs with
standard fields. The `RUNNER_REGISTRY` pattern cleanly dispatches `(variant, method)` pairs
to runner functions. The `BenchmarkConfig` dataclass uses sensible defaults and has
`quick_config`/`full_config` factory methods. No new external dependencies are needed.

**Primary recommendation:** Build `generate_fixtures.py` around the existing simulators,
extend `BenchmarkConfig` with dataclass field defaults that preserve v0.1.0 behavior, and
replace the RMSE-ratio amortization gap proxy with `Trace_ELBO().loss(model, guide, *args)`.

## Standard Stack

No new dependencies required for Phase 9. All work uses existing infrastructure.

### Core (already installed)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| numpy | >=1.24 | `.npz` fixture format (cross-framework interchange) | Common denominator for PyTorch/JAX arrays |
| torch | >=2.0 | Tensor computations, simulator execution | Already used by all simulators |
| pyro-ppl | >=1.9 | `Trace_ELBO().loss()` for ELBO evaluation | Already used for SVI inference |

### Supporting (already installed)
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| dataclasses | stdlib | BenchmarkConfig extension | Already used |
| json | stdlib | manifest.json for fixture metadata | Already used for results |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `.npz` | HDF5/h5py | Overkill; adds dependency; `.npz` is simpler and both PyTorch/JAX read it natively |
| `.npz` | `.pt` files | JAX cannot read `.pt` without torch; violates cross-backend goal |
| `.npz` | pickle | Not cross-framework safe; security concerns |

## Architecture Patterns

### BENCH-01: Fixture Generation

#### Simulator Output Fields (What Each `.npz` Must Contain)

**Task DCM fixture** (from `simulate_task_dcm` return dict):
```
A_true          # (N, N) float64 -- parameterized A matrix (NOT A_free)
C               # (N, M) float64 -- driving input weights
bold            # (T_TR, N) float64 -- noisy BOLD (the "observed" data)
bold_clean      # (T_TR, N) float64 -- noise-free BOLD for reference
stimulus_times  # (K,) float64 -- stimulus onset times
stimulus_values # (K, M) float64 -- stimulus values at each onset
TR              # scalar float64
SNR             # scalar float64
duration        # scalar float64
seed            # scalar int
```

**Spectral DCM fixture** (from `simulate_spectral_dcm` return dict):
```
A_true          # (N, N) float64 -- parameterized A matrix
csd_real        # (F, N, N) float64 -- real part of CSD
csd_imag        # (F, N, N) float64 -- imaginary part of CSD
freqs           # (F,) float64 -- frequency vector
noise_a         # (2, N) float64 -- neuronal noise params
noise_b         # (2, 1) float64 -- global observation noise
noise_c         # (2, N) float64 -- regional observation noise
TR              # scalar float64
n_freqs         # scalar int
seed            # scalar int
```

Note: `.npz` cannot store complex128 directly. Must split into real and
imaginary parts. Runners reconstruct via `torch.complex(real, imag)`.

**rDCM fixture** (from `generate_bold` + `create_regressors`):
```
A_true          # (N, N) float64 -- ground truth A
C_true          # (N, M) float64 -- ground truth C
a_mask          # (N, N) float64 -- architecture mask
c_mask          # (N, M) float64 -- input mask
y               # (N_y, N) float64 -- noisy BOLD
y_clean         # (N_y, N) float64 -- clean BOLD
X               # (N_eff, K, N) complex128 -> stored as X_real, X_imag
Y               # (N_eff, N) complex128 -> stored as Y_real, Y_imag
u               # (N_u, M) float64 -- stimulus
u_dt            # scalar float64
y_dt            # scalar float64
SNR             # scalar float64
seed            # scalar int
```

Note: rDCM regressors X and Y are complex (from FFT). Must split like CSD.
Alternative: only store the BOLD and stimulus, letting each runner call
`create_regressors` itself. This is simpler and avoids storing large complex
arrays. **Recommended: store only (A_true, C_true, a_mask, c_mask, y, y_clean, u,
u_dt, y_dt, SNR, seed) and let runners call `create_regressors` as needed.**

#### Directory Structure
```
benchmarks/
  fixtures/                     # gitignored (generated artifacts)
    task_3region/               # variant_Nregion
      dataset_000.npz
      dataset_001.npz
      ...
      manifest.json             # metadata: n_datasets, seed_base, params
    task_5region/
    task_10region/
    spectral_3region/
    spectral_5region/
    spectral_10region/
    rdcm_3region/
    rdcm_5region/
    rdcm_10region/
```

#### Script API
```python
# benchmarks/generate_fixtures.py
def generate_task_fixtures(
    n_regions: int, n_datasets: int, seed: int, output_dir: str,
) -> None: ...

def generate_spectral_fixtures(
    n_regions: int, n_datasets: int, seed: int, output_dir: str,
) -> None: ...

def generate_rdcm_fixtures(
    n_regions: int, n_datasets: int, seed: int, output_dir: str,
) -> None: ...

# CLI entry:
# python benchmarks/generate_fixtures.py [--variant task|spectral|rdcm|all]
#     [--n-regions 3,5,10] [--n-datasets 50] [--seed 42]
#     [--output-dir benchmarks/fixtures]
```

#### Fixture Loading Helper
```python
def load_fixture(
    variant: str,
    n_regions: int,
    index: int,
    fixtures_dir: str = "benchmarks/fixtures",
) -> dict[str, torch.Tensor]:
    """Load a single .npz fixture as a dict of torch tensors.

    Parameters
    ----------
    variant : str
        One of "task", "spectral", "rdcm".
    n_regions : int
        Number of regions (3, 5, or 10).
    index : int
        Dataset index (0-based).
    fixtures_dir : str
        Root fixtures directory.

    Returns
    -------
    dict
        Tensors keyed by field name (A_true, bold/csd, etc).
    """
```

This helper should live in `benchmarks/fixtures.py` (new module) and be
imported by runners when `config.fixtures_dir` is set.

### BENCH-02: Extended BenchmarkConfig

#### Current Fields (preserve all defaults)
```python
@dataclass
class BenchmarkConfig:
    variant: str
    method: str
    n_datasets: int = 20
    n_regions: int = 3
    n_svi_steps: int = 3000
    seed: int = 42
    quick: bool = False
    output_dir: str = "benchmarks/results"
    save_figures: bool = True
    figure_dir: str = "figures"
```

#### New Fields (with backward-compatible defaults)
```python
    # Phase 9 additions:
    guide_type: str = "mean_field"
    n_regions_list: list[int] = field(default_factory=lambda: [3])
    elbo_type: str = "trace"
    fixtures_dir: str | None = None  # None = inline generation (v0.1.0 behavior)
```

Key design decisions:
- `fixtures_dir: str | None = None` -- `None` preserves inline data generation
  (v0.1.0 behavior). When set, runners load from fixtures instead.
- `n_regions_list` defaults to `[3]` (single element) preserving v0.1.0
  single-size behavior. Runners currently use `config.n_regions`; they should
  continue to do so when `n_regions_list` has one element.
- `guide_type = "mean_field"` matches current `create_guide` which only
  returns `AutoNormal`. Phase 10 will extend the factory.
- `elbo_type = "trace"` matches current `Trace_ELBO`. Phase 10 adds
  `"mean_field"` and `"renyi"`.
- `quick_config` and `full_config` class methods must be updated to
  accept and pass through the new fields.

#### CLI Extension
The CLI in `run_all_benchmarks.py` needs new flags:
```
--fixtures-dir PATH      Load data from fixtures instead of inline generation
--guide-type TYPE         Guide type (default: mean_field)
--n-regions LIST          Comma-separated region sizes (default: 3)
```

### BENCH-03: Amortization Gap Fix

#### What Is Wrong (P4 from PITFALLS.md)

Both `task_amortized.py` (line 403-413) and `spectral_amortized.py` (line 385-395)
compute the amortization gap using this incorrect pattern:

```python
# WRONG: fabricates an "amortized ELBO" from RMSE ratio
gap = compute_amortization_gap(
    svi_result["final_loss"],
    svi_result["final_loss"] * (
        1.0 + max(0.0, rmse_amort / rmse_svi - 1.0)
    ),
)
```

This multiplies the SVI ELBO by an RMSE ratio to fabricate a fake "amortized ELBO."
The resulting "amortization gap" is scientifically meaningless -- it correlates with
RMSE differences, not actual ELBO differences.

#### What the Fix Requires

Replace with actual ELBO evaluation:

```python
# CORRECT: evaluate real ELBO for both guides
from pyro.infer import Trace_ELBO

elbo_fn = Trace_ELBO(num_particles=5)

# Amortized ELBO: evaluate flow guide against wrapper model
# The flow guide.forward() takes (observed_data, *args) and samples _latent
# The wrapper model samples _latent from N(0,I) and runs forward model
amortized_elbo = elbo_fn.loss(
    amortized_task_dcm_model,  # wrapper model
    guide,                      # AmortizedFlowGuide
    bold, stimulus, a_mask, c_mask, t_eval, TR, dt, packer,
)

# SVI ELBO: evaluate per-subject AutoNormal against standard model
svi_elbo = elbo_fn.loss(
    task_dcm_model,            # standard model
    svi_guide,                 # AutoNormal guide (already trained)
    bold, stimulus, a_mask, c_mask, t_eval, TR, dt,
)

gap = compute_amortization_gap(svi_elbo, amortized_elbo)
```

#### Pyro API for ELBO Evaluation (Verified)

`Trace_ELBO` has two relevant methods:
- `loss(model, guide, *args, **kwargs)` -> `float`: Returns the ELBO loss
  (= -ELBO) as a Python float. Uses `num_particles` for MC estimation.
  **Use this for evaluation.**
- `differentiable_loss(model, guide, *args, **kwargs)` -> `torch.Tensor`:
  Returns a differentiable tensor for gradient computation. Use this only
  when you need gradients (SVI training).

For evaluation (no gradients needed), `loss()` is simpler and returns a float.
The ROADMAP says `differentiable_loss` but `loss()` is equivalent for
evaluation and avoids gradient graph construction. Both are valid; `loss()` is
more efficient when we only need the scalar value.

Use `num_particles=5` or higher to reduce MC variance in the ELBO estimate.
Multiple evaluations can be averaged for even more stable estimates.

#### Key Constraint: Different Model Signatures

The amortized guide uses `amortized_task_dcm_model` (wrapper with `_latent`
site and `packer` argument), while the SVI guide uses `task_dcm_model` (with
named sites `A_free`, `C`, `noise_prec`). These models have different argument
signatures:

```python
# Standard model args (SVI):
task_dcm_model(bold, stimulus, a_mask, c_mask, t_eval, TR, dt)

# Wrapper model args (amortized):
amortized_task_dcm_model(bold, stimulus, a_mask, c_mask, t_eval, TR, dt, packer)
```

This means the ELBO values are not directly on the same scale because
the models have different priors (N(0, 1/64) per site vs N(0, I) on packed
vector). However, this is the standard way to measure the total inference gap.
The PURE amortization gap would require fine-tuning the flow on the test
subject (deferred to v0.3+). For Phase 9, computing real ELBO for both is
sufficient and replaces the scientifically invalid RMSE proxy.

#### Same Pattern for Spectral

```python
# Spectral amortized model args:
amortized_spectral_dcm_model(csd, freqs, a_mask, packer)

# Spectral standard model args:
spectral_dcm_model(csd, freqs, a_mask, N)
```

The spectral amortized runner already has the `packer` and `freqs` in scope,
so the fix is mechanical.

### Pattern: Runner Fixture Loading

Runners should detect `config.fixtures_dir` and branch:

```python
def run_task_svi(config: BenchmarkConfig) -> dict[str, Any]:
    # ...
    for i in range(config.n_datasets):
        if config.fixtures_dir is not None:
            data = load_fixture("task", config.n_regions, i, config.fixtures_dir)
            A_true = data["A_true"]
            bold = data["bold"]
            stimulus = {
                "times": data["stimulus_times"],
                "values": data["stimulus_values"],
            }
            # ... etc
        else:
            # Existing inline generation (v0.1.0 path)
            A_true = make_random_stable_A(N, density=0.5, seed=seed_i)
            # ...
```

### Anti-Patterns to Avoid

- **Breaking v0.1.0 behavior:** All new BenchmarkConfig fields MUST have defaults
  that produce identical behavior to v0.1.0 when not specified. Test by running
  existing benchmark CLI commands and verifying identical JSON output.
- **Storing complex tensors directly in .npz:** NumPy `.npz` handles complex
  arrays but they may cause issues with `torch.from_numpy`. Always split to
  real/imag.
- **Aggregating metrics across DCM variants:** Never compute mean RMSE across
  task + spectral + rDCM. Each variant has fundamentally different metrics
  (task: parameterized A space; spectral: A_free space; rDCM: also has F1).
- **Generating fixtures with different seeds than inline generation:** If
  fixtures use `seed + 1000 + i` but inline uses `seed + i`, the comparison
  to v0.1.0 breaks. Match the exact seeding pattern currently used per runner.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| ELBO evaluation | Manual log_prob_sum on traces | `Trace_ELBO().loss(model, guide, *args)` | Pyro handles trace matching, particle averaging, enumeration |
| Cross-framework data exchange | Custom serialization | `numpy.savez` / `numpy.load` | Standard, fast, both PyTorch and JAX can read |
| Config with defaults | Nested dicts / argparse only | `dataclasses.dataclass` with `field(default_factory=...)` | Type safety, IDE support, serialization |
| Fixture manifest | Custom metadata format | JSON file with dict | Human-readable, standard |

## Common Pitfalls

### Pitfall 1: PRNG Mismatch Between Inline and Fixture Data
**What goes wrong:** Runners currently set torch seeds in specific patterns
(e.g., `seed + i` for task_svi, `seed + 1000 + i` for task_amortized). If
`generate_fixtures.py` uses a different seeding pattern, fixtures won't match
inline generation, making the "identical results" success criterion impossible.
**Why it happens:** Each runner has its own seeding convention baked into the loop.
**How to avoid:** Document and match the exact seeding pattern per runner in
`generate_fixtures.py`. Use a manifest.json per fixture directory that records
the seed pattern.
**Warning signs:** Results JSON differs between fixture-loaded and inline runs.

### Pitfall 2: Complex Array Storage in .npz
**What goes wrong:** Saving `torch.complex128` tensors to `.npz` requires
conversion via `.numpy()`. While NumPy handles complex dtypes, torch's
`torch.from_numpy` may produce unexpected results or errors on some platforms.
**Why it happens:** Cross-library dtype handling edge cases.
**How to avoid:** Always split complex tensors into real and imaginary parts
before saving. Reconstruct with `torch.complex(real, imag)` after loading.
**Warning signs:** TypeError or wrong dtype when loading fixtures.

### Pitfall 3: BenchmarkConfig Backward Compatibility
**What goes wrong:** Adding required fields to BenchmarkConfig or changing
default values breaks all existing runner calls and the `quick_config`/`full_config`
factory methods.
**Why it happens:** Dataclass field ordering matters -- fields with defaults
must come after fields without defaults.
**How to avoid:** All new fields MUST have defaults. Place them after existing
fields. Update `quick_config` and `full_config` to pass through new fields.
**Warning signs:** `TypeError: __init__() missing required argument` in existing tests.

### Pitfall 4: Amortized Guide Requires Model Args Including Packer
**What goes wrong:** When computing amortized ELBO via `Trace_ELBO().loss()`,
forgetting to pass the `packer` argument causes the wrapper model to fail.
The wrapper model signature differs from the standard model.
**Why it happens:** The amortized wrapper model has an extra `packer` argument
not present in the standard model.
**How to avoid:** Keep track of which model variant requires which args. The
amortized runners already have all needed objects in scope (guide, packer,
stimulus, masks).
**Warning signs:** `TypeError` about missing `packer` argument during ELBO eval.

### Pitfall 5: ELBO Evaluation on Untrained/Detached Guide
**What goes wrong:** After loading a pre-trained amortized guide, calling
`Trace_ELBO().loss()` may fail if `pyro.module` registrations are stale or
if `pyro.clear_param_store()` was called between training and evaluation.
**Why it happens:** Pyro's module registry and param store are global state.
The amortized guide's `forward()` calls `pyro.module("summary_net", ...)` and
`pyro.module("flow", ...)`, which registers params in the global store.
**How to avoid:** Call `guide.eval()` before ELBO evaluation. Do NOT call
`pyro.clear_param_store()` between loading the guide and evaluating ELBO.
Alternatively, construct a fresh `Trace_ELBO` instance for each evaluation.
**Warning signs:** ELBO returns NaN or very large values after param store clear.

### Pitfall 6: Fixture Generation Takes Too Long for Large Networks
**What goes wrong:** Generating 50 datasets x 10 regions for task DCM involves
50 ODE integrations at dt=0.01 for 300s each, which can take minutes per dataset.
**Why it happens:** Task DCM simulation is compute-intensive for large networks.
**How to avoid:** Use the same reduced simulation parameters as the existing
runners (e.g., `duration=60.0` in quick mode). Document expected generation
times in the script. Consider reducing `dt` for fixtures (0.01 -> 0.05 is
acceptable for benchmarking).
**Warning signs:** `generate_fixtures.py` takes >1 hour for all variants.

## Code Examples

### Loading a fixture and converting to torch tensors
```python
import numpy as np
import torch

data = np.load("benchmarks/fixtures/task_3region/dataset_000.npz")
A_true = torch.from_numpy(data["A_true"])           # (N, N) float64
bold = torch.from_numpy(data["bold"])                 # (T, N) float64
stim_times = torch.from_numpy(data["stimulus_times"]) # (K,) float64
stim_values = torch.from_numpy(data["stimulus_values"]) # (K, M) float64
stimulus = {"times": stim_times, "values": stim_values}
```

### Evaluating real ELBO for amortization gap
```python
from pyro.infer import Trace_ELBO

elbo_fn = Trace_ELBO(num_particles=5)

# After training/loading the amortized guide:
guide.eval()
amortized_elbo_loss = elbo_fn.loss(
    amortized_task_dcm_model,
    guide,
    bold, stimulus, a_mask, c_mask, t_eval, TR, dt, packer,
)

# After running per-subject SVI:
svi_elbo_loss = elbo_fn.loss(
    task_dcm_model,
    svi_guide,
    bold, stimulus, a_mask, c_mask, t_eval, TR, dt,
)

# compute_amortization_gap expects (svi_loss, amortized_loss)
# Both are ELBO losses (= -ELBO), so lower is better
gap = compute_amortization_gap(svi_elbo_loss, amortized_elbo_loss)
```

### Extended BenchmarkConfig with backward-compatible defaults
```python
from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class BenchmarkConfig:
    variant: str
    method: str
    n_datasets: int = 20
    n_regions: int = 3
    n_svi_steps: int = 3000
    seed: int = 42
    quick: bool = False
    output_dir: str = "benchmarks/results"
    save_figures: bool = True
    figure_dir: str = "figures"
    # Phase 9 additions:
    guide_type: str = "mean_field"
    n_regions_list: list[int] = field(default_factory=lambda: [3])
    elbo_type: str = "trace"
    fixtures_dir: str | None = None
```

### Saving a task DCM fixture
```python
import numpy as np

# After simulate_task_dcm returns result dict:
np.savez(
    filepath,
    A_true=result["params"]["A"].numpy(),
    C=result["params"]["C"].numpy(),
    bold=result["bold"].numpy(),
    bold_clean=result["bold_clean"].numpy(),
    stimulus_times=stimulus["times"].numpy(),
    stimulus_values=stimulus["values"].numpy(),
    TR=np.float64(TR),
    SNR=np.float64(SNR),
    duration=np.float64(duration),
    seed=np.int64(seed),
)
```

### Saving a spectral DCM fixture (complex CSD split)
```python
csd = result["csd"]  # complex128
np.savez(
    filepath,
    A_true=A.numpy(),
    csd_real=csd.real.numpy(),
    csd_imag=csd.imag.numpy(),
    freqs=result["freqs"].numpy(),
    noise_a=noise_a.numpy(),
    noise_b=noise_b.numpy(),
    noise_c=noise_c.numpy(),
    TR=np.float64(TR),
    n_freqs=np.int64(n_freqs),
    seed=np.int64(seed),
)
```

## State of the Art

| Old Approach (v0.1.0) | New Approach (Phase 9) | Why |
|----------------------|----------------------|-----|
| Each runner generates its own data inline | Shared `.npz` fixtures loaded by all runners | Bit-identical data ensures fair cross-method comparison |
| Single `n_regions: int = 3` | `n_regions_list: list[int] = [3]` | Enables multi-size benchmarks without CLI hacks |
| `create_guide` returns only AutoNormal | `guide_type` field in config (Phase 10 extends factory) | Config-driven guide selection for systematic comparison |
| RMSE-ratio proxy for amortization gap | `Trace_ELBO().loss()` for real ELBO evaluation | Scientifically valid metric instead of fabricated proxy |
| Hardcoded Trace_ELBO | `elbo_type` field for future ELBO variant comparison | Prepares for TraceMeanField_ELBO, RenyiELBO (Phase 10) |

## Open Questions

1. **Seeding consistency across runners**
   - What we know: task_svi uses `seed + i`, task_amortized uses `seed + 1000 + i`,
     spectral_svi uses `seed + i`, spectral_amortized uses `seed + 5000` base,
     rdcm_vb uses `seed + i` with `seed + 10000` for BOLD noise.
   - What's unclear: Should fixtures use a single canonical seeding pattern, or
     should each variant/runner have its own? If we unify seeds, fixture-loaded
     results won't match v0.1.0 inline results for amortized runners.
   - Recommendation: Use a single canonical pattern (`seed + i`) for fixtures.
     Accept that fixture-loaded results won't exactly match amortized runners'
     inline results (which used different seed offsets). The goal is
     cross-runner comparability, not backward-identical reproduction.

2. **rDCM fixture complexity: store regressors or recompute?**
   - What we know: rDCM regressors (X, Y) are complex-valued and large. Storing
     them in `.npz` requires real/imag splitting. Alternatively, store only BOLD
     and stimulus and let runners call `create_regressors`.
   - What's unclear: Is regressor computation deterministic enough across
     platforms? (FFT implementations vary.)
   - Recommendation: Store BOLD + stimulus + HRF only. Let runners call
     `create_regressors`. The FFT is deterministic within a single platform
     (torch). Cross-platform reproducibility (NumPyro) is deferred to v0.3+.

3. **ELBO scale difference between wrapper and standard models**
   - What we know: The amortized wrapper model uses `N(0, I)` prior on packed
     latent (via standardization), while the standard model uses per-site priors
     (e.g., `N(0, 1/64)` on `A_free`). These produce different log-prob scales.
   - What's unclear: Is the ELBO gap meaningfully interpretable as an
     "amortization gap" when the models differ?
   - Recommendation: For Phase 9, compute and report both ELBOs honestly.
     Label as "total inference gap" not "pure amortization gap." The pure
     amortization gap (fine-tuning approach) is deferred to v0.3+ per PITFALLS.md P4.

## Sources

### Primary (HIGH confidence)
- Pyro `Trace_ELBO.loss()` source (verified via `inspect.getsource`): returns
  `float`, signature `(self, model, guide, *args, **kwargs)`, uses
  `model_trace.log_prob_sum() - guide_trace.log_prob_sum()` with
  `num_particles` averaging
- Pyro `Trace_ELBO.differentiable_loss()` source (verified): returns
  differentiable tensor, same signature
- Existing codebase files (all read directly):
  - `benchmarks/config.py` -- BenchmarkConfig dataclass (50 lines)
  - `benchmarks/metrics.py` -- 5 metric functions including `compute_amortization_gap`
  - `benchmarks/run_all_benchmarks.py` -- CLI and RUNNER_REGISTRY dispatch
  - `benchmarks/runners/task_amortized.py` -- RMSE-ratio proxy at lines 403-413
  - `benchmarks/runners/spectral_amortized.py` -- same proxy at lines 385-395
  - `benchmarks/runners/task_svi.py` -- inline data generation pattern
  - `benchmarks/runners/rdcm_vb.py` -- rDCM-specific constants and data generation
  - `src/pyro_dcm/simulators/task_simulator.py` -- output dict fields
  - `src/pyro_dcm/simulators/spectral_simulator.py` -- output dict fields
  - `src/pyro_dcm/simulators/rdcm_simulator.py` -- output dict fields
  - `src/pyro_dcm/models/guides.py` -- `create_guide`, `run_svi` implementations
  - `src/pyro_dcm/models/amortized_wrappers.py` -- wrapper model signatures
  - `src/pyro_dcm/guides/amortized_flow.py` -- AmortizedFlowGuide.forward()

### Secondary (MEDIUM confidence)
- `.planning/research/v0.2.0/ARCHITECTURE.md` -- fixture format, BenchmarkConfig
  extension design, guide factory pattern
- `.planning/research/v0.2.0/PITFALLS.md` -- P3 (cross-framework fairness),
  P4 (amortization gap conflation with approximation gap)
- `.planning/research/v0.2.0/FEATURES.md` -- TS-6 amortization gap characterization
- [Pyro SVI documentation](https://docs.pyro.ai/en/dev/inference_algos.html)
- [Pyro custom objectives tutorial](https://pyro.ai/examples/custom_objectives.html)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- no new dependencies, verified existing libraries
- Architecture (fixtures): HIGH -- `.npz` format verified, simulator outputs inspected
- Architecture (config): HIGH -- dataclass extension is mechanical
- Architecture (ELBO fix): HIGH -- Pyro API verified via source inspection
- Pitfalls: HIGH -- derived from codebase inspection and v0.2.0 research

**Research date:** 2026-04-07
**Valid until:** 2026-05-07 (stable domain; no fast-moving dependencies)
