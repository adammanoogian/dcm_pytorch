# Phase 4: Pyro Generative Models + Baseline Guides - Research

**Researched:** 2026-03-27
**Domain:** Pyro probabilistic programming for ODE-based and frequency-domain DCM models
**Confidence:** MEDIUM (verified patterns from Pyro docs + codebase analysis; no Context7 available)

## Summary

This phase wires three existing forward models (task DCM with ODE integration, spectral DCM with frequency-domain CSD, regression DCM with analytic frequency-domain likelihood) into Pyro generative models with proper plate structure and priors, plus implements baseline mean-field Gaussian guides.

The standard approach is: (1) define a `model()` function that uses `pyro.sample` for latent variables and `pyro.sample(..., obs=)` for the likelihood, calling existing forward model functions as deterministic computation between the two; (2) use `AutoNormal` as the baseline mean-field guide; (3) train with `SVI` + `ClippedAdam` + `Trace_ELBO`. The key challenges are: handling ODE integration inside the Pyro trace (gradients must flow through torchdiffeq), handling complex-valued CSD data in the spectral likelihood (Pyro distributions do not support complex tensors natively), and applying structural masks to enforce which A/C connections exist.

**Primary recommendation:** Use `AutoNormal` (not `AutoDiagonalNormal`) as the baseline guide for all three variants. Decompose complex CSD into real/imaginary stacked vectors for the spectral likelihood. Use `pyro.deterministic` for the A matrix transform and all forward model outputs. Apply structural masking by zeroing out `A_free` entries for absent connections before the `parameterize_A` transform.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pyro-ppl | >=1.9 | Probabilistic programming, SVI, guides | Already in project deps; native PyTorch integration |
| torch | >=2.0 | Tensor computation, autograd | Already in project deps |
| torchdiffeq | latest | ODE integration with differentiable solvers | Already in project deps; used by Phase 1 |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pyro.optim.ClippedAdam | (built-in) | Optimizer with gradient clipping + LR decay | Default optimizer for all SVI training |
| pyro.infer.Trace_ELBO | (built-in) | ELBO loss computation | Default loss for all variants |
| pyro.infer.autoguide.AutoNormal | (built-in) | Mean-field Gaussian guide | Baseline guide for all DCM variants |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| AutoNormal | AutoDiagonalNormal | AutoNormal is newer, has better plate support and site names; AutoDiagonalNormal is deprecated-ish |
| AutoNormal | Manual diagonal guide | More control, but boilerplate; save for Phase 7 amortized guides |
| Trace_ELBO | TraceMeanField_ELBO | Analytic KL when possible; use if Normal-Normal pairs dominate |
| ClippedAdam | Adam | ClippedAdam adds gradient clipping and LR decay natively; strictly better |

## Architecture Patterns

### Recommended Module Structure
```
src/pyro_dcm/
├── models/                    # NEW: Pyro generative models
│   ├── __init__.py
│   ├── task_dcm_model.py     # PROB-01: Task DCM Pyro model
│   ├── spectral_dcm_model.py # PROB-02: Spectral DCM Pyro model
│   ├── rdcm_model.py         # PROB-03: Regression DCM Pyro model
│   └── guides.py             # PROB-04: Guide factory + SVI runner
├── forward_models/            # Existing Phase 1-3 code (unchanged)
├── simulators/                # Existing simulators (unchanged)
└── utils/                     # Existing utilities (unchanged)
```

### Pattern 1: Task DCM Pyro Model (ODE-based)

**What:** Pyro model that samples A_free and C, applies `parameterize_A`, builds the `CoupledDCMSystem`, runs ODE integration via `integrate_ode`, computes BOLD signal, and conditions on observed BOLD data.

**When to use:** Task-based fMRI with BOLD time series data.

**Key pattern:**
```python
def task_dcm_model(
    observed_bold: torch.Tensor,    # (T, N)
    stimulus: PiecewiseConstantInput,
    a_mask: torch.Tensor,           # (N, N) binary
    c_mask: torch.Tensor,           # (N, M) binary
    t_eval: torch.Tensor,           # (T_fine,) fine time grid
    TR: float = 2.0,
    dt: float = 0.01,
):
    N, M = c_mask.shape
    T = observed_bold.shape[0]

    # --- Sample latent variables ---
    # A_free: element-wise Normal, masked
    A_free_prior = dist.Normal(
        torch.zeros(N, N), (1.0 / 64.0) ** 0.5 * torch.ones(N, N)
    ).to_event(2)
    A_free = pyro.sample("A_free", A_free_prior)
    A_free = A_free * a_mask  # zero out absent connections

    # C: element-wise Normal, masked
    C_prior = dist.Normal(
        torch.zeros(N, M), torch.ones(N, M)
    ).to_event(2)
    C = pyro.sample("C", C_prior)
    C = C * c_mask  # zero out absent inputs

    # --- Deterministic forward model ---
    A = pyro.deterministic("A", parameterize_A(A_free))

    system = CoupledDCMSystem(A, C, stimulus)
    y0 = make_initial_state(N)
    solution = integrate_ode(system, y0, t_eval, method="dopri5",
                            grid_points=stimulus.grid_points)

    # Extract hemodynamic states, compute BOLD
    v = torch.exp(solution[:, 3*N:4*N])
    q = torch.exp(solution[:, 4*N:5*N])
    bold_fine = bold_signal(v, q)

    # Downsample to TR
    step = round(TR / dt)
    indices = torch.arange(0, len(t_eval), step)
    predicted_bold = bold_fine[indices[:T]]
    pyro.deterministic("predicted_bold", predicted_bold)

    # --- Noise precision ---
    noise_prec = pyro.sample(
        "noise_prec",
        dist.Gamma(torch.tensor(1.0), torch.tensor(1.0))
    )
    noise_std = (1.0 / noise_prec).sqrt()

    # --- Likelihood ---
    with pyro.plate("time", T):
        with pyro.plate("region", N):
            pyro.sample("obs", dist.Normal(predicted_bold, noise_std),
                       obs=observed_bold)
```

**Critical notes:**
- `to_event(2)` on the matrix priors declares both dimensions as event (dependent), so log_prob sums over the full matrix.
- Masking via multiplication (`A_free * a_mask`) is simpler and more robust than trying to sample only non-zero elements.
- ODE integration happens as pure PyTorch computation between `pyro.sample` calls -- gradients flow through torchdiffeq automatically.
- The time and region plates declare the observations as independent given the predicted BOLD.

### Pattern 2: Spectral DCM Pyro Model (Frequency-domain)

**What:** Pyro model that samples A_free and noise parameters (a, b, c), computes predicted CSD via `spectral_dcm_forward`, decomposes complex CSD into real/imaginary vector, and conditions on observed CSD.

**Key insight -- complex likelihood workaround:**
Pyro's `dist.Normal` does not support `complex128` tensors. The standard workaround is:
1. Flatten the predicted CSD matrix `(F, N, N)` to a vector.
2. Split into real and imaginary parts, stack to get `(2*F*N*N,)` real vector.
3. Do the same for observed CSD.
4. Use a single multivariate-independent Normal likelihood on the stacked real vector.

```python
def spectral_dcm_model(
    observed_csd: torch.Tensor,     # (F, N, N) complex128
    freqs: torch.Tensor,            # (F,) Hz
    a_mask: torch.Tensor,           # (N, N) binary
    N: int,
):
    # Sample A_free
    A_free = pyro.sample("A_free",
        dist.Normal(torch.zeros(N, N), (1/64)**0.5 * torch.ones(N, N))
        .to_event(2))
    A_free = A_free * a_mask

    A = pyro.deterministic("A", parameterize_A(A_free))

    # Sample noise parameters (SPM priors: N(0, 1/64))
    noise_std = (1.0 / 64.0) ** 0.5
    a = pyro.sample("noise_a",
        dist.Normal(torch.zeros(2, N), noise_std).to_event(2))
    b = pyro.sample("noise_b",
        dist.Normal(torch.zeros(2, 1), noise_std).to_event(2))
    c = pyro.sample("noise_c",
        dist.Normal(torch.zeros(2, N), noise_std).to_event(2))

    # Forward model (all differentiable PyTorch ops)
    predicted_csd = spectral_dcm_forward(A, freqs, a, b, c)
    pyro.deterministic("predicted_csd", predicted_csd)

    # Decompose complex to real for likelihood
    pred_flat = torch.cat([predicted_csd.real.reshape(-1),
                           predicted_csd.imag.reshape(-1)])
    obs_flat = torch.cat([observed_csd.real.reshape(-1),
                          observed_csd.imag.reshape(-1)])

    # Observation noise on CSD
    csd_noise_scale = pyro.sample("csd_noise_scale",
        dist.HalfCauchy(torch.tensor(1.0)))

    pyro.sample("obs_csd",
        dist.Normal(pred_flat, csd_noise_scale).to_event(1),
        obs=obs_flat)
```

### Pattern 3: Regression DCM Pyro Model (Analytic frequency-domain)

**What:** Pyro model wrapping the rDCM frequency-domain regression likelihood. For each region, samples theta (columns of A and C relevant to that region) and evaluates the frequency-domain Gaussian likelihood from Phase 3.

**Key insight:** The rDCM model is region-wise, so the natural Pyro structure uses a region plate. Each region's parameters are independent given the design matrix.

```python
def rdcm_model(
    Y: torch.Tensor,               # (N_eff, nr) frequency-domain data
    X: torch.Tensor,               # (N_eff, D) design matrix
    a_mask: torch.Tensor,          # (nr, nr) binary
    c_mask: torch.Tensor,          # (nr, nu) binary
    confound_cols: int = 1,
):
    nr, nu = a_mask.shape[0], c_mask.shape[1]

    with pyro.plate("region", nr):
        # Per-region parameter vector (A row + C row + confounds)
        # Build dimension per region based on mask
        # ... sample theta_r, noise_prec_r per region
        # ... compute likelihood per region
```

### Pattern 4: Structural Masking

**What:** Enforce which A/C connections exist by zeroing sampled parameters at absent positions.

**Two approaches:**

**Approach A (Recommended): Post-sample masking**
```python
A_free = pyro.sample("A_free", dist.Normal(0, sigma).expand([N,N]).to_event(2))
A_free = A_free * a_mask  # zeros where mask is 0
```
- Simple, works with AutoNormal out of the box.
- The guide will still sample all N*N parameters, but the masked ones have zero effect on the likelihood. Over training, their posteriors will collapse to the prior.
- Slightly wasteful but robust.

**Approach B: Prior with zero variance at absent connections**
```python
prior_std = a_mask * (1/64)**0.5  # 0 std where absent
A_free = pyro.sample("A_free", dist.Normal(0, prior_std + 1e-10).to_event(2))
```
- Tight prior (near-delta) at absent connections forces them to zero.
- Risk: very small std can cause numerical issues in guide optimization.
- Not recommended as primary approach.

**Verdict:** Use Approach A (post-sample masking). It is simpler, numerically safer, and the planner should implement this.

### Pattern 5: SVI Training Loop

**What:** Standard SVI training with convergence monitoring.

```python
def run_svi(model, guide, data_args, num_steps=2000, lr=0.01,
            clip_norm=10.0, num_particles=1):
    pyro.clear_param_store()
    optimizer = pyro.optim.ClippedAdam({
        "lr": lr,
        "betas": (0.9, 0.999),
        "clip_norm": clip_norm,
        "lrd": (0.01) ** (1.0 / num_steps),  # decay to 1% of initial
    })
    elbo = Trace_ELBO(num_particles=num_particles,
                      vectorize_particles=(num_particles > 1))
    svi = SVI(model, guide, optimizer, loss=elbo)

    losses = []
    for step in range(num_steps):
        loss = svi.step(*data_args)
        losses.append(loss)

        if torch.isnan(torch.tensor(loss)):
            raise RuntimeError(f"NaN loss at step {step}")

    return losses
```

### Anti-Patterns to Avoid

- **Sampling hemodynamic parameters:** Decision is to FIX at SPM defaults. Do NOT add pyro.sample for kappa, gamma, tau, alpha, E0. This reduces parameter space and prevents identifiability issues.
- **Using adjoint method for ODE in SVI:** The adjoint method (`odeint_adjoint`) saves memory but can have gradient accuracy issues with SVI's stochastic optimization. Use standard `odeint` (forward mode) for reliability. Switch to adjoint only if memory becomes a bottleneck.
- **Sampling A as a full matrix then masking:** Do NOT sample A directly. Always sample A_free then apply `parameterize_A`. The transform guarantees negative self-connections.
- **Using `pyro.plate` around the ODE integration:** The time points in an ODE trajectory are NOT independent. Only use `pyro.plate` for the final likelihood over independent observations (time points are approximately independent for BOLD given the predicted signal).
- **Complex-valued pyro.sample:** Never pass complex tensors to `pyro.sample` or `dist.Normal`. Always decompose to real/imaginary first.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Mean-field guide | Manual loc/scale params per site | `AutoNormal` | Handles plates, constraints, init automatically |
| Gradient clipping | Manual `torch.nn.utils.clip_grad_norm_` | `ClippedAdam(clip_norm=...)` | Built into optimizer, cleaner |
| LR scheduling | Manual decay loop | `ClippedAdam(lrd=...)` | Per-step multiplicative decay built-in |
| Distribution constraints | Manual exp/softplus transforms | `pyro.param(..., constraint=constraints.positive)` | Pyro's constraint system handles bijections |
| Parameter store management | Manual dict tracking | `pyro.clear_param_store()` + `pyro.get_param_store()` | Pyro's global param store is the standard |

**Key insight:** AutoNormal handles the entire guide creation including constraint transforms, plate structure, and initialization. Writing a manual diagonal Gaussian guide is only needed for the amortized guide in Phase 7.

## Common Pitfalls

### Pitfall 1: NaN Gradients from ODE Integration
**What goes wrong:** ODE solver produces NaN states when A matrix eigenvalues have large positive real parts, causing the exponential growth in neural dynamics to blow up.
**Why it happens:** A_free samples from the prior can produce A matrices with large positive off-diagonal entries that dominate the negative self-connections.
**How to avoid:** The `parameterize_A` function guarantees negative diagonal, but off-diagonal values are unconstrained. The prior N(0, 1/64) has std=0.125, which is tight enough to prevent blow-up for reasonable network sizes (2-10 regions). If NaN appears during SVI, reduce `init_scale` in AutoNormal (default 0.1) to start from tighter guide.
**Warning signs:** Loss jumps to NaN suddenly. Check `pyro.get_param_store()` for any NaN parameters.

### Pitfall 2: ELBO Not Decreasing (Stuck Optimization)
**What goes wrong:** ELBO oscillates without clear downward trend.
**Why it happens:** Learning rate too high for the ODE-based model (gradients through ODE are high-variance), or `num_particles=1` gives too noisy ELBO estimates.
**How to avoid:** Start with `lr=0.01`, use `ClippedAdam` with `clip_norm=10.0`, and `lrd` for decay. Increase `num_particles` to 3-5 if still unstable. Monitor ELBO smoothed over windows of 50-100 steps.
**Warning signs:** Loss variance doesn't decrease over training.

### Pitfall 3: Complex Tensor Errors in Spectral Likelihood
**What goes wrong:** `RuntimeError: Expected all tensors to be on the same device` or `TypeError` when passing complex128 tensor to `dist.Normal`.
**Why it happens:** Pyro distributions do not support complex dtypes. The `log_prob` method internally does operations that fail on complex tensors.
**How to avoid:** Always decompose complex CSD into real/imaginary parts before the `pyro.sample("obs", ...)` call. Use `torch.view_as_real()` or manual `.real` / `.imag` splitting.
**Warning signs:** Error occurs at `svi.step()` call, traceback mentions distribution log_prob.

### Pitfall 4: Plate Shape Mismatch
**What goes wrong:** `ValueError: Shape mismatch` between plate size and tensor dimensions.
**Why it happens:** The observation tensor shape doesn't align with the plate declaration. Pyro plates operate on rightmost dimensions by default.
**How to avoid:** Use explicit `dim=` argument in `pyro.plate`. For 2D observations `(T, N)`: time plate on `dim=-2`, region plate on `dim=-1`. Or use `.to_event()` to declare all observation dimensions as a single event (simpler, slightly less efficient).
**Warning signs:** Error at model trace time, mentioning plate dimension mismatch.

### Pitfall 5: Guide-Model Mismatch with Masked Parameters
**What goes wrong:** AutoNormal creates variational parameters for ALL elements of A_free (including masked-out ones), wasting computation.
**Why it happens:** AutoNormal sees the `pyro.sample("A_free", ...)` statement and creates N*N variational parameters regardless of masking.
**How to avoid:** This is acceptable for small networks (N<=10). The masked parameters will have posteriors close to the prior (no gradient signal). For large networks, consider sampling only the non-zero elements using a flat vector and reshaping.
**Warning signs:** No error, just slightly slower training. Not a real problem for DCM network sizes.

### Pitfall 6: Forgetting pyro.clear_param_store()
**What goes wrong:** Parameters from a previous SVI run persist, causing incorrect initialization or shape errors.
**Why it happens:** Pyro's global param store accumulates parameters across runs.
**How to avoid:** Always call `pyro.clear_param_store()` before each SVI run.
**Warning signs:** Unexpected parameter shapes, or optimization starts from non-random point.

### Pitfall 7: ODE Integration Memory with Long Time Series
**What goes wrong:** Out-of-memory when integrating with fine time grid (e.g., dt=0.01 for 300s = 30,000 time points).
**Why it happens:** Standard `odeint` stores all intermediate states for backpropagation.
**How to avoid:** Use coarser dt (0.1s is often sufficient for fMRI), or switch to `odeint_adjoint` for O(1) memory. For Phase 4, use `dt=0.5` or `1.0` initially to validate the model structure, then tune.
**Warning signs:** GPU/CPU memory spikes during `svi.step()`.

## Code Examples

### Example 1: Complete Minimal Task DCM Model + SVI

```python
from __future__ import annotations
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal

from pyro_dcm.forward_models.neural_state import parameterize_A
from pyro_dcm.forward_models.bold_signal import bold_signal
from pyro_dcm.forward_models.coupled_system import CoupledDCMSystem
from pyro_dcm.utils.ode_integrator import (
    PiecewiseConstantInput, integrate_ode, make_initial_state
)

def task_dcm_model(obs_bold, stimulus, a_mask, c_mask, t_eval, TR, dt):
    N, M = c_mask.shape
    T = obs_bold.shape[0]

    # Priors (SPM convention)
    A_free = pyro.sample("A_free",
        dist.Normal(torch.zeros(N, N, dtype=torch.float64),
                    (1/64)**0.5 * torch.ones(N, N, dtype=torch.float64))
        .to_event(2))
    A_free = A_free * a_mask

    C = pyro.sample("C",
        dist.Normal(torch.zeros(N, M, dtype=torch.float64),
                    torch.ones(N, M, dtype=torch.float64))
        .to_event(2))
    C = C * c_mask

    A = pyro.deterministic("A", parameterize_A(A_free))

    # Forward model
    system = CoupledDCMSystem(A, C, stimulus)
    y0 = make_initial_state(N, dtype=torch.float64)
    sol = integrate_ode(system, y0, t_eval, method="dopri5",
                       grid_points=stimulus.grid_points)

    v = torch.exp(sol[:, 3*N:4*N])
    q = torch.exp(sol[:, 4*N:5*N])
    pred_bold_fine = bold_signal(v, q)

    step = round(TR / dt)
    pred_bold = pred_bold_fine[::step][:T]

    # Noise
    noise_prec = pyro.sample("noise_prec",
        dist.Gamma(torch.tensor(1.0, dtype=torch.float64),
                   torch.tensor(1.0, dtype=torch.float64)))
    noise_std = (1.0 / noise_prec).sqrt()

    # Likelihood (flatten to single event for simplicity)
    pyro.sample("obs",
        dist.Normal(pred_bold, noise_std).to_event(2),
        obs=obs_bold)

# Usage:
guide = AutoNormal(task_dcm_model, init_scale=0.01)
optimizer = pyro.optim.ClippedAdam({"lr": 0.01, "clip_norm": 10.0})
svi = SVI(task_dcm_model, guide, optimizer, Trace_ELBO())
```

### Example 2: Complex CSD Decomposition for Spectral Likelihood

```python
def decompose_csd_for_likelihood(csd_complex: torch.Tensor) -> torch.Tensor:
    """Convert complex CSD (F, N, N) to real vector for Pyro likelihood.

    Stacks real and imaginary parts into a single real-valued vector.
    """
    return torch.cat([csd_complex.real.reshape(-1),
                      csd_complex.imag.reshape(-1)])
```

### Example 3: Structural Mask Application

```python
# Define masks: 1 where connection exists, 0 where absent
a_mask = torch.tensor([
    [1, 1, 0],  # region 0 -> 0, 0 -> 1 exist; 0 -> 2 absent
    [0, 1, 1],  # 1 -> 0 absent; 1 -> 1, 1 -> 2 exist
    [1, 0, 1],  # 2 -> 0, 2 -> 2 exist; 2 -> 1 absent
], dtype=torch.float64)

c_mask = torch.tensor([
    [1],  # input drives region 0
    [0],  # input does NOT drive region 1
    [0],  # input does NOT drive region 2
], dtype=torch.float64)

# In model:
A_free = pyro.sample("A_free", prior)
A_free = A_free * a_mask   # <-- this zeros absent connections
A = parameterize_A(A_free)  # diagonal transform still applies correctly
```

### Example 4: SVI Runner with Convergence Monitoring

```python
def run_svi(model, guide, model_args, num_steps=2000,
            lr=0.01, clip_norm=10.0, lr_decay_factor=0.01):
    pyro.clear_param_store()

    lrd = lr_decay_factor ** (1.0 / max(num_steps, 1))
    optimizer = pyro.optim.ClippedAdam({
        "lr": lr,
        "clip_norm": clip_norm,
        "lrd": lrd,
    })
    elbo = Trace_ELBO(num_particles=1, vectorize_particles=False)
    svi = SVI(model, guide, optimizer, loss=elbo)

    losses = []
    for step in range(num_steps):
        loss = svi.step(*model_args)
        losses.append(loss)
        if torch.isnan(torch.tensor(loss)):
            raise RuntimeError(f"NaN ELBO at step {step}")

    return {"losses": losses, "final_loss": losses[-1]}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|---|---|---|---|
| AutoDiagonalNormal | AutoNormal | Pyro 1.8+ | Better plate support, cleaner param names |
| Adam + manual clip | ClippedAdam | Pyro 1.5+ | Built-in gradient clipping + LR decay |
| Manual guide for mean-field | AutoNormal | Pyro 1.5+ | Eliminates boilerplate for baseline guides |
| Separate real/imag tensors | torch.complex128 native | PyTorch 2.0+ | Native complex autograd, view_as_real/view_as_complex |

**Deprecated/outdated:**
- `AutoDiagonalNormal`: Still works but `AutoNormal` is preferred (better site names, plate support).
- `pyro.infer.SVI` with raw `Adam`: `ClippedAdam` should be used instead for robustness.

## Open Questions

### 1. Noise Precision Prior for Task DCM
- **What we know:** SPM uses a specific noise precision model (log-precision with Gaussian prior). The CONTEXT.md says "suitable weakly informative prior."
- **What's unclear:** Exact Gamma shape/rate for the noise precision. Gamma(1,1) is weakly informative but may be too broad.
- **Recommendation:** Start with `Gamma(1.0, 1.0)` (mean=1, broad). If ELBO is unstable, tighten to `Gamma(2.0, 1.0)`. Validate by checking that posterior precision is reasonable (SNR 1-20 range for fMRI).

### 2. CSD Noise Scale for Spectral DCM
- **What we know:** The spectral DCM forward model already includes neuronal and observation noise. An additional observation noise on the CSD prediction-vs-data mismatch is needed.
- **What's unclear:** What scale is appropriate for the CSD observation noise? CSD values can span orders of magnitude.
- **Recommendation:** Sample a `csd_noise_scale` from `HalfCauchy(1.0)` to be weakly informative. Alternatively, compute a per-frequency-bin scale from the data, but this adds complexity.

### 3. rDCM Pyro Model Granularity
- **What we know:** rDCM is region-wise: each region has an independent regression. The Pyro model should reflect this.
- **What's unclear:** Should we use a single `pyro.plate("region", nr)` with vectorized operations, or loop over regions? The plate approach requires all regions to have the same number of active connections (same D_r), which they may not if masks differ per region.
- **Recommendation:** Use a Python loop over regions inside the model (not a plate for parameters), since each region can have different active columns. Use a plate for the frequency-domain data points within each region. This matches the existing `rigid_inversion` structure.

### 4. ODE Solver Step Size for SVI
- **What we know:** Phase 1 uses dt=0.01 for accuracy. SVI calls the forward model many times per training run.
- **What's unclear:** What dt is fast enough for SVI without losing too much accuracy?
- **Recommendation:** Use dt=0.5 for initial development and testing. Validate that predicted BOLD is reasonable at this resolution. Drop to dt=0.1 if needed. Use `euler` or `rk4` fixed-step methods for more predictable runtime (adaptive `dopri5` can be slow with stiff A matrices).

### 5. float64 vs float32 in Pyro SVI
- **What we know:** Project convention is float64 throughout. ODE integration and spectral computations need float64 for numerical stability.
- **What's unclear:** Whether Pyro's SVI (especially AutoNormal guide) works reliably with float64. Most Pyro examples use float32.
- **Recommendation:** Keep float64 as decided. Pyro supports float64 fully. Set `torch.set_default_dtype(torch.float64)` or pass dtype explicitly to all tensor constructors. If performance is an issue later, consider float32 for the guide only (Phase 7).

## Sources

### Primary (HIGH confidence)
- Pyro SVI Tutorials (pyro.ai/examples/svi_part_iv.html) -- SVI tips, ClippedAdam, convergence
- Pyro Tensor Shapes Tutorial (pyro.ai/examples/tensor_shapes.html) -- plate structure, to_event, batch/event dims
- Pyro Bayesian Regression Tutorial (pyro.ai/examples/bayesian_regression.html) -- model/guide pattern, AutoDiagonalNormal
- Pyro Introduction Tutorial (pyro.ai/examples/intro_long.html) -- pyro.sample, pyro.deterministic, pyro.plate
- Existing codebase: Phase 1-3 forward models, simulators, and their interfaces

### Secondary (MEDIUM confidence)
- Pyro Forum: ODE integration discussion (forum.pyro.ai/t/pyro-and-differential-equations/4266) -- torchdiffeq is recommended approach
- Pyro Forum: Plates and ODEs (forum.pyro.ai/t/parallelization-plate-and-odes/2763) -- plates not for ODE time steps
- Medium article: Probabilistic programming with DEs -- general pattern for DE in Pyro
- PyTorch Complex Numbers docs (docs.pytorch.org/docs/stable/complex_numbers.html) -- autograd through complex128
- Pyro AutoGuide source (github.com/pyro-ppl/pyro autoguide/guides.py) -- AutoNormal implementation details

### Tertiary (LOW confidence)
- GitHub Issue #2109 (pyro-ppl/pyro) -- gradient clipping feature request/discussion
- GitHub Issue #83376 (pytorch/pytorch) -- complex-valued Gaussian distributions not natively supported

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- Pyro, torchdiffeq, and their integration is well-established
- Architecture (model structure): MEDIUM -- ODE-in-Pyro pattern is validated by forum + tutorials but no canonical DCM-in-Pyro example exists
- Architecture (spectral complex workaround): MEDIUM -- real/imag decomposition is standard practice but not documented in Pyro tutorials specifically
- Pitfalls: MEDIUM -- based on training knowledge of common Pyro/ODE issues, partially verified by forum posts
- SVI configuration: MEDIUM -- ClippedAdam params from tutorials, but DCM-specific tuning is uncharted

**Research date:** 2026-03-27
**Valid until:** 2026-04-27 (stable domain, 30-day validity)
