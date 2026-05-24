# Phase 20: Direct Observation Forward Model, Simulator & Synthetic Validation - Research

**Researched:** 2026-05-24
**Domain:** Direct-observation bilinear DCM forward model, simulator, Pyro generative model, multi-start SVI, synthetic parameter recovery
**Confidence:** HIGH (all load-bearing interfaces verified by direct source read of coupled_system.py, task_dcm_model.py, guides.py, task_simulator.py, bilinear_metrics.py, task_bilinear.py runner, and Phase 16.1 research)

---

## Summary

Phase 20 builds a direct-observation DCM model that fits bilinear dynamics to N-dimensional neural state trajectories without hemodynamic convolution. This is the "does the math work?" phase -- correctly-specified bilinear ground truth, correctly-specified bilinear model, parameter recovery validation.

The codebase already has all the building blocks. CoupledDCMSystem in coupled_system.py implements the full 5N-dimensional neural+hemodynamic ODE. NeuralStateEquation in neural_state.py implements bilinear dx/dt = (A + sum_j u_j B_j) x + Cu with parameterize_A, parameterize_B, compute_effective_A. The task_dcm_model.py Pyro model shows the exact pattern for A/B/C sampling, prior specification, ODE integration, and likelihood. The benchmark infrastructure (bilinear_metrics.py, task_bilinear.py runner, generate_fixtures.py, BenchmarkConfig) provides reusable metric computation and fixture generation. Phase 16.1 research provides critical lessons about init_scale x prior_variance interaction that directly inform the prior recalibration strategy.

**Primary recommendation:** Extend CoupledDCMSystem with a `hemodynamic=False` toggle (N-dim state, identity observation, no TR downsampling), fork task_dcm_model.py into latent_circuit_dcm_model.py with separate LC_A_PRIOR_VARIANCE/LC_B_PRIOR_VARIANCE constants, extend run_svi() with n_restarts kwarg, and replicate the task_bilinear.py runner pattern for the validation benchmark.

---

## Standard Stack

No new libraries needed. Phase 20 reuses the existing stack entirely.

### Core (already in project)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.x | Tensor computation, autograd | Foundation of entire project |
| Pyro | 1.9+ | Probabilistic model, SVI, guides | Inference backbone |
| torchdiffeq | latest | ODE integration (odeint) | Already used by CoupledDCMSystem |

### Supporting (already in project)

| Library | Version | Purpose | When Used |
|---------|---------|---------|-----------|
| numpy | latest | Fixture serialization (.npz) | Fixture generation |
| matplotlib | latest | Recovery diagnostics, forest plots | Validation figures |

### No New Dependencies

Phase 20 requires zero new library installations. All code is built on existing infrastructure.

---

## Architecture Patterns

### Recommended Project Structure

```
src/pyro_dcm/
  forward_models/
    coupled_system.py        # EXTEND: add hemodynamic=False toggle
    neural_state.py          # REUSE AS-IS (OBS-04)
  models/
    latent_circuit_dcm_model.py  # NEW: fork of task_dcm_model pattern
    guides.py                    # EXTEND: n_restarts kwarg to run_svi
  simulators/
    latent_circuit_simulator.py  # NEW: fork of task_simulator pattern

benchmarks/
  latent_circuit_metrics.py      # NEW: extends bilinear_metrics pattern
  runners/
    latent_circuit_bilinear.py   # NEW: fork of task_bilinear runner
  generate_fixtures.py           # EXTEND: add latent_circuit variant
```

### Pattern 1: CoupledDCMSystem Hemodynamic Toggle

**What:** Add `hemodynamic: bool = True` parameter to CoupledDCMSystem.__init__. When False, the state is N-dimensional (neural only), forward() returns N derivatives, and make_initial_state returns N-dim zero vector.

**When to use:** All latent circuit / direct observation DCM fitting.

**Implementation approach (verified against source):**

The current CoupledDCMSystem.__init__ (coupled_system.py:134-249) creates self.hemo = BalloonWindkessel() and the forward() method (line 251-306) unpacks 5N state, computes neural dx, computes hemodynamic ds/dlnf/dlnv/dlnq, and returns torch.cat([dx, ds, dlnf, dlnv, dlnq]).

When `hemodynamic=False`:
- Skip BalloonWindkessel construction (self.hemo = None)
- State is N-dimensional: y0 = torch.zeros(N)
- forward() only computes neural derivatives dx (lines 286-300 are reused as-is)
- forward() returns dx directly (not concatenated with hemodynamic derivatives)
- No changes to NeuralStateEquation or BalloonWindkessel classes

```python
# In CoupledDCMSystem.__init__:
self.hemodynamic = hemodynamic
if hemodynamic:
    self.hemo = BalloonWindkessel(**hemo_params) if hemo_params else BalloonWindkessel()
else:
    self.hemo = None

# In CoupledDCMSystem.forward():
if self.hemodynamic:
    # existing 5N path unchanged
    x = state[:N]
    s = state[N:2*N]
    # ... existing code ...
    return torch.cat([dx, ds, dlnf, dlnv, dlnq])
else:
    # Direct observation: state IS neural activity
    x = state  # shape (N,)
    u_all = self.input_fn(t)
    if self.B is None or self.B.shape[0] == 0:
        dx = self.A @ x + self.C @ u_all
    else:
        M_d = self.n_driving_inputs
        u_drive = u_all[:M_d]
        u_mod = u_all[M_d:]
        A_eff = compute_effective_A(self.A, self.B, u_mod)
        dx = A_eff @ x + self.C @ u_drive
        self._maybe_check_stability(t, A_eff, u_mod)
    return dx
```

**Key design constraint:** The hemodynamic=True path MUST remain bit-exact to current behavior. No existing tests should break. The toggle is purely additive.

### Pattern 2: Pyro Model Fork (latent_circuit_dcm_model)

**What:** New latent_circuit_dcm_model.py that copies the A/B/C sampling pattern from task_dcm_model.py but uses CoupledDCMSystem(hemodynamic=False) and identity observation.

**Key differences from task_dcm_model (verified by source read):**

1. **Prior constants:** LC_A_PRIOR_VARIANCE (not 1/64), LC_B_PRIOR_VARIANCE (not 1.0). Separate calibration needed because RNN-scale dynamics have different magnitude than BOLD-scale DCM.
2. **No hemodynamic parameters:** No BalloonWindkessel, no BOLD equation, no TR downsampling.
3. **State dimension:** N (not 5N). Initial state y0 = torch.zeros(N).
4. **Observation model:** y = x + noise (identity C_obs for v0.6.0). No bold_signal() call.
5. **Time grid:** Same as trajectory time grid (no TR downsampling step).
6. **Likelihood:** Gaussian on neural trajectories directly (not downsampled BOLD).

**Sample site names must enable create_guide() auto-discovery (MODEL-03).** Verified: create_guide() (guides.py:77-189) calls `GUIDE_REGISTRY[guide_type](model, **ctor_kwargs)` -- all AutoGuide variants auto-discover sample sites from a single model trace. As long as the new model uses standard pyro.sample() calls with unique site names, no factory changes are needed. The B_free_j naming convention from task_dcm_model (lines 303-311) should be reused exactly.

```python
# Signature pattern:
def latent_circuit_dcm_model(
    observed_trajectories: torch.Tensor,  # (T, N) -- neural state trajectories
    stimulus: PiecewiseConstantInput,
    a_mask: torch.Tensor,
    c_mask: torch.Tensor,
    t_eval: torch.Tensor,
    dt: float = 0.01,
    *,
    b_masks: list[torch.Tensor] | None = None,
    stim_mod: PiecewiseConstantInput | None = None,
) -> None:
```

Note: no TR parameter (no downsampling). dt matches the trajectory time grid.

### Pattern 3: Multi-Start SVI (n_restarts)

**What:** Extend run_svi() in guides.py with `n_restarts: int = 1` kwarg. When n_restarts > 1, clear param store, re-init guide, run SVI, repeat N times, select best by final ELBO.

**Implementation approach (verified against run_svi source, guides.py:192-363):**

run_svi() already calls pyro.clear_param_store() at line 313. The extension adds an outer loop:

```python
def run_svi(
    model, guide, model_args,
    ...,
    n_restarts: int = 1,  # NEW: default preserves backward compat
) -> dict[str, Any]:
    best_result = None
    all_results = []
    for restart in range(n_restarts):
        pyro.clear_param_store()
        # ... existing SVI loop ...
        result = {"losses": losses, "final_loss": losses[-1], ...}
        all_results.append(result)
        if best_result is None or result["final_loss"] < best_result["final_loss"]:
            best_result = result
    best_result["all_restarts"] = all_results
    best_result["n_restarts"] = n_restarts
    return best_result
```

**Critical detail:** The guide must be RE-CREATED for each restart (not just clearing param store). AutoGuide parameters live in Pyro's param store AND the guide object. After pyro.clear_param_store(), the guide needs to be re-initialized. The cleanest approach: accept a guide_factory callable instead of a guide instance for the multi-restart path.

Actually, re-reading the Pyro internals: pyro.clear_param_store() removes all learned parameters. When SVI.step() is called next, the guide lazily re-initializes its parameters from scratch (AutoGuide._setup_prototype is called on first forward pass). So clearing the param store IS sufficient for re-initialization -- the guide object itself stores configuration (model reference, init_scale, etc.) but not the learned parameters.

**Verification:** In guides.py line 313, `pyro.clear_param_store()` is already called once before the SVI loop. For multi-restart, we need to call it before each restart iteration. The guide's _setup_prototype flag needs to be reset: `guide._prototype_trace = None` (forces re-initialization).

**Alternative (safer):** Re-create the guide from scratch each restart. The create_guide() function is lightweight. The CONTEXT decision says "extend run_svi() with n_restarts=1 kwarg" -- so the run_svi function needs access to the guide creation args. This suggests either:
- Accept guide_factory: Callable (cleanest)
- Accept the same create_guide kwargs and re-create internally
- Re-use the existing guide but reset its internal state

**Recommendation:** Accept `guide_factory: Callable[[], AutoGuide] | None = None` parameter. When n_restarts > 1, call guide_factory() for each restart. When n_restarts == 1, use the existing guide argument (backward compat). This is the safest pattern -- no reliance on internal Pyro state management.

### Pattern 4: Simulator (simulate_latent_circuit)

**What:** New simulate_latent_circuit() function. Follows the same structure as simulate_task_dcm() but uses CoupledDCMSystem(hemodynamic=False).

**Key differences from simulate_task_dcm (verified by source read of task_simulator.py:130-443):**

1. **No hemodynamic parameters**: No hemo_params, no E0, no V0
2. **No BOLD computation**: No bold_signal(), no v/q extraction
3. **No TR downsampling**: Output time grid = ODE integration time grid
4. **State dimension**: N (not 5N)
5. **Initial state**: torch.zeros(N) (not make_initial_state(N) which returns 5N zeros)
6. **Noise**: Added directly to neural trajectories
7. **Return dict**: trajectories (T, N), neural = trajectories (same thing), times, params

```python
def simulate_latent_circuit(
    A: torch.Tensor,
    C: torch.Tensor,
    stimulus: dict | PiecewiseConstantInput,
    duration: float = 100.0,
    dt: float = 0.01,
    SNR: float = 10.0,
    solver: str = "dopri5",
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    seed: int | None = None,
    *,
    B_list: torch.Tensor | list[torch.Tensor] | None = None,
    stimulus_mod: dict | PiecewiseConstantInput | None = None,
    n_driving_inputs: int | None = None,
) -> dict:
```

### Anti-Patterns to Avoid

- **Don't inherit from CoupledDCMSystem.** The CONTEXT says extend with a toggle, not create a subclass. A subclass would introduce fragile coupling and override confusion.
- **Don't share prior constants between task_dcm_model and latent_circuit_dcm_model.** The prior variances MUST be independent -- the Phase 16.1 lesson shows prior_variance x init_scale interaction is the primary failure mode.
- **Don't bypass pyro.clear_param_store() in multi-restart.** Each restart must start from a clean parameter state. Pyro's param store is global mutable state.
- **Don't use adaptive ODE solvers (dopri5) in the Pyro model during SVI.** Use rk4 fixed-step for predictable computation graphs. Adaptive solvers can cause variable-length forward passes that destabilize gradient estimation. (The simulator can use dopri5 for ground-truth generation.)

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| ODE integration | Custom Euler/RK4 loop | torchdiffeq.odeint via integrate_ode() | Handles adjoint, grid_points, solver selection |
| A matrix parameterization | Custom diagonal transform | parameterize_A() from neural_state.py | Guarantees negative self-connections, SPM12 convention |
| B matrix masking | Manual element-wise zeroing | parameterize_B() from neural_state.py | Handles shape validation, diagonal warnings |
| Piecewise stimulus | Custom interpolation | PiecewiseConstantInput from ode_integrator.py | Handles searchsorted, grid_points for adaptive solvers |
| Stimulus merging (drive + mod) | Manual column concatenation | merge_piecewise_inputs() from ode_integrator.py | Handles breakpoint union, dtype/device validation |
| Recovery metrics (RMSE) | Custom metric computation | compute_rmse() from benchmarks/metrics.py | Consistent with all other benchmark runners |
| Coverage metrics | Custom CI computation | compute_coverage_multi_level() from metrics.py | Empirical quantiles, works with non-Gaussian posteriors |
| B-specific metrics | New metric functions | bilinear_metrics.py functions | Magnitude-masked B-RMSE, sign recovery, coverage-of-zero |
| Posterior extraction | Manual guide sampling | extract_posterior_params() from guides.py | Handles all 6 guide types, Predictive wrapper |
| Fixture serialization | Custom save/load | .npz pattern from generate_fixtures.py | Consistent with project convention, generic loader |

**Key insight:** Phase 20 is a reassembly of existing building blocks in a new configuration, not new mathematics.

---

## Common Pitfalls

### Pitfall 1: init_scale x prior_variance Interaction (CRITICAL)

**What goes wrong:** The variational posterior collapses to zero for parameters whose prior_std >> init_scale, especially when likelihood signal is weak. This was the root cause of Phase 16.1's RECOV-04 failure.

**Why it happens:** Phase 16.1 research (16.1-RESEARCH.md) showed that init_scale=0.005 with B_PRIOR_VARIANCE=1.0 (sigma_prior=1.0) produces init/sigma_prior ratio of 0.5%, trapping B posteriors near zero. Meanwhile A with sigma_prior=0.125 had a 4% ratio and recovered fine.

**How to avoid:** For the latent circuit model, the prior variances (LC_A_PRIOR_VARIANCE, LC_B_PRIOR_VARIANCE) and init_scale must be calibrated jointly. The CONTEXT decision specifies a "joint prior_var x init_scale empirical sweep" on cluster. The recommended sweep grid:
- prior_var: {1/64, 1/16, 1/4, 1.0} for A; {0.25, 1.0, 4.0} for B
- init_scale: {0.01, 0.05, 0.1, 0.5}
- Metric: B-RMSE and A-RMSE on a single seed at N=4

**Warning signs:** All seeds produce near-identical B-RMSE close to the "collapse-to-zero" theoretical value (sqrt(sum(B_true^2) / n_elements)). RECOV-07 shrinkage near 1.0 means posterior hasn't moved from prior.

### Pitfall 2: State Dimension Mismatch in Toggle

**What goes wrong:** When hemodynamic=False, make_initial_state(N) returns 5N-dim zeros. The ODE integrates a 5N state through an N-dim derivative function, producing shape errors or silent zero-padding.

**Why it happens:** make_initial_state() always returns 5*n_regions zeros (ode_integrator.py:199-241).

**How to avoid:** For hemodynamic=False path, use torch.zeros(N) directly as initial state, NOT make_initial_state(). Or create a new helper make_initial_state_neural(N) that returns N zeros.

### Pitfall 3: Multi-Restart Guide Re-initialization

**What goes wrong:** Running SVI multiple times with the same guide object accumulates learned parameters across restarts, so "restart 2" doesn't actually restart from a fresh initialization.

**Why it happens:** Pyro AutoGuides store learned parameters in both the global param store AND internal guide attributes (_setup_prototype caching). pyro.clear_param_store() removes the param store entries, but the guide's internal prototype trace may still reference stale tensors.

**How to avoid:** For multi-restart, either (a) create a fresh guide from a factory callable for each restart, or (b) reset the guide's _prototype_trace to None after clearing the param store. Option (a) is safer and more explicit.

### Pitfall 4: NaN in Direct Observation ODE (Bilinear Instability)

**What goes wrong:** Bilinear coupling can push max Re(eig(A_eff)) positive during sustained modulator-ON periods, causing exponential growth in neural states. Without hemodynamic damping (which acts as a low-pass filter on neural activity), the direct observation model is MORE sensitive to instability than the task DCM model.

**Why it happens:** The hemodynamic states (s, f, v, q) in the full CoupledDCMSystem act as a natural stabilizer -- the balloon-Windkessel dynamics have their own damping. Without them, unstable neural dynamics grow unchecked.

**How to avoid:**
1. Ground-truth generation: verify max Re(eig(A + sum_j B_j)) < 0 for the ground truth A and B at maximum u_mod=1.
2. The seed-pool rejection pattern from task_bilinear.py runner (lines 681-704) should be replicated.
3. The NaN-safe BOLD guard pattern from task_dcm_model.py (lines 379-381) should be adapted for trajectories.
4. Consider tighter A self-connection magnitudes for direct observation (e.g., self_inhibition >= 0.8 Hz instead of 0.5 Hz).

### Pitfall 5: Prior Scale for RNN-Scale vs BOLD-Scale

**What goes wrong:** Using SPM12 task-DCM priors (A_PRIOR_VARIANCE = 1/64, B_PRIOR_VARIANCE = 1.0) for RNN-scale data causes severe under/over-regularization.

**Why it happens:** BOLD signal is ~1-5% signal change; RNN hidden states can be order 1-10. The A matrix eigenvalues may also differ in scale: task DCM A is typically ~0.1-0.5 Hz, while RNN linearized dynamics can have larger eigenvalues.

**How to avoid:** The CONTEXT decision specifies empirical calibration on 5+ synthetic scenarios. Since Phase 20 uses synthetic bilinear ground truth (not actual RNNs), the ground truth A/B magnitudes determine the appropriate prior scale. Start with A_free ~ N(0, 1/16) and B_free ~ N(0, 1.0) and sweep from there.

### Pitfall 6: Trajectory R-squared Computed on Training Data

**What goes wrong:** R-squared computed on the same trajectories used for fitting gives an overoptimistic estimate, especially with flexible models.

**Why it happens:** Standard overfitting concern. The SYNTH-02 requirement specifies "held-out trials."

**How to avoid:** Generate multiple "trials" (different stimulus realizations or different temporal segments). Fit on a subset, evaluate trajectory R-squared on the held-out subset. In the synthetic bilinear setting with correctly specified model, R-squared should be >= 0.95 on held-out data.

### Pitfall 7: ELBO Model Selection Requires Consistent Priors

**What goes wrong:** Comparing ELBO across models with different N (number of regions) is only valid if the priors are defined consistently. Adding more regions adds more parameters, each contributing KL divergence.

**Why it happens:** ELBO = log p(data | theta) - KL(q || p). More parameters = larger KL term. The ELBO penalizes complexity through the KL, but only correctly if priors are comparable.

**How to avoid:** Use the same per-element prior variance across all candidate N values. The ELBO then automatically penalizes over-parameterized models through the aggregate KL. Verify on synthetic data that true N=4 scores better than N=2 (underfitting) and N=6/8 (overfitting).

---

## Code Examples

### Example 1: CoupledDCMSystem with hemodynamic=False

```python
# Source: Extension of coupled_system.py pattern
import torch
from pyro_dcm.forward_models.coupled_system import CoupledDCMSystem
from pyro_dcm.utils.ode_integrator import integrate_ode, PiecewiseConstantInput

N = 4
A = -0.5 * torch.eye(N, dtype=torch.float64)
A[0, 1] = 0.2
C = torch.zeros(N, 1, dtype=torch.float64)
C[0, 0] = 1.0

stimulus = PiecewiseConstantInput(
    torch.tensor([0.0, 10.0, 20.0], dtype=torch.float64),
    torch.tensor([[1.0], [0.0], [1.0]], dtype=torch.float64),
)

system = CoupledDCMSystem(A, C, stimulus, hemodynamic=False)
y0 = torch.zeros(N, dtype=torch.float64)  # N-dim, not 5N
t_eval = torch.arange(0, 30, 0.01, dtype=torch.float64)
solution = integrate_ode(system, y0, t_eval, method="dopri5", step_size=0.01)
# solution shape: (3000, 4) -- (T, N), not (T, 5N)
```

### Example 2: Latent Circuit Pyro Model (pattern)

```python
# Source: Fork of task_dcm_model.py pattern
import pyro
import pyro.distributions as dist
import torch

LC_A_PRIOR_VARIANCE: float = 1 / 16  # Calibrated for RNN-scale
LC_B_PRIOR_VARIANCE: float = 1.0     # Calibrated empirically

def latent_circuit_dcm_model(
    observed_trajectories: torch.Tensor,  # (T, N)
    stimulus: PiecewiseConstantInput,
    a_mask: torch.Tensor,
    c_mask: torch.Tensor,
    t_eval: torch.Tensor,
    dt: float = 0.01,
    *,
    b_masks: list[torch.Tensor] | None = None,
    stim_mod: PiecewiseConstantInput | None = None,
) -> None:
    N = a_mask.shape[0]
    T = observed_trajectories.shape[0]

    # Sample A_free with LC-specific prior
    A_free = pyro.sample("A_free", dist.Normal(
        torch.zeros(N, N, dtype=torch.float64),
        LC_A_PRIOR_VARIANCE ** 0.5 * torch.ones(N, N, dtype=torch.float64),
    ).to_event(2))
    A_free = A_free * a_mask
    A = pyro.deterministic("A", parameterize_A(A_free))

    # Sample C
    C = pyro.sample("C", dist.Normal(
        torch.zeros_like(c_mask), torch.ones_like(c_mask),
    ).to_event(2)) * c_mask

    # B sampling (same pattern as task_dcm_model)
    # ... [identical to task_dcm_model lines 293-337]

    # ODE integration with hemodynamic=False
    system = CoupledDCMSystem(A, C, merged_input_fn, hemodynamic=False,
                               B=B_stacked, n_driving_inputs=c_mask.shape[1])
    y0 = torch.zeros(N, dtype=torch.float64)
    solution = integrate_ode(system, y0, t_eval, method="rk4", step_size=dt)
    # solution shape: (T, N) -- neural trajectories directly

    predicted = solution[:T]  # No downsampling

    # NaN guard (same pattern as task_dcm_model)
    if torch.isnan(predicted).any() or torch.isinf(predicted).any():
        predicted = torch.zeros_like(predicted).detach()
    pyro.deterministic("predicted_trajectories", predicted)

    # Noise precision
    noise_prec = pyro.sample("noise_prec", dist.Gamma(
        torch.tensor(1.0, dtype=torch.float64),
        torch.tensor(1.0, dtype=torch.float64),
    ))
    noise_std = (1.0 / noise_prec).sqrt()

    # Likelihood on neural trajectories (identity C_obs)
    pyro.sample("obs", dist.Normal(predicted, noise_std).to_event(2),
                obs=observed_trajectories)
```

### Example 3: Multi-Start SVI Extension

```python
# Source: Extension of guides.py run_svi()
def run_svi(
    model, guide, model_args,
    ...,
    n_restarts: int = 1,
    guide_factory: Callable[[], AutoGuide] | None = None,
) -> dict[str, Any]:
    if n_restarts <= 1:
        # Existing single-run path (backward compat)
        pyro.clear_param_store()
        # ... existing code ...
        return result

    # Multi-restart path
    best_result = None
    all_results = []
    for r in range(n_restarts):
        pyro.clear_param_store()
        if guide_factory is not None:
            guide = guide_factory()
        # ... existing SVI loop ...
        result = {"losses": losses, "final_loss": losses[-1], "restart": r}
        all_results.append(result)
        if best_result is None or result["final_loss"] < best_result["final_loss"]:
            best_result = result
            best_guide = guide
    best_result["all_restarts"] = all_results
    best_result["n_restarts"] = n_restarts
    best_result["guide"] = best_guide
    return best_result
```

### Example 4: Trajectory R-squared

```python
# Source: Standard metric computation
def compute_trajectory_r_squared(
    predicted: torch.Tensor,  # (T, N)
    observed: torch.Tensor,   # (T, N)
) -> float:
    """Per-region R-squared averaged across regions."""
    ss_res = ((observed - predicted) ** 2).sum(dim=0)  # (N,)
    ss_tot = ((observed - observed.mean(dim=0)) ** 2).sum(dim=0)  # (N,)
    r2_per_region = 1.0 - ss_res / ss_tot.clamp(min=1e-12)
    return r2_per_region.mean().item()
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| LatentCircuitSystem (new class) | CoupledDCMSystem(hemodynamic=False) | Phase 20 CONTEXT decision | Avoids fork drift, enables easy switch to BOLD fitting |
| Fixed init_scale across all params | Joint prior_var x init_scale sweep | Phase 16.1 lesson (2026-04-24) | Prevents posterior collapse from init_scale/prior_var mismatch |
| Single SVI run | Multi-start SVI (10+ restarts) | Phase 20 decision | Addresses multi-modality in ELBO landscape |
| Per-variant recovery metrics | Reuse bilinear_metrics.py infrastructure | Phase 16 established pattern | Consistent metric computation across model types |

---

## Design Decisions Requiring Research Confirmation

### 1. CoupledDCMSystem vs OBS-04 Tension

**Finding:** OBS-04 in REQUIREMENTS-v0.6.0.md says "Zero edits to neural_state.py, balloon_model.py, bold_signal.py, or coupled_system.py." But the CONTEXT decision explicitly says "Make hemodynamic stage OPTIONAL in existing CoupledDCMSystem."

**Resolution:** The CONTEXT decision (from user discussion) takes priority over the pre-discussion requirements wording. OBS-04's intent is "don't break existing forward model behavior" -- which is satisfied by making the toggle purely additive with hemodynamic=True preserving bit-exact existing behavior. The zero-edit constraint applies to neural_state.py, balloon_model.py, and bold_signal.py (which truly should not be touched). coupled_system.py MUST be edited per the CONTEXT decision.

**Confidence:** HIGH -- CONTEXT decisions are locked.

### 2. Noise Model (Claude's Discretion)

**Finding:** CONTEXT says noise model details are Claude's discretion (scalar noise_prec vs per-dimension).

**Recommendation:** Start with scalar noise_prec (single precision shared across all N dimensions), matching the task_dcm_model pattern. This is simpler and sufficient for v0.6.0 synthetic validation where all dimensions have similar signal scale.

Rationale: Per-dimension noise_prec adds N parameters and makes multi-start SVI harder (more modes). For synthetic validation with uniform SNR, scalar noise is adequate. Per-dimension noise can be added in v0.6.1 if needed for real RNN data.

**Confidence:** HIGH -- scalar noise is the simplest correct choice.

### 3. ELBO Model Selection Candidate N Set (Claude's Discretion)

**Finding:** CONTEXT says candidate N set is Claude's discretion.

**Recommendation:** Use {2, 3, 4, 5, 6} with true N=4. This gives two underfitting candidates (2, 3), the true model (4), and two overfitting candidates (5, 6). Spacing of 1 makes the resolution clear. Wider spacing (e.g., {2, 4, 6, 8}) might miss the transition.

**Confidence:** MEDIUM -- the clearest figure may depend on how sharp the ELBO differences are. Start with {2, 3, 4, 5, 6}; if differences are subtle, try wider spacing.

### 4. Stimulus Design for Synthetic Data (Claude's Discretion)

**Finding:** CONTEXT says stimulus design details are Claude's discretion.

**Recommendation:** Use block design with epoch modulators, matching the v0.3.0 bilinear validation pattern. Specifically:
- Driving: 5 blocks of 10s ON + 10s OFF over 100s total
- Modulator: 3 epochs of 8s at [10, 40, 70]s
- Duration: 100s (shorter than v0.3.0's 200s because no hemodynamic lag to wait for)
- dt: 0.01s for simulator, 0.01s for model (no TR mismatch)

Shorter durations are appropriate because direct observation has no hemodynamic lag (6-8s HRF delay); the neural response is instantaneous.

**Confidence:** MEDIUM -- exact timing may need adjustment during calibration sweep.

### 5. Ground Truth A/B Magnitudes

**Finding:** For N=4 generic bilinear system, ground truth should be stable under bilinear coupling.

**Recommendation:**
- A: self-inhibition = 0.5 Hz (SPM default). Off-diagonal: sparse (density ~0.5), magnitudes 0.1-0.3 Hz.
- B: 1 modulator, J=1. Sparse: 2-3 non-zero elements at positions forming a directed path. Magnitudes 0.3-0.5 (detectable but not destabilizing).
- C: single driving input, one non-zero entry (amplitude 1.0).
- Stability check: verify max Re(eig(A + B @ 1.0)) < -0.1 (safe margin).

The v0.3.0 bilinear benchmark used B[1,0]=0.4, B[2,1]=0.3 for N=3. For N=4, a similar directed-chain pattern works: B[1,0]=0.4, B[2,1]=0.3, B[3,2]=0.2 (descending magnitudes along the chain).

**Confidence:** HIGH for the pattern; specific magnitudes will be validated during calibration.

---

## CoupledDCMSystem Toggle: Internal Implementation Detail

### make_initial_state Consideration

Current `make_initial_state(n_regions)` returns `torch.zeros(5 * n_regions)`. For the hemodynamic=False path, this is wrong.

**Options:**
1. Create a new `make_initial_state_neural(n_regions)` that returns `torch.zeros(n_regions)`.
2. Add a `hemodynamic: bool = True` parameter to `make_initial_state`.
3. Just use `torch.zeros(N)` directly in the model and simulator.

**Recommendation:** Option 3 (just use torch.zeros(N) directly). It's a one-liner, avoids editing ode_integrator.py unnecessarily, and the intent is clear. The make_initial_state helper was designed for the 5N hemodynamic state and doesn't need to be generalized.

### Forward Method Dispatch

The forward() method needs to branch on self.hemodynamic early. The neural derivative computation (lines 287-300 in the current code) can be shared -- the key difference is:
- hemodynamic=True: unpack x from 5N state, compute hemodynamic derivatives, pack 5N output
- hemodynamic=False: x IS the state, compute only neural derivatives, return N output

The bilinear vs linear branching logic (checking self.B) is identical in both paths.

---

## Multi-Start SVI: Detailed Design

### Guide Factory Pattern

```python
# At call site:
from functools import partial

guide_factory = partial(
    create_guide,
    latent_circuit_dcm_model,
    guide_type="auto_normal",
    init_scale=0.1,
)

result = run_svi(
    latent_circuit_dcm_model,
    guide_factory(),  # first guide instance
    model_args=(...),
    n_restarts=10,
    guide_factory=guide_factory,
)
```

### Return Contract

```python
{
    "losses": [...],           # losses from BEST restart
    "final_loss": float,       # BEST final loss
    "num_steps": int,
    "n_restarts": int,
    "best_restart_idx": int,
    "all_restarts": [
        {"losses": [...], "final_loss": float, "restart": 0},
        {"losses": [...], "final_loss": float, "restart": 1},
        ...
    ],
}
```

### Backward Compatibility

When n_restarts=1 (default), run_svi() must return EXACTLY the same dict structure as today: {"losses", "final_loss", "num_steps", optionally "guide"}. No "all_restarts" key, no "n_restarts" key. This ensures all existing callers (task_svi.py, spectral_svi.py, rdcm_vb.py, task_bilinear.py, amortized runners) work unchanged.

---

## Benchmark Infrastructure Integration

### Fixture Generation Pattern

Extend generate_fixtures.py with `generate_latent_circuit_fixtures()`:

```python
def generate_latent_circuit_fixtures(
    n_regions: int, n_datasets: int, seed: int, output_dir: str,
) -> None:
    # N=4 default, bilinear ground truth
    # Uses simulate_latent_circuit() instead of simulate_task_dcm()
    # Saves trajectories (not BOLD)
    # Same .npz + manifest.json pattern
```

Add `"latent_circuit": generate_latent_circuit_fixtures` to `_GENERATORS` dict.

### Runner Pattern

New `benchmarks/runners/latent_circuit_bilinear.py` following task_bilinear.py pattern:

```python
def run_latent_circuit_bilinear_svi(config: BenchmarkConfig) -> dict[str, Any]:
    # Same structure as run_task_bilinear_svi
    # Key differences:
    # 1. Uses simulate_latent_circuit (or loads latent_circuit fixtures)
    # 2. Uses latent_circuit_dcm_model (not task_dcm_model)
    # 3. No TR/BOLD parameters
    # 4. Multi-start SVI (n_restarts=10)
    # 5. Additional metrics: trajectory R-squared, ELBO model selection
```

### Metrics Extension

New `benchmarks/latent_circuit_metrics.py`:

```python
# Reuse from bilinear_metrics.py:
# - compute_b_rmse_magnitude
# - compute_sign_recovery_nonzero
# - compute_coverage_of_zero
# - compute_shrinkage

# New:
def compute_trajectory_r_squared(predicted, observed) -> float: ...
def compute_elbo_model_selection(elbos: dict[int, float]) -> dict: ...
def compute_latent_circuit_acceptance_gates(runner_result) -> dict: ...
```

### BenchmarkConfig Extension

Add to quick_config and full_config defaults dicts in config.py:

```python
# In quick_config:
defaults["latent_circuit"] = {"n_datasets": 3, "n_svi_steps": 500}
# In full_config:
defaults["latent_circuit"] = {"n_datasets": 10, "n_svi_steps": 2000}
```

---

## Open Questions

### 1. Prior Variance Calibration (requires empirical sweep)

**What we know:** Phase 16.1 showed init_scale=0.005 with B_PRIOR_VARIANCE=1.0 caused collapse. Direct observation has no hemodynamic attenuation, so parameter scales may differ.

**What's unclear:** The exact LC_A_PRIOR_VARIANCE and LC_B_PRIOR_VARIANCE values for RNN-scale dynamics. CONTEXT says "calibrated empirically on 5+ synthetic RNNs" but Phase 20 uses synthetic bilinear (not RNNs). The calibration should use synthetic bilinear ground truth at the expected scale of the Phase 21/22 RNN-derived data.

**Recommendation:** Start with LC_A_PRIOR_VARIANCE = 1/16 (wider than SPM's 1/64) and LC_B_PRIOR_VARIANCE = 1.0 (same as task DCM). Run calibration sweep on cluster to find the joint optimum. Document the winning combination as fixed constants.

### 2. Number of SVI Steps for Direct Observation

**What we know:** Task DCM uses 500 (quick) to 3000 (full) steps. Direct observation has no hemodynamic lag, so the ELBO landscape may be simpler (fewer local optima from hemodynamic convolution).

**What's unclear:** Whether direct observation converges faster or needs more steps.

**Recommendation:** Start with 1000 steps per restart, 10 restarts. Monitor convergence across restarts. Adjust during calibration sweep.

### 3. Multi-Start vs Single-Start with Better Init

**What we know:** CONTEXT locks 10 restarts for v0.6.0. L&E uses 100.

**What's unclear:** Whether 10 restarts suffices for N=4 bilinear or whether the ELBO landscape is benign enough for fewer.

**Recommendation:** Implement 10 as the default. The calibration sweep will reveal how much variance exists across restarts. If all 10 converge to similar ELBO, the landscape is well-behaved and 10 is sufficient.

---

## Sources

### Primary (HIGH confidence)
- `src/pyro_dcm/forward_models/coupled_system.py` -- CoupledDCMSystem architecture, forward(), bilinear gate
- `src/pyro_dcm/forward_models/neural_state.py` -- NeuralStateEquation, parameterize_A/B, compute_effective_A
- `src/pyro_dcm/models/task_dcm_model.py` -- Pyro model pattern, A/B/C sampling, NaN guard, likelihood
- `src/pyro_dcm/models/guides.py` -- create_guide, run_svi, extract_posterior_params, GUIDE_REGISTRY
- `src/pyro_dcm/simulators/task_simulator.py` -- simulate_task_dcm, make_block_stimulus, stimulus helpers
- `src/pyro_dcm/utils/ode_integrator.py` -- integrate_ode, PiecewiseConstantInput, make_initial_state, merge_piecewise_inputs
- `benchmarks/bilinear_metrics.py` -- compute_acceptance_gates, B-RMSE, sign recovery, coverage-of-zero
- `benchmarks/runners/task_bilinear.py` -- run_task_bilinear_svi, seed-pool rejection, init_scale retry
- `benchmarks/generate_fixtures.py` -- Fixture generation pattern, _GENERATORS registry
- `benchmarks/config.py` -- BenchmarkConfig, quick_config, full_config
- `benchmarks/metrics.py` -- compute_rmse, coverage, R-squared helpers
- `.planning/phases/16.1-recov-04-b-rmse-diagnostic/16.1-RESEARCH.md` -- init_scale x prior_variance root cause
- `.planning/phases/20-latent-circuit-forward-model/20-CONTEXT.md` -- User decisions (locked)

### Secondary (MEDIUM confidence)
- `.planning/REQUIREMENTS-v0.6.0.md` -- OBS/SIM/MODEL/SYNTH requirements
- `.planning/phases/16-bilinear-recovery-benchmark/16-RESEARCH.md` -- Benchmark infrastructure architecture

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- no new libraries, all existing infrastructure verified by source read
- Architecture: HIGH -- CoupledDCMSystem toggle design verified against source; Pyro model pattern verified by task_dcm_model source read
- Forward model: HIGH -- hemodynamic=False toggle is a clean branch in existing code
- Multi-start SVI: HIGH -- run_svi internals verified; guide factory pattern is standard Pyro practice
- Pitfalls: HIGH -- Phase 16.1 init_scale lesson is documented and verified; bilinear instability well-understood from Phase 16
- Prior calibration: MEDIUM -- exact values require empirical sweep (known unknown)
- ELBO model selection: MEDIUM -- standard approach but threshold depends on empirical data

**Research date:** 2026-05-24
**Valid until:** 2026-06-24 (30 days; stable domain, no upstream library changes expected)
