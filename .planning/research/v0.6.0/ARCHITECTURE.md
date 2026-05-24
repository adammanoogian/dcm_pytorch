# Architecture Patterns: Latent Circuit DCM (v0.6.0)

**Domain:** Latent circuit inference from RNN hidden states via bilinear DCM
**Researched:** 2026-05-24
**Overall confidence:** HIGH (direct codebase analysis + Langdon & Engel 2025 paper)

---

## 1. Existing Architecture Summary

The current Pyro-DCM codebase follows a strict component pattern per DCM variant:

```
forward_model  -->  pyro_model  -->  guide (create_guide)  -->  SVI (run_svi)
     |                   |
 simulator          extract_posterior_params
```

Each variant (task, spectral, rDCM) implements all four layers:

| Layer | Task DCM | Spectral DCM | rDCM |
|-------|----------|--------------|------|
| Forward model | `neural_state.py` + `balloon_model.py` + `bold_signal.py` via `coupled_system.py` | `spectral_transfer.py` + `spectral_noise.py` + `csd_computation.py` | `rdcm_forward.py` + `rdcm_posterior.py` |
| Pyro model | `task_dcm_model.py` | `spectral_dcm_model.py` | `rdcm_model.py` |
| Guide | `create_guide()` in `guides.py` (shared) | Same shared factory | Same shared factory |
| Simulator | `task_simulator.py` | `spectral_simulator.py` | `rdcm_simulator.py` |

**Critical invariant:** The guide factory (`create_guide`) and SVI runner (`run_svi`) are generic -- they work with ANY Pyro model that uses `pyro.sample` sites. The new latent circuit DCM model automatically inherits all 6 guide types and 3 ELBO objectives if it follows the `pyro.sample` naming convention.

---

## 2. What Changes: Direct Observation vs Hemodynamic Convolution

### Current task DCM observation path

```
neural_state.py          balloon_model.py        bold_signal.py
dx/dt = A_eff @ x + Cu  -->  hemodynamic ODE  -->  y = V0*(k1*(1-q) + ...)
                              (s, f, v, q)           BOLD %signal change
                              
State: 5N-dimensional (x, s, lnf, lnv, lnq)
Observation: BOLD = nonlinear function of (v, q)
ODE: torchdiffeq.odeint with CoupledDCMSystem (nn.Module)
```

### New latent circuit DCM observation path

```
neural_state.py          direct_observation.py
dx/dt = A_eff @ x + Cu  -->  y = C_obs @ x + noise
                              Linear readout of neural states
                              
State: N-dimensional (x only -- NO hemodynamic states)
Observation: Linear mixing of neural activity
ODE: torchdiffeq.odeint with LatentCircuitSystem (nn.Module)
```

**Key architectural differences:**

| Dimension | Task DCM | Latent Circuit DCM |
|-----------|----------|--------------------|
| State dimensionality | 5N (neural + hemodynamic) | N (neural only) |
| Observation model | Nonlinear (balloon + BOLD equation) | Linear (C_obs @ x + noise) |
| `CoupledDCMSystem` | Wraps neural + hemodynamic ODE | NOT used; new `LatentCircuitSystem` wraps neural ODE only |
| ODE right-hand side | 5N derivatives (dx, ds, dlnf, dlnv, dlnq) | N derivatives (dx only) |
| Initial state | `make_initial_state(N)` -> zeros(5N) | zeros(N) |
| Observation noise | Single noise_prec on BOLD | Noise per observed dimension (M_obs) |
| Downsampling | BOLD downsampled to TR resolution | No TR downsampling; match RNN time grid directly |

---

## 3. New Modules Needed

### 3.1 Forward Models

#### `forward_models/direct_observation.py` (NEW)

**Purpose:** The observation equation `y = C_obs @ x + noise` that replaces the balloon-Windkessel + BOLD pathway when fitting DCM to RNN hidden states.

**Why new file, not extending `bold_signal.py`:** The BOLD signal equation is a specific biophysical model (Buxton balloon, Stephan 2007 Eq. 6) that maps hemodynamic states `(v, q)` to BOLD percent signal change. The direct observation model is a fundamentally different observation equation (linear readout matrix). Conflating them in one file violates the "one equation, one module" citation convention.

```python
def direct_observation(
    x: torch.Tensor,          # (T, N) neural states
    C_obs: torch.Tensor,      # (M_obs, N) observation matrix
) -> torch.Tensor:
    """Direct linear observation of neural states.
    
    y(t) = C_obs @ x(t)
    
    When M_obs == N and C_obs == I, this is full-state observation.
    When M_obs < N, C_obs selects/mixes latent dimensions.
    """
    # y shape: (T, M_obs)
    return (C_obs @ x.unsqueeze(-1)).squeeze(-1)  # or einsum
```

**Complexity:** LOW. This is a single matrix multiply -- the simplest observation model in the codebase.

#### `forward_models/latent_circuit_system.py` (NEW)

**Purpose:** An `nn.Module` ODE right-hand side for torchdiffeq that computes ONLY neural state derivatives (N-dimensional, not 5N). Mirrors `CoupledDCMSystem` but without the hemodynamic wrapper.

**Why not reuse `CoupledDCMSystem`:** The coupled system hardcodes the 5N state layout `[x, s, lnf, lnv, lnq]`, calls `BalloonWindkessel.derivatives()`, and packs 5 derivative blocks. Removing the hemodynamic states from `CoupledDCMSystem` would break every existing caller. A separate N-dimensional system is cleaner.

**Can `NeuralStateEquation` be reused directly?** YES -- the `NeuralStateEquation.derivatives()` method computes `dx/dt = A_eff @ x + C @ u` (bilinear or linear). The new `LatentCircuitSystem` wraps it identically to how `CoupledDCMSystem` wraps it, but without the hemodynamic states.

```python
class LatentCircuitSystem(nn.Module):
    """Neural-only ODE system for latent circuit DCM.
    
    State vector: [x(N)] -- neural activity only.
    No hemodynamic states. Uses NeuralStateEquation.derivatives()
    for the bilinear form dx/dt = A_eff @ x + C @ u.
    
    Drop-in replacement for CoupledDCMSystem in torchdiffeq.odeint,
    but with N-dimensional (not 5N) state vector.
    """
    
    def __init__(self, A, C, input_fn, *, B=None, n_driving_inputs=None):
        super().__init__()
        self.register_buffer("A", A)
        self.register_buffer("C", C)
        self.B = B  # same pattern as CoupledDCMSystem
        self.input_fn = input_fn
        self.n_regions = A.shape[0]
        self.neural = NeuralStateEquation(self.A, self.C)
    
    def forward(self, t, state):
        x = state  # N-dimensional, no unpacking needed
        u_all = self.input_fn(t)
        # Same bilinear/linear branching as CoupledDCMSystem
        if self.B is None or self.B.shape[0] == 0:
            dx = self.A @ x + self.C @ u_all
        else:
            u_drive = u_all[:self.n_driving_inputs]
            u_mod = u_all[self.n_driving_inputs:]
            A_eff = compute_effective_A(self.A, self.B, u_mod)
            dx = A_eff @ x + self.C @ u_drive
        return dx  # N-dimensional
```

**Complexity:** LOW-MEDIUM. Mostly copied structure from `CoupledDCMSystem` with hemodynamic code removed.

### 3.2 Models

#### `models/latent_circuit_dcm_model.py` (NEW)

**Purpose:** Pyro generative model for latent circuit DCM. Samples `A_free`, `C`, and optionally `B_free_j` from priors, runs the neural-only ODE, applies the direct observation model, and evaluates likelihood against observed RNN hidden state trajectories.

**Structure mirrors `task_dcm_model.py`** but replaces the observation path:

```python
def latent_circuit_dcm_model(
    observed_trajectories: torch.Tensor,  # (T, M_obs) -- RNN hidden states
    stimulus: PiecewiseConstantInput,
    a_mask: torch.Tensor,
    c_mask: torch.Tensor,
    t_eval: torch.Tensor,
    dt: float = 0.01,
    *,
    C_obs: torch.Tensor | None = None,   # (M_obs, N) observation matrix
    b_masks: list[torch.Tensor] | None = None,
    stim_mod: PiecewiseConstantInput | None = None,
) -> None:
    """Pyro generative model for latent circuit DCM.
    
    Generative process:
    1. Sample A_free ~ N(0, 1/64), apply mask, parameterize_A
    2. Sample C ~ N(0, 1), apply mask
    3. (Optional bilinear) Sample B_free_j ~ N(0, 1.0) per modulator
    4. Run neural-only ODE: dx/dt = A_eff @ x + C @ u
    5. Apply direct observation: y = C_obs @ x  (or y = x if C_obs=I)
    6. Gaussian likelihood on observed trajectories
    
    NO balloon-Windkessel. NO hemodynamic convolution.
    NO TR downsampling (time grid matches RNN output directly).
    """
```

**Key differences from `task_dcm_model`:**

1. **Initial state:** `torch.zeros(N)` not `make_initial_state(N)` (which returns `zeros(5N)`)
2. **ODE system:** `LatentCircuitSystem` not `CoupledDCMSystem`
3. **Observation:** `C_obs @ x` not `bold_signal(v, q)`
4. **No downsampling:** Time grid matches RNN output directly
5. **Noise model:** May want per-dimension noise (not single `noise_prec`)
6. **C_obs optionality:** When `C_obs=None`, observe neural states directly (`y = x`)

**Reused from existing infrastructure:**
- `parameterize_A`, `parameterize_B`, `compute_effective_A` from `neural_state.py`
- `integrate_ode` from `ode_integrator.py`
- `PiecewiseConstantInput` and `merge_piecewise_inputs` from `ode_integrator.py`
- All priors (A_free, C, B_free_j) identical to `task_dcm_model.py`
- NaN-safe guard pattern from `task_dcm_model.py`

**Automatically gets:** All 6 guide types, 3 ELBO variants, `extract_posterior_params` -- zero changes to `guides.py` needed.

### 3.3 RNN Module

#### `rnn/` package (NEW -- separate from forward_models)

**Rationale for separate package:** The RNN is NOT a DCM forward model. It is a tool-neural-network trained to reproduce neural activity from task stimuli. It lives outside `forward_models/` because:
1. It is trained separately (supervised learning on neural data, not Bayesian inference)
2. It is not part of the DCM generative model
3. Its parameters are not sampled by Pyro
4. It has its own training loop (Adam, MSE loss) distinct from SVI

**Proposed structure:**

```
src/pyro_dcm/rnn/
    __init__.py
    continuous_time_rnn.py   # ContinuousTimeRNN class (nn.Module)
    rnn_trainer.py           # Training loop, loss, early stopping
    latent_extraction.py     # Extract h(t), reduce dimensionality
    synthetic_rnn.py         # Known-connectivity RNN for validation
```

#### `rnn/continuous_time_rnn.py`

**The RNN architecture.** Based on the Langdon & Engel 2025 architecture and PROJECT.md specification ("neural data prediction RNN"):

```python
class ContinuousTimeRNN(nn.Module):
    """Continuous-time RNN for neural data prediction.
    
    Dynamics: tau * dh/dt = -h + f(W_rec @ h + W_in @ u(t))
    
    Discretized (Euler): h(t+dt) = (1 - alpha) * h(t) 
                                   + alpha * f(W_rec @ h(t) + W_in @ u(t))
    where alpha = dt / tau.
    
    Parameters
    ----------
    n_hidden : int
        Number of hidden units.
    n_inputs : int
        Number of external input channels.
    n_outputs : int
        Number of output channels (for supervised training).
    tau : float
        Time constant (seconds). Default 0.1.
    activation : str
        'relu' (default, matching Langdon & Engel) or 'tanh'.
    """
    
    def __init__(self, n_hidden, n_inputs, n_outputs, tau=0.1, ...):
        self.W_rec = nn.Linear(n_hidden, n_hidden, bias=False)
        self.W_in = nn.Linear(n_inputs, n_hidden, bias=False)
        self.W_out = nn.Linear(n_hidden, n_outputs, bias=False)
```

**Why NOT use torchdiffeq for the RNN?** The RNN training loop needs discrete-time Euler integration for efficiency (thousands of epochs). The Langdon & Engel implementation uses explicit Euler with `alpha=dt/tau`. Using torchdiffeq's `odeint` for RNN training would be unnecessarily slow (adaptive solvers, adjoint method overhead). torchdiffeq is reserved for the DCM fitting stage where ODE accuracy matters for parameter recovery.

**Langdon & Engel comparison point:** Their RNN uses `alpha=0.2`, `sigma_rec=0.15`, ReLU activation, 50 units (40E/10I with Dale's law). Our implementation should be configurable but default to comparable settings.

#### `rnn/rnn_trainer.py`

**Training loop for the RNN.** Standard supervised training:

```python
def train_rnn(
    rnn: ContinuousTimeRNN,
    stimuli: torch.Tensor,       # (n_trials, T, n_inputs)
    targets: torch.Tensor,       # (n_trials, T, n_outputs)
    *,
    n_epochs: int = 1000,
    lr: float = 0.02,
    batch_size: int = 128,
    patience: int = 25,
) -> dict:
    """Train RNN to predict neural activity from task stimuli.
    
    Loss: MSE between W_out @ h(t) and target activity.
    Optimizer: Adam with weight decay.
    Early stopping with patience.
    
    Returns dict with 'losses', 'rnn' (trained), 'best_epoch'.
    """
```

**This is NOT Pyro SVI.** The RNN training is plain PyTorch supervised learning. It produces a trained RNN whose hidden states are then used as observed data for the DCM.

#### `rnn/latent_extraction.py`

**Extract and reduce RNN hidden states for DCM fitting.**

```python
def extract_latent_trajectories(
    rnn: ContinuousTimeRNN,
    stimuli: torch.Tensor,       # (n_trials, T, n_inputs)
) -> torch.Tensor:
    """Run trained RNN forward, collect hidden state trajectories.
    
    Returns h(t) shape (n_trials, T, n_hidden).
    """

def reduce_dimensionality(
    trajectories: torch.Tensor,  # (n_trials, T, n_hidden)
    n_components: int,
    method: str = 'pca',
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce hidden state dimensionality for DCM fitting.
    
    Returns:
    - reduced: (n_trials, T, n_components) -- reduced trajectories
    - C_obs: (n_hidden, n_components) -- the projection matrix
             such that h_approx = C_obs @ x_reduced
    
    Methods:
    - 'pca': Standard PCA (torch.pca_lowrank)
    - 'task_aligned': Langdon & Engel style task-variable alignment
    """
```

**PCA vs task-aligned dimensionality reduction:** PCA is the default because it is rotation-agnostic and does not require task-variable labels. The Langdon & Engel "task-aligned" approach (Cayley-transform orthonormal embedding) requires knowing which latent dimensions correspond to which task variables a priori. PCA is appropriate for our initial implementation; task-aligned can be added later.

#### `rnn/synthetic_rnn.py`

**Known-connectivity synthetic RNN for parameter recovery validation.**

```python
def make_known_circuit_rnn(
    A_true: torch.Tensor,        # (N, N) ground-truth connectivity
    C_true: torch.Tensor,        # (N, M) input weights
    *,
    n_hidden: int = 50,
    B_true: torch.Tensor | None = None,
) -> ContinuousTimeRNN:
    """Create an RNN whose hidden dynamics embed a known circuit.
    
    Constructs W_rec such that the low-rank structure 
    Q^T @ W_rec @ Q approximately equals A_true, where Q is a 
    known embedding matrix. This provides ground truth for 
    parameter recovery: fit DCM to h(t), verify recovered A 
    matches A_true.
    """
```

### 3.4 Simulators

#### `simulators/latent_circuit_simulator.py` (NEW)

**Purpose:** Generate synthetic latent circuit data (neural-only trajectories + observations) for testing the DCM model WITHOUT needing to train an RNN first. This is the fast path for unit tests and parameter recovery.

```python
def simulate_latent_circuit_dcm(
    A: torch.Tensor,
    C: torch.Tensor,
    stimulus: PiecewiseConstantInput,
    C_obs: torch.Tensor | None = None,
    duration: float = 10.0,       # shorter than BOLD (no hemodynamic lag)
    dt: float = 0.01,
    SNR: float = 10.0,
    *,
    B_list: torch.Tensor | None = None,
    stimulus_mod: PiecewiseConstantInput | None = None,
) -> dict:
    """Generate synthetic latent circuit observations.
    
    Returns dict with:
    - 'observed': (T, M_obs) noisy observations
    - 'neural': (T, N) clean neural activity
    - 'times': (T,) time grid
    - 'params': {A, C, C_obs, ...}
    """
```

**Mirrors `simulate_task_dcm` but:**
- Uses `LatentCircuitSystem` instead of `CoupledDCMSystem`
- Uses `direct_observation` instead of `bold_signal`
- No TR downsampling
- Shorter default duration (RNN dynamics are faster than hemodynamic)
- Initial state is `zeros(N)` not `zeros(5N)`

---

## 4. Component Integration Map

### What is REUSED (zero changes)

| Component | File | Why Reusable |
|-----------|------|-------------|
| `parameterize_A` | `neural_state.py` | Same A-matrix parameterization |
| `parameterize_B` | `neural_state.py` | Same B-matrix masking |
| `compute_effective_A` | `neural_state.py` | Same bilinear composition |
| `NeuralStateEquation` | `neural_state.py` | Same `dx/dt` computation |
| `PiecewiseConstantInput` | `ode_integrator.py` | Same stimulus handling |
| `merge_piecewise_inputs` | `ode_integrator.py` | Same input merging for bilinear |
| `integrate_ode` | `ode_integrator.py` | Same ODE solver wrapper |
| `create_guide` | `guides.py` | Auto-discovers sample sites |
| `run_svi` | `guides.py` | Generic SVI runner |
| `extract_posterior_params` | `guides.py` | Generic posterior extraction |
| Stimulus utilities | `task_simulator.py` | `make_block_stimulus`, `make_event_stimulus`, `make_epoch_stimulus` |
| All 6 guide types | `guides.py` | AutoDelta through AutoIAF |
| All 3 ELBO variants | `guides.py` | Trace, TraceMeanField, Renyi |
| Benchmark infrastructure | `benchmarks/` | `.npz` fixtures, `BenchmarkConfig`, figure pipeline |

### What is NEW (must be built)

| Component | File | Complexity | Depends On |
|-----------|------|-----------|------------|
| Direct observation equation | `forward_models/direct_observation.py` | LOW | Nothing new |
| Neural-only ODE system | `forward_models/latent_circuit_system.py` | LOW | `neural_state.py` |
| Latent circuit Pyro model | `models/latent_circuit_dcm_model.py` | MEDIUM | `latent_circuit_system.py`, `direct_observation.py` |
| Continuous-time RNN | `rnn/continuous_time_rnn.py` | MEDIUM | PyTorch only |
| RNN trainer | `rnn/rnn_trainer.py` | MEDIUM | `continuous_time_rnn.py` |
| Latent extraction | `rnn/latent_extraction.py` | LOW | Trained RNN |
| Synthetic RNN | `rnn/synthetic_rnn.py` | MEDIUM | `continuous_time_rnn.py` |
| Latent circuit simulator | `simulators/latent_circuit_simulator.py` | LOW | `latent_circuit_system.py`, `direct_observation.py` |

### What is MODIFIED (minimal edits)

| Component | File | Change | Why |
|-----------|------|--------|-----|
| `forward_models/__init__.py` | Re-export new modules | Convention |
| `models/__init__.py` | Re-export `latent_circuit_dcm_model` | Convention |
| `simulators/__init__.py` | Re-export `simulate_latent_circuit_dcm` | Convention |
| New `rnn/__init__.py` | Package init | New package |

---

## 5. Data Flow: End-to-End Pipeline

### Path A: Synthetic RNN Ground Truth (Parameter Recovery)

```
1. Define ground truth: A_true, C_true, B_true
   |
2. make_known_circuit_rnn(A_true, C_true, B_true)
   |  -> ContinuousTimeRNN with known low-rank W_rec structure
   |
3. train_rnn(rnn, task_stimuli, neural_targets)
   |  -> Trained RNN (plain PyTorch, NOT Pyro)
   |
4. extract_latent_trajectories(trained_rnn, stimuli)
   |  -> h(t) shape (n_trials, T, n_hidden)
   |
5. reduce_dimensionality(h, n_components=N)
   |  -> x_reduced (n_trials, T, N), C_obs (n_hidden, N)
   |
6. latent_circuit_dcm_model(x_reduced, stimulus, a_mask, c_mask, ...)
   |  Pyro model: sample A, C, (B); neural ODE; direct obs; likelihood
   |
7. create_guide(latent_circuit_dcm_model) + run_svi(...)
   |  -> Trained variational posterior
   |
8. extract_posterior_params(guide)
   |  -> A_recovered, C_recovered, (B_recovered)
   |
9. Compare: A_recovered vs A_true (RMSE, correlation, coverage)
```

### Path B: Direct Simulator (Fast Unit Tests)

```
1. Define ground truth: A_true, C_true, (B_true)
   |
2. simulate_latent_circuit_dcm(A_true, C_true, stimulus, ...)
   |  -> observed_trajectories, neural_true
   |  (No RNN needed -- direct ODE simulation + observation noise)
   |
3. latent_circuit_dcm_model(observed_trajectories, stimulus, ...)
   |
4. create_guide(...) + run_svi(...)
   |
5. Compare: recovered vs true
```

**Path B is critical** for fast iteration. Path A (full RNN pipeline) is needed for the milestone's scientific claim, but Path B enables rapid development and debugging of the DCM model itself.

### Path C: Langdon & Engel Comparison

```
1. Train task-RNN on cognitive task (same paradigm as L&E 2025)
   |
2. Extract h(t) from trained RNN
   |
3a. Fit Langdon & Engel latent circuit model (their code)
    |  -> w_rec (nonlinear f()), Q embedding
    |
3b. Fit bilinear DCM (our model)
    |  -> A, B_j, C with posterior uncertainty
    |
4. Compare:
   - Do both recover similar circuit structure?
   - Bilinear B_j vs nonlinear f() interpretability
   - Posterior uncertainty (our advantage -- L&E has no uncertainty)
```

---

## 6. Detailed Observation Model Design

### When `C_obs = I` (Full State Observation)

The simplest case: observe all N latent dimensions directly. Appropriate when:
- `n_components == N` after PCA
- The DCM latent dimension matches the RNN reduced dimension
- No additional mixing/selection needed

```python
# In latent_circuit_dcm_model:
if C_obs is None:
    predicted_obs = neural_activity  # (T, N) -- identity observation
else:
    predicted_obs = (C_obs @ neural_activity.unsqueeze(-1)).squeeze(-1)
```

### When `C_obs != I` (Partial/Mixed Observation)

Needed when:
- Fitting DCM with fewer latent states than observed dimensions
- M_obs > N: more observed channels than latent states
- C_obs becomes part of the generative model (could be sampled or fixed)

**Recommendation for v0.6.0:** Start with `C_obs = I` (full state observation). This avoids the identifiability headache of jointly inferring A and C_obs. Defer partial observation to v0.7.0+.

### Noise Model Options

**Option 1: Scalar noise precision (like task DCM)**
```python
noise_prec = pyro.sample("noise_prec", Gamma(1, 1))
noise_std = (1.0 / noise_prec).sqrt()
pyro.sample("obs", Normal(predicted, noise_std).to_event(2), obs=observed)
```

**Option 2: Per-dimension noise (more flexible)**
```python
noise_prec = pyro.sample("noise_prec", 
    Gamma(ones(N), ones(N)).to_event(1))
noise_std = (1.0 / noise_prec).sqrt()
pyro.sample("obs", Normal(predicted, noise_std.unsqueeze(0)).to_event(2), obs=observed)
```

**Recommendation:** Start with scalar noise (Option 1) for consistency with existing models. Per-dimension noise is a natural extension if recovery benchmarks show noise heterogeneity is a bottleneck.

---

## 7. Comparison: Pyro-DCM vs Langdon & Engel 2025

| Aspect | Langdon & Engel 2025 | Pyro-DCM v0.6.0 |
|--------|---------------------|------------------|
| **Latent dynamics** | x_dot = -x + f(w_rec @ x + w_in @ u) | dx/dt = A_eff @ x + C @ u |
| **Nonlinearity** | ReLU f() on entire RHS | Bilinear B_j only (linear in x given u) |
| **Observation model** | y = Q @ x (orthonormal embedding Q) | y = C_obs @ x (or identity) |
| **Inference** | MSE loss, Adam optimizer, point estimate | ELBO, SVI, full posterior |
| **Uncertainty** | None (ensemble of 100 fits for stability) | Per-parameter posterior (mean, std, CI) |
| **Connectivity** | w_rec (unconstrained after fitting) | A with negative-diagonal guarantee + B_j masks |
| **Model comparison** | Not supported | ELBO-based Bayesian model comparison |
| **Task modulation** | Implicit in nonlinear f() | Explicit bilinear B_j per condition |
| **Dimensionality** | 8 latent nodes (task-structure-guided) | N latent states (PCA or fixed) |
| **Code** | PyTorch, custom optimizer | Pyro PPL, torchdiffeq, reuses existing DCM infrastructure |

**Our key advantages:**
1. Posterior uncertainty on A and B (not just point estimates)
2. Explicit bilinear parameterization of task modulation (B_j interpretable as condition-specific connectivity change)
3. ELBO-based model comparison (which connectivity architecture best explains the data?)
4. Inherited infrastructure (6 guide types, 3 ELBO variants, benchmark pipeline)

**Their key advantages:**
1. Nonlinear dynamics (ReLU) may capture phenomena bilinear cannot
2. Task-aligned dimensionality reduction assigns meaning to each latent dimension
3. Published and validated on real neural data (monkey PFC)

---

## 8. Suggested Build Order

Based on dependency analysis and the principle of testing each component before integration:

### Phase 1: Direct Observation Forward Model + Simulator
**Build:** `direct_observation.py`, `latent_circuit_system.py`, `latent_circuit_simulator.py`
**Test:** Simulate known A matrix, verify ODE integration, check observation shapes
**Rationale:** These are the simplest new components and enable Path B (fast unit tests) immediately. Everything else depends on the forward model being correct.

### Phase 2: Latent Circuit Pyro Model + Recovery
**Build:** `latent_circuit_dcm_model.py`
**Test:** Parameter recovery via Path B (simulator -> DCM fit -> compare). Verify A recovery RMSE, then add bilinear B recovery.
**Rationale:** This is the core scientific deliverable. Must pass recovery benchmarks before RNN work begins. Reuses all existing Pyro infrastructure.

### Phase 3: RNN Training + Latent Extraction
**Build:** `continuous_time_rnn.py`, `rnn_trainer.py`, `latent_extraction.py`, `synthetic_rnn.py`
**Test:** Train RNN on synthetic task, extract h(t), verify dimensionality reduction
**Rationale:** RNN is a tool, not the scientific contribution. Build it after the DCM model is proven to work on clean simulated data.

### Phase 4: End-to-End Pipeline + Comparison
**Build:** Full Path A pipeline, comparison to Langdon & Engel
**Test:** End-to-end: Train RNN -> extract latents -> fit DCM -> recover parameters. Compare to L&E results.
**Rationale:** Integration test. Only meaningful after all components are individually validated.

### Phase 5: Publication Artifacts
**Build:** Figures, methods section, benchmark documentation
**Test:** Publication-quality outputs

---

## 9. Anti-Patterns to Avoid

### Anti-Pattern 1: Modifying CoupledDCMSystem to support "no hemodynamics" mode
**Why bad:** CoupledDCMSystem is battle-tested with 5N state layout. Adding a "skip hemodynamics" flag creates a fragile branching nightmare and risks breaking the 40+ existing tests. The latent circuit system is architecturally simpler (N states, not 5N) and deserves its own clean module.
**Instead:** New `LatentCircuitSystem` with N-dimensional state vector.

### Anti-Pattern 2: Training the RNN inside the Pyro model
**Why bad:** The RNN is a tool for generating observed data, not a component of the DCM generative model. Mixing RNN training (MSE, Adam, epochs) with DCM inference (ELBO, SVI, posterior) creates a confused training objective. The Langdon & Engel approach is explicitly two-stage: train RNN, then fit interpretable model to RNN hidden states.
**Instead:** Two completely separate training loops. RNN training is plain PyTorch. DCM inference is Pyro SVI.

### Anti-Pattern 3: Making C_obs a sampled parameter in v0.6.0
**Why bad:** Jointly inferring A (connectivity) and C_obs (observation matrix) creates a rotation ambiguity: any orthogonal transform R satisfies `C_obs @ x = (C_obs @ R) @ (R^T @ x)` with `A' = R^T @ A @ R`. This is the "latent space identifiability" problem that Durstewitz et al. 2023 (REF-052) document extensively.
**Instead:** Fix C_obs = I for v0.6.0 (full state observation). Defer partial observation to v0.7.0+ with explicit identifiability constraints.

### Anti-Pattern 4: Using torchdiffeq for RNN training
**Why bad:** RNN training needs thousands of epochs of gradient descent. Using torchdiffeq's odeint (with error control, adaptive stepping) for each forward pass would be 10-100x slower than simple Euler integration. The Langdon & Engel code uses explicit Euler for this reason.
**Instead:** Euler discretization for RNN training. Reserve torchdiffeq for DCM fitting where ODE accuracy matters for parameter recovery.

### Anti-Pattern 5: Putting RNN code in `forward_models/`
**Why bad:** The `forward_models/` package contains DCM generative model components -- each file traces to a specific equation in REFERENCES.md. The RNN is not part of the DCM generative model; it is a data-generation tool.
**Instead:** New `rnn/` package at `src/pyro_dcm/rnn/`.

---

## 10. Directory Structure After v0.6.0

```
src/pyro_dcm/
    forward_models/
        __init__.py
        neural_state.py           # EXISTING (reused: A, B, bilinear)
        balloon_model.py          # EXISTING (not used by latent circuit)
        bold_signal.py            # EXISTING (not used by latent circuit)
        coupled_system.py         # EXISTING (not used by latent circuit)
        direct_observation.py     # NEW: y = C_obs @ x + noise
        latent_circuit_system.py  # NEW: N-dimensional ODE system
        spectral_transfer.py      # EXISTING
        spectral_noise.py         # EXISTING
        csd_computation.py        # EXISTING
        rdcm_forward.py           # EXISTING
        rdcm_posterior.py         # EXISTING
    models/
        __init__.py
        task_dcm_model.py             # EXISTING
        spectral_dcm_model.py         # EXISTING
        rdcm_model.py                 # EXISTING
        latent_circuit_dcm_model.py   # NEW
        guides.py                     # EXISTING (no changes)
        amortized_wrappers.py         # EXISTING (no changes)
    rnn/                              # NEW PACKAGE
        __init__.py
        continuous_time_rnn.py        # ContinuousTimeRNN class
        rnn_trainer.py                # Training loop
        latent_extraction.py          # h(t) extraction + dim reduction
        synthetic_rnn.py              # Known-connectivity test RNN
    simulators/
        __init__.py
        task_simulator.py                 # EXISTING
        spectral_simulator.py             # EXISTING
        rdcm_simulator.py                 # EXISTING
        latent_circuit_simulator.py       # NEW
    io/
        __init__.py
        mne_loader.py             # EXISTING
        bids_loader.py            # EXISTING
    utils/
        __init__.py
        ode_integrator.py         # EXISTING (reused)
        circuit_viz.py            # EXISTING
    guides/
        __init__.py
        amortized_flow.py         # EXISTING
        parameter_packing.py      # EXISTING
        summary_networks.py       # EXISTING
```

---

## 11. Scalability Considerations

| Concern | N=3 (unit test) | N=8 (L&E comparison) | N=20+ (ambitious) |
|---------|-----------------|---------------------|-------------------|
| ODE state dimension | 3 (fast) | 8 (fast) | 20 (still fast -- no hemodynamics) |
| A_free parameters | 9 | 64 | 400 |
| B_free per modulator | 9 | 64 | 400 |
| SVI step time | <0.1s | ~0.5s | ~2s |
| RNN training | <1 min | ~5 min | ~20 min |

**Key insight:** Latent circuit DCM is MUCH faster than task DCM because the ODE is N-dimensional (not 5N) and there is no hemodynamic nonlinearity. SVI convergence should also be faster because the observation model is linear.

---

## Sources

- Langdon & Engel (2025). [Latent circuit inference from heterogeneous neural responses during cognitive tasks](https://www.nature.com/articles/s41593-025-01869-7). Nature Neuroscience.
- [engellab/latentcircuit GitHub](https://github.com/engellab/latentcircuit) -- reference implementation.
- [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC11893458/) -- technical details of RNN + latent circuit equations.
- Existing Pyro-DCM codebase: `neural_state.py`, `coupled_system.py`, `task_dcm_model.py`, `guides.py`, `task_simulator.py` (direct code analysis).
- PROJECT.md v0.6.0 milestone specification.
- Durstewitz et al. (2023), REF-052 -- identifiability / rotational degeneracy in neural state spaces.
