# Phase 21: CT-RNN Training & Latent Extraction - Research

**Researched:** 2026-05-25
**Domain:** Continuous-time RNN training on CDDM, fixed-point analysis, PCA dimensionality reduction
**Confidence:** HIGH (stack + architecture verified from v0.6.0 STACK.md, ARCHITECTURE.md, PITFALLS.md; neurogym API verified via official docs; PyTorch APIs stable)

---

## Summary

Phase 21 builds the RNN data pipeline that produces the observed data bilinear DCM fits to: train
continuous-time RNNs on a cognitive task, extract hidden state trajectories, and reduce them to a
low-dimensional space with quality diagnostics. This phase is fully independent of Phase 20 (no
shared code under construction) and requires only two new optional dependencies (neurogym>=2.3,
scikit-learn>=1.3) not yet in pyproject.toml.

The architecture is entirely clear from the v0.6.0 milestone research: the CT-RNN is a custom
`nn.Module` implementing Euler integration with `alpha = dt/tau`, trained with standard PyTorch
BPTT (no Pyro involved), using neurogym's `ContextDecisionMaking-v0` task. Fixed-point finding is
~50 lines of PyTorch (minimize `||dx/dt||^2` via Adam). PCA dimensionality reduction uses
`sklearn.decomposition.PCA` for offline analysis plus an output-R-squared quality gate. Everything
needed is standard PyTorch + two new optional packages.

The biggest implementation decision is already locked by the v0.6.0 milestone research: write a
custom ~100-line `ContinuousTimeRNN` module referencing the Engel lab's `trainRNNbrain` for
correctness but NOT depending on it. The task environment uses `neurogym.Dataset` for batched
trial generation. Training 20 RNNs with H=64-256 requires cluster routing (estimated 30-60 min
per RNN on M3, or 10-20 hours total -- clearly >3 min threshold).

**Primary recommendation:** New `src/pyro_dcm/rnn/` package with four files:
`continuous_time_rnn.py`, `rnn_trainer.py`, `latent_extraction.py`, `fixed_point_analysis.py`.
Add `[latent]` optional extra to pyproject.toml with `neurogym>=2.3` and `scikit-learn>=1.3`.
All 20-RNN training runs go to M3 cluster via sbatch.

---

## Standard Stack

### Core (existing in project -- no changes needed)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.x | CT-RNN nn.Module, BPTT, fixed-point optimization | Foundation; `torch.linalg.eig` available since 1.9 |
| `torch.autograd.functional` | 2.x | Jacobian computation at fixed points | Built-in, differentiable |
| `torch.optim.Adam` | 2.x | RNN training + fixed-point optimization | Standard optimizer |

### New Dependencies (add to pyproject.toml `[latent]` extra)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| neurogym | >=2.3 (current: 2.3.1) | CDDM task environment (Gymnasium API) | Exact task Langdon & Engel 2025 used; 33 standard tasks; framework-agnostic numpy output |
| scikit-learn | >=1.3 | PCA for offline dimensionality reduction + `explained_variance_ratio_` | Standard; provides scree diagnostic; not differentiable (offline use only) |

### Libraries Explicitly Rejected

| Library | Reason |
|---------|--------|
| trainRNNbrain | Adds hydra-core, opinionated config -- read for correctness, do not depend |
| latentcircuit | Benchmark target, not a dependency; Python 3.8 target mismatches ours |
| FixedPointFinder | Stale (2022), TensorFlow-dependent; reimplementable in ~50 PyTorch lines |
| PsychRNN | TensorFlow-based, unmaintained since 2021 |
| gymnasium | Transitive dependency of neurogym; do not list separately |

**Installation:**
```toml
[project.optional-dependencies]
latent = [
    "neurogym>=2.3",
    "scikit-learn>=1.3",
]
```

Add `latent` alongside existing `benchmark`, `mne`, `dev` groups. Mirrors the `[mne]` pattern exactly.

---

## Architecture Patterns

### Recommended Project Structure

```
src/pyro_dcm/
  rnn/
    __init__.py
    continuous_time_rnn.py   # ContinuousTimeRNN(nn.Module), Euler alpha=dt/tau
    rnn_trainer.py           # train_rnn() function, Adam, cross-entropy loss
    latent_extraction.py     # extract_trajectories(), pca_reduce(), output_r_squared_gate()
    fixed_point_analysis.py  # find_fixed_points(), compute_jacobian(), classify_stability()

checkpoints/
  rnn/                       # Trained RNN weights (.pt files)
    seed_{N}_H{H}.pt

data/                        # (new) RNN trajectory data
  rnn_trajectories/
    seed_{N}_condition_{C}.npz
```

The `rnn/` package lives in `src/pyro_dcm/` alongside `models/`, `simulators/`, `forward_models/`.
It is a pure PyTorch package -- no Pyro imports anywhere in it. The RNN is a data-generation tool;
the DCM model (Phase 20) is the scientific contribution.

### Pattern 1: ContinuousTimeRNN (Euler Discrete-Time Form)

**What:** `nn.Module` implementing `tau * dh/dt = -h + f(W_rec @ h + W_in @ u + b)` in Euler
discrete-time as `h_{t+1} = (1 - alpha) * h_t + alpha * f(W_rec @ h_t + W_in @ u_t + b)` where
`alpha = dt / tau`. This matches Langdon & Engel (2025) exactly (their Eq. matching `trainRNNbrain`).

**Key design choices:**
- `alpha = dt / tau` is a fixed scalar (not a learnable parameter in v0.6.0)
- Activation: ReLU as required by RNN-01; tanh is a secondary option for diagnostics (LC1)
- W_rec initialized with random Gaussian std = 1/sqrt(H) (stable spectrum)
- W_in, W_out initialized with Gaussian std = 1/sqrt(H) or 1/sqrt(input_dim)
- Noise injection during training: optional additive Gaussian noise on h(t) at each step
  (standard regularization per trainRNNbrain; prevents overfitting to deterministic dynamics)
- Output readout: `z_t = W_out @ h_t` (linear; softmax applied in loss, not in module)

**Shape conventions:**
```
u: (T, M_in)     -- input time series (neurogym obs)
h: (T, H)        -- hidden state trajectories
z: (T, M_out)    -- output (pre-softmax logits)
```

**Implementation template:**
```python
class ContinuousTimeRNN(nn.Module):
    """Continuous-time RNN with Euler integration.

    Implements tau * dh/dt = -h + f(W_rec @ h + W_in @ u + b).
    Discretized as h[t+1] = (1-alpha)*h[t] + alpha*f(W_rec@h[t] + W_in@u[t] + b)
    where alpha = dt / tau.

    Implements [REF-NEW-001] Langdon & Engel (2025) Eq. matching trainRNNbrain.

    Parameters
    ----------
    n_input : int  -- input dimension (neurogym obs_size)
    n_hidden : int -- hidden units H (64-256)
    n_output : int -- output dimension (neurogym act_size)
    tau : float    -- time constant (default 1.0 in normalized units)
    dt : float     -- integration step size (alpha = dt/tau)
    activation : str -- 'relu' (required by RNN-01) or 'tanh'
    noise_std : float -- training noise on hidden states (default 0.0)
    """
    def forward(
        self,
        u: torch.Tensor,           # (T, M_in) or (T, B, M_in) batched
        h0: torch.Tensor | None,   # Initial hidden state; None -> zeros
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Returns (z, h_traj): (T, M_out), (T, H) or (T, B, M_out), (T, B, H)
        ...
```

**Why Euler, not torchdiffeq.odeint for training:**
- neurogym produces fixed-dt observations; adaptive stepping provides no benefit
- Euler is O(T) not O(T * steps_per_interval); deterministic computation graph
- Langdon & Engel and trainRNNbrain both use Euler
- For analysis (long autonomous runs), `torchdiffeq.odeint` wrapping the same dynamics is optional

### Pattern 2: neurogym CDDM Task Environment

**What:** `ContextDecisionMaking-v0` from neurogym provides the CDDM task. The Gymnasium API
returns numpy arrays; we convert to tensors.

**API (verified via official docs + neurogym.github.io):**
```python
import neurogym as ngym

# Create environment
env = ngym.make('ContextDecisionMaking-v0', dt=100)  # dt in ms

# Use Dataset for batched supervised learning
dataset = ngym.Dataset(
    'ContextDecisionMaking-v0',
    env_kwargs={'dt': 100},
    batch_size=32,
    seq_len=100,
)

# Each call returns a batch
inputs, labels = dataset()
# inputs: (seq_len, batch_size, obs_size) -- numpy, convert to tensor
# labels: (seq_len * batch_size,) -- ground truth actions, flattened
```

**CDDM task structure (verified from neurogym docs + literature):**
- Two sensory modalities (e.g., color coherence, motion coherence)
- One context cue indicating which modality is relevant
- Trial periods: fixation, stimulus, decision
- Observation channels: [fixation signal, modality_1_A, modality_1_B, modality_2_A, modality_2_B, context]
  (exact count depends on neurogym version; `ob_size = env.observation_space.shape[0]`)
- Action space: 3 choices (fixation=0, choice_A=1, choice_B=2) or similar 3-class
  (`act_size = env.action_space.n`)
- The context cue (`context_id`) indicates which modality's coherence to use for the decision
- dt parameter controls trial timing in milliseconds (default 100ms = 0.1s per step)

**Important:** Do NOT hardcode `ob_size` or `act_size`. Always query from the environment:
```python
env = ngym.make('ContextDecisionMaking-v0', dt=100)
ob_size = env.observation_space.shape[0]   # query at runtime
act_size = env.action_space.n              # query at runtime
```

**Why not write custom CDDM:**
- Reimplementing one task correctly (stimulus statistics, timing, coherence levels) is
  200-400 lines of fragile code (per v0.6.0 STACK.md)
- neurogym tasks are peer-reviewed and match Yang et al. 2019 conventions
- Langdon & Engel 2025 and Yang et al. 2019 both use these exact environments

### Pattern 3: RNN Training Loop (BPTT)

**What:** Standard PyTorch supervised training. Loss = cross-entropy on action logits vs
neurogym ground truth. Optimizer = Adam. Gradient clipping applied.

**Training loop structure:**
```python
def train_rnn(
    rnn: ContinuousTimeRNN,
    dataset: ngym.Dataset,
    n_steps: int = 2000,
    lr: float = 1e-3,
    grad_clip: float = 1.0,
    criterion_acc: float = 0.85,    # early stopping on accuracy
) -> dict:
    optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for step in range(n_steps):
        inputs, labels = dataset()
        inputs = torch.tensor(inputs, dtype=torch.float32)  # (T, B, obs_size)
        labels = torch.tensor(labels, dtype=torch.long)     # (T*B,)

        z, h = rnn(inputs, h0=None)  # (T, B, act_size), (T, B, H)
        loss = criterion(z.reshape(-1, act_size), labels)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(rnn.parameters(), grad_clip)
        optimizer.step()
```

**Hyperparameters (from neurogym official PyTorch example + trainRNNbrain reference):**
- Learning rate: 1e-3 (neurogym example) -- may need 1e-4 for large H
- Batch size: 16-32 trials per step
- Sequence length: determined by neurogym task (typically 80-200 steps per trial at dt=100ms)
- Gradient clip: 1.0 (standard for RNNs with BPTT)
- Hidden units H: 64-256 per RNN-03
- Training steps: 2000-5000 (neurogym example achieves 83% at 2000 steps)
- Criterion: cross-entropy on choice readout (RNN-01)

**BPTT across full trial (truncated BPTT is not needed):** Each trial is typically 10-30s of
cognitive time at dt=100ms = 100-300 steps. This is manageable for full BPTT without truncation.
For longer trials, truncate at 200 steps.

**Performance criterion:** 85%+ accuracy on held-out trials before saving weights and extracting
trajectories. This matches the L&E convention and ensures RNNs have learned the task.

### Pattern 4: Trajectory Extraction (Per Trial Condition)

**What:** After training, run the RNN on each trial condition to extract full h(t) trajectories.
Store per-condition (context x coherence combinations) for PCA and DCM fitting.

**Trial conditions for CDDM:**
- Context 1 (attend modality 1) x coherences [0.04, 0.08, 0.16, 0.32, 0.64] = 5 conditions
- Context 2 (attend modality 2) x coherences [0.04, 0.08, 0.16, 0.32, 0.64] = 5 conditions
- Total: 10 conditions, typically 20+ trials per condition for averaging

**Extraction pattern:**
```python
def extract_trajectories(
    rnn: ContinuousTimeRNN,
    env: gym.Env,
    n_trials_per_condition: int = 50,
    conditions: list[dict],  # list of {context_id, coherence} dicts
) -> dict[str, np.ndarray]:
    """Returns {condition_key: h_trajectories (n_trials, T, H)}."""
    ...
```

**Save format:** `checkpoints/rnn/seed_{N}_H{H}.pt` for weights; `data/rnn_trajectories/seed_{N}_condition_{C}.npz` for trajectories (numpy `.npz` for consistency with existing fixture pattern).

### Pattern 5: PCA Dimensionality Reduction

**What:** Offline PCA on concatenated h(t) trajectories across conditions and trials. Uses
`sklearn.decomposition.PCA` for `explained_variance_ratio_` (scree diagnostic). The PCA is fitted
ONCE on the full trajectory set, then applied to each condition.

**Critical:** PCA is fitted on the training trajectories, then applied to held-out test trajectories.
Fitting PCA on all trajectories and evaluating on the same set is data leakage for the output-R²
gate.

```python
from sklearn.decomposition import PCA

def pca_reduce(
    h_all: np.ndarray,    # (n_samples, H) -- all training trajectories stacked
    n_components: int,    # N target dimensions
) -> tuple[PCA, np.ndarray]:
    """Returns (fitted_pca, projected) where projected is (n_samples, N)."""
    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(h_all)
    return pca, projected

def variance_explained_diagnostic(pca: PCA) -> dict:
    """Reports cumulative variance explained vs N; recommends N where marginal < 5%."""
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    marginal = np.diff(pca.explained_variance_ratio_, prepend=0)
    recommended_n = np.searchsorted(marginal[::-1], 0.05)  # where marginal drops below 5%
    return {"cumulative": cumulative, "marginal": marginal, "recommended_n": recommended_n}
```

**Output R² gate (DIM-02):** Verify that PCA-projected states reconstruct RNN behavioral readout:
```python
def output_r_squared_gate(
    h_projected: np.ndarray,    # (n_samples, N) PCA-projected
    z_true: np.ndarray,         # (n_samples, act_size) behavioral readout
    w_out: np.ndarray,          # (act_size, H) RNN output weights
    pca: PCA,                   # fitted PCA to reconstruct W_out in reduced space
    threshold: float = 0.90,    # DIM-02 gate: R² >= 0.90
) -> tuple[float, bool]:
    """Check if PCA-projected states reconstruct behavioral readout at R² >= threshold.

    The output weights W_out applied to h recover z = W_out @ h.
    In PCA space: z ≈ (W_out @ pca.components_.T) @ h_projected.
    R² measures how well this reconstruction matches the true output.
    """
    w_out_pca = w_out @ pca.components_.T           # (act_size, N)
    z_pred = h_projected @ w_out_pca.T              # (n_samples, act_size)
    ss_res = ((z_true - z_pred) ** 2).sum()
    ss_tot = ((z_true - z_true.mean(0)) ** 2).sum()
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    return float(r2), r2 >= threshold
```

### Pattern 6: Fixed-Point Analysis

**What:** Find fixed points of the CT-RNN dynamics via gradient descent on `||dx/dt||²`. Compute
Jacobians at fixed points. Classify stability via eigenvalue decomposition. This is the RNN-04
linearization quality diagnostic.

**Fixed-point finding (DO NOT use FixedPointFinder; implement in PyTorch):**
```python
def find_fixed_points(
    rnn: ContinuousTimeRNN,
    u_context: torch.Tensor,    # (M_in,) context input at which to find FPs
    n_inits: int = 100,         # multiple random initializations
    n_steps: int = 5000,        # optimization steps per init
    tol: float = 1e-12,         # convergence criterion on ||dh/dt||^2
) -> list[torch.Tensor]:
    """Find fixed points by minimizing ||f(h, u) - h||^2 via Adam.

    For CT-RNN: fixed point satisfies tau * dh/dt = -h + f(W_rec @ h + W_in @ u + b) = 0
    i.e., h* = f(W_rec @ h* + W_in @ u + b)

    Objective: minimize ||h - f(W_rec @ h + W_in @ u + b)||^2
    """
    fixed_points = []
    for _ in range(n_inits):
        h = nn.Parameter(torch.randn(rnn.n_hidden) * 0.1)
        optimizer = torch.optim.Adam([h], lr=1e-3)
        for step in range(n_steps):
            dh = -h + rnn.activation(rnn.W_rec @ h + rnn.W_in @ u_context + rnn.b)
            loss = (dh ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if loss.item() < tol:
                break
        if loss.item() < 1e-6:  # converged
            fixed_points.append(h.detach().clone())
    return _deduplicate_fixed_points(fixed_points)
```

**Jacobian at fixed point (use `torch.autograd.functional.jacobian`):**
```python
import torch.autograd.functional as F

def compute_jacobian_at_fp(
    rnn: ContinuousTimeRNN,
    h_star: torch.Tensor,    # (H,) fixed point
    u: torch.Tensor,         # (M_in,) input at fixed point
) -> torch.Tensor:
    """Compute Jacobian d(dh/dt)/dh at fixed point h_star.

    J = d/dh[-h + f(W_rec @ h + W_in @ u + b)] = -I + diag(f'(net)) @ W_rec
    """
    def dynamics(h: torch.Tensor) -> torch.Tensor:
        return -h + rnn.activation(rnn.W_rec @ h + rnn.W_in @ u + rnn.b)

    return F.jacobian(dynamics, h_star)  # (H, H)
```

**Eigenvalue stability classification:**
```python
def classify_stability(jacobian: torch.Tensor) -> dict:
    """Classify fixed point stability from Jacobian eigenvalues.

    A fixed point is stable if all eigenvalues have negative real parts.
    Returns eigenvalues and stability classification.
    """
    eigenvalues = torch.linalg.eig(jacobian).eigenvalues  # (H,) complex
    real_parts = eigenvalues.real
    return {
        "eigenvalues": eigenvalues,
        "stable": bool((real_parts < 0).all()),
        "n_unstable": int((real_parts >= 0).sum()),
        "max_real_part": float(real_parts.max()),
    }
```

**Linearization quality diagnostic (feeds into Phase 22 PIPE-03):**
The Jacobian at the fixed point provides `J(h*)`. The bilinear DCM approximation is `A_eff = A + sum_j u_j B_j`. The linearization quality index is `||J(h*) - A_eff(h*)||_F / ||J(h*)||_F`. Values > 0.5 indicate poor local fit (pitfall LC1).

### Anti-Patterns to Avoid

- **Don't use `torchdiffeq.odeint` for RNN training.** Euler is faster, deterministic, matches
  the published reference implementation (Langdon & Engel, trainRNNbrain). Reserve torchdiffeq
  for DCM fitting (Phase 20).
- **Don't hardcode `ob_size` or `act_size`.** Always query from `env.observation_space.shape[0]`
  and `env.action_space.n`. neurogym environments may vary across versions.
- **Don't fit PCA on the same trials used for the output-R² gate.** This inflates R². Fit on
  training split, evaluate on held-out split.
- **Don't use `explained_variance_ratio_` alone to choose N.** Always check output-R² too
  (DIM-02). PCA can explain 90% of variance while capturing 0% of task-relevant computation
  (pitfall LC3, Dubreuil et al. 2024 eLife).
- **Don't save trajectories as raw tensors.** Use `.npz` format (consistent with existing
  fixture pattern in `benchmarks/generate_fixtures.py`).
- **Don't run 20 RNN training runs locally.** Each RNN training (2000-5000 steps, H=256) takes
  >3 minutes. All 20 seeds go to M3 cluster via sbatch.
- **Don't use truncated BPTT.** Neurogym trials are short enough (~100-300 steps) for full
  BPTT. Truncated BPTT complicates the training loop without benefit at this trial length.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Cognitive task environment | Custom CDDM implementation | `neurogym.make('ContextDecisionMaking-v0')` | 200-400 lines of fragile code vs proven library; exact L&E task |
| Batched trial generation | Custom DataLoader | `ngym.Dataset(task, batch_size, seq_len)` | Returns (seq_len, batch_size, obs_size) batches directly |
| Gradient clipping | Manual norm computation | `nn.utils.clip_grad_norm_(rnn.parameters(), clip)` | Standard PyTorch utility |
| PCA explained variance | Manual SVD + ratio | `sklearn.decomposition.PCA.explained_variance_ratio_` | Correct normalization, handles edge cases |
| Fixed-point finding | External library (FixedPointFinder) | Custom Adam on `||dh/dt||^2` (~50 lines) | FixedPointFinder is TF-based, stale; the math is simple |
| Jacobian computation | Manual chain rule | `torch.autograd.functional.jacobian` | Exact, handles activations automatically |
| Eigenvalue decomposition | Manual QR iteration | `torch.linalg.eig` | Stable, GPU-compatible, handles complex eigenvalues |

**Key insight:** The RNN training is standard supervised learning. All complexity is in the
infrastructure around it (task environment, trajectory saving, PCA quality gate), not in the
training loop itself.

---

## Common Pitfalls

### Pitfall 1: PCA Explains Variance But Not Task Output (LC3 -- CRITICAL)

**What goes wrong:** PCA selects axes that maximize variance. Task-relevant dynamics can live in
low-variance oblique directions (Dubreuil et al. 2024 eLife). You can have 90% cumulative
variance explained in top-N PCs but near-zero output R² (behavioral readout reconstructability).

**Why it happens:** The CDDM task requires only a low-dimensional context-selective computation.
The high-variance directions may reflect noise or spontaneous fluctuations unrelated to the task.

**How to avoid:** Always compute the output-R² gate (DIM-02) before reporting PCA variance
diagnostic (DIM-03). The gate is: `R² = 1 - ||z_true - W_out_pca @ h_pca||² / ||z_true||²`.
If R² < 0.90 (DIM-02 threshold), the PCA subspace is insufficient for the task. Try larger N or
consider task-informed reduction (demixed PCA). Do NOT proceed to DCM fitting (Phase 22) without
passing this gate.

**Warning signs:** High cumulative variance (>90%) but adding N+1 dramatically improves output R².

### Pitfall 2: Solution Degeneracy Across Seeds (LC14)

**What goes wrong:** Different RNN training seeds learn qualitatively different internal solutions
for the same task -- different fixed-point structures, different attractor topologies. DCM fitted
to one seed may not generalize to another.

**Why it happens:** The CDDM task is a fixed-point task (Mante et al. 2013 architecture). RNNs
can solve it via different dynamical mechanisms (point attractors, slow manifolds, etc.). With
ReLU activation, multiple solutions exist.

**How to avoid:** Train >=20 seeds (RNN-03) and report DCM fit statistics across seeds. Run
fixed-point analysis on each seed (RNN-04) and check that the number of stable fixed points is
consistent. Use L2 regularization during training (`weight_decay` in Adam) to push solutions
toward a consistent dynamical regime.

**Warning signs:** Trajectory R² (DCM fit quality in Phase 22) varies wildly across seeds.
Fixed-point count differs across seeds.

### Pitfall 3: RNN dt/tau Mismatch with DCM Time Grid (LC10)

**What goes wrong:** The RNN uses an arbitrary `dt` (e.g., `dt=100ms` from neurogym, with
`tau=1.0` in normalized units giving `alpha=0.1`). The DCM ODE integrator (Phase 20) expects
dt in seconds. If trajectories are passed to DCM without documenting the time normalization,
the DCM ODE integrates on the wrong timescale, producing wrong A matrix values.

**Why it happens:** neurogym's `dt` parameter is in milliseconds; DCM's existing dt conventions
are in seconds. The RNN `alpha = dt/tau` is dimensionless, but the physical interpretation
depends on `tau`.

**How to avoid:**
1. Document the RNN `dt` and `tau` explicitly in the saved trajectory `.npz` metadata.
2. Normalize: treat `tau=1.0` as the base timescale and rescale trajectories to seconds in
   `latent_extraction.py` before saving. A trajectory of T=100 steps at dt=100ms is 10 seconds
   of real time. The DCM dt should be set to `dt_seconds = rnn_dt_ms / 1000`.
3. The trajectory `.npz` file must include: `dt_seconds`, `tau`, `alpha`, `n_steps`.
4. Phase 22 will use `dt_seconds` as the DCM `dt` parameter.

### Pitfall 4: Scale of RNN Hidden States vs DCM Priors (LC4 -- CRITICAL)

**What goes wrong:** After PCA reduction, the latent trajectories may have amplitude O(1)-O(10).
DCM priors calibrated for BOLD data (A_free ~ N(0, 1/64)) will heavily shrink A toward zero,
repeating the Phase 16.1 RECOV-04 failure mode.

**Why it happens:** The LC_A_PRIOR_VARIANCE and LC_B_PRIOR_VARIANCE in Phase 20 are calibrated
for synthetic bilinear ground truth. Real RNN latents after PCA may have different amplitude.

**How to avoid:**
1. Z-score each PC trajectory to zero mean, unit variance BEFORE passing to Phase 20/22 DCM.
   Document the mean and std for inverse-normalization (needed for perturbation experiments).
2. Save both raw and z-scored trajectories; Phase 22 decides which to pass to DCM.
3. Report the amplitude statistics of raw PCA trajectories in the DIM-01/DIM-02/DIM-03 output.

### Pitfall 5: Performance Gate Before Trajectory Extraction

**What goes wrong:** Extracting trajectories from an undertrained RNN (e.g., 40% accuracy)
produces trajectories that don't reflect the task-relevant computation. DCM fitted to these
trajectories learns noise, not circuit dynamics.

**Why it happens:** Early in training, RNN weights are random and the dynamics are not
task-meaningful.

**How to avoid:** Gate trajectory extraction on task performance. Require >=85% accuracy on
held-out test trials before extracting and saving trajectories. The `train_rnn()` function
should return the trained model + performance metrics; `extract_trajectories()` should assert
performance >= threshold before running.

### Pitfall 6: Memory and Time for 20-Seed Ensemble on Cluster

**What goes wrong:** Training 20 RNNs sequentially takes 10-20+ hours on a single node.
Saving full h(t) trajectories for H=256 units x 10 conditions x 50 trials x 200 steps =
25.6M floats per RNN = ~100MB per RNN x 20 = ~2GB total trajectory storage.

**Why it happens:** Phase 21 is explicitly designed to produce 20 trained RNNs with trajectories.
This is the only phase in v0.6.0 that is primarily compute-bound (not algorithm-bound).

**How to avoid:**
1. Submit an array job to M3 (sbatch --array=0-19) with one RNN per job. 20 parallel 30-60min
   jobs (GPU or CPU-heavy) is standard HPC usage.
2. Use float32 for trajectory storage (not float64) -- DCM fitting uses float64 internally but
   trajectories are data, not numerics.
3. Store trajectories in compressed `.npz` format (`np.savez_compressed`).
4. Include trajectory storage estimation in the sbatch script comment.

---

## Code Examples

Verified patterns from official sources and existing codebase conventions:

### neurogym Dataset Usage (verified from neurogym official docs)

```python
# Source: neurogym.github.io/example_neurogym_pytorch.html
import neurogym as ngym
import torch

task = 'ContextDecisionMaking-v0'
kwargs = {'dt': 100}  # dt in ms
dataset = ngym.Dataset(task, env_kwargs=kwargs, batch_size=32, seq_len=100)

env = dataset.env
ob_size = env.observation_space.shape[0]  # always query, never hardcode
act_size = env.action_space.n

for step in range(2000):
    inputs, labels = dataset()
    inputs = torch.tensor(inputs, dtype=torch.float32)   # (seq_len, batch_size, ob_size)
    labels = torch.tensor(labels, dtype=torch.long)      # (seq_len * batch_size,)
    # ... forward pass, loss, backward ...
```

### CT-RNN Euler Integration

```python
# Source: Reference from trainRNNbrain Euler formulation (alpha = dt/tau)
# Verified against Langdon & Engel (2025) conventions

class ContinuousTimeRNN(nn.Module):
    def forward(self, u, h0=None):
        T, B, M_in = u.shape
        if h0 is None:
            h = torch.zeros(B, self.n_hidden, device=u.device, dtype=u.dtype)
        else:
            h = h0
        h_traj = []
        for t in range(T):
            pre_act = h @ self.W_rec.T + u[t] @ self.W_in.T + self.b
            h = (1 - self.alpha) * h + self.alpha * self.f(pre_act)
            if self.training and self.noise_std > 0:
                h = h + self.noise_std * torch.randn_like(h)
            h_traj.append(h)
        h_traj = torch.stack(h_traj, dim=0)  # (T, B, H)
        z = h_traj @ self.W_out.T            # (T, B, act_size)
        return z, h_traj
```

### sklearn PCA Workflow

```python
# Source: scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
from sklearn.decomposition import PCA
import numpy as np

# h_all: (n_samples, H) where n_samples = n_trials * T
pca = PCA(n_components=None)  # fit all components for scree analysis
pca.fit(h_all_train)

# Variance diagnostic (DIM-03)
cum_var = np.cumsum(pca.explained_variance_ratio_)
marginal = np.diff(pca.explained_variance_ratio_, prepend=0.0)
recommended_n = int(np.argmax(marginal[1:] < 0.05)) + 1  # first N where marginal < 5%

# Choose N, refit
pca_n = PCA(n_components=recommended_n)
h_reduced = pca_n.fit_transform(h_all_train)  # (n_samples, N)

# Apply to test data (no refitting)
h_test_reduced = pca_n.transform(h_all_test)  # (n_test_samples, N)
```

### Fixed-Point Finding (PyTorch Adam)

```python
# Source: Standard optimization pattern; verified against fixed-point finding literature
# (Golub et al. 2018 Neuron; tripdancer0916/pytorch-fixed-point-analysis conventions)
import torch
import torch.nn as nn

def find_fixed_points(rnn, u_fixed, n_inits=100, n_steps=5000, tol=1e-12):
    """Minimize ||dh/dt||^2 = ||-h + f(Wh + Wu + b)||^2 via Adam."""
    fixed_points = []
    for _ in range(n_inits):
        h = nn.Parameter(torch.randn(rnn.n_hidden) * 0.1)
        opt = torch.optim.Adam([h], lr=1e-3)
        for _ in range(n_steps):
            net = rnn.W_rec @ h + rnn.W_in @ u_fixed + rnn.b
            dh = -h + rnn.f(net)
            loss = (dh ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if loss.item() < tol:
                break
        if loss.item() < 1e-6:
            fixed_points.append(h.detach())
    return fixed_points

# Jacobian computation (torch.autograd.functional.jacobian)
from torch.autograd.functional import jacobian

def jacobian_at_fp(rnn, h_star, u):
    def dyn(h):
        return -h + rnn.f(rnn.W_rec @ h + rnn.W_in @ u + rnn.b)
    J = jacobian(dyn, h_star)   # (H, H)
    eigs = torch.linalg.eig(J).eigenvalues  # complex (H,)
    return J, eigs
```

### Output R² Gate (DIM-02)

```python
import numpy as np

def output_r_squared_gate(h_reduced, w_out, pca, z_true, threshold=0.90):
    """
    h_reduced: (n_samples, N) PCA-projected hidden states
    w_out: (act_size, H) RNN output weights
    pca: fitted sklearn PCA object
    z_true: (n_samples, act_size) true output activations (pre-softmax)
    threshold: DIM-02 gate
    """
    w_out_pca = w_out @ pca.components_.T  # (act_size, N)
    z_pred = h_reduced @ w_out_pca.T       # (n_samples, act_size)
    ss_res = np.sum((z_true - z_pred) ** 2)
    ss_tot = np.sum((z_true - z_true.mean(0)) ** 2)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    passed = r2 >= threshold
    return {"r_squared": float(r2), "passed": passed, "threshold": threshold}
```

---

## Directory Structure & File Naming

```
src/pyro_dcm/
  rnn/
    __init__.py                  # exports: ContinuousTimeRNN, train_rnn, extract_trajectories
                                 #          pca_reduce, output_r_squared_gate, find_fixed_points
    continuous_time_rnn.py       # ContinuousTimeRNN(nn.Module)
    rnn_trainer.py               # train_rnn(), eval_rnn_performance()
    latent_extraction.py         # extract_trajectories(), pca_reduce(),
                                 # variance_explained_diagnostic(), output_r_squared_gate()
    fixed_point_analysis.py      # find_fixed_points(), compute_jacobian_at_fp(), classify_stability()

checkpoints/
  rnn/
    seed_0000_H064.pt            # torch.save(rnn.state_dict(), ...)
    seed_0000_H064_meta.json     # {"seed": 0, "H": 64, "accuracy": 0.91, "n_steps": 3000, ...}
    ...
    seed_0019_H256.pt

data/
  rnn_trajectories/              # new directory
    seed_0000_H064_condition_ctx1_coh0.32.npz  # np.savez_compressed
    ...

tests/
  test_ctrnn.py                  # RNN-01: unit tests (forward pass, Euler, gradient flow)
  test_rnn_trainer.py            # RNN-02/03: integration test (train on CDDM, check accuracy)
  test_fixed_point_analysis.py   # RNN-04: find FPs, check Jacobian shape, eigenvalue real parts
  test_latent_extraction.py      # DIM-01/02/03: PCA reduction, output R², variance diagnostic
```

**Pytest markers for Phase 21 tests:**
- Fast unit tests (no neurogym, no RNN training): no special marker
- Slow tests requiring neurogym + RNN training: `@pytest.mark.slow`
- Full 20-seed ensemble: run on cluster only, not in CI

---

## Cluster Routing Strategy

**Estimated runtime per RNN training run:**
- H=64, 2000 steps: ~5-10 min on CPU (M3 node without GPU)
- H=256, 5000 steps: ~30-60 min on M3 CPU node; ~5-10 min on GPU node
- 20 seeds x 60 min = 20 hours sequential → use array job

**Recommended M3 sbatch structure:**
```bash
#!/bin/bash
#SBATCH --job-name=rnn_train_phase21
#SBATCH --array=0-19         # 20 seeds, one per job
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:30:00      # 90 min for H=256, 5000 steps
#SBATCH --gres=gpu:1         # Optional: if GPU node available

python scripts/train_rnn_seed.py --seed $SLURM_ARRAY_TASK_ID --hidden 256
```

**Fixed-point analysis per RNN seed:**
- H=256, 100 inits, 5000 steps each: ~20-40 min on CPU per RNN
- 20 seeds x 30 min = 10 hours → separate array job after training

**Local dev testing:**
- Run single seed, H=64, 500 steps for smoke test: ~1-2 min OK on laptop
- Use `pytest tests/test_ctrnn.py -m "not slow"` for fast unit tests on laptop

---

## pyproject.toml Changes Required

```toml
[project.optional-dependencies]
benchmark = [
    "matplotlib",
    "tabulate",
]
mne = [
    "mne>=1.6",
    "mne-bids>=0.14",
]
latent = [
    "neurogym>=2.3",
    "scikit-learn>=1.3",
]
dev = [
    "pytest",
    "ruff",
    "mypy",
]
```

Add `pytest` marker for `latent` tests (analogous to existing `mne` marker):

```toml
[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow",
    "spm: requires MATLAB + SPM12",
    "tapas: requires tapas rDCM toolbox",
    "mne: requires MNE-Python",
    "latent: requires neurogym + scikit-learn",
]
```

---

## References to Add to REFERENCES.md

Phase 21 introduces one new primary reference not yet in REFERENCES.md:

**Langdon & Engel (2025) [REF-071 or next available]:**
- Already slated for PUB-03 / REF-076 in current REQUIREMENTS-v0.6.0.md
- Needed in Phase 21 for `continuous_time_rnn.py` docstring (Euler formulation)
- Cite for: RNN dynamics equation, CDDM task convention, training protocol

**trainRNNbrain (Engel lab):**
- Reference implementation; cite in `rnn_trainer.py` docstring
- Not a separate REFERENCES.md entry -- inline URL citation acceptable

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| FixedPointFinder (TF, stale) | Custom PyTorch Adam on `||dh/dt||^2` | v0.6.0 design decision | Eliminates TF dependency; ~50 lines |
| Custom task environments | neurogym CDDM v0 | v0.6.0 stack research | Exact L&E task; peer-reviewed; saves 300+ lines |
| torch.pca_lowrank for analysis | sklearn PCA + explained_variance_ratio_ | v0.6.0 stack research | Explained variance diagnostic natively available |
| trainRNNbrain full package | Custom ContinuousTimeRNN (~100 lines) | v0.6.0 design decision | Avoids hydra-core; clean interface |

---

## Open Questions

### 1. neurogym ContextDecisionMaking-v0 Observation Space Dimensions

**What we know:** The task has two sensory modalities + context cue + fixation signal.
Literature typically reports 5-7 input channels for this task variant. The `ob_size` must be
queried at runtime: `env.observation_space.shape[0]`.

**What's unclear:** Exact channel count and layout in neurogym 2.3.1. Cannot verify without
running the environment.

**Recommendation:** Query at runtime as shown in Pattern 2. The ContinuousTimeRNN takes
`n_input` as a constructor argument -- never hardcode it. Include an assertion in the test:
```python
assert env.observation_space.shape[0] == ob_size
assert env.action_space.n == act_size
```

**Confidence:** LOW on exact channel count; HIGH on the API pattern.

### 2. Training Hyperparameters for H=256, CDDM

**What we know:** neurogym official PyTorch example achieves 83% at 2000 steps with H=64 LSTM,
lr=0.01 on PerceptualDecisionMaking. trainRNNbrain uses lr=1e-3, grad_clip=1.0 for CT-RNN.

**What's unclear:** Whether CDDM (harder than PDM: two modalities + context) requires more steps
or lower lr at H=256.

**Recommendation:** Start with `lr=1e-3, n_steps=3000, grad_clip=1.0`. Tune during the first
3-5 training seeds before submitting the full 20-seed array job. Check convergence curve and
final accuracy.

**Confidence:** MEDIUM -- standard hyperparameters; task-specific calibration needed.

### 3. Data Directory Convention

**What we know:** Existing project uses `checkpoints/` for model weights and `benchmarks/results/`
for benchmark outputs. There is no existing `data/` directory.

**What's unclear:** Where to store RNN trajectories. Options: `checkpoints/rnn_trajectories/`,
`data/rnn_trajectories/`, `benchmarks/rnn_trajectories/`.

**Recommendation:** Create `data/rnn_trajectories/` as a new top-level directory (separate from
`checkpoints/` which is for model weights and `benchmarks/` which is for metric results). Add to
`.gitignore` or document as large-file storage. Trajectories are data, not code artifacts.

**Confidence:** MEDIUM -- convention choice; no existing pattern in project.

---

## Sources

### Primary (HIGH confidence)

- `.planning/research/v0.6.0/STACK.md` -- neurogym v2.3.1 verified, sklearn PCA rationale,
  trainRNNbrain rejection rationale, CT-RNN Euler formulation, fixed-point finding approach
- `.planning/research/v0.6.0/PITFALLS.md` -- LC1-LC14 pitfalls, all directly relevant to
  Phase 21 implementation (LC3 PCA gate, LC4 scale, LC9 task choice, LC14 seed degeneracy)
- `.planning/research/v0.6.0/SUMMARY.md` -- Phase 2 (CT-RNN) scope, architecture, dependencies
- `.planning/research/v0.6.0/ARCHITECTURE.md` -- rnn/ package structure, component boundaries
- `.planning/REQUIREMENTS-v0.6.0.md` -- RNN-01 through DIM-03 exact requirements
- `neurogym.github.io/example_neurogym_pytorch.html` -- Dataset API, (seq_len, batch_size, ob_size)
  shape confirmed, training loop structure
- `scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html` -- explained_variance_ratio_
  attribute, fit/transform/inverse_transform API
- `pyproject.toml` (project) -- existing optional dependency group pattern ([mne], [benchmark])

### Secondary (MEDIUM confidence)

- neurogym.github.io/envs/ContextDecisionMaking-v0.html -- task description, parameters
  (dt, timing, sigma); exact channel count not returned by web fetch, must query at runtime
- trainRNNbrain GitHub README -- tau*dh/dt = -x + f(W_rec x + W_inp u) dynamics confirmed,
  Euler discrete-time form with alpha=dt/tau, Adam optimizer, gradient clipping
- tripdancer0916/pytorch-fixed-point-analysis -- repository structure confirms PyTorch fixed-point
  analysis is standard ~50-line pattern; specific implementation not fetchable but pattern clear

### Tertiary (LOW confidence)

- WebSearch results confirming neurogym Dataset class `seq_len` and `batch_size` parameters
- WebSearch results on fixed-point finding in PyTorch (Golub FixedPointFinder README confirms
  `||dx/dt||^2` objective is standard; PyTorch implementation is our own)

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- neurogym v2.3.1 confirmed on PyPI, sklearn PCA verified, all existing
  stack verified by codebase source read
- Architecture: HIGH -- rnn/ package structure directly from v0.6.0 ARCHITECTURE.md; component
  boundaries clearly defined
- CT-RNN implementation: HIGH -- Euler formulation from trainRNNbrain confirmed; alpha=dt/tau
  from Langdon & Engel 2025
- neurogym API: HIGH -- Dataset pattern + shape confirmed from official docs; ob_size must be
  queried at runtime (LOW confidence on exact channel count)
- Fixed-point finding: HIGH -- torch.optim + torch.autograd.functional.jacobian + torch.linalg.eig
  is the correct PyTorch-native approach
- PCA + output gate: HIGH -- sklearn PCA API verified; output-R² formula directly from
  LC3 mitigation strategy in PITFALLS.md
- Pitfalls: HIGH -- LC3, LC4, LC10 directly from v0.6.0 PITFALLS.md; LC14 from Huang et al. 2025
- Cluster routing: HIGH -- project rule (>3 min = cluster); 20x H=256 clearly exceeds threshold
- Training hyperparameters: MEDIUM -- lr=1e-3, grad_clip=1.0 from trainRNNbrain; CDDM-specific
  calibration needed on first few seeds

**Research date:** 2026-05-25
**Valid until:** 2026-06-25 (30 days; neurogym and sklearn are stable; PyTorch 2.x API stable)
