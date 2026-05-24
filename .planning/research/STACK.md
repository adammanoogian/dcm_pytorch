---
type: "research"
scope: "stack"
milestone: "v0.6.0"
updated: "2026-05-24"
---

# Technology Stack: Latent Circuit DCM (v0.6.0)

**Project:** Pyro-DCM
**Milestone:** v0.6.0 -- Latent Circuit DCM
**Researched:** 2026-05-24

## Summary

The Latent Circuit DCM milestone adds four new capabilities: (a) training
continuous-time RNNs on cognitive tasks, (b) extracting and reducing RNN hidden
dynamics to a low-dimensional circuit, (c) fitting existing bilinear DCM to
those dynamics with a direct observation model (no hemodynamics), and
(d) benchmarking against Langdon & Engel 2025. The good news: **almost
everything needed is already in the project's dependency tree.** The core RNN,
dimensionality reduction, and observation model are pure PyTorch. Two new
optional dependencies are recommended: `neurogym` for task environments and
`scikit-learn` for PCA validation/analysis.

---

## Recommended Stack Additions

### 1. neurogym (NEW -- optional dependency)

| Field | Value |
|-------|-------|
| Version | `>=2.3` (current: 2.3.1, released 2026-03-31) |
| Python | >=3.10 (matches our minimum) |
| Purpose | Curated neuroscience task environments (Gymnasium API) |
| Install group | `[latent]` optional extra |

**Why neurogym:**
- Provides 33 standard cognitive task environments used across the
  computational neuroscience literature, including the Yang et al. 2019
  20-task battery.
- Critical tasks for our use case: `ContextDecisionMaking-v0` (the exact
  task Langdon & Engel 2025 used), `PerceptualDecisionMaking-v0`,
  `DelayMatchSample-v0`, `GoNogo-v0`.
- Gymnasium API means tasks produce `(obs, reward, terminated, truncated,
  info)` tuples compatible with standard training loops.
- Neurogym handles trial structure, timing, and coherence levels -- saving
  us from reimplementing experimental paradigms.

**Why not write our own task environments:**
- Reimplementing even one task correctly (stimulus statistics, timing,
  coherence levels) is 200-400 lines of fragile code.
- Neurogym tasks are peer-reviewed and match the literature conventions.
- Yang et al. 2019 and Langdon & Engel 2025 both use these exact tasks.

**Why not PsychRNN:**
- PsychRNN is TensorFlow-based, would add a TF dependency for no reason.
- Not actively maintained (last major release: 2021).
- Neurogym is framework-agnostic -- it returns numpy arrays, we convert
  to tensors ourselves.

**Confidence:** HIGH (verified on PyPI: v2.3.1 released 2026-03-31,
requires Python >=3.10, depends on Gymnasium).

**Source:** [neurogym PyPI](https://pypi.org/project/neurogym/),
[GitHub](https://github.com/neurogym/neurogym)

---

### 2. scikit-learn (NEW -- optional dependency)

| Field | Value |
|-------|-------|
| Version | `>=1.3` |
| Purpose | PCA for dimensionality reduction analysis and validation |
| Install group | `[latent]` optional extra |

**Why scikit-learn PCA over torch.pca_lowrank:**
- `sklearn.decomposition.PCA` provides explained variance ratios, which
  are essential for choosing the number of latent dimensions (scree plot).
- We need PCA primarily as an analysis/validation tool (offline, on numpy
  arrays of extracted hidden states), not as a differentiable component
  in the training loop.
- The actual dimensionality reduction in the latent circuit model is a
  *learned* orthonormal projection Q (Cayley-parameterized), not PCA.
  PCA is used only for: (1) choosing n (number of latent nodes) via
  explained variance, (2) initializing Q, (3) validation visualization.
- torch.pca_lowrank returns U, S, V but not explained variance ratios.
  We would end up reimplementing sklearn's analysis anyway.

**Why not torch.pca_lowrank for everything:**
- torch.pca_lowrank is appropriate when PCA must be differentiable or
  when data is already on GPU. Our PCA operates on extracted hidden state
  trajectories (numpy arrays, offline).
- We *will* use torch.pca_lowrank if we ever need differentiable PCA
  in the forward pass. For now, that is not the design.

**Why not a heavier dimensionality reduction library (TorchDR, UMAP):**
- Linear PCA is sufficient and interpretable for our use case. The
  Langdon & Engel approach explicitly uses linear projections.
- Nonlinear methods (UMAP, t-SNE) are visualization aids, not part of
  the inference pipeline. matplotlib handles scatter plots fine.

**Confidence:** HIGH (scikit-learn is stable, widely used, well-tested).

---

## Existing Stack -- No Changes Needed

### PyTorch 2.x (EXISTING -- `torch>=2.0`)

**Covers these new capabilities:**

| Capability | PyTorch API | Notes |
|------------|-------------|-------|
| Continuous-time RNN | `nn.Module`, `nn.Linear`, `nn.ReLU` | Custom CTRNN as nn.Module subclass |
| ODE integration for RNN | `torchdiffeq.odeint` | Already a dependency |
| Learned projection Q | `nn.Parameter` + Cayley transform | `torch.linalg.solve` for (I-A)^-1 |
| Fixed-point finding | `torch.autograd.functional.jacobian` | Built-in Jacobian computation |
| Eigenvalue analysis | `torch.linalg.eig` | Stability analysis at fixed points |
| Direct observation model | `torch.matmul` | y = C_obs @ x + noise, trivially linear |
| Gradient-based optimization | `torch.optim.Adam` | RNN training |
| Batched training | `torch.utils.data.DataLoader` | Standard |

**The continuous-time RNN is pure PyTorch.** The dynamics:

```
tau * dx/dt = -x + f(W_rec @ x + W_in @ u + b)
```

are implemented as an `nn.Module` whose `forward()` either: (a) uses Euler
integration with a fixed dt (matching Langdon & Engel's alpha=dt/tau
formulation), or (b) wraps the right-hand side in a `torchdiffeq.odeint`
call for adaptive stepping.

**No specialized RNN library is needed.** The trainRNNbrain package (from
Engel lab) is MIT-licensed and provides training + analysis utilities, but
it adds hydra-core as a dependency and its API is opinionated. Writing a
clean 100-line CTRNN module in PyTorch is preferable for our codebase
style. We can reference their implementation for correctness.

### Pyro 1.9+ (EXISTING -- `pyro-ppl>=1.9`)

**Covers the direct observation model:**

The new "latent circuit DCM" Pyro model replaces the hemodynamic forward
model with a simple linear observation:

```python
y_pred = C_obs @ x  # shape (T, N_obs)
pyro.sample("obs", dist.Normal(y_pred, sigma), obs=y_observed)
```

This is strictly simpler than the existing task-DCM model. No new Pyro
APIs needed. The existing `create_guide` factory (6 guide types) works
unchanged.

### torchdiffeq (EXISTING)

Already used for balloon-model ODE integration. Same `odeint` call works
for continuous-time RNN dynamics if we choose adaptive stepping over
fixed-dt Euler. No version change needed.

### Zuko (EXISTING -- `zuko>=1.2`)

Amortized guides for latent circuit DCM can reuse the existing
`AmortizedFlowGuide` with a different summary network. No changes.

### scipy (EXISTING)

May use `scipy.optimize.fsolve` as a cross-check for fixed-point finding,
but this is optional -- PyTorch gradient descent is the primary method.

### matplotlib (EXISTING -- in `[benchmark]` extra)

Needed for: RNN trajectory visualization, PCA scree plots, latent circuit
diagrams, parameter recovery plots. Already available.

---

## Libraries Explicitly NOT Recommended

### trainRNNbrain (DO NOT ADD)

| Reason | Detail |
|--------|--------|
| Unnecessary dependency | Adds hydra-core, tqdm, sklearn as transitive deps |
| API mismatch | Opinionated training pipeline doesn't fit our Pyro-centric design |
| Maintenance | Small research codebase, not a maintained library |
| Implementation effort | Our CTRNN module is ~100 lines of PyTorch |

The Engel lab's `trainRNNbrain` package provides a full training pipeline
for continuous-time RNNs, but it is a research codebase designed for their
specific workflow. We should *read* their code for correctness reference
(especially the noise injection and Dale's law constraint logic), but not
depend on it.

**Source:** [GitHub](https://github.com/engellab/trainRNNbrain) -- MIT license,
depends on matplotlib, numpy, torch, hydra-core, scikit-learn, tqdm.

### latentcircuit (DO NOT ADD as dependency -- USE as benchmark reference)

| Reason | Detail |
|--------|--------|
| Benchmark target | We compare *against* their results, not wrap their code |
| Different design | Their Net + LatentNet are standalone; ours integrates with Pyro |
| Python 3.8 target | Their code targets Python 3.8, we target 3.10+ |
| Vendored reference | We may vendor their LatentNet for benchmark comparison only |

The Langdon & Engel `latentcircuit` package (MIT license, PyTorch-based)
implements their Net (teacher RNN) and LatentNet (latent circuit fitter).
We should:
1. Clone and run their Tutorial notebook to reproduce their published results.
2. Use their trained Net weights as a benchmark target.
3. Implement our own latent circuit extraction that integrates with our
   bilinear DCM + Pyro model.
4. Compare parameter recovery between their LatentNet and our approach.

**Source:** [GitHub](https://github.com/engellab/latentcircuit) -- MIT license,
requires torch==2.4.1, jupyter, pandas, scipy, seaborn.
Also: [latent_circuit_inference](https://github.com/engellab/latent_circuit_inference)
-- newer repo for continuous-time RNN circuits, depends on trainRNNbrain.

### PsychRNN (DO NOT ADD)

TensorFlow-based, not maintained since 2021. Neurogym covers the same
task environments with a framework-agnostic API.

### FixedPointFinder (DO NOT ADD)

| Reason | Detail |
|--------|--------|
| Stale | Last release December 2022, tests broken |
| Heavy deps | Requires RecurrentWhisperer + TensorFlow |
| Reimplementable | Fixed-point finding is ~50 lines with torch.optim |

Fixed-point finding for RNNs is straightforward: minimize
`||dx/dt||^2 = ||-x + f(W_rec @ x + W_in @ u)||^2` using Adam. The
Jacobian at each fixed point is computed via
`torch.autograd.functional.jacobian`. Eigenvalues via `torch.linalg.eig`.
No external library needed.

**Source:** [GitHub](https://github.com/mattgolub/fixed-point-finder)

### gymnasium (DO NOT ADD directly)

Gymnasium is a transitive dependency of neurogym. No need to list it
separately.

---

## Proposed pyproject.toml Changes

```toml
[project.optional-dependencies]
# ... existing groups unchanged ...
latent = [
    "neurogym>=2.3",
    "scikit-learn>=1.3",
]
```

**Why a separate optional group:**
- Not all users need RNN training capabilities.
- Keeps the core `pyro-dcm` install lean (PyTorch + Pyro + torchdiffeq +
  Zuko + scipy + numpy).
- Mirrors the existing `[mne]` and `[benchmark]` optional groups.

**Total new direct dependencies: 2** (neurogym, scikit-learn).
Both are stable, well-maintained, pure-Python (no C compilation issues).

---

## Integration Points with Existing Stack

### RNN Training Pipeline (NEW code, existing tools)

```
neurogym task env  -->  PyTorch DataLoader  -->  CTRNN (nn.Module)
                                                    |
                                                    v
                                            torch.optim.Adam
                                            (standard training loop)
```

- neurogym produces `(obs, action, reward)` tuples as numpy arrays.
- We wrap these in a `torch.utils.data.Dataset` for batched training.
- The CTRNN module is a vanilla `nn.Module` with `forward()` that runs
  Euler integration or calls `torchdiffeq.odeint`.
- Training uses standard PyTorch `loss.backward()` + `optimizer.step()`.

### Latent Circuit Extraction (NEW code, existing tools)

```
Trained CTRNN  -->  Extract hidden trajectories  -->  PCA (sklearn)
                                                        |
                                                        v
                                                  Choose n_latent
                                                        |
                                                        v
                                            Fit Q via Cayley transform
                                            (torch.nn.Parameter + Adam)
```

- Hidden states extracted as numpy arrays, analyzed with sklearn PCA.
- Orthonormal Q parameterized via Cayley transform using `nn.Parameter`.
- Loss: `||y - Qx||^2 + lambda * ||z - w_out @ x||^2` (Langdon & Engel
  Eq. matching).

### Direct Observation DCM (NEW model, existing Pyro infrastructure)

```
Latent trajectories (x)  -->  Bilinear DCM model (EXISTING neural_state.py)
                                    |
                                    v
                              y = C_obs @ x + noise  (NEW, replaces balloon+BOLD)
                                    |
                                    v
                              Pyro SVI (EXISTING guides.py, create_guide)
```

- The `neural_state.py` bilinear equation `dx/dt = Ax + sum_j u_j B_j x + Cu`
  is reused unchanged.
- The observation model becomes `y = C_obs @ x + Normal(0, sigma)` instead
  of the balloon-Windkessel + BOLD chain.
- Pyro model structure mirrors `task_dcm_model.py` but replaces the
  hemodynamic forward model with a single linear layer.
- All 6 existing guide types work without modification.

### Dynamical Systems Analysis (NEW code, existing PyTorch)

```
Trained CTRNN  -->  Fixed-point finding (torch.optim)
                        |
                        v
                  Jacobian computation (torch.autograd.functional.jacobian)
                        |
                        v
                  Eigenvalue analysis (torch.linalg.eig)
                        |
                        v
                  Stability classification (pure Python)
```

No external libraries needed. The entire dynamical systems analysis
pipeline uses built-in PyTorch functions.

---

## Version Compatibility Matrix

| Dependency | Min Version | Tested With | Notes |
|------------|-------------|-------------|-------|
| Python | 3.10 | 3.10+ | Unchanged |
| torch | 2.0 | 2.4+ | Need `torch.linalg.eig` (available since 1.9) |
| pyro-ppl | 1.9 | 1.9+ | Unchanged |
| torchdiffeq | any | latest | Unchanged |
| zuko | 1.2 | latest | Unchanged |
| scipy | any | latest | Unchanged |
| numpy | any | latest | Unchanged |
| neurogym | 2.3 | 2.3.1 | NEW -- requires gymnasium |
| scikit-learn | 1.3 | latest | NEW -- PCA, explained_variance_ratio_ |

---

## Key Implementation Decisions (Stack-Driven)

### D1: Euler vs Adaptive ODE for RNN Training

**Recommendation: Fixed-dt Euler integration for RNN training.**

Rationale:
- Langdon & Engel use discrete-time Euler with `alpha = dt/tau = 0.2`.
- trainRNNbrain uses the same formulation.
- Euler is faster (no adaptive step overhead) and deterministic.
- Neurogym tasks produce fixed-dt observations, so adaptive stepping
  provides no benefit during training.
- For *analysis* (e.g., long autonomous runs), we can optionally wrap
  the same dynamics in `torchdiffeq.odeint`.

### D2: Custom CTRNN vs External Library

**Recommendation: Custom nn.Module, ~100 lines.**

Rationale:
- Our CTRNN needs: (a) standard forward pass, (b) hidden state extraction,
  (c) optional Dale's law constraints, (d) noise injection during training.
- trainRNNbrain provides all this but adds hydra-core and an opinionated
  config system we don't want.
- PsychRNN is TensorFlow. neurogym doesn't provide networks, only tasks.
- A custom module gives us full control over the interface with our
  existing DCM infrastructure.

### D3: PCA Library Choice

**Recommendation: sklearn PCA for analysis, torch for any differentiable path.**

Rationale:
- The primary use of PCA is offline analysis of extracted hidden states.
- sklearn provides explained_variance_ratio_ for dimension selection.
- If we later need differentiable PCA (e.g., end-to-end training of Q
  initialized from PCA), we use `torch.pca_lowrank` at that point.

### D4: Fixed-Point Finding Approach

**Recommendation: Custom implementation using torch.optim.Adam.**

Rationale:
- Minimize `||f(x) - x||^2` (or `||dx/dt||^2` for continuous-time) using
  Adam with multiple random initializations.
- ~50 lines of PyTorch code.
- FixedPointFinder is stale (2022) and adds TF as a dependency.
- The Engel lab's DynamicSystemAnalyzer (in trainRNNbrain) is a good
  reference implementation but not worth the dependency.

---

## References

### Papers
- Langdon & Engel (2025). Latent circuit inference from heterogeneous
  neural responses during cognitive tasks. *Nature Neuroscience*, 28,
  665-675. [PubMed](https://pubmed.ncbi.nlm.nih.gov/39930096/)
- Yang et al. (2019). Task representations in neural networks trained
  to perform many cognitive tasks. *Nature Neuroscience*, 22, 297-306.
  [PDF](https://www.cns.nyu.edu/wanglab/publications/pdf/yang.nn2019.pdf)

### Code Repositories
- [engellab/latentcircuit](https://github.com/engellab/latentcircuit) --
  MIT, PyTorch, reference implementation of LatentNet + Net.
- [engellab/latent_circuit_inference](https://github.com/engellab/latent_circuit_inference) --
  Newer repo for continuous-time variant, depends on trainRNNbrain.
- [engellab/trainRNNbrain](https://github.com/engellab/trainRNNbrain) --
  MIT, PyTorch, continuous-time RNN training pipeline.
- [neurogym/neurogym](https://github.com/neurogym/neurogym) --
  v2.3.1, Gymnasium-based neuroscience task environments.

### PyTorch APIs (verified against docs)
- [torch.linalg.eig](https://docs.pytorch.org/docs/stable/generated/torch.linalg.eig.html) --
  eigenvalue decomposition for stability analysis.
- [torch.pca_lowrank](https://docs.pytorch.org/docs/stable/generated/torch.pca_lowrank.html) --
  differentiable low-rank PCA approximation.
- [torch.autograd.functional.jacobian](https://docs.pytorch.org/docs/stable/generated/torch.autograd.functional.jacobian.html) --
  Jacobian computation for dynamical systems analysis.
