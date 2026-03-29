# Phase 7 Research: Amortized Neural Inference Guides

**Researched:** 2026-03-28
**Mode:** Ecosystem
**Overall confidence:** HIGH (core patterns well-established; DCM-specific amortization is novel)

---

## Executive Summary

Amortized inference via normalizing flows is a mature technique in the simulation-based inference (SBI) literature, with clear architectural patterns and well-understood trade-offs. The core idea: train a conditional normalizing flow on simulated (data, parameter) pairs so that at test time, a single forward pass through the flow produces an approximate posterior -- no iterative optimization required.

For Pyro-DCM Phase 7, the technology stack is settled: **Zuko** for normalizing flows (already a project dependency decision), **Pyro's `pyro.contrib.zuko.ZukoToPyro`** wrapper for integration, and the existing three simulators (`task_simulator`, `spectral_simulator`, `rdcm_simulator`) for generating training data. The key engineering challenges are: (1) designing summary networks that compress variable-length BOLD/CSD data into fixed-dimension embeddings, (2) choosing flow architecture hyperparameters, (3) generating a sufficiently diverse training dataset (10,000+ simulated subjects), and (4) handling the amortization gap gracefully.

This phase is architecturally straightforward -- the building blocks exist -- but requires careful empirical tuning. The main risk is the multimodal posterior problem flagged in STATE.md: standard normalizing flows with unimodal base distributions struggle with disconnected posterior modes.

---

## Technology Stack

### Zuko Normalizing Flows

**Version:** 1.4.0 (latest stable as of March 2025)
**Confidence:** HIGH (official GitHub, PyPI)
**Source:** [GitHub - probabilists/zuko](https://github.com/probabilists/zuko)

Zuko provides 12 flow architectures. The relevant ones for Phase 7:

| Flow | Class | Use Case | Notes |
|------|-------|----------|-------|
| Neural Spline Flow | `zuko.flows.NSF` | **Primary recommendation** | Rational-quadratic splines; more expressive than affine MAF |
| Masked Autoregressive Flow | `zuko.flows.MAF` | Baseline/fallback | Affine transforms; simpler, faster, less expressive |
| Neural Autoregressive Flow | `zuko.flows.NAF` | If NSF insufficient | Unconstrained monotonic networks |

**Recommendation: Use NSF (Neural Spline Flow) as the primary flow architecture.**

Rationale:
- NSF extends MAF with monotonic rational-quadratic spline transforms instead of simple affine transforms
- More expressive: can capture non-Gaussian, skewed, heavy-tailed posteriors
- Analytically invertible (exact log-prob computation)
- NSF is the de facto standard in the SBI literature (used by sbi, BayesFlow, and Pyro's own tutorial)
- Only ~2x overhead vs MAF per transform, negligible for inference-time forward pass

**NSF Constructor API** (verified from documentation):

```python
import zuko

flow = zuko.flows.NSF(
    features=D,           # int: dimensionality of latent (posterior) space
    context=C,            # int: dimensionality of summary embedding
    bins=8,               # int: number of spline bins (default 8; 8-16 typical)
    transforms=5,         # int: number of autoregressive transforms (3-8 typical)
    hidden_features=[256, 256],  # list[int]: hidden layer sizes per transform
    randperm=False,       # bool: random permutation between transforms
    passes=None,          # int: None = full autoregressive; 2 = coupling
)
```

**Key domain consideration:** Spline transforms are defined over [-5, 5]. Features outside this range are identity-mapped. The recommendation is to **standardize all features to zero mean, unit variance** before passing to the flow. This is critical for DCM parameters which live on different scales (A_free ~ N(0, 1/64), C ~ N(0, 1), noise params).

### ZukoToPyro Wrapper

**Confidence:** HIGH (official Pyro source code verified)
**Source:** [pyro.contrib.zuko](https://docs.pyro.ai/en/dev/contrib.zuko.html), [source on GitHub](https://github.com/pyro-ppl/pyro/blob/dev/pyro/contrib/zuko.py)

```python
from pyro.contrib.zuko import ZukoToPyro
```

The wrapper is minimal (~30 lines). Key behavior:
- Wraps any `torch.distributions.Distribution` (including Zuko flows) as a `pyro.distributions.TorchDistribution`
- If the distribution has `rsample_and_log_prob` (all Zuko flows do), it caches the log-prob during sampling to avoid redundant computation during ELBO scoring
- Properties `has_rsample`, `event_shape`, `batch_shape` delegated to wrapped distribution

```python
class ZukoToPyro(pyro.distributions.TorchDistribution):
    def __init__(self, dist):
        self.dist = dist
        self.cache = {}

    def __call__(self, shape=()):
        if hasattr(self.dist, "rsample_and_log_prob"):
            x, self.cache[x] = self.dist.rsample_and_log_prob(shape)
        elif self.has_rsample:
            x = self.dist.rsample(shape)
        else:
            x = self.dist.sample(shape)
        return x

    def log_prob(self, x):
        if x in self.cache:
            return self.cache[x]
        return self.dist.log_prob(x)
```

**Integration pattern** (from Pyro official tutorial):

```python
def guide(observed_data):
    pyro.module("flow", flow)
    context = summary_net(observed_data)  # embed data
    pyro.sample("z", ZukoToPyro(flow(context)))
```

This pattern is directly applicable to our DCM guides.

### Pyro SVI Integration

**Confidence:** HIGH (official Pyro tutorial)
**Source:** [SVI with a Normalizing Flow guide](https://pyro.ai/examples/svi_flow_guide.html)

The Pyro tutorial demonstrates the exact pattern we need:
1. Define a Zuko conditional flow with `context` parameter
2. Register with `pyro.module("flow", flow)` in the guide
3. Sample using `pyro.sample("site_name", ZukoToPyro(flow(context)))`
4. Train with standard `SVI` + `Trace_ELBO` + `ClippedAdam`

Key training configuration from the tutorial:
- `ClippedAdam({"lr": 1e-3, "clip_norm": 10.0})`
- `Trace_ELBO(num_particles=16, vectorize_particles=True)` -- multiple particles reduce ELBO variance for flow-based guides
- 4096 training steps in the tutorial (our case will need more due to higher dimensionality)

**Critical note:** The tutorial inverts the flow transform (`flow.transform = flow.transform.inv`) to get an Inverse Autoregressive Flow (IAF) for efficient sampling. This is important: MAF is fast for density evaluation but slow for sampling; IAF is fast for sampling but slow for evaluation. For amortized inference where we primarily sample, **IAF mode is preferred**.

---

## Architecture Patterns

### 1. Summary Network Design

The summary network compresses raw observed data (BOLD time series or CSD matrix) into a fixed-dimensional embedding vector that serves as the flow's context.

**Confidence:** MEDIUM (architecture patterns from SBI literature; DCM-specific choices are novel)
**Sources:** [sbi embedding_net tutorial](https://sbi-dev.github.io/sbi/0.22/tutorial/05_embedding_net/), [BayesFlow](https://arxiv.org/html/2306.16015), [Cranmer et al. 2020](https://doi.org/10.1073/pnas.1912789117)

#### For Task DCM (BOLD time series input)

Input: `(T, N)` BOLD tensor where T ~ 100-500 time points, N = 3-10 regions.

**Recommended: 1D-CNN + Global Pooling**

```
Input (T, N) -> reshape to (N, T) [channels=regions, length=time]
-> Conv1d(N, 64, kernel_size=5, padding=2)
-> BatchNorm1d + ReLU
-> Conv1d(64, 128, kernel_size=5, padding=2)
-> BatchNorm1d + ReLU
-> Conv1d(128, 256, kernel_size=5, padding=2)
-> BatchNorm1d + ReLU
-> AdaptiveAvgPool1d(1)  # squeeze temporal dimension
-> Flatten -> Linear(256, embed_dim)
```

Rationale:
- 1D convolutions naturally capture temporal patterns (hemodynamic response shape)
- `AdaptiveAvgPool1d` handles variable-length time series
- BatchNorm stabilizes training with heterogeneous simulation data
- Embedding dim should be 64-256 (128 is a good starting point)

Alternative considered: Set Transformer / DeepSet. These are better for unordered sets, but BOLD data has temporal ordering that convolutions exploit. Transformers also require more parameters and more training data. **Use 1D-CNN for BOLD.**

#### For Spectral DCM (CSD matrix input)

Input: `(F, N, N)` complex CSD tensor, F ~ 32 frequency bins, N = 3-10 regions.

**Recommended: Flatten + MLP (after real/imag decomposition)**

```
Input (F, N, N) complex
-> decompose to (2*F*N*N,) real vector (matching existing pattern)
-> Linear(2*F*N*N, 512) + ReLU
-> Linear(512, 256) + ReLU
-> Linear(256, embed_dim)
```

Rationale:
- CSD is already a compact summary statistic (frequency-domain)
- The matrix structure is small enough that flattening + MLP is sufficient
- For N=3, F=32: input dim = 2*32*3*3 = 576, very manageable
- For N=10, F=32: input dim = 6400, still tractable with MLP
- More complex architectures (graph neural networks for the N*N structure) are overkill at this scale

#### For Regression DCM (frequency-domain regressors)

Input: `(Y, X)` where Y is `(N_eff, nr)` and X is `(N_eff, D)`.

rDCM has closed-form analytic posteriors via VB, so the amortized guide is **optional** (as stated in the phase description). If implemented:

**Recommended: Per-region MLP with shared weights**

```
For each region r:
  Input: (Y_r, X_r) concatenated -> (N_eff, D_r + 1)
  -> Global average over N_eff dimension
  -> MLP -> embed_r

Stack embed_r across regions -> full embedding
```

### 2. Flow Architecture Per Variant

#### Task DCM Flow

**Latent dimensions:** Count the Pyro sample sites in `task_dcm_model`:
- `A_free`: N*N parameters (e.g., 3*3 = 9 for 3 regions)
- `C`: N*M parameters (e.g., 3*1 = 3)
- `noise_prec`: 1 parameter
- **Total for 3 regions, 1 input: 13 dimensions**

For 5 regions, 2 inputs: 5*5 + 5*2 + 1 = 36 dimensions.

**Recommended NSF configuration for 3-region task DCM:**

```python
flow = zuko.flows.NSF(
    features=13,           # A_free(9) + C(3) + noise_prec(1)
    context=128,           # summary network output dim
    bins=8,                # spline bins
    transforms=5,          # number of autoregressive layers
    hidden_features=[256, 256],
)
```

#### Spectral DCM Flow

**Latent dimensions:**
- `A_free`: N*N (e.g., 9)
- `noise_a`: 2*N (e.g., 6)
- `noise_b`: 2*1 = 2
- `noise_c`: 2*N (e.g., 6)
- `csd_noise_scale`: 1
- **Total for 3 regions: 24 dimensions**

```python
flow = zuko.flows.NSF(
    features=24,
    context=128,
    bins=8,
    transforms=5,
    hidden_features=[256, 256],
)
```

#### Regression DCM Flow

**Latent dimensions:** Variable per region (D_r differs). This is a complication.

Options:
1. **Pad to max D_r** and mask: simplest, but wastes capacity
2. **Separate flow per region**: clean but N separate networks
3. **Skip amortized guide**: use analytic VB (the primary rDCM inference path)

**Recommendation: Skip amortized guide for rDCM.** The analytic VB posterior is exact (conjugate model). An amortized guide adds complexity with no accuracy benefit. Implement only if the comparison study (AMR-04) specifically requires it. If needed, option 2 (separate per-region flows) is cleanest.

### 3. Parameter Packing/Unpacking

The flow outputs a single vector of dimension D (total parameters). This must be unpacked into individual Pyro sample sites.

**Pattern:**

```python
def guide(observed_data, ...):
    pyro.module("flow", flow)
    pyro.module("summary_net", summary_net)

    # Embed observed data
    embedding = summary_net(observed_data)

    # Sample from flow
    z = pyro.sample("_latent", ZukoToPyro(flow(embedding)))

    # Unpack into named sites using pyro.deterministic
    # (or use separate samples and combine -- see alternatives below)
    A_free = z[..., :N*N].reshape(..., N, N)
    C = z[..., N*N:N*N+N*M].reshape(..., N, M)
    noise_prec = z[..., -1].exp()  # ensure positive
```

**Critical issue:** Pyro's ELBO requires the guide to have sample sites with the **same names** as the model's sample sites. The flow outputs a single vector, not individual named samples.

**Solution approaches:**

*Approach A: Single flow, delta-distribute individual sites*
```python
z = flow(embedding).rsample()
A_free_val = z[:N*N].reshape(N, N)
pyro.sample("A_free", dist.Delta(A_free_val).to_event(2))
pyro.sample("C", dist.Delta(C_val).to_event(2))
```
This works because Delta distributions pass through the KL computation. The flow's log_prob handles the variational entropy.

*Approach B: Multiple independent flows per site*
Each site gets its own conditional flow. Loses inter-site correlations but simpler.

*Approach C: Single flow with manual ELBO*
Bypass Pyro's automatic ELBO matching and compute it manually.

**Recommendation: Approach A (single flow + Delta sites).** This is the pattern used in the Pyro official tutorial and preserves posterior correlations between A_free, C, and noise_prec. However, the entropy term must be handled carefully -- the flow's log_prob provides the full q(z|x) entropy, and the Delta distributions contribute zero entropy, avoiding double-counting.

Actually, reviewing more carefully: the cleanest Pyro pattern is to have the guide sample from the flow and use `pyro.sample` with the flow distribution directly for a single encompassing latent, then use `pyro.deterministic` to expose the unpacked values. But this requires the model to also have a single latent... which it does not.

**Revised recommendation:** The guide must produce samples with matching site names. The correct pattern:

```python
def amortized_guide(observed_data, ...):
    pyro.module("flow", flow)
    pyro.module("summary_net", summary_net)

    embedding = summary_net(observed_data)
    # Get joint sample from flow (not registered as pyro.sample)
    z = flow(embedding).rsample()

    # Register each component as a pyro.sample with Delta
    idx = 0
    A_free = z[idx:idx + N*N].reshape(N, N)
    idx += N*N
    pyro.sample("A_free", dist.Delta(A_free).to_event(2))

    C_val = z[idx:idx + N*M].reshape(N, M)
    idx += N*M
    pyro.sample("C", dist.Delta(C_val).to_event(2))

    noise_prec_raw = z[idx]
    idx += 1
    pyro.sample("noise_prec", dist.Delta(noise_prec_raw.exp()))
```

This requires a **custom ELBO** that accounts for the flow entropy instead of the Delta entropy. Standard `Trace_ELBO` will see Delta log-probs (which are 0 or -inf) and fail.

**Alternative (simpler, tested):** Restructure the model to sample a single packed latent vector and deterministically unpack. This avoids the site-matching problem entirely.

**Final recommendation:** This is a known pain point. The Pyro tutorial sidesteps it by having both model and guide use the same single `pyro.sample("z", ...)` site. **We should follow the same pattern**: modify the model (or create a wrapper model) that samples a single packed latent vector, then deterministically unpacks it. This keeps Pyro's automatic ELBO machinery working.

### 4. Training Data Generation

**Confidence:** HIGH (existing simulators are well-tested)

The three simulators can generate training data directly:

| Variant | Simulator | Key Inputs | Output |
|---------|-----------|------------|--------|
| Task DCM | `simulate_task_dcm()` | A, C, stimulus | BOLD `(T, N)` |
| Spectral DCM | `simulate_spectral_dcm()` | A, noise_params | CSD `(F, N, N)` |
| Regression DCM | `simulate_rdcm()` | A, C, u | Y, X |

**Training dataset generation pipeline:**

```python
for i in range(n_simulations):
    # 1. Sample parameters from prior
    A_free = torch.randn(N, N) * (1/64)**0.5
    A = parameterize_A(A_free * a_mask)
    C = torch.randn(N, M) * c_mask
    # ... noise params for spectral

    # 2. Run simulator
    result = simulate_task_dcm(A, C, stimulus, ...)

    # 3. Store (data, params) pair
    dataset.append((result['bold'], A_free, C, noise_prec))
```

**Dataset size recommendations** (from SBI literature):
- Minimum: 1,000 simulations (proof of concept)
- Target: 10,000 simulations (stated in success criteria)
- Ideal: 50,000-100,000 for high-dimensional posteriors
- The 10,000 target is reasonable for 13-36 dimensional posteriors

**Simulation speed estimate:**
- Task DCM simulation: ~1-5 seconds per subject (ODE integration)
- Spectral DCM simulation: ~0.01 seconds per subject (algebraic, no ODE)
- 10,000 task DCM simulations: ~3-14 hours on single CPU
- 10,000 spectral DCM simulations: ~2 minutes

**Recommendation:** Pre-generate and cache training datasets to disk. Use `torch.save`/`torch.load` for the tensor pairs. Spectral DCM is fast enough to generate on-the-fly; task DCM must be pre-generated.

---

## Feature Landscape

### Table Stakes (Must Have)

| Feature | Why Required | Complexity | Notes |
|---------|-------------|------------|-------|
| NSF flow per DCM variant | Core deliverable (AMR-01, AMR-02) | Medium | One flow architecture, two summary nets |
| Summary network for BOLD | Task DCM input encoding | Medium | 1D-CNN, variable-length handling |
| Summary network for CSD | Spectral DCM input encoding | Low | MLP on flattened real/imag |
| Training data generator | Feed the flow | Low | Wraps existing simulators |
| Training loop | Amortized SVI | Medium | Standard Pyro SVI with flow guide |
| Posterior sampling | Single forward pass | Low | `flow(embedding).rsample()` |
| RMSE comparison vs SVI | AMR-04 success criterion | Low | Already have SVI infrastructure |
| Amortization gap metric | Success criterion | Low | Per-subject ELBO comparison |

### Differentiators

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Calibration diagnostics (SBC) | Validates posterior quality | Medium | Simulation-based calibration |
| Multi-subject batch inference | Main selling point of amortization | Low | Trivial once flow is trained |
| Transfer to real data | Scientific utility | High | Requires domain shift handling |
| Online fine-tuning | Closes amortization gap | Medium | Few SVI steps after flow init |
| Mixture base distribution | Handles multimodal posteriors | Medium | GMM base instead of Gaussian |

### Anti-Features (Do NOT Build)

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Amortized guide for rDCM | Analytic VB is exact and fast | Use closed-form posterior |
| Complex custom ELBO | Fragile, hard to debug | Restructure model for single-site compatibility |
| Graph neural network summary | Overkill for N < 20 regions | Use MLP/CNN |
| Continuous normalizing flow (CNF) | Slow (ODE-based), unstable training | Use discrete NSF |
| Diffusion-based posterior | Too new, not Pyro-compatible | Stick with normalizing flows |

---

## Architecture Recommendations

### Overall Component Architecture

```
Training Phase:
  Prior -> Simulator -> (data, params) pairs
                            |
                            v
  data -> [Summary Network] -> embedding
                                    |
                                    v
  params <- [Normalizing Flow (NSF)] <- embedding
                    |
                    v
            Train via SVI (ELBO)

Inference Phase:
  new_data -> [Summary Network] -> embedding
                                       |
                                       v
              [Normalizing Flow] -> posterior samples (single forward pass)
```

### File Structure

```
src/pyro_dcm/guides/
    __init__.py
    amortized_flow.py       # AmortizedFlowGuide class
    summary_networks.py     # BoldSummaryNet, CsdSummaryNet
    parameter_packing.py    # Pack/unpack Pyro sites to/from vectors

src/pyro_dcm/models/
    amortized_task_model.py     # Wrapper model with single packed latent
    amortized_spectral_model.py # Wrapper model with single packed latent

scripts/
    generate_training_data.py   # Pre-generate (data, params) pairs
    train_amortized_guide.py    # Training script

tests/
    test_amortized.py           # Unit + integration tests
```

### Component Boundaries

| Component | Responsibility | Interface |
|-----------|---------------|-----------|
| `BoldSummaryNet` | BOLD (T,N) -> embedding (D,) | `nn.Module`, forward returns tensor |
| `CsdSummaryNet` | CSD (F,N,N) -> embedding (D,) | `nn.Module`, forward returns tensor |
| `AmortizedFlowGuide` | Wraps summary net + flow + packing | Callable compatible with Pyro guide |
| `pack_params` / `unpack_params` | Convert named dict <-> flat vector | Pure functions |
| `generate_training_data` | Run simulator N times, store to disk | Script, outputs .pt file |
| `train_amortized_guide` | Full training pipeline | Script, outputs trained model |

### Data Flow

1. **Training:** Prior samples -> Simulator -> Dataset of (bold/csd, params) -> DataLoader -> Summary net + Flow -> ELBO loss -> Backprop
2. **Inference:** New BOLD/CSD -> Summary net -> Flow -> Posterior samples -> Unpack to A, C, etc.

---

## Domain Pitfalls

### Critical Pitfalls

#### Pitfall 1: Pyro Site Name Mismatch

**What goes wrong:** The flow outputs a single vector, but the Pyro model has multiple named sample sites (A_free, C, noise_prec). Standard `Trace_ELBO` cannot match these.

**Why it happens:** Pyro's ELBO computation aligns model and guide traces by site name. A flow-based guide naturally produces a single joint sample.

**Consequences:** NaN ELBO, incorrect gradient computation, silent incorrect inference.

**Prevention:** Create a wrapper model that samples a single packed latent vector and deterministically unpacks it. Both model and guide then have a single matching sample site. Alternatively, implement a custom ELBO that handles the flow entropy correctly.

**Detection:** Check that `Trace_ELBO` produces finite values on the first step. If NaN or extremely large, site matching is broken.

#### Pitfall 2: Multimodal Posteriors from Weakly Identifiable Configurations

**What goes wrong:** Standard normalizing flows with unimodal Gaussian base distributions cannot represent disconnected posterior modes. The flow places probability mass between modes ("probability bridges").

**Why it happens:** DCM parameters can be weakly identifiable -- multiple A matrices produce similar BOLD/CSD patterns. This creates multimodal posteriors.

**Consequences:** Overconfident, inaccurate posteriors. Coverage drops below target [0.85, 0.99].

**Prevention:**
1. Start with well-identifiable configurations (strong inputs, distinct connectivity)
2. Monitor calibration via SBC during training
3. If multimodality detected: use a Gaussian Mixture Model base distribution instead of a single Gaussian (recent literature supports this approach)
4. Alternatively: use more flow transforms (deeper flow) which can approximate multimodality to some degree

**Detection:** Coverage < 0.85 on held-out simulated data; bimodal parameter histograms that the flow smooths into a single bump.

#### Pitfall 3: Spline Domain Truncation

**What goes wrong:** NSF spline transforms operate on [-5, 5]. Parameters outside this range pass through unchanged (identity transform), losing expressiveness.

**Why it happens:** DCM parameters may have different scales. A_free ~ N(0, 1/8) is small, but noise_prec ~ Gamma(1,1) can be large.

**Consequences:** The flow cannot learn the full posterior for out-of-range parameters. Posterior is truncated or distorted.

**Prevention:** **Standardize all parameters to zero mean, unit variance** before packing into the flow's feature vector. Store the mean and std used for standardization. Unstandardize after sampling.

**Detection:** Check that >99% of training parameter samples fall within [-5, 5] after standardization.

### Moderate Pitfalls

#### Pitfall 4: Slow Task DCM Simulation for Training Data

**What goes wrong:** Generating 10,000 task DCM simulations takes 3-14 hours due to ODE integration.

**Prevention:**
- Pre-generate training data and cache to disk
- Use coarser dt during simulation (0.1 instead of 0.01) for training data -- slight accuracy loss is acceptable since the flow learns from the distribution, not individual points
- Parallelize across CPU cores using `torch.multiprocessing`
- Start with 1,000 simulations for prototyping, scale up later

#### Pitfall 5: Unstable Simulations in Training Data

**What goes wrong:** Some prior samples produce A matrices with eigenvalues near zero, causing ODE blow-up or NaN BOLD. These corrupt the training dataset.

**Prevention:**
- Filter out simulations with NaN/Inf in the BOLD output
- Use the existing `make_random_stable_A` which enforces eigenvalue stability
- Clip A_free prior samples to a reasonable range before parameterize_A
- Target: <1% rejection rate

#### Pitfall 6: ELBO Variance with Flow Guides

**What goes wrong:** Single-particle ELBO has high variance with expressive flows, causing noisy gradients and slow convergence.

**Prevention:** Use `num_particles=8-16` in `Trace_ELBO` with `vectorize_particles=True`. This increases memory but dramatically stabilizes training. The Pyro tutorial uses 16 particles.

#### Pitfall 7: Mode Collapse During Flow Training

**What goes wrong:** The flow collapses to a narrow region of parameter space, ignoring parts of the prior.

**Prevention:**
- Use forward KL (standard SVI/ELBO) not reverse KL for training
- Monitor training ELBO for sudden drops
- Use learning rate warmup or annealing
- Gradient clipping (already standard in our `run_svi`)

### Minor Pitfalls

#### Pitfall 8: Transform Inversion Direction

**What goes wrong:** MAF is fast for log_prob but slow for sampling. In amortized inference, we primarily sample (one forward pass per subject).

**Prevention:** Invert the transform: `flow.transform = flow.transform.inv` to get IAF behavior (fast sampling, slower log_prob). NSF with `passes=2` (coupling mode) is fast in both directions -- **prefer coupling NSF over autoregressive NSF**.

#### Pitfall 9: Forgetting to Register Modules

**What goes wrong:** Summary network and flow parameters are not optimized because they are not registered with Pyro.

**Prevention:** Always use `pyro.module("summary_net", summary_net)` and `pyro.module("flow", flow)` inside the guide function. This registers parameters for gradient computation.

---

## Comparison: NSF vs MAF vs Coupling NSF

| Criterion | MAF (autoregressive) | NSF (autoregressive) | NSF (coupling, passes=2) |
|-----------|---------------------|---------------------|--------------------------|
| Expressiveness | Low (affine) | High (spline) | High (spline) |
| Log-prob speed | Fast (parallel) | Fast (parallel) | Fast (parallel) |
| Sampling speed | Slow (sequential) | Slow (sequential) | **Fast (parallel)** |
| Training stability | Good | Good | Good |
| Parameters | Fewer | More (~2x MAF) | More (~2x MAF) |
| Best for | Density estimation | Posterior estimation | **Amortized inference** |

**Recommendation: Coupling NSF (passes=2) for amortized inference.** Fast in both directions, highly expressive. Use autoregressive NSF only if coupling mode is insufficient.

---

## Training Protocol Recommendations

### Hyperparameters

| Parameter | Recommended Value | Range to Explore | Notes |
|-----------|------------------|------------------|-------|
| Flow transforms | 5 | 3-8 | More = more expressive, slower |
| Spline bins | 8 | 4-16 | 8 is standard, 16 for complex posteriors |
| Hidden features | [256, 256] | [128,128] to [512,512] | Scale with problem dimension |
| Summary embed dim | 128 | 64-256 | Match flow context dim |
| Learning rate | 1e-3 | 1e-4 to 3e-3 | With cosine or exponential decay |
| Gradient clip norm | 10.0 | 5.0-20.0 | Standard for flow training |
| ELBO particles | 16 | 4-32 | 16 balances variance and cost |
| Batch size | 128 | 64-512 | Training subjects per batch |
| Training epochs | 100-500 | - | Early stop on validation ELBO |
| Training dataset size | 10,000 | 1,000-100,000 | 10K for 13D, more for higher D |

### Training Loop Structure

```python
# Pseudocode
dataset = load_training_data()  # pre-generated (data, params) pairs
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

optimizer = ClippedAdam({"lr": 1e-3, "clip_norm": 10.0})
elbo = Trace_ELBO(num_particles=16, vectorize_particles=True)
svi = SVI(amortized_model, amortized_guide, optimizer, loss=elbo)

for epoch in range(n_epochs):
    for batch in dataloader:
        loss = svi.step(batch)
    # Validate on held-out set
    val_elbo = evaluate(val_set)
    # Early stopping
```

### Evaluation Protocol

1. **RMSE comparison (AMR-04):** For 100 held-out simulated subjects, compare:
   - Amortized posterior mean vs ground truth RMSE
   - Per-subject SVI posterior mean vs ground truth RMSE
   - Target: amortized RMSE < 1.5x SVI RMSE

2. **Amortization gap:** For each held-out subject:
   - Compute per-subject ELBO (run SVI for that subject)
   - Compute amortized ELBO (forward pass through flow)
   - Gap = (per-subject ELBO - amortized ELBO) / |per-subject ELBO|
   - Target: gap < 10%

3. **Coverage:** Using simulation-based calibration (SBC):
   - Generate 1000 (theta*, x*) pairs from prior
   - For each: compute posterior credible intervals via flow
   - Check if theta* falls within 90% CI
   - Target: coverage in [0.85, 0.99]

4. **Inference speed:**
   - Time a single forward pass through summary net + flow
   - Target: < 1 second per subject

---

## Implications for Roadmap

### Suggested Phase Structure

Based on the research, Phase 7 should be structured as **three plans**:

1. **Plan 07-01: Infrastructure and Summary Networks**
   - Implement `BoldSummaryNet` (1D-CNN for task DCM)
   - Implement `CsdSummaryNet` (MLP for spectral DCM)
   - Implement parameter packing/unpacking utilities
   - Implement `generate_training_data.py` script
   - Pre-generate 1,000 training datasets for each variant (prototyping)
   - Tests: summary net output shapes, packing round-trips

2. **Plan 07-02: Amortized Flow Guide for Task DCM (AMR-01)**
   - Implement `AmortizedFlowGuide` class with NSF + ZukoToPyro
   - Create wrapper model with single packed latent site
   - Implement training loop in `train_amortized_guide.py`
   - Train on 10,000 simulated task DCM datasets
   - Evaluate RMSE, amortization gap, coverage
   - Tests: ELBO convergence, RMSE < 1.5x SVI, coverage in range

3. **Plan 07-03: Amortized Flow Guide for Spectral DCM (AMR-02) + Benchmarks (AMR-04)**
   - Implement spectral DCM amortized guide
   - Train on 10,000 simulated spectral DCM datasets
   - Cross-variant comparison (AMR-04): amortized vs SVI vs NUTS
   - rDCM decision: skip amortized guide (analytic VB is primary)
   - Final benchmark report
   - Tests: all success criteria met

### Phase Ordering Rationale

- **01 before 02:** Summary nets and data generation are shared infrastructure needed by both variants
- **02 before 03:** Task DCM is the more challenging variant (ODE-based); validate the pattern here first
- **03 last:** Spectral DCM is faster (no ODE), so iteration is quick; final benchmarks cap the phase

### Research Flags

- **Plan 07-02 may need deeper research** on the site-matching problem if the wrapper model approach introduces subtle ELBO issues
- **Plan 07-02 may need iteration** on training data size if 10,000 is insufficient
- **AMR-03 (rDCM amortized guide) should be explicitly deferred** -- the analytic VB posterior is the correct inference method for rDCM

---

## Confidence Assessment

| Area | Confidence | Reason |
|------|------------|--------|
| Zuko API / architecture | HIGH | Official docs, GitHub, PyPI verified |
| ZukoToPyro integration | HIGH | Official Pyro source code verified |
| Summary network design | MEDIUM | Standard SBI patterns; DCM-specific choices are novel |
| Site-matching pattern | MEDIUM | Pyro tutorial uses simple case; multi-site DCM model needs wrapper |
| Training hyperparameters | MEDIUM | Literature ranges; will need empirical tuning |
| Multimodal handling | MEDIUM | Known problem, known solutions, but untested in DCM context |
| Amortization gap targets | LOW-MEDIUM | 10% gap is ambitious; may need relaxation for task DCM |
| rDCM skip recommendation | HIGH | Analytic VB is provably exact for conjugate rDCM model |

---

## Open Questions

1. **How to handle the Pyro site-matching problem cleanly?** The wrapper model approach (single packed latent) is the simplest path but requires modifying the model interface. Need to verify ELBO computation is correct.

2. **Is 10,000 simulations sufficient for task DCM?** The posterior is 13D (3 regions) to 36D (5 regions). Literature suggests 10K is sufficient for moderate dimensions, but validation is needed.

3. **Should we implement online fine-tuning?** After amortized inference, running a few SVI steps initialized from the flow's output could close the amortization gap. This is a "nice to have" if the gap exceeds 10%.

4. **What about variable network sizes?** The current design assumes fixed N (number of regions). A truly amortized guide would handle variable N. This is future work (requires set-based architectures like DeepSet/Transformer).

5. **How does the amortized guide interact with model comparison?** The ELBO from amortized inference could potentially be used for model comparison (different masks), but the amortization gap adds noise. Need to validate that ELBO differences between models exceed the amortization gap.

---

## Key References

| ID | Reference | Relevance |
|----|-----------|-----------|
| REF-042 | Papamakarios et al. (2021), JMLR | Normalizing flow theory |
| REF-043 | Cranmer et al. (2020), PNAS | Simulation-based inference framework |
| REF-062 | Zuko, GitHub | Flow implementation library |
| REF-060 | Pyro, Bingham et al. (2019) | Probabilistic programming framework |
| -- | [Pyro SVI flow guide tutorial](https://pyro.ai/examples/svi_flow_guide.html) | ZukoToPyro integration pattern |
| -- | [sbi embedding_net tutorial](https://sbi-dev.github.io/sbi/0.22/tutorial/05_embedding_net/) | Summary network patterns |
| -- | [BayesFlow, Radev et al. (2023)](https://arxiv.org/html/2306.16015) | Amortized Bayesian workflows |
| -- | [Multimodal flows](https://arxiv.org/abs/2512.04954) | GMM base for multimodal posteriors |
| -- | [Stable NF training](https://arxiv.org/html/2402.16408v1) | Training stability techniques |
| -- | [SBC diagnostics](https://sbi-dev.github.io/sbi/0.22/tutorial/13_diagnostics_simulation_based_calibration/) | Posterior calibration checking |
| -- | [Mode collapse mitigation](https://arxiv.org/abs/2505.03652) | Annealing for normalizing flows |
