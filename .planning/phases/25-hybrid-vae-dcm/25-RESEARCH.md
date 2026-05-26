# Phase 25: Hybrid VAE-DCM - Research

**Researched:** 2026-05-26
**Domain:** Physics-informed variational autoencoders with ODE-based decoders for neural timeseries
**Confidence:** MEDIUM

## Summary

Phase 25 builds a sequential VAE where the decoder IS the DCM forward model (bilinear
neural state ODE + observation equation). The encoder maps M/EEG source-localized ROI
timeseries to approximate posterior parameters (A, B_j, C, initial conditions), and the
DCM ODE integrator produces predicted timeseries. Training uses ELBO maximization --
reconstruction loss plus KL divergence against DCM-calibrated priors.

This architecture sits at the intersection of three well-established lines of work:
(1) Latent ODE models (Rubanova et al. 2019, Chen et al. 2018) that place neural ODEs
inside VAE decoders; (2) Physics-informed VAEs (PI-VAE, Phi-DVAE) that embed known
differential equations as structured decoders; and (3) the TREND architecture (Imaging
Neuroscience 2024) which is the closest neuroscience precedent -- a Transformer encoder
with a full DCM as decoder for fMRI effective connectivity estimation. The project
already has all the forward model components (CoupledDCMSystem with hemodynamic toggle,
bilinear neural state equation, Pyro-based generative models, amortized flow guides
with summary networks, parameter packing). Phase 25 combines these into a new VAE
architecture rather than building from scratch.

The key technical challenge is training stability: ODE-based decoders can produce NaN
or divergent trajectories during early training when the encoder outputs extreme
parameter values. The project has existing mitigation patterns (NaN guard in
`latent_circuit_dcm_model`, stability monitor in `CoupledDCMSystem`) that should be
reused. KL annealing (warmup) and gradient clipping are standard additional measures.

**Primary recommendation:** Build the hybrid VAE-DCM as a Pyro model/guide pair
reusing the existing `CoupledDCMSystem(hemodynamic=False)` as the decoder, a new
`DCMEncoderNet` (1D-CNN or transformer encoder) as the guide's recognition network,
and the existing parameter packing infrastructure for the latent space. Start with
synthetic validation using `simulate_latent_circuit`, then extend to real M/EEG.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | >=2.0 | Tensor computation, autograd, nn.Module | Already in project |
| Pyro | >=1.9 | Probabilistic programming, SVI, ELBO | Already in project; VAE is a first-class Pyro pattern |
| torchdiffeq | 0.2.5 | ODE integration with adjoint backprop | Already in project; O(1) memory backprop through ODE |
| Zuko | >=1.2 | Normalizing flows for flow-enhanced guide | Already in project; optional for richer posterior |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| matplotlib | any | Training curves, reconstruction plots, A-matrix visualization | Diagnostics and publication figures |
| scipy | any | Signal processing, CSD computation for spectral variant | If spectral VAE-DCM variant needed |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Pyro VAE pattern | Pure PyTorch VAE | Lose probabilistic semantics, prior specification, automatic ELBO; must hand-roll KL |
| torchdiffeq adjoint | torchdiffeq direct backprop | Direct is simpler but O(T) memory; adjoint is O(1) but requires smooth activations |
| 1D-CNN encoder | Transformer encoder | Transformer better at long-range temporal dependencies but heavier; CNN proven in existing BoldSummaryNet |
| torchode | torchdiffeq | torchode supports batched ODE but less mature; torchdiffeq already in project |

**Installation:**
```bash
# No new dependencies required -- all are already in pyproject.toml
pip install -e ".[latent]"
```

## Architecture Patterns

### Recommended Project Structure
```
src/pyro_dcm/
  models/
    hybrid_vae_dcm.py          # Pyro model (decoder = DCM ODE) + guide (encoder)
  guides/
    dcm_encoder_net.py         # Recognition network: timeseries -> DCM params
    parameter_packing.py       # Extend with LatentCircuitDCMPacker
  simulators/
    latent_circuit_simulator.py  # Already exists -- synthetic validation data
  forward_models/
    coupled_system.py           # Already exists -- CoupledDCMSystem(hemodynamic=False)
    latent_observation.py       # Already exists -- direct_observation
tests/
  test_hybrid_vae_dcm.py       # Unit + integration tests
  test_dcm_encoder_net.py      # Encoder architecture tests
scripts/
  train_hybrid_vae_dcm.py      # Training script (cluster)
```

### Pattern 1: Pyro VAE with Physics-Informed Decoder
**What:** The standard Pyro VAE pattern (model=generative process, guide=recognition
network) where the model's decoder is a deterministic DCM ODE integration rather
than a learned neural network. The encoder (guide) outputs approximate posterior
parameters for DCM connectivity and initial conditions.

**When to use:** When the generative process has known physics (differential
equations) and the goal is both reconstruction AND interpretable parameters.

**Architecture:**
```
Encoder (Guide):                    Decoder (Model):
M/EEG timeseries (T, N)            z = {A_free, C, x0, noise_prec}
       |                                   |
  [1D-CNN / Transformer]            parameterize_A(A_free * a_mask)
       |                                   |
  [z_loc, z_scale]                  CoupledDCMSystem(hemodynamic=False)
       |                                   |
  pyro.sample("_latent", N(z_loc,   torchdiffeq.odeint(system, x0, t_eval)
              z_scale))                     |
                                    direct_observation(x, C_obs, noise_prec)
                                           |
                                    pyro.sample("obs", N(y_mean, noise_std),
                                                obs=observed)
```

**Key implementation detail:** The model and guide must share the same sample site
names. Following the existing `amortized_wrappers.py` pattern, use a single
packed `_latent` site in both model and guide. The model samples `_latent` from a
standard normal prior, unstandardizes and unpacks into DCM parameters, then runs
the ODE decoder. The guide (encoder) samples `_latent` from the recognition
network's output distribution.

**Example (model side):**
```python
def hybrid_vae_dcm_model(
    observed_trajectories: torch.Tensor,
    stimulus: PiecewiseConstantInput,
    a_mask: torch.Tensor,
    c_mask: torch.Tensor,
    t_eval: torch.Tensor,
    dt: float,
    packer: LatentCircuitDCMPacker,
    *,
    b_masks: list[torch.Tensor] | None = None,
    stim_mod: PiecewiseConstantInput | None = None,
) -> None:
    """Pyro generative model: DCM ODE as decoder."""
    pyro.module("hybrid_vae_dcm", ...)  # register modules

    # Sample packed latent from prior (standard normal in standardized space)
    n = packer.n_features
    z_std = pyro.sample(
        "_latent",
        dist.Normal(torch.zeros(n), torch.ones(n)).to_event(1),
    )
    z = packer.unstandardize(z_std)
    params = packer.unpack(z)

    # Extract DCM parameters
    A_free = params["A_free"] * a_mask
    A = parameterize_A(A_free)
    C = params["C"] * c_mask
    x0 = params["x0"]  # initial conditions (NEW: not in existing packer)
    noise_prec = params["noise_prec"].exp()

    # DCM ODE decoder
    system = CoupledDCMSystem(A, C, stimulus, hemodynamic=False)
    solution = integrate_ode(system, x0, t_eval, method="rk4", step_size=dt)

    # NaN guard (existing pattern)
    if torch.isnan(solution).any() or torch.isinf(solution).any():
        solution = torch.zeros_like(solution).detach()

    # Observation likelihood
    y_mean, noise_std = direct_observation(solution, C_obs, noise_prec)
    pyro.sample("obs", dist.Normal(y_mean, noise_std).to_event(2),
                obs=observed_trajectories)
```

**Example (guide/encoder side):**
```python
class HybridVAEDCMGuide(nn.Module):
    """Encoder (recognition network) for hybrid VAE-DCM."""

    def __init__(self, encoder_net, packer):
        super().__init__()
        self.encoder_net = encoder_net
        self.packer = packer

    def forward(self, observed_trajectories, *args, **kwargs):
        pyro.module("encoder_net", self.encoder_net)
        # Encode observations to latent distribution parameters
        z_loc, z_scale = self.encoder_net(observed_trajectories)
        # Sample in standardized space
        z_std = pyro.sample(
            "_latent",
            dist.Normal(z_loc, z_scale).to_event(1),
        )
        return z_std
```

### Pattern 2: KL Annealing for ODE Decoder Stability
**What:** Gradually increase the KL divergence weight from 0 to 1 during the
first N training epochs. This lets the encoder-decoder pair first learn good
reconstructions (decoder learns to integrate ODEs stably), then slowly adds
regularization pressure toward the prior.

**When to use:** Always with ODE-based decoders. Without annealing, the KL
term can dominate early training, pushing the encoder to output near-prior
parameters that produce unstable ODE solutions.

**Implementation in Pyro:** Use a custom ELBO or `TraceMeanField_ELBO` with
a loss scaling factor:
```python
# Pyro supports custom ELBO via poutine.scale
def scaled_model(beta, *args, **kwargs):
    with pyro.poutine.scale(scale=beta):
        return hybrid_vae_dcm_model(*args, **kwargs)

# In training loop:
for epoch in range(n_epochs):
    beta = min(1.0, epoch / warmup_epochs)
    loss = svi.step(beta, observed, ...)
```

**Alternative (cleaner Pyro pattern):** Use `Trace_ELBO` and manually scale the
KL in a custom loss class. See Pyro's `custom_objectives` tutorial.

### Pattern 3: Multi-Subject Amortized Training
**What:** Train the encoder on simulated or multi-subject data so that at
inference time, a single forward pass through the encoder produces posterior
DCM parameters for a new subject without per-subject SVI.

**When to use:** The main value proposition of Phase 25 -- amortized inference.

**Training data sources:**
1. **Synthetic (Phase 1):** Generate diverse DCM parameter sets, simulate
   trajectories via `simulate_latent_circuit`, train encoder to recover them.
2. **Multi-subject real data (Phase 2):** Train on a cohort (e.g., Cam-CAN MEG
   subjects), each providing one training example.

### Anti-Patterns to Avoid
- **Sampling initial conditions from the prior every SVI step without encoder
  guidance:** The ODE is extremely sensitive to initial conditions; randomly
  sampling x0 from a broad prior produces wildly divergent trajectories. The
  encoder must provide informed x0 estimates.
- **Using ReLU activations in the encoder network:** torchdiffeq's adjoint
  method requires smooth activations. Use Softplus, ELU, or GELU. This applies
  to the ODE right-hand side, but the encoder itself can use any activation.
- **Direct backprop through long ODE solves without adjoint:** Memory grows
  linearly with integration steps. For T=1000+ time points at dt=0.01, this is
  prohibitive. Use `odeint_adjoint` for the decoder.
- **Ignoring the packer standardization:** The NSF flow and the standard normal
  prior in the packed latent space require standardized parameters. Skipping
  `fit_standardization` will produce a model that trains poorly.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| ODE integration with backprop | Custom RK4 with autograd tape | `torchdiffeq.odeint` or `odeint_adjoint` | Adjoint method gives O(1) memory; custom solvers have subtle gradient bugs |
| Parameter packing/unpacking | Manual tensor slicing | Extend `TaskDCMPacker` or create `LatentCircuitDCMPacker` | Standardization, log-space contract, batch-dim handling already tested |
| DCM forward model | New ODE system class | `CoupledDCMSystem(hemodynamic=False)` | Already has bilinear support, stability monitor, device alignment |
| NaN handling in ODE decoder | Custom NaN detection | Existing NaN guard pattern from `latent_circuit_dcm_model` | Zero-fill + detach pattern is proven; produces finite ELBO penalty |
| Summary network for timeseries | Custom CNN | Extend `BoldSummaryNet` or create variant | Architecture proven, handles unbatched/batched, adaptive pooling |
| ELBO computation | Manual ELBO | `pyro.infer.Trace_ELBO()` with `SVI` | Automatic gradient computation, plate handling, vectorized |
| A-matrix parameterization | Custom diagonal constraint | `parameterize_A()` | Guarantees negative diagonal (stability), tested extensively |

**Key insight:** Phase 25 is primarily an *integration* phase, not a *new
primitives* phase. The DCM forward model, parameter packing, summary networks,
Pyro model/guide pattern, and torchdiffeq integration all exist. The new work
is: (1) a new encoder network architecture, (2) a new packer that includes
initial conditions, (3) wiring the encoder as a Pyro guide paired with the
ODE decoder model, and (4) training infrastructure with KL annealing.

## Common Pitfalls

### Pitfall 1: ODE Decoder Divergence During Early Training
**What goes wrong:** The encoder outputs extreme A matrix values (large positive
eigenvalues) in early training before it has learned reasonable parameter
ranges. The ODE integrator diverges, producing NaN/Inf, corrupting gradients.
**Why it happens:** Random initialization of encoder weights maps to arbitrary
DCM parameter space. Even small deviations from stable A matrices cause
exponential blowup in the neural state ODE.
**How to avoid:**
1. Initialize encoder output layers with near-zero weights so initial outputs
   are near the prior mean (stable A from parameterize_A).
2. Use KL annealing (warmup) to keep early training close to the prior.
3. Reuse the existing NaN guard pattern: zero-fill divergent trajectories and
   detach to produce a large but finite ELBO penalty.
4. Clip encoder output to a reasonable range (e.g., A_free elements in [-3, 3]
   which maps to connectivity values within 3 sigma of the prior).
**Warning signs:** Training loss is NaN or oscillates wildly in first 50 steps.

### Pitfall 2: Posterior Collapse (KL Vanishing)
**What goes wrong:** The encoder learns to output near-constant parameters
regardless of input, and the KL term goes to zero. The decoder produces the
same trajectory for all inputs.
**Why it happens:** When the decoder is powerful (accurate ODE), it can
reconstruct well even with uninformative latent codes. The ELBO optimization
then collapses the posterior to the prior.
**How to avoid:**
1. Use cyclical KL annealing (Hadjeres et al. 2020) -- cycle beta between 0
   and 1 multiple times during training.
2. Alternatively, use a fixed beta < 1 (beta-VAE) to allow more informative
   latent codes at the cost of slightly worse KL.
3. Monitor the KL term separately during training; it should be non-trivial
   (not near zero).
**Warning signs:** KL divergence term drops to near zero; encoder outputs
have very low variance (z_scale << 0.1).

### Pitfall 3: Initial Condition Sensitivity
**What goes wrong:** Small changes in the initial condition x0 produce
wildly different trajectories, making the loss landscape jagged and training
unstable.
**Why it happens:** Chaotic or near-chaotic dynamics amplify initial
condition perturbations exponentially.
**How to avoid:**
1. Use stable A matrices (parameterize_A ensures negative diagonal, but
   off-diagonal elements can still cause instability).
2. Keep integration time short for initial validation (2-5 seconds, not 100s).
3. Use the adjoint method which can handle sensitivity better than direct
   backprop.
4. Consider learning x0 as a deterministic function of the first few time
   points rather than treating it as a free latent variable.
**Warning signs:** Gradient norms spike orders of magnitude between steps.

### Pitfall 4: Prior-Posterior Mismatch in Packed Latent Space
**What goes wrong:** The standard normal prior in standardized space does
not match the true DCM parameter distribution, causing poor training.
**Why it happens:** The standardization (mean/std from simulated data) may
not capture the true distribution of parameters in real M/EEG data. The
prior in packed space is N(0, I), but DCM parameters have structured
correlations and non-Gaussian marginals (e.g., noise_prec is positive).
**How to avoid:**
1. Fit standardization on a large diverse set of simulated DCM parameters
   spanning the expected range of real-data parameters.
2. For positive parameters (noise_prec), use log-space in the packed vector
   (existing contract in TaskDCMPacker).
3. Consider a flow-based prior (Zuko NSF) instead of standard normal for
   richer prior expressiveness -- but only if the simpler approach fails.
**Warning signs:** Reconstruction is good but posterior samples are
physiologically implausible.

### Pitfall 5: Adjoint Method Requires Smooth ODE RHS
**What goes wrong:** Using `odeint_adjoint` with a non-smooth right-hand
side (e.g., piecewise-constant stimulus) causes incorrect gradients or
solver failures.
**Why it happens:** The adjoint method solves a backward ODE that requires
differentiability of the forward ODE RHS with respect to state and time.
Piecewise-constant inputs create discontinuities.
**How to avoid:**
1. Use `odeint` (direct backprop) instead of `odeint_adjoint` when the
   integration is short enough to fit in memory.
2. For `odeint_adjoint`, ensure the stimulus interpolation is smooth (e.g.,
   use linear interpolation or softened step functions).
3. The existing `PiecewiseConstantInput` is technically non-smooth at
   transitions, but in practice torchdiffeq handles this because the ODE
   RHS is smooth in the state variable even if the input jumps.
**Warning signs:** Adjoint solver reports "step size too small" errors.

### Pitfall 6: Confusing Amortized VAE with Per-Subject SVI
**What goes wrong:** Treating the hybrid VAE-DCM as a per-subject model
(running SVI for each subject) defeats the purpose of amortization.
**Why it happens:** Conceptual confusion between Phase 20's per-subject SVI
approach and Phase 25's amortized approach.
**How to avoid:**
1. **Training phase:** Train on a dataset of many subjects/simulations.
   SVI runs over the full dataset, training the encoder weights.
2. **Inference phase:** For a new subject, run a single forward pass
   through the trained encoder. No SVI needed.
3. The Pyro SVI loop in training optimizes the encoder (guide) parameters,
   NOT per-subject DCM parameters.
**Warning signs:** Training takes the same time per subject as Phase 20 SVI.

## Code Examples

### Encoder Network for Source-Localized ROI Timeseries
```python
# Adapted from existing BoldSummaryNet pattern
class DCMEncoderNet(nn.Module):
    """Temporal CNN encoder for M/EEG ROI timeseries.

    Maps (T, N) timeseries to (z_loc, z_scale) for the packed
    DCM parameter vector. Architecture follows BoldSummaryNet
    with an additional output head for z_scale.

    Parameters
    ----------
    n_regions : int
        Number of ROIs (N).
    n_features : int
        Dimension of packed parameter vector (from packer.n_features).
    embed_dim : int
        Intermediate embedding dimension before final projection.
    """

    def __init__(self, n_regions: int, n_features: int,
                 embed_dim: int = 256) -> None:
        super().__init__()
        # Reuse BoldSummaryNet-style temporal CNN backbone
        self.conv1 = nn.Conv1d(n_regions, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, embed_dim, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(embed_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.act = nn.ELU()  # Smooth activation (not ReLU)

        # Two output heads: location and scale
        self.fc_loc = nn.Linear(embed_dim, n_features)
        self.fc_scale = nn.Linear(embed_dim, n_features)

        # Initialize output layers near zero for stability
        nn.init.zeros_(self.fc_loc.weight)
        nn.init.zeros_(self.fc_loc.bias)
        nn.init.constant_(self.fc_scale.bias, -2.0)  # softplus(-2) ~ 0.13

        self.double()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode timeseries to latent distribution parameters.

        Parameters
        ----------
        x : torch.Tensor
            Input timeseries, shape (T, N) or (batch, T, N).

        Returns
        -------
        z_loc : torch.Tensor
            Mean of approximate posterior, shape (n_features,).
        z_scale : torch.Tensor
            Std of approximate posterior, shape (n_features,).
        """
        unbatched = x.dim() == 2
        if unbatched:
            x = x.unsqueeze(0)
        x = x.transpose(1, 2)  # (batch, N, T)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)  # (batch, embed_dim)
        z_loc = self.fc_loc(x)
        z_scale = F.softplus(self.fc_scale(x)) + 1e-5
        if unbatched:
            z_loc = z_loc.squeeze(0)
            z_scale = z_scale.squeeze(0)
        return z_loc, z_scale
```

### LatentCircuitDCMPacker with Initial Conditions
```python
class LatentCircuitDCMPacker:
    """Pack/unpack latent-circuit DCM params including initial conditions.

    Extends the TaskDCMPacker pattern with:
    - A_free: (N, N) connectivity
    - C: (N, M) driving input weights
    - x0: (N,) initial conditions
    - noise_prec: scalar (log-space)

    Packed vector: [A_free.flat, C.flat, x0, log(noise_prec)]
    n_features = N*N + N*M + N + 1
    """

    def __init__(self, n_regions, n_inputs, a_mask, c_mask):
        self.n_regions = n_regions
        self.n_inputs = n_inputs
        self.a_mask = a_mask
        self.c_mask = c_mask
        self.n_features = (
            n_regions * n_regions  # A_free
            + n_regions * n_inputs  # C
            + n_regions            # x0
            + 1                    # log(noise_prec)
        )
        self.mean_ = None
        self.std_ = None
```

### Training Loop with KL Annealing
```python
def train_hybrid_vae_dcm(
    model_fn, guide, optimizer, train_data, n_epochs, warmup_epochs,
):
    """Train hybrid VAE-DCM with KL annealing.

    Parameters
    ----------
    model_fn : callable
        Pyro model function (DCM decoder).
    guide : HybridVAEDCMGuide
        Encoder (recognition network).
    optimizer : pyro.optim
        Pyro optimizer (Adam or ClippedAdam).
    train_data : list of dict
        Training examples with 'trajectories', 'stimulus', etc.
    n_epochs : int
        Total training epochs.
    warmup_epochs : int
        Number of epochs for KL warmup (beta: 0 -> 1).
    """
    svi = SVI(model_fn, guide, optimizer, loss=Trace_ELBO())
    losses = []
    for epoch in range(n_epochs):
        beta = min(1.0, epoch / max(1, warmup_epochs))
        epoch_loss = 0.0
        for batch in train_data:
            # Scale KL via poutine.scale on the model
            with pyro.poutine.scale(scale=beta):
                loss = svi.step(**batch)
            epoch_loss += loss
        losses.append(epoch_loss / len(train_data))
    return losses
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Per-subject VL/EM (SPM12 DCM) | Amortized encoder (TREND, VBI) | 2023-2025 | Single forward pass vs minutes of optimization per subject |
| Neural ODE VAE (Rubanova 2019) | Physics-informed ODE VAE (PI-VAE, Phi-DVAE) | 2022-2024 | Known equations as decoder, not learned dynamics |
| Separate encoder + SBI (VBI) | Joint encoder-decoder training (TREND) | 2024 | End-to-end differentiable; no simulation budget |
| Mean-field VL posterior (SPM12) | Normalizing flow posterior (Zuko NSF) | 2020+ | Captures posterior correlations and non-Gaussianity |
| Standard ReLU encoder | Smooth activations (ELU, Softplus, GELU) | 2020+ | Required for adjoint ODE backprop stability |

**Deprecated/outdated:**
- Direct backprop through long ODE solves: replaced by adjoint method for memory efficiency
- Fixed beta=1 in VAE ELBO: replaced by KL annealing / cyclical annealing for training stability

**Closely related published work:**
- **TREND (Imaging Neuroscience 2024):** Transformer encoder + DCM decoder for fMRI.
  Closest architectural precedent. Key differences from Phase 25: TREND uses
  P-DCM (two-state, hemodynamic), not bilinear; uses Transformer attention, not
  CNN; does not use Pyro or produce full posteriors (point estimates only).
- **VBI (eLife 2025):** Simulation-based inference for whole-brain models. Uses
  SBI (SNPE) rather than VAE; trains neural density estimator on simulated data
  pairs. Phase 26 (SBI for spectral DCM) is the VBI-style approach.
- **Phi-DVAE (JCP 2024):** Physics-informed dynamical VAE with ODE/PDE decoder.
  Most similar framework design. Uses extended Kalman filter for latent state
  estimation + VAE for unstructured data assimilation. Joint estimation of
  encoding, latent states, and ODE parameters.
- **Latent ODE (Rubanova et al. 2019, NeurIPS):** ODE-RNN encoder + Neural ODE
  decoder. Key difference: dynamics are learned (neural ODE), not known physics.
  Phase 25 uses known DCM equations.

## Open Questions

1. **Encoder architecture: CNN vs Transformer?**
   - What we know: BoldSummaryNet (1D-CNN with AdaptiveAvgPool) works for the
     existing amortized guides. TREND uses Transformer with dual attention
     (across-sample + across-node). TCFormer (2025) shows CNN-Transformer
     hybrids work well for EEG.
   - What's unclear: Whether the added complexity of a Transformer encoder
     provides meaningful benefit over a CNN for source-localized ROI timeseries
     (typically N=4-8 regions, T=100-1000 time points). CNN is simpler and proven.
   - Recommendation: Start with 1D-CNN (extend BoldSummaryNet). Add Transformer
     variant as an optional upgrade if CNN performance is insufficient.

2. **Should x0 (initial conditions) be in the latent space or deterministic?**
   - What we know: Rubanova et al. encode x0 as part of the latent space. The
     existing `latent_circuit_dcm_model` uses fixed zeros for x0. For M/EEG data,
     initial conditions are informative (first time point of the recording).
   - What's unclear: Whether treating x0 as a latent variable (with prior and
     posterior) helps or hurts. It adds N dimensions to the latent space.
   - Recommendation: Include x0 in the latent space for the full VAE formulation.
     For comparison, also support a deterministic x0 path where x0 is extracted
     from the first few time points of the observation.

3. **Should the bilinear B path be included from the start?**
   - What we know: The existing `latent_circuit_dcm_model` supports bilinear B.
     However, adding B_free_j to the packed latent increases dimensionality by
     J * N * N per modulator, and the v0.3.0 amortized_wrappers explicitly refuse
     bilinear (deferred to v0.3.1).
   - What's unclear: Whether the encoder can learn a useful mapping to a
     high-dimensional latent that includes B.
   - Recommendation: Start linear-only (A, C, x0, noise_prec). Add bilinear as
     a documented extension point. This matches the project's pattern of shipping
     linear first, adding bilinear incrementally.

4. **Training data strategy for amortization**
   - What we know: VBI and SNPE use simulation-based training (generate many
     parameter sets, simulate, train). TREND uses real fMRI data directly. The
     project has `simulate_latent_circuit` for synthetic data.
   - What's unclear: How many training examples are needed for good amortization.
     VBI uses 10k-100k simulations. Whether real M/EEG data (limited subjects)
     can supplement or replace synthetic training.
   - Recommendation: Phase 1 (synthetic validation) uses 10k+ simulated
     trajectories. Phase 2 (real data) uses a mix of simulated + real.

5. **Integration time and dt for the ODE decoder during training**
   - What we know: Phase 20's acceptance run timed out at 100s integration
     (20-04-D2: 100s/dt=0.01 = ~16s per SVI step). VAE training requires many
     more forward passes than single-subject SVI.
   - What's unclear: Whether shorter integration times (2-10s) are sufficient
     for M/EEG dynamics, or whether the decoder needs to handle full trial
     durations.
   - Recommendation: Use short integration times (2-5s) during training for
     speed. Validate on longer durations after training.

## Sources

### Primary (HIGH confidence)
- Pyro VAE tutorial (https://pyro.ai/examples/vae.html) -- model/guide pattern
- Pyro CVAE tutorial (https://pyro.ai/examples/cvae.html) -- conditional VAE pattern
- torchdiffeq FAQ (https://github.com/rtqichen/torchdiffeq/blob/master/FAQ.md) -- training tips, adjoint method
- Existing codebase: `amortized_wrappers.py`, `amortized_flow.py`, `summary_networks.py`, `parameter_packing.py`, `coupled_system.py`, `latent_circuit_dcm_model.py`, `latent_observation.py` -- verified patterns

### Secondary (MEDIUM confidence)
- TREND: Transformer-aided dynamic causal model (Imaging Neuroscience 2024, https://direct.mit.edu/imag/article/doi/10.1162/imag_a_00290/) -- encoder-DCM-decoder architecture
- VBI: Virtual Brain Inference (eLife 2025, https://elifesciences.org/articles/106194) -- SBI for brain models
- Phi-DVAE (JCP 2024, https://arxiv.org/abs/2209.15609) -- physics-informed dynamical VAE
- Latent ODE (Rubanova et al. 2019, https://arxiv.org/abs/1907.03907) -- ODE-RNN encoder + latent ODE
- ODE2VAE (Yildiz et al. 2019, https://github.com/cagatayyildiz/ODE2VAE) -- second-order ODE VAE
- Cyclical KL annealing (Hadjeres et al. 2020, https://arxiv.org/abs/1903.10145) -- training stability

### Tertiary (LOW confidence)
- PI-VAE (CMAME 2022, https://arxiv.org/abs/2203.11363) -- physics-informed VAE for SDEs
- NeurIPS 2021 Physics-Integrated VAEs (https://proceedings.neurips.cc/paper/2021/file/7ca57a9f85a19a6e4b9a248c1daca185-Paper.pdf)
- CommsVAE (https://arxiv.org/abs/2210.03667) -- coupled sequential VAEs for brain communication

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries already in project; no new dependencies
- Architecture: MEDIUM -- pattern is well-established (Latent ODE, PI-VAE, TREND) but
  specific DCM-as-decoder combination is novel; no published Pyro+DCM VAE reference
- Pitfalls: MEDIUM -- drawn from general Neural ODE VAE literature + project experience
  (Phase 16.1 RECOV-04, Phase 20 training); M/EEG-specific pitfalls need validation
- Code examples: MEDIUM -- adapted from existing codebase patterns but untested

**Research date:** 2026-05-26
**Valid until:** 60 days (stable domain; libraries not changing rapidly)
