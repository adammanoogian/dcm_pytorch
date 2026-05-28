# Phase 26: SBI for Spectral DCM - Research

**Researched:** 2026-05-26
**Domain:** Simulation-based inference (SBI) for spectral DCM posterior estimation
**Confidence:** MEDIUM-HIGH

## Summary

This phase implements simulation-based inference (SBI) as an alternative to SVI for
spectral DCM. The approach trains a neural density estimator (NPE) on simulated
cross-spectral densities from the existing spectral DCM forward model
(`spectral_dcm_forward`), producing amortized posteriors that perform inference in a
single forward pass (<1s per subject) rather than iterative SVI (~minutes per subject).

The standard tool for this is the `sbi` Python package (v0.26.1, April 2026), which
provides a PyTorch-native implementation of Neural Posterior Estimation (NPE), Neural
Likelihood Estimation (NLE), and Neural Ratio Estimation (NRE). The key references are
VBI (eLife 2025) for whole-brain SBI methodology and Bernardo et al. (2024, Comms
Physics) for SBI applied to spectral graph models. Both papers confirm that NPE/TSNPE
with learned summary statistics is the standard approach for spectral neuroimaging
models.

The project already has a Pyro-based amortized inference pipeline (Phase 7/8:
`AmortizedFlowGuide` + `amortized_spectral_dcm_model`). Phase 26 uses the standalone
`sbi` package instead of Pyro's SVI machinery because SBI does not require a
likelihood function -- it only needs a simulator mapping parameters to observations.
This decouples the inference network from the generative model, making it simpler,
more flexible, and directly comparable to the VBI/spectral graph model literature.

**Primary recommendation:** Use `sbi` v0.26.1 with NPE and an embedding network (MLP
or 1D CNN) to compress decomposed CSD vectors into summary statistics, trained on
50,000-100,000 simulations from `spectral_dcm_forward`. Validate with SBC (200 runs)
before applying to real data.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| sbi | 0.26.1 | Neural density estimation (NPE/NLE/NRE) | De facto standard SBI package; PyTorch native; built-in SBC diagnostics; used by VBI, spectral graph model papers |
| torch | 2.x | Tensor computation, autograd | Already in project stack |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| arviz | >=0.17 | Posterior analysis, convergence diagnostics | Analyzing SBI posteriors alongside SVI posteriors |
| joblib | >=1.3 | Parallel simulation generation | Parallelizing simulator calls for training data |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| sbi NPE | Pyro AmortizedFlowGuide (existing) | Existing code works but couples guide to Pyro model; sbi is simpler for pure SBI and matches literature |
| sbi NPE | sbijax (JAX-based) | JAX performance but project is PyTorch; would require JAX dependency |
| sbi NPE | BayesFlow | Similar capability but sbi has larger community, better docs, built-in SBC |
| NPE (amortized) | SNPE (sequential) | Sequential is more simulation-efficient but not amortized; use NPE for our use case (many subjects) |
| NPE | NLE | NLE requires MCMC sampling step after training; NPE gives direct posterior samples |
| NPE | TSNPE (truncated sequential) | TSNPE has better calibration at parameter bounds per Bernardo et al.; consider if NPE shows boundary leakage |

**Installation:**
```bash
pip install sbi>=0.26.0
```

## Architecture Patterns

### Recommended Project Structure
```
src/pyro_dcm/
    inference/
        __init__.py
        sbi_spectral.py          # SBI wrapper: simulator, training, posterior
        sbi_embedding.py         # Custom embedding network for CSD
        sbi_diagnostics.py       # SBC and coverage analysis
    simulators/
        spectral_simulator.py    # (existing) -- used as-is for SBI simulator
```

### Pattern 1: SBI Simulator Wrapper
**What:** Wrap `spectral_dcm_forward` as an sbi-compatible simulator function
**When to use:** Always -- this is the core interface between our forward model and sbi

The sbi simulator must be a callable that takes a 1D parameter tensor and returns a
1D observation tensor. Our spectral DCM forward model takes structured parameters
(A, noise_a, noise_b, noise_c) and returns complex CSD (F, N, N). The wrapper
handles parameter packing/unpacking and CSD-to-real decomposition.

**Example:**
```python
# Source: sbi documentation + project spectral_dcm_forward API
import torch
from sbi.inference import NPE
from sbi.utils import BoxUniform

from pyro_dcm.forward_models.neural_state import parameterize_A
from pyro_dcm.forward_models.spectral_transfer import spectral_dcm_forward
from pyro_dcm.models.spectral_dcm_model import decompose_csd_for_likelihood


def make_spectral_dcm_simulator(
    n_regions: int,
    a_mask: torch.Tensor,
    freqs: torch.Tensor,
) -> callable:
    """Create sbi-compatible simulator from spectral DCM forward model.

    Parameters
    ----------
    n_regions : int
        Number of brain regions N.
    a_mask : torch.Tensor
        Binary structural mask for A, shape (N, N).
    freqs : torch.Tensor
        Frequency vector in Hz, shape (F,).

    Returns
    -------
    callable
        Simulator function: theta (D,) -> x (2*F*N*N,).
    """
    N = n_regions

    def simulator(theta: torch.Tensor) -> torch.Tensor:
        # Unpack theta into structured parameters
        # Layout: [A_free (N*N), noise_a (2*N), noise_b (2), noise_c (2*N)]
        idx = 0
        A_free = theta[idx : idx + N * N].reshape(N, N) * a_mask
        idx += N * N
        noise_a = theta[idx : idx + 2 * N].reshape(2, N)
        idx += 2 * N
        noise_b = theta[idx : idx + 2].reshape(2, 1)
        idx += 2
        noise_c = theta[idx : idx + 2 * N].reshape(2, N)

        A = parameterize_A(A_free)
        csd = spectral_dcm_forward(A, freqs, noise_a, noise_b, noise_c)
        return decompose_csd_for_likelihood(csd)

    return simulator
```

### Pattern 2: Prior Definition
**What:** Define sbi-compatible prior matching SPM12 spectral DCM conventions
**When to use:** Always -- prior must match the generative model

```python
# Source: spectral_dcm_model.py prior conventions
import torch
from torch.distributions import Independent, Normal

def make_spectral_dcm_prior(
    n_regions: int,
) -> Independent:
    """Create sbi-compatible prior for spectral DCM parameters.

    Matches SPM12 spm_dcm_fmri_priors.m: N(0, 1/64) for all params.

    Parameters
    ----------
    n_regions : int
        Number of brain regions N.

    Returns
    -------
    Independent
        Prior distribution over packed parameter vector.
    """
    N = n_regions
    # Parameter count: N*N (A) + 2*N (noise_a) + 2 (noise_b) + 2*N (noise_c)
    n_params = N * N + 4 * N + 2
    prior_std = (1.0 / 64.0) ** 0.5  # SPM12 convention

    return Independent(
        Normal(
            torch.zeros(n_params, dtype=torch.float64),
            prior_std * torch.ones(n_params, dtype=torch.float64),
        ),
        reinterpreted_batch_ndims=1,
    )
```

### Pattern 3: Training Pipeline
**What:** Full NPE training workflow with sbi
**When to use:** One-time training before amortized inference

```python
# Source: sbi getting started tutorial + project conventions
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn

# 1. Create simulator and prior
simulator = make_spectral_dcm_simulator(N, a_mask, freqs)
prior = make_spectral_dcm_prior(N)

# 2. Generate training simulations
n_sims = 50_000
theta = prior.sample((n_sims,))
x = torch.stack([simulator(t) for t in theta])  # or parallel

# 3. Configure NPE with embedding network for high-dim observations
# For N=3, F=32: observation dim = 576; for N=10: obs dim = 6400
embedding_net = CSDEmbeddingNet(obs_dim=2 * F * N * N, embed_dim=64)
density_estimator = posterior_nn(
    model="zuko_nsf",
    embedding_net=embedding_net,
    hidden_features=128,
    num_transforms=5,
)
inference = NPE(prior=prior, density_estimator=density_estimator)

# 4. Train
inference.append_simulations(theta, x)
estimator = inference.train(training_batch_size=256, max_num_epochs=200)
posterior = inference.build_posterior(estimator)

# 5. Amortized inference (single forward pass per subject)
samples = posterior.sample((10_000,), x=x_observed)
```

### Pattern 4: Embedding Network for CSD
**What:** Custom neural network to compress high-dimensional CSD to summary statistics
**When to use:** Always for spectral DCM SBI -- raw CSD is too high-dimensional

```python
# Source: sbi embedding_net tutorial + VBI summary statistics approach
import torch.nn as nn


class CSDEmbeddingNet(nn.Module):
    """Embedding network for cross-spectral density observations.

    Compresses decomposed CSD vector (2*F*N*N real values) into a
    fixed-dimensional embedding for the NPE density estimator.
    """

    def __init__(
        self,
        obs_dim: int,
        embed_dim: int = 64,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
```

### Pattern 5: SBC Validation
**What:** Simulation-based calibration to verify posterior coverage
**When to use:** After training, before applying to real data

```python
# Source: sbi SBC tutorial
from sbi.diagnostics import run_sbc
from sbi.analysis.plot import sbc_rank_plot

# Generate SBC test data
n_sbc = 200
sbc_theta = prior.sample((n_sbc,))
sbc_x = torch.stack([simulator(t) for t in sbc_theta])

# Run SBC
ranks, dap_samples = run_sbc(
    sbc_theta,
    sbc_x,
    posterior,
    num_posterior_samples=1000,
)

# Visualize
fig, ax = sbc_rank_plot(ranks, num_posterior_samples=1000, num_bins=20)
```

### Anti-Patterns to Avoid
- **Using raw complex CSD tensors:** sbi expects real-valued 1D tensors. Always
  decompose complex CSD via `decompose_csd_for_likelihood` before passing to sbi.
- **Using the Pyro amortized pipeline for SBI:** The existing
  `AmortizedFlowGuide` + `amortized_spectral_dcm_model` is a Pyro SVI pipeline,
  not SBI. SBI does not use ELBO or require a likelihood function -- it only needs
  a simulator. Do not mix the two approaches.
- **Training on too few simulations:** VBI found 50k-500k simulations needed
  depending on parameter dimensionality. For spectral DCM with N=3 (24 params),
  start with 50k minimum.
- **Skipping the embedding network:** Without an embedding network, NPE must learn
  from raw 576+ dimensional observations. Always use an embedding network.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Neural density estimation | Custom normalizing flow training loop | `sbi.inference.NPE` | Handles training, validation, early stopping automatically |
| Posterior sampling | Custom MCMC on learned density | `posterior.sample()` from sbi NPE | NPE gives direct samples; no MCMC needed |
| Simulation-based calibration | Custom rank statistics | `sbi.diagnostics.run_sbc` | Handles rank computation, DAP samples, visualization |
| Parameter packing/unpacking | New packer class | Reuse layout from `SpectralDCMPacker` | Same parameter structure, just different interface |
| Embedding network | Custom learned summary stats | sbi's built-in `embedding_net` parameter | Jointly trained with density estimator; well-tested |
| Prior distribution | Custom distribution class | `torch.distributions.Independent(Normal(...))` | sbi accepts standard PyTorch distributions directly |

**Key insight:** The `sbi` package handles the entire density estimation pipeline
(training loop, early stopping, posterior construction, diagnostics). The only
custom code needed is the simulator wrapper and optionally the embedding network.

## Common Pitfalls

### Pitfall 1: Complex-Valued CSD Incompatibility
**What goes wrong:** sbi expects real-valued 1D tensors as observations. Passing
complex CSD (F, N, N) directly fails silently or produces nonsense.
**Why it happens:** The spectral DCM forward model returns complex128 tensors.
**How to avoid:** Always decompose via `decompose_csd_for_likelihood` which stacks
real and imaginary parts into a single float64 vector of shape (2*F*N*N,).
**Warning signs:** NaN in training loss, or posterior samples that are uniform/prior-like.

### Pitfall 2: Simulation Budget Too Small
**What goes wrong:** NPE produces poorly calibrated posteriors (SBC fails).
**Why it happens:** Insufficient training data for the parameter dimensionality.
VBI needed 2k simulations for 2 params but 500k for 89 params.
**How to avoid:** Scale simulation budget with parameter count. For spectral DCM:
- N=3 (24 params): 50,000 simulations minimum
- N=5 (48 params): 100,000 simulations minimum
- N=10 (143 params): 200,000-500,000 simulations
**Warning signs:** SBC rank histograms showing U-shape (posterior too narrow) or
bell shape (posterior too wide).

### Pitfall 3: Boundary Leakage with NPE
**What goes wrong:** NPE posterior probability leaks outside the support of the prior,
especially at parameter bounds.
**Why it happens:** Normalizing flows have infinite support; the prior bounds are not
enforced in the learned posterior.
**How to avoid:** Use `posterior.set_default_x(x_obs)` and consider TSNPE if
boundary leakage is severe (per Bernardo et al. 2024 who found TSNPE superior to
NPE for bounded parameters). Alternatively, use Normal priors (unbounded) rather
than BoxUniform priors -- our spectral DCM uses Normal priors, so this is less of
a concern.
**Warning signs:** Posterior samples outside prior support; SBC showing skewed ranks.

### Pitfall 4: Eigenvalue Instability in Simulations
**What goes wrong:** Some parameter samples from the prior produce unstable A matrices
(positive eigenvalues), causing `spectral_dcm_forward` to produce NaN or infinite
CSD values.
**Why it happens:** `parameterize_A` ensures negative diagonal but does not guarantee
all eigenvalues are negative for off-diagonal entries drawn from the prior.
**How to avoid:** Add eigenvalue stability check in the simulator wrapper. If
`spectral_dcm_forward` returns NaN, return a flag value (e.g., zeros) and
potentially reject the simulation. Alternatively, use `compute_transfer_function`'s
built-in eigenvalue clamping (which clamps real parts to max(-1/32)).
**Warning signs:** NaN values in simulated CSD; training loss diverging.

### Pitfall 5: Amortization Gap
**What goes wrong:** SBI posterior is systematically worse than per-subject SVI,
especially for subjects with unusual data.
**Why it happens:** The neural density estimator learns an average mapping across all
training data; it cannot adapt to individual subjects the way SVI does.
**How to avoid:** Quantify the amortization gap by comparing SBI posteriors to SVI
posteriors on held-out synthetic subjects. VBI found amortization gap is acceptable
when training data covers the prior well.
**Warning signs:** SBI posterior means consistently further from ground truth than
SVI posterior means; wider credible intervals.

### Pitfall 6: Simulation Speed Bottleneck
**What goes wrong:** Generating 50k-500k simulations takes too long on CPU.
**Why it happens:** Each `spectral_dcm_forward` call involves eigendecomposition of
A (O(N^3)) and matrix operations at each frequency.
**How to avoid:** Time a single simulation first. For N=3, expect ~1ms per call
(~50s for 50k). For N=10, expect ~5-10ms per call (~5-10 min for 100k). This is
fast enough on CPU. For larger N, batch simulations on GPU or parallelize with
joblib.
**Warning signs:** Training data generation taking >1 hour for small N.

### Pitfall 7: Forgetting to Match Prior Between SBI and SVI
**What goes wrong:** SBI and SVI posteriors are not comparable because priors differ.
**Why it happens:** The sbi prior is defined independently from the Pyro model prior.
**How to avoid:** Extract prior parameters from `spectral_dcm_model` and use the
exact same values when constructing the sbi prior. Both use N(0, 1/64) per SPM12.
**Warning signs:** SBI posterior systematically shifted relative to SVI posterior.

## Code Examples

Verified patterns from official sources:

### Complete NPE Workflow (sbi v0.26.1)
```python
# Source: sbi getting started + embedding net tutorials
import torch
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn

# Prior: N(0, 1/64) for all D parameters
from torch.distributions import Independent, Normal

D = 24  # N=3 spectral DCM: 9 + 6 + 2 + 6 + 1
prior_std = (1.0 / 64.0) ** 0.5
prior = Independent(
    Normal(torch.zeros(D), prior_std * torch.ones(D)),
    reinterpreted_batch_ndims=1,
)

# Simulate training data
theta = prior.sample((50_000,))
x = torch.stack([simulator(t) for t in theta])

# NPE with Neural Spline Flow + embedding
density_estimator = posterior_nn(
    model="zuko_nsf",
    embedding_net=CSDEmbeddingNet(obs_dim=576, embed_dim=64),
    hidden_features=128,
    num_transforms=5,
)
inference = NPE(prior=prior, density_estimator=density_estimator)
inference.append_simulations(theta, x)
estimator = inference.train()
posterior = inference.build_posterior(estimator)
```

### SBC Validation
```python
# Source: sbi SBC how-to guide
from sbi.diagnostics import run_sbc
from sbi.analysis.plot import sbc_rank_plot

sbc_theta = prior.sample((200,))
sbc_x = torch.stack([simulator(t) for t in sbc_theta])
ranks, dap = run_sbc(sbc_theta, sbc_x, posterior, num_posterior_samples=1000)
fig, ax = sbc_rank_plot(ranks, num_posterior_samples=1000)
```

### Comparison with SVI Posteriors
```python
# Source: project convention (spectral_dcm_model.py + models/__init__.py)
from pyro_dcm.models import spectral_dcm_model, create_guide, run_svi

# SVI posterior (per-subject, iterative)
guide = create_guide(spectral_dcm_model, guide_type="AutoNormal")
result = run_svi(
    spectral_dcm_model, guide,
    model_args=(observed_csd, freqs, a_mask),
    n_steps=500,
)

# SBI posterior (amortized, single forward pass)
x_obs = decompose_csd_for_likelihood(observed_csd)
sbi_samples = posterior.sample((10_000,), x=x_obs)

# Compare: posterior means, credible intervals, RMSE vs ground truth
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Per-subject SVI/VB (SPM12) | Amortized SBI via NPE | 2020-2025 | O(minutes) -> O(seconds) per subject inference |
| SNPE (sequential, per-observation) | NPE (amortized, reusable) | sbi v0.23.0 (2024) | Class renamed from SNPE to NPE; amortized is default |
| Manual summary statistics | Learned embedding networks | 2020+ | Embedding net jointly trained with density estimator |
| No calibration checking | SBC built into sbi | 2022+ | `sbi.diagnostics.run_sbc` provides automated validation |

**Deprecated/outdated:**
- `SNPE` class name: renamed to `NPE` in sbi v0.23.0. `SNPE` still works as alias.
- `delfi` package: predecessor to `sbi`; unmaintained.
- Manual feature engineering for SBI: learned embeddings now standard.

## Relationship to Existing Amortized Infrastructure

The project already has a Pyro-based amortized inference pipeline for spectral DCM:

| Component | Existing (Phase 7/8) | Phase 26 (SBI) |
|-----------|---------------------|----------------|
| Density estimator | Zuko NSF via `AmortizedFlowGuide` | sbi NPE (NSF or MAF) |
| Summary network | `CsdSummaryNet` in `summary_networks.py` | sbi embedding network |
| Parameter packing | `SpectralDCMPacker` | Flat 1D vector (simpler) |
| Training | Pyro SVI (ELBO minimization) | sbi `inference.train()` |
| Posterior sampling | `guide.sample_posterior()` | `posterior.sample()` |
| Requires likelihood? | Yes (Pyro model) | No (simulator only) |
| Calibration | Manual | `sbi.diagnostics.run_sbc` |

**Key architectural decision:** Phase 26 uses `sbi` as a standalone package rather
than extending the existing Pyro amortized pipeline. Reasons:
1. SBI is a fundamentally different paradigm (no likelihood needed)
2. Matches the VBI and spectral graph model literature directly
3. sbi provides built-in SBC diagnostics
4. Cleaner comparison between SVI and SBI approaches
5. The existing amortized pipeline remains available for Pyro-based amortized SVI

## Parameter Dimensionality Analysis

For spectral DCM with N regions:

| N | A_free | noise_a | noise_b | noise_c | Total params | CSD obs dim | Suggested sims |
|---|--------|---------|---------|---------|-------------|-------------|----------------|
| 3 | 9 | 6 | 2 | 6 | 23 | 576 | 50,000 |
| 5 | 25 | 10 | 2 | 10 | 47 | 3,200 | 100,000 |
| 10 | 100 | 20 | 2 | 20 | 142 | 6,400 | 200,000-500,000 |
| 20 | 400 | 40 | 2 | 40 | 482 | 25,600 | 500,000+ |

Note: `csd_noise_scale` from the Pyro model is NOT included in SBI parameters.
In SBI, the observation noise is implicitly handled by the density estimator
learning from noisy simulations. If explicit noise is desired, add Gaussian noise
to simulated CSD observations during training data generation.

## Open Questions

Things that couldn't be fully resolved:

1. **Optimal simulation budget for spectral DCM**
   - What we know: VBI used 2k-500k depending on dimensionality. Bernardo et al.
     used up to 1M for spectral graph model but with TSNPE (sequential).
   - What's unclear: Exact number needed for N=3 spectral DCM (24 params). The
     VBI 50k for Jansen-Rit (similar dimensionality) is a reasonable starting point.
   - Recommendation: Start with 50k for N=3; run SBC; increase if calibration fails.

2. **NSF vs MAF for density estimator**
   - What we know: VBI found MAF 2-4x faster to train than NSF; NSF slightly more
     accurate. sbi supports both via `model="zuko_nsf"` and `model="zuko_maf"`.
   - What's unclear: Which is better for spectral DCM's parameter structure.
   - Recommendation: Default to NSF (matches existing project convention with Zuko);
     benchmark MAF if training is too slow.

3. **Whether to add observation noise to simulated CSD**
   - What we know: The existing spectral DCM Pyro model adds a `csd_noise_scale`
     (HalfCauchy prior) for model-data mismatch. Pure SBI simulators can be
     deterministic or stochastic.
   - What's unclear: Whether adding Gaussian noise to simulated CSD improves SBI
     robustness or just adds unnecessary variance.
   - Recommendation: Start with deterministic simulator (no added noise). If SBI
     posteriors are overconfident on real data, add noise to simulations and retrain.

4. **Cluster routing for simulation generation**
   - What we know: 50k simulations at ~1ms each = ~50s (laptop-safe for N=3). But
     200k+ simulations or N>=10 may exceed 3-minute threshold.
   - What's unclear: Exact timing for larger N values.
   - Recommendation: Time single simulation; if total > 3 min, route to M3.
     NPE training itself (GPU-accelerated) is likely laptop-safe.

5. **Float64 vs Float32 for SBI training**
   - What we know: The project uses float64 everywhere (SPM12 convention). sbi
     works with float32 by default.
   - What's unclear: Whether float64 is necessary for SBI training or if float32
     suffices (as it does for most neural network training).
   - Recommendation: Generate simulations in float64 (matching forward model), but
     allow sbi training in float32 if it significantly speeds up training. Cast
     back to float64 for posterior analysis.

## Sources

### Primary (HIGH confidence)
- sbi GitHub (https://github.com/sbi-dev/sbi) - v0.26.1, API, supported methods
- sbi documentation (https://sbi-dev.github.io/sbi/v0.23.3/) - tutorials, embedding nets, SBC
- sbi PyPI (https://pypi.org/project/sbi/) - version 0.26.1, Python >=3.10
- Project source code: `spectral_dcm_model.py`, `spectral_simulator.py`,
  `spectral_transfer.py`, `amortized_flow.py`, `amortized_wrappers.py` - existing API

### Secondary (MEDIUM confidence)
- VBI (eLife 2025, PMC12700528) - SBI methodology for whole-brain models, MAF/NSF
  architecture, simulation budgets (2k-500k), amortized inference speed
- Bernardo et al. (2024, Comms Physics) - SBI for spectral graph model, TSNPE vs
  NPE calibration, spectral summary statistics, SBC validation
- sbi reloaded (arXiv 2411.17337) - updated toolkit features, diagnostics

### Tertiary (LOW confidence)
- Simulation budget scaling rules (extrapolated from VBI's reports; not directly
  validated for spectral DCM's parameter structure)
- Simulation timing estimates (~1ms per spectral_dcm_forward for N=3; based on
  forward model complexity, not benchmarked)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - sbi v0.26.1 is the established SBI package; confirmed via
  PyPI and GitHub; PyTorch-native; matches project stack
- Architecture: MEDIUM-HIGH - Patterns verified from sbi tutorials and VBI/spectral
  graph model papers; adapted to project's spectral DCM API
- Pitfalls: MEDIUM - Drawn from VBI paper (simulation budget, summary statistics),
  Bernardo et al. (boundary leakage, TSNPE), and project knowledge (eigenvalue
  stability, float64 convention)
- Simulation budget: MEDIUM - Scaling from VBI; not directly benchmarked for
  spectral DCM

**Research date:** 2026-05-26
**Valid until:** 2026-06-26 (30 days; sbi is a stable, well-maintained package)
