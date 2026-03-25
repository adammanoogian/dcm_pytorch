---
type: "research"
scope: "architecture"
updated: "2026-03-24"
---

# Architecture

## Module Interfaces

The framework is built around three swappable interfaces. Each DCM variant
composes one connectivity prior + one observation model + one inference guide.

### ConnectivityPrior Protocol

```python
class ConnectivityPrior(Protocol):
    def sample(self, n_regions: int, mask: torch.Tensor | None) -> torch.Tensor:
        """Sample A matrix from prior. Returns (n_regions, n_regions) tensor."""
        ...

    def log_prob(self, A: torch.Tensor) -> torch.Tensor:
        """Log probability of A under this prior."""
        ...
```

**Implementations (v0.1):**
- `StaticA`: N(0, σ²I) with optional binary mask. Simplest prior.

**Future (v0.2):**
- `GPPriorA`: Gaussian process over time → A(t)
- `SwitchingA`: HMM with discrete A states
- `RNNPriorA`: RNN outputting A at each timestep

### ObservationModel Protocol

```python
class ObservationModel(Protocol):
    def forward(self, neural_states: torch.Tensor, params: dict) -> torch.Tensor:
        """Map neural states to observations (BOLD, CSD, or freq-domain)."""
        ...

    def log_likelihood(self, observed: torch.Tensor, predicted: torch.Tensor,
                       noise_params: dict) -> torch.Tensor:
        """Log likelihood of observed data given predicted."""
        ...
```

**Implementations:**
- `BalloonBOLD`: Balloon-Windkessel → BOLD (task DCM)
- `SpectralCSD`: Transfer function → predicted CSD (spectral DCM)
- `FreqDomainLinear`: Frequency-domain regression (rDCM)

### InferenceGuide Protocol

```python
class InferenceGuide(Protocol):
    def __call__(self, observed_data: torch.Tensor) -> dict[str, Distribution]:
        """Return variational distributions for all latent variables."""
        ...
```

**Implementations:**
- `MeanFieldGaussian`: Independent Gaussian per parameter (Laplace baseline)
- `NormalizingFlowGuide`: Zuko MAF/NSF conditioned on data summary

## Data Flow

```
Input data (BOLD / CSD / freq-domain)
    │
    ├── Generative Model (Pyro model)
    │   ├── Sample A ~ ConnectivityPrior
    │   ├── Sample hemodynamic/noise params ~ Priors
    │   ├── Compute predicted data via ObservationModel.forward()
    │   └── Score: ObservationModel.log_likelihood(observed, predicted)
    │
    └── Inference Guide (Pyro guide)
        ├── MeanFieldGaussian: q(A) = N(μ_A, diag(σ²_A))
        └── NormalizingFlowGuide: q(A|data) = Flow(data → A)
```

## Pyro Model Registration Pattern

Each DCM variant follows this pattern:

```python
def task_dcm_model(observed_bold, input_u, n_regions, mask=None):
    # --- Priors ---
    A = pyro.sample("A", dist.Normal(0, 1).expand([n, n]).to_event(2))
    if mask is not None:
        A = A * mask  # structural masking

    C = pyro.sample("C", dist.Normal(0, 1).expand([n, m]).to_event(2))
    log_kappa = pyro.sample("log_kappa", dist.Normal(log(0.65), 0.5).expand([n]))
    # ... other hemodynamic params

    # --- Forward model ---
    kappa = torch.exp(log_kappa)
    predicted_bold = integrate_dcm(A, C, input_u, kappa, ...)

    # --- Likelihood ---
    with pyro.plate("time", T):
        pyro.sample("obs", dist.Normal(predicted_bold, sigma), obs=observed_bold)
```

## Tensor Shape Conventions

| Tensor | Shape | Description |
|--------|-------|-------------|
| A | (N, N) | Effective connectivity matrix |
| C | (N, M) | Driving input weights |
| B_j | (N, N) | Modulatory input j |
| u | (T, M) | Experimental inputs over time |
| bold | (T, N) | BOLD time series |
| csd | (F, N, N) | Cross-spectral density (complex) |
| hemo_state | (T, N, 4) | (s, f, v, q) per region |

N = number of regions, M = number of inputs, T = time points, F = frequency bins
