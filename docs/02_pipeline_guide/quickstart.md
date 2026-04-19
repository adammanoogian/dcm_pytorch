# Quickstart: Pyro-DCM in 5 Minutes

This tutorial walks you through the full Pyro-DCM workflow: simulate synthetic
fMRI data, run Bayesian inference with SVI, inspect posterior distributions,
and compare competing connectivity models. You will touch all three DCM variants
(task, spectral, regression) and see that the same API pattern applies to each.

**Prerequisites:** You know what DCM is (effective connectivity, A matrix,
hemodynamic model). You have Python 3.10+ installed.

---

## Installation

```bash
pip install pyro-dcm
```

For development with benchmark and plotting tools:

```bash
pip install -e ".[benchmark]"
```

---

## Step 1: Simulate Task-DCM Data

Generate synthetic BOLD time series from a known connectivity matrix.

```python
import torch
from pyro_dcm.simulators.task_simulator import (
    simulate_task_dcm,
    make_random_stable_A,
    make_block_stimulus,
)

# 3-region network with 50% connection density
A_true = make_random_stable_A(n_regions=3, density=0.5, seed=42)

# Driving input to region 0 only
C = torch.tensor([[0.25], [0.0], [0.0]], dtype=torch.float64)

# Block design: 6 blocks of 30s ON / 30s OFF = 360s total
stimulus = make_block_stimulus(n_blocks=6, block_duration=30.0, rest_duration=30.0)

# Run the full forward model (neural -> hemodynamic -> BOLD -> noise)
result = simulate_task_dcm(A_true, C, stimulus, duration=360.0, SNR=5.0, seed=42)

print(f"BOLD shape: {result['bold'].shape}")     # (180, 3) at TR=2.0
print(f"True A:\n{A_true}")
```

The simulator runs the neural state equation (dx/dt = Ax + Cu), integrates the
Balloon-Windkessel hemodynamic ODEs, computes the BOLD signal, downsamples to
TR resolution, and adds Gaussian noise at the requested SNR.

---

## Step 2: Run SVI Inference

Fit the generative model to the simulated BOLD data using stochastic
variational inference.

```python
import pyro
from pyro_dcm.models.task_dcm_model import task_dcm_model
from pyro_dcm.models.guides import create_guide, run_svi

# Prepare model inputs
observed_bold = result["bold"]
stim = result["stimulus"]
N = 3

# Structural masks: which connections exist (1=present, 0=absent)
a_mask = torch.ones(N, N, dtype=torch.float64)   # fully connected
c_mask = torch.tensor([[1.0], [0.0], [0.0]], dtype=torch.float64)

# Fine time grid and TR for ODE integration
t_eval = torch.arange(0, 360.0, 0.5, dtype=torch.float64)
TR = 2.0

# Create mean-field Gaussian guide and run SVI
model_args = (observed_bold, stim, a_mask, c_mask, t_eval, TR)
guide = create_guide(task_dcm_model)
svi_result = run_svi(task_dcm_model, guide, model_args, num_steps=500, lr=0.01)

print(f"Final ELBO loss: {svi_result['final_loss']:.1f}")
```

The `run_svi` function handles ClippedAdam optimization, learning rate decay,
gradient clipping, and NaN detection automatically.

---

## Step 3: Inspect Posteriors

Extract the fitted parameters and compare against ground truth.

```python
from pyro_dcm.models.guides import extract_posterior_params
from pyro_dcm.forward_models.neural_state import parameterize_A

# Get posterior medians and variational parameters
posterior = extract_posterior_params(guide, model_args)

# Transform A_free -> A (negative diagonal for self-inhibition)
A_free_median = posterior["median"]["A_free"]
A_inferred = parameterize_A(A_free_median)

print("True A:")
print(A_true.numpy().round(3))
print("\nInferred A:")
print(A_inferred.detach().numpy().round(3))

# Compute RMSE
rmse = torch.sqrt(torch.mean((A_true - A_inferred.detach()) ** 2)).item()
print(f"\nRMSE(A): {rmse:.4f}")
```

Plot true vs inferred connectivity:

```python
import matplotlib.pyplot as plt

a_true_flat = A_true.numpy().flatten()
a_inf_flat = A_inferred.detach().numpy().flatten()

plt.figure(figsize=(5, 5))
plt.scatter(a_true_flat, a_inf_flat, s=80, edgecolors="k", zorder=3)
lims = [min(a_true_flat.min(), a_inf_flat.min()) - 0.1,
        max(a_true_flat.max(), a_inf_flat.max()) + 0.1]
plt.plot(lims, lims, "k--", alpha=0.5, label="identity")
plt.xlabel("True A elements")
plt.ylabel("Inferred A elements")
plt.title("Task DCM: Parameter Recovery")
plt.legend()
plt.tight_layout()
plt.savefig("task_dcm_recovery.png", dpi=150)
plt.show()
```

---

## Step 4: Model Comparison

Compare two models with different structural assumptions using ELBO.

```python
# Model 1: fully connected (all A elements free)
mask_full = torch.ones(N, N, dtype=torch.float64)

guide_full = create_guide(task_dcm_model)
args_full = (observed_bold, stim, mask_full, c_mask, t_eval, TR)
res_full = run_svi(task_dcm_model, guide_full, args_full, num_steps=500)

# Model 2: sparse (only diagonal + region 0->1 connection)
mask_sparse = torch.eye(N, dtype=torch.float64)
mask_sparse[0, 1] = 1.0  # only A[0,1] off-diagonal connection

guide_sparse = create_guide(task_dcm_model)
args_sparse = (observed_bold, stim, mask_sparse, c_mask, t_eval, TR)
res_sparse = run_svi(task_dcm_model, guide_sparse, args_sparse, num_steps=500)

# Lower (more negative) ELBO = better model fit
print(f"Full model  ELBO: {-res_full['final_loss']:.1f}")
print(f"Sparse model ELBO: {-res_sparse['final_loss']:.1f}")

winner = "Full" if res_full["final_loss"] < res_sparse["final_loss"] else "Sparse"
print(f"Winner: {winner} model (higher ELBO)")
```

ELBO-based model comparison selects the model with the best balance between
data fit and complexity (Occam's razor through the variational bound).

---

## Step 5: Spectral DCM

The same API pattern works for spectral (resting-state) DCM. Instead of BOLD
time series, the data is cross-spectral density (CSD) in the frequency domain.

```python
from pyro_dcm.simulators.spectral_simulator import (
    simulate_spectral_dcm,
    make_stable_A_spectral,
)
from pyro_dcm.models.spectral_dcm_model import spectral_dcm_model

# Simulate CSD from a 3-region spectral DCM
A_spec = make_stable_A_spectral(3, seed=42)
spec_result = simulate_spectral_dcm(A_spec, TR=2.0, n_freqs=32)

observed_csd = spec_result["csd"]     # shape (32, 3, 3), complex128
freqs = spec_result["freqs"]          # shape (32,)

# Run SVI on spectral model
a_mask_spec = torch.ones(3, 3, dtype=torch.float64)
spec_args = (observed_csd, freqs, a_mask_spec)

guide_spec = create_guide(spectral_dcm_model)
spec_svi = run_svi(
    spectral_dcm_model, guide_spec, spec_args, num_steps=500, lr=0.01,
)

# Extract inferred A
post_spec = extract_posterior_params(guide_spec, spec_args)
A_spec_inf = parameterize_A(post_spec["median"]["A_free"])

print(f"Spectral ELBO: {-spec_svi['final_loss']:.1f}")
print(f"True A:\n{A_spec.numpy().round(3)}")
print(f"Inferred A:\n{A_spec_inf.detach().numpy().round(3)}")
```

---

## Next Steps

- **Guide selection:** Which variational guide should you use? See
  [guide_selection.md](guide_selection.md) for a decision tree based on your DCM
  variant, network size, and compute budget.
- **Methods reference:** See `docs/03_methods_reference/methods.md` for
  the full mathematical framework (paper-ready).
- **Equations quick-reference:** See `docs/03_methods_reference/equations.md`
  for a single-page lookup of all implemented equations.
- **Regression DCM:** The `pyro_dcm.simulators.rdcm_simulator` and
  `pyro_dcm.forward_models.rdcm_posterior` modules provide a
  frequency-domain regression variant with analytic VB inference.
- **Amortized inference:** Train a normalizing flow guide once, then
  get instant posteriors for new subjects. See
  `scripts/train_amortized_guide.py` and `pyro_dcm.guides.AmortizedFlowGuide`.
- **Benchmarks:** Run `python benchmarks/run_all_benchmarks.py` for the
  full benchmark suite comparing all methods and variants.
- **API docstrings:** Every public function has NumPy-style docstrings
  with parameter descriptions, return types, and reference citations.
