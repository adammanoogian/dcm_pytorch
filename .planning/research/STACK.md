---
type: "research"
scope: "stack"
updated: "2026-03-24"
---

# Stack Research

## Primary Stack

### PyTorch 2.x
- **Role:** Tensor computation backbone, autograd for gradient-based inference
- **Why:** Pyro requires PyTorch; torchdiffeq is native; Zuko flows are PyTorch
- **Version:** 2.2+ (for `torch.compile` potential, improved complex tensor support)
- **Risk:** Complex-valued tensors needed for spectral DCM — verify dtype support

### Pyro 1.9+
- **Role:** Probabilistic programming — `pyro.sample`, SVI, ELBO, model/guide pattern
- **Why:** Only PPL that supports both manual guides AND amortized neural guides
  with explicit generative models for ELBO computation
- **Key APIs:**
  - `pyro.poutine.trace` — trace model for ELBO
  - `pyro.infer.SVI` — stochastic variational inference
  - `pyro.infer.Trace_ELBO` — standard ELBO estimator
  - `pyro.distributions` — priors
  - `pyro.nn.PyroModule` — for neural guides

### torchdiffeq
- **Role:** ODE integration within PyTorch computation graph
- **Why:** Backprop through ODE solutions via adjoint method
- **Key APIs:**
  - `odeint(func, y0, t, method='dopri5')` — forward solve
  - `odeint_adjoint` — memory-efficient backprop for long time series
- **Risk:** Stiff Balloon model may need implicit solvers not in torchdiffeq.
  Fallback: `torchsde` or manual implicit Euler.

### Zuko
- **Role:** Normalizing flow architectures for amortized inference guides
- **Why:** Clean PyTorch-native flows, compatible with Pyro guide interface
- **Key architectures:**
  - `zuko.flows.MAF` — Masked Autoregressive Flow
  - `zuko.flows.NSF` — Neural Spline Flow
  - `zuko.flows.NAF` — Neural Autoregressive Flow
- **Integration:** Wrap Zuko flow as `pyro.nn.PyroModule` guide

### NumPyro (validation only)
- **Role:** NUTS/HMC sampling for ground-truth posterior validation
- **Why:** JAX backend gives fast NUTS; validates that SVI posterior is correct
- **Not primary** because amortized SVI requires PyTorch guide networks

## Secondary Stack

### scipy.signal
- **Role:** CSD computation (Welch periodogram, multi-taper)
- **Why:** Battle-tested spectral estimation; validate against before reimplementing in torch

### matplotlib / seaborn
- **Role:** Diagnostic plots, benchmark figures
- **Plots needed:** posterior marginals, ELBO traces, A matrix heatmaps, coverage plots

### pytest
- **Role:** Test framework
- **Plugins:** pytest-benchmark for timing, pytest-xdist for parallel tests

### h5py / scipy.io
- **Role:** Save/load .mat files for SPM validation, HDF5 for large simulation datasets

## Stack Decision Log

| Decision | Chosen | Rejected | Reason |
|----------|--------|----------|--------|
| PPL | Pyro | sbi, BayesFlow | Need explicit generative model for ELBO model comparison |
| ODE solver | torchdiffeq | diffrax, torchode | PyTorch native, adjoint method, proven |
| Flows | Zuko | nflows, normflows | Cleaner API, actively maintained, Pyro-compatible |
| Validation | NumPyro NUTS | Stan, emcee | Same API conventions as Pyro, JAX speed |
| CSD | scipy.signal | custom torch | Validated reference first, torch port later if needed |
