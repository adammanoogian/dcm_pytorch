# Equations Quick-Reference

All equations implemented in Pyro-DCM, organized by DCM variant. Each entry
links the mathematical formula to its reference paper and source file.

---

## Task-Based DCM

| Name | Formula | Reference | Implementation |
|------|---------|-----------|----------------|
| Neural state equation | dx/dt = Ax + Cu | [REF-001] Eq. 1 | `neural_state.py` |
| A matrix parameterization | a_ii = -exp(A_free_ii) / 2; a_ij = A_free_ij | SPM12 convention | `neural_state.py` |
| Vasodilatory signal | ds/dt = x - kappa*s - gamma*(f - 1) | [REF-002] Eq. 2 | `balloon_model.py` |
| Blood flow (log-space) | d(ln f)/dt = s / f | [REF-002] Eq. 3 | `balloon_model.py` |
| Blood volume (log-space) | d(ln v)/dt = (f - v^(1/alpha)) / (tau * v) | [REF-002] Eq. 4 | `balloon_model.py` |
| Deoxyhemoglobin (log-space) | d(ln q)/dt = (f * E_f/E0 - v^(1/alpha) * q/v) / (tau * q) | [REF-002] Eq. 5 | `balloon_model.py` |
| Oxygen extraction | E_f = 1 - (1 - E0)^(1/f) | [REF-002] Eq. 5 | `balloon_model.py` |
| BOLD signal | y = V0 * (k1*(1 - q) + k2*(1 - q/v) + k3*(1 - v)) | [REF-002] Eq. 6 | `bold_signal.py` |
| BOLD constants | k1 = 7*E0, k2 = 2, k3 = 2*E0 - 0.2, V0 = 0.02 | [REF-002] Eq. 6 | `bold_signal.py` |
| Coupled ODE system | 5N-dim state: [x, s, ln f, ln v, ln q] per region | [REF-001]+[REF-002] | `coupled_system.py` |

### Hemodynamic Parameter Defaults (SPM12 Code)

| Parameter | Symbol | Default | Unit | Source |
|-----------|--------|---------|------|--------|
| Signal decay | kappa | 0.64 | s^-1 | SPM12 spm_fx_fmri.m |
| Autoregulation | gamma | 0.32 | s^-1 | SPM12 spm_fx_fmri.m |
| Transit time | tau | 2.00 | s | SPM12 spm_fx_fmri.m |
| Grubb's exponent | alpha | 0.32 | -- | SPM12 spm_fx_fmri.m |
| O2 extraction | E0 | 0.40 | -- | SPM12 spm_fx_fmri.m |

---

## Spectral DCM

| Name | Formula | Reference | Implementation |
|------|---------|-----------|----------------|
| Transfer function | H(w) = C_out @ (iwI - A)^-1 @ C_in | [REF-010] Eq. 3 | `spectral_transfer.py` |
| Modal decomposition | H(w) = sum_k dgdv_k * dvdu_k / (i*2*pi*w - lambda_k) | [REF-010] Eq. 3 | `spectral_transfer.py` |
| Eigenvalue stabilization | Re(lambda_k) <= -1/32 | SPM12 convention | `spectral_transfer.py` |
| Predicted CSD | S(w) = H(w) @ Gu(w) @ H(w)^H + Gn(w) | [REF-010] Eq. 4 | `spectral_transfer.py` |
| Neuronal noise (1/f) | Gu_i(w) = C * exp(a[0,i]) * w^(-exp(a[1,i])) * 4 | [REF-010] Eq. 5-6 | `spectral_noise.py` |
| Observation noise (global) | Gn_global(w) = C * exp(b[0]) * w^(-exp(b[1])/2) / 8 | [REF-010] Eq. 7 | `spectral_noise.py` |
| Observation noise (regional) | Gn_i(w) = C * exp(c[0,i]) * w^(-exp(c[1,i])/2) | [REF-010] Eq. 7 | `spectral_noise.py` |
| SPM scaling constant | C = 1/256 | SPM12 spm_csd_fmri_mtf.m | `spectral_noise.py` |
| Standard spDCM convention | C_in = C_out = I_N | [REF-010] | `spectral_transfer.py` |

### Noise Parameters

| Group | Shape | Description | Prior |
|-------|-------|-------------|-------|
| a | (2, N) | Neuronal: [log amplitude, log exponent] per region | N(0, 1/64) |
| b | (2, 1) | Global observation: [log amplitude, log exponent] | N(0, 1/64) |
| c | (2, N) | Regional observation: [log amplitude, log exponent] per region | N(0, 1/64) |

Total noise parameters: 4N + 2.

---

## Regression DCM

| Name | Formula | Reference | Implementation |
|------|---------|-----------|----------------|
| Neural dynamics (Euler) | x_new = x + dt*(A @ x + C @ u_t) | [REF-020] Eq. 1-2 | `rdcm_forward.py` |
| Hemodynamic Euler step | s_new = s + dt*(x - kappa*s - gamma*(f-1)) | [REF-020] / Julia | `rdcm_forward.py` |
| rDCM BOLD signal | y = V0*(k1*(1-q) + k2*(1-q/v) + k3*(1-v)) | Julia convention | `rdcm_forward.py` |
| rDCM BOLD constants | k1=25*0.4*0.04, k2=40.3*0.4*0.04, k3=1, V0=4 | Julia convention | `rdcm_forward.py` |
| HRF generation | Euler integrate 1-region DCM with A=-1, C=16 | Julia get_hrf() | `rdcm_forward.py` |
| DFT derivative coefficients | coef[k] = exp(2*pi*i*k/N) - 1 | [REF-020] Eq. 6-7 | `rdcm_forward.py` |
| Frequency-domain design matrix | X_j = [DFT(HRF*BOLD_others), DFT(HRF*u), confounds] | [REF-020] Eq. 4-8 | `rdcm_forward.py` |
| Linear regression model | y_j = X_j @ theta_j + epsilon_j (per region) | [REF-020] Eq. 5-6 | `rdcm_forward.py` |
| Posterior covariance | Sigma_j = (l0 + lambda_j * X^H X)^-1 | [REF-020] Eq. 12 | `rdcm_posterior.py` |
| Posterior mean | mu_j = Sigma_j @ (l0*m0 + lambda_j * X^H y) | [REF-020] Eq. 11 | `rdcm_posterior.py` |
| Noise precision update | a_N = a0 + N_y/2 | [REF-020] Eq. 13 | `rdcm_posterior.py` |
| Noise rate update | b_N = b0 + (y^H y - mu^H Sigma^-1 mu + m0^H l0 m0)/2 | [REF-020] Eq. 14 | `rdcm_posterior.py` |
| Free energy (rigid) | F = sum of 5 components (log-likelihood, prior, complexity) | [REF-020] Eq. 15 | `rdcm_posterior.py` |
| Free energy (sparse) | F = rigid F + 2 ARD indicator terms | [REF-021] | `rdcm_posterior.py` |
| Prior mean | A: -0.5*I (self-inhibition); C: zeros | [REF-020] Eq. 9 | `rdcm_posterior.py` |
| Prior precision | A off-diag: nr/8; A diag: 8*nr; C: 1 for present | [REF-020] Eq. 10 | `rdcm_posterior.py` |
| ARD sparsity prior | Bernoulli(p0=0.5) binary indicators z_j | [REF-020] Eq. 9-10 | `rdcm_posterior.py` |

### rDCM Hemodynamic Constants (Julia Convention)

| Parameter | Symbol | Value | Source |
|-----------|--------|-------|--------|
| Signal decay | kappa | 0.64 | Julia dcm_euler_integration.jl |
| Autoregulation | gamma | 0.32 | Julia dcm_euler_integration.jl |
| Transit time | tau | 2.00 | Julia dcm_euler_integration.jl |
| Grubb's exponent | alpha | 0.32 | Julia dcm_euler_integration.jl |
| O2 extraction (rho) | rho | 0.32 | Julia dcm_euler_integration.jl |
| Relaxation rate slope | -- | 25.0 | Julia dcm_euler_integration.jl |
| Frequency offset | -- | 40.3 | Julia dcm_euler_integration.jl |
| Echo time | TE | 0.04 | Julia dcm_euler_integration.jl |
| Resting venous volume | V0 | 4.0 | Julia dcm_euler_integration.jl |

---

## Inference

| Name | Formula | Reference | Implementation |
|------|---------|-----------|----------------|
| ELBO objective | L(phi) = E_q[log p(y,theta)] - E_q[log q(theta)] | [REF-041] | `guides.py` |
| Mean-field guide | q(theta) = prod_i N(mu_i, sigma_i^2) | [REF-041] | `guides.py` (AutoNormal) |
| SVI optimization | ClippedAdam with LR decay: lr_t = lr_0 * decay^(t/T) | [REF-060] Pyro | `guides.py` |
| Amortized guide | q(theta\|x) = NSF(summary_net(x)) | [REF-042]+[REF-043] | `amortized_flow.py` |
| Summary network (BOLD) | 1D-CNN: Conv1d -> ReLU -> Pool -> FC | [REF-043] | `summary_networks.py` |
| Summary network (CSD) | MLP: flatten -> FC -> ReLU -> FC | [REF-043] | `summary_networks.py` |
| Neural Spline Flow | Monotonic rational-quadratic splines on [-5, 5] | [REF-042] | Zuko NSF |

---

## Model Comparison

| Name | Formula | Reference | Implementation |
|------|---------|-----------|----------------|
| ELBO comparison | Best model: argmax_m ELBO(m) | [REF-040] | `model_comparison.py` |
| rDCM free energy | F_total = sum over regions F_r | [REF-020] Eq. 15 | `rdcm_posterior.py` |
| Free energy comparison | Best model: argmax_m F(m) | [REF-020] | `model_comparison.py` |

---

## Reference Key

| ID | Citation |
|----|----------|
| [REF-001] | Friston, Harrison & Penny (2003). Dynamic causal modelling. NeuroImage 19(4), 1273-1302. |
| [REF-002] | Stephan et al. (2007). Comparing hemodynamic models with DCM. NeuroImage 38(3), 387-401. |
| [REF-010] | Friston, Kahan, Biswal & Razi (2014). A DCM for resting state fMRI. NeuroImage 94, 396-407. |
| [REF-020] | Frassle et al. (2017). A generative model of whole-brain effective connectivity. NeuroImage 145, 270-275. |
| [REF-021] | Frassle et al. (2018). Regression DCM for fMRI. NeuroImage 155, 406-421. |
| [REF-040] | Friston et al. (2007). Variational free energy and the Laplace approximation. NeuroImage 34(1), 220-234. |
| [REF-041] | Blei, Kucukelbir & McAuliffe (2017). Variational inference: A review for statisticians. JASA 112(518), 859-877. |
| [REF-042] | Papamakarios et al. (2021). Normalizing flows for probabilistic modeling and inference. JMLR 22(57), 1-64. |
| [REF-043] | Cranmer, Brehmer & Louppe (2020). The frontier of simulation-based inference. PNAS 117(48), 30055-30062. |

## Source File Index

| Source File | DCM Variant | Key Equations |
|-------------|-------------|---------------|
| `forward_models/neural_state.py` | Task | [REF-001] Eq. 1 |
| `forward_models/balloon_model.py` | Task | [REF-002] Eq. 2-5 |
| `forward_models/bold_signal.py` | Task | [REF-002] Eq. 6 |
| `forward_models/coupled_system.py` | Task | [REF-001] Eq. 1 + [REF-002] Eq. 2-5 |
| `forward_models/spectral_transfer.py` | Spectral | [REF-010] Eq. 3-4 |
| `forward_models/spectral_noise.py` | Spectral | [REF-010] Eq. 5-7 |
| `forward_models/rdcm_forward.py` | rDCM | [REF-020] Eq. 4-8 |
| `forward_models/rdcm_posterior.py` | rDCM | [REF-020] Eq. 9-15 |
| `models/task_dcm_model.py` | Task | [REF-001] Eq. 1, [REF-002] Eq. 2-6 |
| `models/spectral_dcm_model.py` | Spectral | [REF-010] Eq. 3-10 |
| `models/rdcm_model.py` | rDCM | [REF-020] Eq. 4-8 |
| `models/guides.py` | All | [REF-041] (SVI), [REF-060] (Pyro) |
| `guides/amortized_flow.py` | Task, Spectral | [REF-042], [REF-043] |
| `guides/summary_networks.py` | Task, Spectral | [REF-043] |
