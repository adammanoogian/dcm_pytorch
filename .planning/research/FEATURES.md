---
type: "research"
scope: "features"
updated: "2026-03-24"
---

# Feature Catalog: Mathematical Specifications

This file contains the EXACT mathematics to implement. Every equation here must
appear in code with the corresponding [REF-XXX] citation.

---

## Feature 1: Task-Based DCM

### 1.1 Neural State Equation [REF-001 Eq. 1]

```
dx/dt = A·x + Σⱼ uⱼ(t)·B⁽ʲ⁾·x + C·u(t)
```

Where:
- x ∈ ℝᴺ: neural states (one per region)
- A ∈ ℝᴺˣᴺ: effective connectivity (intrinsic coupling)
- B⁽ʲ⁾ ∈ ℝᴺˣᴺ: modulatory effect of input j on connections
- C ∈ ℝᴺˣᴹ: driving input weights
- u(t) ∈ ℝᴹ: experimental inputs (stimulus functions)

For v0.1 without modulatory inputs: dx/dt = A·x + C·u(t)

### 1.2 Balloon-Windkessel Model [REF-002 Eq. 2-5]

Per region i, four hemodynamic state variables:

```
ds/dt = x - κ·s - γ·(f - 1)        [vasodilatory signal]
df/dt = s                            [blood inflow]
dv/dt = (1/τ)·(f - v^(1/α))        [blood volume]
dq/dt = (1/τ)·(f·E(f,E₀)/E₀ - v^(1/α)·q/v)  [deoxyhemoglobin]
```

Where:
- E(f, E₀) = 1 - (1 - E₀)^(1/f)   [oxygen extraction fraction]
- κ ≈ 0.65 s⁻¹: signal decay rate
- γ ≈ 0.41 s⁻¹: autoregulation feedback
- τ ≈ 0.98 s: hemodynamic transit time
- α ≈ 0.32: Grubb's exponent (stiffness)
- E₀ ≈ 0.34: resting oxygen extraction fraction

### 1.3 BOLD Signal Equation [REF-002 Eq. 6]

```
y = V₀ · [k₁·(1 - q) + k₂·(1 - q/v) + k₃·(1 - v)]
```

Where:
- V₀ = 0.02: resting venous blood volume fraction
- k₁ = 7·E₀ (intravascular signal)
- k₂ = 2 (extravascular signal)
- k₃ = 2·E₀ - 0.2 (volume-related signal)

At 3T with TE=0.04s: k₁ = 4.3·ν₀·E₀·TE, k₂ = ε·r₀·E₀·TE, k₃ = 1 - ε

### 1.4 Task DCM Likelihood

```
p(y | A, C, θ_hemo) = N(y; BOLD(A, C, θ_hemo, u), σ²·I)
```

BOLD(·) = integrate neural state eq. → feed through Balloon model → BOLD signal eq.

### 1.5 Task DCM Test Case

**Setup:** 3 regions (V1 → V5 → PPC), forward chain connectivity
```
A_true = [[-0.5,  0.0,  0.0],
          [ 0.4, -0.5,  0.0],
          [ 0.0,  0.3, -0.5]]

C_true = [[0.8],    # V1 receives input
          [0.0],
          [0.0]]
```

**Stimulus:** Block design, 30s on / 30s off, 10 blocks, TR = 2s, total 600s
**SNR:** 5 (σ_obs = peak_bold / 5)
**Recovery criteria:** RMSE(A) < 0.05, coverage ∈ [0.90, 0.99]

---

## Feature 2: Spectral DCM (spDCM)

### 2.1 Transfer Function [REF-010 Eq. 3]

```
g(ω, θ) = (iωI - A)⁻¹
```

Full transfer from neuronal fluctuations to BOLD CSD:

```
H(ω) = C_out · g(ω) · C_in
```

Where C_out, C_in are output/input matrices (often identity for standard spDCM).

### 2.2 Predicted Cross-Spectral Density [REF-010 Eq. 4]

```
S(ω) = H(ω) · Σ_n(ω) · H(ω)* + Σ_ε(ω)
```

Where:
- Σ_n(ω) = diag(σ²ₙ · ω^(-αₙ)): neuronal fluctuation power spectrum (1/f)
- Σ_ε(ω) = diag(σ²_ε): observation noise (white, or 1/f)
- H(ω)* denotes conjugate transpose

### 2.3 Neuronal Fluctuation Spectrum [REF-010 Eq. 5-6]

```
Gₙ(ω) = σ²ₙ / (ω/ω₀)^αₙ
```

Where:
- σ²ₙ: neuronal noise amplitude per region
- αₙ ≈ 1: spectral exponent (1/f noise)
- ω₀: reference frequency (2π Hz)

### 2.4 Spectral DCM Likelihood [REF-010 Eq. 8-10]

CSD likelihood treating vec(CSD) as multivariate Gaussian:

```
p(CSD_obs | θ) = N(vec(CSD_obs); vec(S(ω, θ)), Σ_noise)
```

Summed over frequency bins: L = Σ_ω log p(CSD_obs(ω) | θ)

### 2.5 Spectral DCM Test Case

**Setup:** 3 regions with known CSD structure
```
A_true = [[-0.5,  0.2,  0.0],
          [ 0.0, -0.5,  0.3],
          [ 0.1,  0.0, -0.5]]
```

**Parameters:**
- Frequency range: 0.01 - 0.5 Hz (128 bins)
- α_neuronal = 1.0 (1/f noise)
- σ²_neuronal = 0.01 per region
- σ²_observation = 0.001
- TR = 2s, 300 time points (simulate time series → compute empirical CSD)

**Recovery criteria:** RMSE(A) < 0.05, CSD prediction matches within 10%

---

## Feature 3: Regression DCM (rDCM)

### 3.1 Convolution Model [REF-020 Eq. 1-2]

In the time domain:
```
y(t) = Σⱼ aⱼ · (hrf * yⱼ)(t) + Σₖ cₖ · (hrf * xₖ)(t) + ε(t)
```

For region j, this is a multiple regression on HRF-convolved inputs.

### 3.2 Frequency Domain Formulation [REF-020 Eq. 4-8]

After DFT:
```
Y_j(ω) = H(ω) · [A_j · Y(ω) + C_j · X(ω)] + E_j(ω)
```

Where H(ω) = DFT of canonical HRF.

In matrix form per region j:
```
y_j = X_j · θ_j + ε_j
```

Where:
- y_j ∈ ℂᶠ: DFT of BOLD for region j (F frequency bins)
- X_j ∈ ℂᶠˣ⁽ᴺ⁺ᴹ⁾: design matrix = H(ω) · [Y_{-j}(ω), X(ω)]
- θ_j ∈ ℝᴺ⁺ᴹ: [a_j1, ..., a_jN, c_j1, ..., c_jM] connectivity parameters
- ε_j ~ N(0, σ²_j I): observation noise

### 3.3 Analytic Posterior [REF-020 Eq. 11-14]

With Gaussian prior θ_j ~ N(0, Λ_j⁻¹) and Gaussian likelihood:

```
Σ_j = (Λ_j + σ_j⁻² · X_j^H · X_j)⁻¹
μ_j = σ_j⁻² · Σ_j · X_j^H · y_j
```

Where X_j^H is conjugate transpose of design matrix.

### 3.4 Free Energy (Model Evidence) [REF-020 Eq. 15]

```
F_j = -½ [N_f · log(2π) + N_f · log(σ²_j) + σ_j⁻² · ||y_j - X_j·μ_j||²
       + log|Σ_j| - log|Λ_j⁻¹| + μ_j^T · Λ_j · μ_j]
```

Total free energy: F = Σ_j F_j

### 3.5 ARD Sparsity Prior [REF-020 Eq. 9-10]

Automatic Relevance Determination:
```
Λ_j = diag(α_j1, ..., α_jP)      [precision per parameter]
α_jp ~ Gamma(a₀, b₀)              [hyperprior]
```

Update: α_jp = (a₀ + ½) / (b₀ + ½(μ²_jp + Σ_j[p,p]))

### 3.6 Regression DCM Test Case

**Setup:** 5 regions, sparse connectivity (60% zero entries)
```
A_true = [[-0.5,  0.3,  0.0,  0.0,  0.0],
          [ 0.0, -0.5,  0.2,  0.0,  0.0],
          [ 0.0,  0.0, -0.5,  0.4,  0.0],
          [ 0.2,  0.0,  0.0, -0.5,  0.3],
          [ 0.0,  0.0,  0.0,  0.0, -0.5]]

C_true = [[0.5, 0.0],
          [0.0, 0.0],
          [0.0, 0.3],
          [0.0, 0.0],
          [0.0, 0.0]]
```

**Design:** 2 experimental conditions, event-related design, TR=2s, 400 TRs
**SNR:** 3 (realistic for whole-brain rDCM)
**Recovery criteria:**
- RMSE(A) < 0.05
- Sparsity pattern recovery: F1 > 0.85 for zero/nonzero classification
- Closed-form posterior matches iterative VB within numerical precision

---

## Feature 4: Pyro Generative Models

### 4.1 Prior Specifications (All Variants)

| Parameter | Distribution | Hyperparameters | Reference |
|-----------|-------------|-----------------|-----------|
| A_ij (off-diag) | N(0, σ²_A) | σ_A = 0.5 | REF-001 |
| A_ii (diagonal) | N(-0.5, 0.1²) | Stabilizing prior | REF-002 |
| C_ij | N(0, σ²_C) | σ_C = 0.5 | REF-001 |
| log(κ) | N(log(0.65), 0.3²) | | REF-002 Table 1 |
| log(γ) | N(log(0.41), 0.3²) | | REF-002 Table 1 |
| log(τ) | N(log(0.98), 0.3²) | | REF-002 Table 1 |
| logit(α) | N(logit(0.32), 0.5²) | α ∈ (0,1) | REF-002 Table 1 |
| logit(E₀) | N(logit(0.34), 0.5²) | E₀ ∈ (0,1) | REF-002 Table 1 |
| log(σ_obs) | N(log(0.1), 1²) | | Weakly informative |
| log(σ²_n) | N(log(0.01), 1²) | spDCM only | REF-010 |
| α_n (spectral) | N(1.0, 0.5²) | spDCM only | REF-010 |

### 4.2 ELBO for Model Comparison

```
ELBO(m) = E_q(θ)[log p(y|θ,m)] - KL[q(θ) || p(θ|m)]
```

For model comparison between connectivity architectures:
- Model m₁: A has all connections (full model)
- Model m₂: A has subset of connections (reduced model)
- ΔF = ELBO(m₁) - ELBO(m₂) > 3 → strong evidence for m₁

This replaces SPM's free energy comparison (same quantity in the limit).

---

## Feature 5: Amortized Inference Architecture

### 5.1 Summary Network

```
Input: observed data (T×N for BOLD, F×N×N for CSD)
  → 1D-CNN / Set Transformer
  → summary embedding h ∈ ℝᵈ (d = 128 or 256)
```

For task DCM: 1D-CNN over time dimension, pooled across regions
For spectral DCM: 2D-CNN over frequency × region dimensions
For rDCM: MLP on sufficient statistics (X^H X, X^H y per region)

### 5.2 Conditional Flow

```
h → Zuko MAF/NSF → q(θ|h)
```

The flow transforms a base Gaussian z ~ N(0, I) into the posterior:
```
θ = flow(z; h)
log q(θ|h) = log p(z) - log |det ∂flow/∂z|
```

### 5.3 Training

Amortized SVI:
```
for each batch of simulated datasets {(y_i, θ_i)}:
    θ_i ~ Prior
    y_i ~ p(y | θ_i)
    L = -ELBO(y_i, θ_i; flow)
    update flow parameters to minimize L
```

### 5.4 Validation Protocol

1. Train on 10,000 simulated datasets
2. Hold out 1,000 datasets
3. Compare: amortized posterior vs per-subject SVI (500 steps) vs NUTS (gold standard)
4. Metrics: RMSE, coverage, ELBO, wall time
5. Report amortization gap per dataset
