# Phase 3: Regression DCM Forward Model + rDCM Simulator - Research

**Researched:** 2026-03-26
**Domain:** Frequency-domain Bayesian linear regression, variational inference, ARD sparsity
**Confidence:** HIGH

## Summary

This research covers the complete technical domain for implementing Phase 3: the regression DCM (rDCM) frequency-domain likelihood, analytic posterior, ARD sparsity priors, free energy computation, and a simulator for synthetic frequency-domain data. The authoritative reference is the Julia implementation at `RegressionDynamicCausalModeling.jl` (v0.2.1), supplemented by the MATLAB tapas/rDCM toolbox and Frassle et al. (2017) [REF-020].

The key technical finding is that rDCM is fundamentally different from both task-DCM (Phase 1) and spectral DCM (Phase 2). It reformulates DCM as **Bayesian linear regression in the frequency domain**: the DFT of the time derivative of BOLD is regressed onto a design matrix composed of the DFT of BOLD signals (for endogenous connectivity A) and DFT of HRF-convolved inputs (for driving inputs C). The posterior is **analytic per region** -- each region j has an independent Gaussian posterior N(mu_j, Sigma_j) for connectivity parameters and a Gamma posterior for noise precision tau_j. The entire inference reduces to iterating closed-form VB update equations until convergence. The sparse variant adds Bernoulli indicator variables z per connection, updated via sigmoid functions with random sweep ordering.

The Julia implementation splits complex-valued frequency-domain data into real and imaginary parts (excluding zero-valued imaginary components at DC and Nyquist) before regression. This is the "mathematically correct version" per the Julia codebase comments. The HRF is computed via Euler integration of a minimal 1-region DCM with unit impulse input (not a parametric double-gamma). Priors on A have mean -I/2 on diagonal (self-inhibition) and zero off-diagonal, with precision scaling of 8/nr for off-diagonal and 1/(8*nr) for diagonal. Noise precision priors are Gamma(a0=2, b0=1).

**Primary recommendation:** Follow the Julia `RegressionDynamicCausalModeling.jl` implementation exactly for all algorithmic choices. Use PyTorch `torch.fft.rfft` for the DFT, `torch.linalg.inv` with `torch.complex128` for posterior covariance, and `torch.special.digamma`/`torch.lgamma` for free energy computation. Implement both rigid (fixed architecture) and sparse (ARD with binary indicators) variants. The design matrix construction, real/imaginary splitting, and VB update equations should match the Julia code line-for-line.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.2+ | Tensor ops, FFT, linalg, complex128 | `torch.fft.rfft`, `torch.linalg.inv`, `torch.special.digamma` |
| numpy | 1.24+ | Test fixtures, reference computations | Cross-validation against Julia reference values |
| scipy.special | 1.11+ | `digamma`, `gammaln` for validation | Verify `torch.special.digamma` matches scipy |
| pytest | 8.0+ | Test framework | All unit and integration tests |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| scipy.linalg | 1.11+ | Reference matrix inversion for validation | Cross-validate `torch.linalg.inv` on complex data |
| torch.fft | (in torch) | rfft, irfft for DFT operations | All frequency-domain transforms |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| torch.fft.rfft | numpy.fft.rfft | numpy breaks autograd graph; torch is native |
| torch.linalg.inv | torch.linalg.solve | solve is more stable but inv matches Julia/MATLAB exactly |
| torch.special.digamma | scipy.special.digamma | scipy not differentiable; torch works with autograd |

**Installation:**
```bash
pip install torch scipy numpy pytest
```

## Architecture Patterns

### Recommended Project Structure
```
src/pyro_dcm/
  forward_models/
    __init__.py              # Add new rDCM exports
    rdcm_forward.py          # Design matrix construction + likelihood [REF-020 Eq. 4-8]
    rdcm_posterior.py         # Analytic posterior + free energy [REF-020 Eq. 11-15]
  simulators/
    __init__.py              # Add rDCM simulator export
    rdcm_simulator.py        # Synthetic frequency-domain data generation
tests/
  test_rdcm_forward.py      # Design matrix, regressor creation, likelihood
  test_rdcm_posterior.py     # VB inversion (rigid + sparse), free energy
  test_rdcm_simulator.py    # End-to-end simulator tests
```

### Pattern 1: Frequency-Domain Design Matrix Construction (Core rDCM)

**What:** The rDCM model transforms BOLD time series into frequency domain, constructs a design matrix X from DFT of BOLD and HRF-convolved inputs, and regresses the DFT of the BOLD temporal derivative (Y) onto X. This is the fundamental transformation that turns DCM into linear regression.

**When to use:** Always -- this is the core of rDCM.

**How it works (from Julia `create_regressors.jl`):**

```python
# Source: Julia RegressionDynamicCausalModeling.jl/src/create_regressors.jl
# Implements [REF-020] Eq. 4-8

def create_regressors(
    hrf: torch.Tensor,       # (N_u,) hemodynamic response function
    y: torch.Tensor,          # (N_y, nr) BOLD signal, nr regions
    u: torch.Tensor,          # (N_u, nu) stimulus inputs, nu inputs
    u_dt: float,              # input sampling interval
    y_dt: float,              # BOLD sampling interval (TR)
    X0: torch.Tensor,         # (N_u, nc) confound regressors
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create frequency-domain design matrix and data.

    Steps:
    1. FFT of HRF, BOLD, and stimulus
    2. Convolve stimulus with HRF in frequency domain: uh = ifft(fft(u) * fft(h))
    3. Subsample convolved input to match BOLD sampling rate
    4. Compute DFT derivative coefficients: coef = exp(2*pi*i*k/N) - 1
    5. Y = coef * fft(y) / y_dt  (DFT of temporal derivative)
    6. X = [fft(y), fft(uh)/r_dt, fft(X0d)]  (design matrix)

    Returns X (complex), Y (complex) in frequency domain.
    """
```

**Critical detail -- real/imaginary splitting:** After creating complex X, Y, the Julia code splits them into real and imaginary parts before regression, discarding the zero-valued imaginary parts at DC (index 0) and Nyquist (last index for even N):

```python
# Source: Julia rigid_inversion.jl, sparse_inversion.jl
# Split complex data for real-valued regression

if N_freqs_is_even:
    Y_real = torch.cat([Y_complex.real, Y_complex.imag[1:-1]], dim=0)
    X_real = torch.cat([X_complex.real, X_complex.imag[1:-1]], dim=0)
else:
    Y_real = torch.cat([Y_complex.real, Y_complex.imag[1:]], dim=0)
    X_real = torch.cat([X_complex.real, X_complex.imag[1:]], dim=0)
```

### Pattern 2: Region-Wise Analytic Posterior (Rigid rDCM)

**What:** For each region r, the posterior over connectivity parameters theta_r is Gaussian with closed-form mean and covariance. The noise precision tau_r has a Gamma posterior. VB iterates between updating the Gaussian and Gamma until convergence.

**When to use:** Always for rigid (fixed-architecture) rDCM inversion.

**Implementation (from Julia `rigid_inversion.jl`):**

```python
# Source: Julia rigid_inversion.jl
# Implements [REF-020] Eq. 11-14

def update_posterior_rigid(
    tau_r: float,          # current noise precision estimate
    W: torch.Tensor,       # X_r^T @ X_r, precomputed
    l0_r: torch.Tensor,    # prior precision matrix (diagonal)
    mu0_r: torch.Tensor,   # prior mean
    V: torch.Tensor,       # X_r^T @ Y_r, precomputed
    Y_r: torch.Tensor,     # data for region r
    X_r: torch.Tensor,     # design matrix for region r
    a0: float,             # prior Gamma shape
    beta0: float,          # prior Gamma rate
    N_eff: int,            # effective number of data points
) -> tuple:
    """
    VB update equations per region [REF-020] Eq. 11-14:

    Posterior covariance [REF-020] Eq. 12:
        Sigma_r = inv(tau_r * X_r^T @ X_r + Lambda0_r)

    Posterior mean [REF-020] Eq. 11:
        mu_r = Sigma_r @ (tau_r * X_r^T @ Y_r + Lambda0_r @ mu0_r)

    Posterior Gamma shape [REF-020] Eq. 13:
        a_r = a0 + N_eff / 2

    Posterior Gamma rate [REF-020] Eq. 14:
        QF = 0.5 * ((Y_r - X_r @ mu_r)^T @ (Y_r - X_r @ mu_r) + tr(W @ Sigma_r))
        beta_r = beta0 + QF

    Posterior precision mean:
        tau_r = a_r / beta_r
    """
```

### Pattern 3: Free Energy in Closed Form (Rigid rDCM)

**What:** The negative free energy F_r for each region has a closed-form expression with 5 additive components.

**Implementation (from Julia `rigid_inversion.jl`):**

```python
# Source: Julia rigid_inversion.jl, compute_F()
# Implements [REF-020] Eq. 15

def compute_free_energy_rigid(
    N_eff, a_r, beta_r, QF, tau_r, l0_r, mu_r, mu0_r, Sigma_r, a0, beta0, dim_r
) -> float:
    """
    Negative free energy [REF-020] Eq. 15:

    F_r = log_lik + log_p_weight + log_p_prec + log_q_weight + log_q_prec

    where:
    log_lik = 0.5*(N_eff*(digamma(a_r) - log(beta_r)) - N_eff*log(2*pi)) - QF*tau_r

    log_p_weight = 0.5*(logdet(l0_r) - dim_r*log(2*pi)
                   - (mu_r - mu0_r)^T @ l0_r @ (mu_r - mu0_r)
                   - tr(l0_r @ Sigma_r))

    log_p_prec = a0*log(beta0) - lgamma(a0)
                 + (a0 - 1)*(digamma(a_r) - log(beta_r)) - beta0*tau_r

    log_q_weight = 0.5*(logdet(Sigma_r) + dim_r*(1 + log(2*pi)))

    log_q_prec = a_r - log(beta_r) + lgamma(a_r) + (1 - a_r)*digamma(a_r)
    """
```

### Pattern 4: Sparse rDCM with Binary Indicators (ARD)

**What:** Extends rigid rDCM with Bernoulli indicator variables z_r for each connection. Connections with z close to 0 are pruned. The update uses a sigmoid function with random sweep ordering.

**Implementation (from Julia `sparse_inversion.jl`):**

```python
# Source: Julia sparse_inversion.jl, update_posterior_sparse!()
# Implements Frassle et al. (2018) sparse rDCM

def update_posterior_sparse(
    # ... same as rigid plus:
    z_r: torch.Tensor,     # binary indicator probabilities
    p0: torch.Tensor,      # Bernoulli prior probabilities
    Z: torch.Tensor,       # diag(z_r) matrix
    G: torch.Tensor,       # E[Z @ X^T @ X @ Z]
) -> tuple:
    """
    Sparse VB update equations:

    Posterior covariance:
        Sigma_r = inv(tau_r * G + Lambda0_r)

    Posterior mean:
        mu_r = Sigma_r @ (tau_r * Z @ V + Lambda0_r @ mu0_r)

    Binary indicator update (sequential with random order):
        A = W * (mu_r @ mu_r^T) + W * Sigma_r
        g_i = log(p0_i / (1 - p0_i)) + tau_r * mu_r_i * V_i + tau_r * A_ii / 2
        For each i in random_order:
            z_r[i] = 1  (temporarily)
            g[i] -= tau_r * z_r^T @ A[:, i]
            z_r[i] = sigmoid(g[i])

    Z matrix and G matrix update:
        Z = diag(z_r)
        G = Z @ W @ Z
        G.diag() = z_r * W.diag()

    Posterior Gamma rate (with sparsity):
        QF = 0.5*(Y^T @ Y - 2*mu^T @ Z @ V + mu^T @ G @ mu + tr(G @ Sigma))
        beta_r = beta0 + QF
        tau_r = a_r / beta_r

    Reruns: The sparse inversion runs multiple times (default 100 in Julia)
    and selects the run with highest free energy.
    """
```

### Pattern 5: HRF Generation via Euler Integration

**What:** The HRF is NOT a parametric double-gamma. It is computed by Euler-integrating a minimal 1-region DCM with A=-1, C=16, unit impulse input, and extracting the resulting BOLD signal. This matches the SPM/tapas convention.

**Implementation (from Julia `generate_BOLD.jl` and `dcm_euler_integration.jl`):**

```python
# Source: Julia generate_BOLD.jl get_hrf()
# HRF computed by simulating a minimal DCM

def get_hrf(N: int, u_dt: float) -> torch.Tensor:
    """Generate HRF by Euler integrating a 1-region DCM.

    Creates a minimal LinearDCM with:
    - A = [[-1.0]]  (single region, self-inhibition)
    - C = [[16.0]]  (strong driving input, divided by 16 internally -> C_eff = 1.0)
    - Unit impulse input at t=0
    - Euler integration with step size u_dt

    Returns the BOLD signal as the HRF.
    """
```

**Critical: Hemodynamic constants in Julia Euler integration:**
```python
# From Julia dcm_euler_integration.jl
H = [0.64, 0.32, 2.00, 0.32, 0.32]
# H[1] = kappa (signal decay)
# H[2] = gamma (autoregulation)
# H[3] = tau (transit time)
# H[4] = alpha (Grubb's exponent)
# H[5] = rho (oxygen extraction fraction) = 0.32

# BOLD signal constants:
# relaxationRateSlope = 25.0
# frequencyOffset = 40.3
# oxygenExtractionFraction = 0.4 (different from H[5]!)
# echoTime = 0.04
# restingVenousVolume = 4.0
```

### Pattern 6: BOLD Data Simulation Pipeline

**What:** Synthetic BOLD is generated by: (1) Euler-integrating the full DCM to get neuronal activity, (2) convolving with HRF via FFT, (3) downsampling to TR, (4) adding Gaussian noise scaled by SNR.

**Implementation (from Julia `generate_BOLD.jl`):**

```python
# Source: Julia generate_BOLD.jl generate_BOLD()

def generate_bold_rdcm(
    A: torch.Tensor,       # (nr, nr) endogenous connectivity
    C: torch.Tensor,       # (nr, nu) driving input weights
    u: torch.Tensor,       # (N_u, nu) stimulus inputs
    u_dt: float,           # input sampling interval
    y_dt: float,           # BOLD sampling interval (TR)
    SNR: float,            # signal-to-noise ratio
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate synthetic BOLD for rDCM.

    Steps:
    1. Compute HRF via get_hrf(N_u, u_dt)
    2. Euler-integrate DCM to get neuronal signal x
    3. Zero-pad x and hrf to 3*N to avoid circular convolution artifacts
    4. Convolve: y = ifft(fft(x) * fft(hrf)) per region
    5. Truncate back to N, subsample to y_dt
    6. Add noise: noise = randn * diag(std(y) / SNR)

    Returns (y_noisy, y_clean, x_neural, hrf).
    """
```

**Critical detail -- zero-padding:** Julia zero-pads to 3*N before FFT convolution to avoid circular convolution artifacts, then truncates back.

### Anti-Patterns to Avoid

- **Using parametric double-gamma HRF:** The Julia/tapas implementations compute HRF by Euler-integrating a minimal DCM. Do NOT substitute a canned double-gamma function -- the shape will differ subtly.
- **Keeping data in complex form for regression:** The Julia implementation explicitly splits into real and imaginary parts. Performing regression on complex-valued data would double-count some frequencies and give wrong results.
- **Forgetting to exclude DC/Nyquist imaginary parts:** The imaginary part at index 0 (DC) is always zero for real signals, and the imaginary part at the Nyquist frequency (last index for even N) is also zero. These must be excluded when stacking real/imaginary parts.
- **Using full design matrix for all regions:** Each region r only uses the columns of X corresponding to connections present in the architecture mask `idx[r, :]`. Remove irrelevant columns per region to reduce dimensionality.
- **Multiplying QF by 0.5 twice in sparse variant:** The Julia code notes that the MATLAB implementation incorrectly multiplied QF by 0.5 an extra time. QF already includes the 0.5 factor. The Julia implementation is the corrected version.
- **Not doing multiple reruns for sparse rDCM:** The sparse variant is sensitive to initialization of z. Julia defaults to 100 reruns and selects the best by free energy. Skipping this leads to poor solutions.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| FFT/DFT computation | Custom DFT loop | `torch.fft.rfft` | Optimized FFT, correct normalization, GPU support |
| Matrix inversion for posterior | Cholesky + solve | `torch.linalg.inv(Hermitian(...))` | Matches Julia exactly; Hermitian wrapper ensures symmetry |
| Digamma function | Custom implementation | `torch.special.digamma` | Standard special function, differentiable |
| Log-gamma function | Custom implementation | `torch.lgamma` | Standard, differentiable |
| Log-determinant | `torch.det` + `torch.log` | `torch.linalg.slogdet` | Numerically stable for near-singular matrices |
| HRF computation | Parametric double-gamma | Euler integration of minimal DCM | Matches Julia/tapas exactly |
| Random permutation for z update | Fixed ordering | `torch.randperm` | Random sweep ordering is essential for sparse convergence |

**Key insight:** rDCM's entire inference is closed-form linear algebra -- matrix inversion, matrix multiplication, special functions. There are no ODE solvers or iterative optimizers needed during inference. The only iteration is the VB convergence loop (typically <50 iterations) and the z-update sweep.

## Common Pitfalls

### Pitfall 1: FFT Normalization and Convention Mismatch (CRITICAL)
**What goes wrong:** PyTorch `torch.fft.rfft` uses a different normalization convention than Julia's `FFTW.rfft`. If normalization differs, all downstream regression coefficients and free energies will be wrong.
**Why it happens:** Julia FFTW uses unnormalized FFT by default. PyTorch rfft also uses unnormalized by default ("backward" normalization).
**How to avoid:** Use `torch.fft.rfft(x, dim=0)` with default normalization (no `norm` argument). Verify against Julia reference values from the test files. Both Julia FFTW.rfft and torch.fft.rfft should produce identical results for the same input.
**Warning signs:** Posterior means (mu) or free energies (F) differ from Julia reference values by more than 1e-6.

### Pitfall 2: Design Matrix Column Ordering (CRITICAL)
**What goes wrong:** The design matrix X has columns: [fft(y) for A connections, fft(uh)/r_dt for C connections, fft(X0d) for confounds]. The column ordering must match the parameter vector ordering for the architecture mask to work correctly.
**Why it happens:** Easy to accidentally reorder columns or forget the 1/r_dt scaling on convolved inputs.
**How to avoid:** Follow Julia `create_regressors_core` exactly: `X = [y_fft, uh_fft ./ r_dt]` where `uh_fft` already includes confound regressors. The architecture mask `idx = [a, c, conf_weight_idx]` must match.
**Warning signs:** Posterior mean matrix has values in wrong positions.

### Pitfall 3: Prior Specification Differences Between Rigid and Sparse (HIGH)
**What goes wrong:** Rigid and sparse rDCM use different priors. Rigid uses the actual architecture mask to zero out absent connections. Sparse uses `ones(size(a))` for the mask (all connections possible) but sets the prior mean of off-diagonal A to zero.
**Why it happens:** In sparse rDCM, the sparsity is learned via z indicators, so the prior allows all connections.
**How to avoid:** Check Julia `get_priors(rdcm::SparseRdcm)` vs `get_priors(rdcm::RigidRdcm)`. Sparse calls `get_prior_stats(ones(size(a)), ones(size(c)))` while rigid uses the actual masks.
**Warning signs:** Sparse inversion gives wrong free energy or fails to prune connections.

### Pitfall 4: Real/Imaginary Splitting for Even vs Odd N (HIGH)
**What goes wrong:** For even-length signals, the Nyquist frequency component has zero imaginary part. For odd-length signals, there is no Nyquist frequency. The splitting logic must handle both cases.
**Why it happens:** `rfft` returns `N//2 + 1` complex values for length-N input. For even N, the last value (Nyquist) has zero imaginary part. For odd N, all values except DC may have nonzero imaginary parts.
**How to avoid:** Follow Julia exactly:
```python
if N_y_is_even:
    Y = cat([Y_c.real, Y_c.imag[1:-1]], dim=0)  # exclude DC and Nyquist imag
else:
    Y = cat([Y_c.real, Y_c.imag[1:]], dim=0)   # exclude only DC imag
```
**Warning signs:** Off-by-one errors in the number of effective data points N_eff.

### Pitfall 5: NaN Handling in Frequency Domain (MEDIUM)
**What goes wrong:** The `reduce_zeros!` function in Julia sets some rows to NaN when zero-valued frequencies outnumber informative ones. These NaN rows must be excluded from regression (via `idx_y = ~isnan(Y[:, r])`).
**Why it happens:** Balancing zero vs non-zero frequencies improves regression quality.
**How to avoid:** Implement `reduce_zeros` and filter NaN rows per-region before computing W = X^T @ X and V = X^T @ Y.
**Warning signs:** NaN in posterior mean or free energy.

### Pitfall 6: Convergence Criterion Difference (MEDIUM)
**What goes wrong:** Rigid and sparse rDCM use different convergence criteria. Rigid: `(F_old - F_r)^2 < tol^2`. Sparse: `(F_old - F_r)^2 < tol^2` (same formula but `tol` is not squared again -- the Julia code uses `pr = tol` for sparse vs `pr = tol^2` for rigid).
**Why it happens:** Subtle difference in variable naming between rigid and sparse.
**How to avoid:** In rigid: `pr = tol**2`, convergence when `(F_old - F_r)**2 < pr`. In sparse: `pr = tol`, convergence when `(F_old - F_r)**2 < pr**2`. Net effect: both use `(F_old - F_r)**2 < tol**2`.
**Warning signs:** Too many or too few iterations.

### Pitfall 7: Sparsity Thresholding in Sparse Variant (LOW)
**What goes wrong:** After each VB iteration in sparse rDCM, connections with |mu| < 1e-5 are set to zero, and z is set to zero where mu is zero. This hard thresholding is essential for clean pruning.
**Why it happens:** Without thresholding, numerically tiny connections persist.
**How to avoid:** Apply the same thresholding as Julia after each VB update and after convergence.
**Warning signs:** Too many non-zero connections in sparse solution.

## Code Examples

### Complete VB Inversion Loop (Rigid)

```python
# Source: Julia rigid_inversion.jl
# Implements [REF-020] Eq. 11-15

import torch
from torch.special import digamma

def rigid_inversion_per_region(
    X_r: torch.Tensor,   # (N_eff, D_r) design matrix for region r
    Y_r: torch.Tensor,   # (N_eff,) data for region r
    mu0_r: torch.Tensor, # (D_r,) prior mean
    l0_r: torch.Tensor,  # (D_r, D_r) prior precision (diagonal)
    a0: float,           # prior Gamma shape (2.0)
    beta0: float,        # prior Gamma rate (1.0)
    max_iter: int = 500,
    tol: float = 1e-5,
) -> dict:
    """Region-wise VB inversion for rigid rDCM.

    Cite: [REF-020] Eq. 11-15.
    """
    N_eff = X_r.shape[0]
    D_r = X_r.shape[1]

    # Precompute [REF-020] sufficient statistics
    W = X_r.T @ X_r                      # (D_r, D_r)
    V = X_r.T @ Y_r                      # (D_r,)

    # Initialize
    tau_r = a0 / beta0                     # posterior precision mean
    a_r = a0 + N_eff * 0.5                # posterior Gamma shape [REF-020] Eq. 13
    F_old = float('-inf')
    pr = tol ** 2

    mu_r = torch.zeros_like(mu0_r)
    Sigma_r = torch.zeros(D_r, D_r, dtype=X_r.dtype)

    for iteration in range(max_iter):
        # Posterior covariance [REF-020] Eq. 12
        Sigma_r = torch.linalg.inv(tau_r * W + l0_r)
        Sigma_r = 0.5 * (Sigma_r + Sigma_r.T)  # enforce symmetry

        # Posterior mean [REF-020] Eq. 11
        mu_r = Sigma_r @ (tau_r * V + l0_r @ mu0_r)

        # Posterior Gamma rate [REF-020] Eq. 14
        residual = Y_r - X_r @ mu_r
        QF = 0.5 * (residual @ residual + torch.trace(W @ Sigma_r))
        beta_r = beta0 + QF
        tau_r = a_r / beta_r

        # Free energy [REF-020] Eq. 15
        F_r = compute_free_energy_rigid(
            N_eff, a_r, beta_r, QF, tau_r,
            l0_r, mu_r, mu0_r, Sigma_r, a0, beta0, D_r
        )

        # Convergence check
        if (F_old - F_r) ** 2 < pr:
            break
        F_old = F_r

    return {
        'mu': mu_r,
        'Sigma': Sigma_r,
        'a': a_r,
        'beta': beta_r,
        'F': F_r,
        'iterations': iteration + 1,
    }
```

### Prior Specification

```python
# Source: Julia get_priors.jl
# Implements [REF-020] Eq. 9-10

def get_priors_rigid(
    a_mask: torch.Tensor,  # (nr, nr) binary architecture mask for A
    c_mask: torch.Tensor,  # (nr, nu) binary mask for C
) -> tuple:
    """Compute rDCM priors for rigid inversion.

    Prior mean [REF-020] Eq. 9:
        A_mean: zeros - I/2  (self-inhibition = -0.5)
        C_mean: zeros

    Prior covariance [REF-020] Eq. 10:
        A_cov: (a_off * 8) / nr + I / (8*nr)
        C_cov: 1.0 for present connections, 0 for absent

    Noise precision:
        Gamma(a0=2, b0=1)
    """
    nr = a_mask.shape[0]
    fac = 8  # scaling factor

    a_off = a_mask.clone().float()
    a_off.fill_diagonal_(0)  # remove diagonal from mask

    # Prior mean
    pE_A = torch.zeros(nr, nr) - 0.5 * torch.eye(nr)
    pE_C = torch.zeros_like(c_mask, dtype=torch.float64)

    # Prior covariance
    pC_A = (a_off * fac) / nr + torch.eye(nr) / (8 * nr)
    pC_C = torch.zeros_like(c_mask, dtype=torch.float64)
    pC_C[c_mask.bool()] = 1.0

    # Convert to precision (1/variance)
    l0_A = 1.0 / pC_A
    l0_C = 1.0 / pC_C  # inf for absent connections (handled by mask)

    # Concatenate into parameter vectors
    m0 = torch.cat([pE_A, pE_C], dim=1)  # (nr, nr+nu)
    l0 = torch.cat([l0_A, l0_C], dim=1)  # (nr, nr+nu)

    a0 = 2.0
    b0 = 1.0

    return m0, l0, a0, b0


def get_priors_sparse(
    a_mask: torch.Tensor,  # (nr, nr) - ignored, use ones
    c_mask: torch.Tensor,  # (nr, nu) - ignored, use ones
) -> tuple:
    """Compute rDCM priors for sparse inversion.

    Key difference from rigid: uses ones(size(a)) and ones(size(c))
    as the architecture mask, since sparsity is learned via z indicators.
    The prior mean on A off-diagonal is zero (not mask-dependent).
    Self-connection prior mean remains -0.5.
    """
    nr = a_mask.shape[0]
    nu = c_mask.shape[1]
    fac = 8

    # Use full connectivity for prior computation
    a_full = torch.ones(nr, nr)
    c_full = torch.ones(nr, nu)

    a_off = a_full.clone()
    a_off.fill_diagonal_(0)

    pE_A = torch.zeros(nr, nr)
    pE_A.fill_diagonal_(-0.5)  # only diagonal has -0.5; off-diag is 0
    pE_C = torch.zeros(nr, nu)

    pC_A = (a_off * fac) / nr + torch.eye(nr) / (8 * nr)
    pC_C = torch.ones(nr, nu)  # all 1.0 since all connections possible

    l0_A = 1.0 / pC_A
    l0_C = 1.0 / pC_C

    m0 = torch.cat([pE_A, pE_C], dim=1)
    l0 = torch.cat([l0_A, l0_C], dim=1)

    a0 = 2.0
    b0 = 1.0

    return m0, l0, a0, b0
```

### Derivative Coefficient Computation

```python
# Source: Julia create_regressors.jl
# The key transformation from BOLD to temporal derivative in frequency domain

def compute_derivative_coefficients(N: int) -> torch.Tensor:
    """Compute DFT derivative coefficients.

    For a signal y of length N, the DFT of dy/dt is:
        Y_dot[k] = coef[k] * Y[k] / dt

    where coef[k] = exp(2*pi*i*k/N) - 1

    Only positive frequencies (rfft output) are kept.

    Cite: [REF-020] Eq. 6-7 (frequency domain differentiation).
    """
    k = torch.arange(N, dtype=torch.float64)
    coef = torch.exp(2 * torch.pi * 1j * k / N) - 1.0
    N_rfft = N // 2 + 1  # rfft output length
    return coef[:N_rfft]
```

### Free Energy Components (Sparse)

```python
# Source: Julia sparse_inversion.jl, compute_F_sparse()

def compute_free_energy_sparse(
    N_eff, a_r, b_r, QF, tau_r, l0_r, mu_r, mu0_r, Sigma_r,
    a0, b0, D, z_r, z_idx, p0,
) -> float:
    """Sparse free energy with 7 components.

    Additional components vs rigid:
    log_p_z = sum(log(1 - p0[z_idx]) + z_r[z_idx] * log(p0[z_idx]/(1-p0[z_idx])))
    log_q_z = sum(-(1-z[z_idx])*log(1-z[z_idx]) - z[z_idx]*log(z[z_idx]))

    z_idx: indices where z is in (tol^2, 1) -- "present" connections
    """
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Full bilinear DCM (time domain) | rDCM frequency domain regression | Frassle 2017 | Orders of magnitude faster, scales to hundreds of regions |
| MATLAB-only implementation | Julia + MATLAB implementations | Julia v0.1.0 (2024) | Cleaner code, corrected bugs (e.g., QF*0.5 error) |
| Complex-valued regression | Real/imaginary split regression | Julia v0.2.0+ | "Mathematically correct version" per Julia code comments |
| Single run sparse inversion | Multiple reruns (default 100) | Julia implementation | Better solutions via best-of-N selection |

**Deprecated/outdated:**
- MATLAB tapas rDCM has a known bug where QF is incorrectly multiplied by 0.5 in the sparse variant. The Julia implementation is the corrected version.
- Older MATLAB versions may use complex-valued regression without splitting; the real/imaginary split is the correct approach.

## Julia Implementation Key Constants and Defaults

From direct reading of the Julia source code:

| Parameter | Value | Source File | Purpose |
|-----------|-------|-------------|---------|
| maxIter (rigid) | 500 | constructors.jl | Max VB iterations |
| maxIter (sparse) | 500 | constructors.jl | Max VB iterations per rerun |
| tol | 1e-5 | constructors.jl | Convergence tolerance |
| reruns (sparse) | 100 | constructors.jl | Number of random restarts |
| restrictInputs | true | constructors.jl | Whether to fix C matrix sparsity |
| a0 (noise shape) | 2.0 | get_priors.jl | Gamma prior shape |
| b0 (noise rate) | 1.0 | get_priors.jl | Gamma prior rate |
| prior A scaling | 8 | get_priors.jl | Off-diagonal covariance factor |
| prior A self-mean | -0.5 | get_priors.jl | Self-connection prior mean |
| p0 (sparse prior) | 0.15 (typical) | test files | Bernoulli prior on connection presence |
| FIXEDSEED | 42 | main module | Reproducibility seed |
| H (hemodynamics) | [0.64, 0.32, 2.0, 0.32, 0.32] | dcm_euler_integration.jl | [kappa, gamma, tau, alpha, rho] |

## Design Decisions for Implementation

Based on the CONTEXT.md constraints and Julia reference analysis:

### 1. Module Organization
- **`rdcm_forward.py`**: Contains `create_regressors()`, `get_hrf()`, `reduce_zeros()`, and derivative coefficient computation. These transform time-domain BOLD + stimulus into frequency-domain design matrix and data.
- **`rdcm_posterior.py`**: Contains `rigid_inversion()`, `sparse_inversion()`, `compute_free_energy_rigid()`, `compute_free_energy_sparse()`, `update_posterior_rigid()`, `update_posterior_sparse()`, and prior computation functions.
- **`rdcm_simulator.py`**: Contains `simulate_rdcm()` which generates synthetic BOLD, applies the rDCM regressor pipeline, and returns frequency-domain data for testing.

### 2. Tensor vs NumPy
Use PyTorch tensors throughout (not numpy). Even though rDCM is closed-form and does not require autograd, keeping everything in PyTorch ensures:
- Consistent API with Phases 1-2
- Future Pyro integration (Phase 4) without conversion
- GPU acceleration for large networks (hundreds of regions)
- `torch.special.digamma` and `torch.lgamma` are differentiable

### 3. dtype
Use `torch.float64` for all real-valued computations and `torch.complex128` for frequency-domain operations, consistent with Phases 1-2.

### 4. Euler Integration for HRF and BOLD Generation
The Phase 1 `CoupledDCMSystem` + `integrate_ode` pipeline can potentially be reused for HRF computation and BOLD generation. However, the Julia implementation uses a simple Euler integrator (not adaptive), so for exact matching, a dedicated Euler integrator matching Julia's state update order may be needed. Evaluate whether Phase 1's `integrate_ode(..., method='euler')` produces identical results. If not, implement a standalone Euler stepper matching the Julia code.

### 5. Existing Phase 1 Reuse
The existing `task_simulator.py` generates BOLD via ODE integration. The rDCM simulator needs a similar pipeline but with Julia-compatible Euler integration and the specific HRF generation method. Consider whether to reuse or create a dedicated rDCM-specific BOLD generator.

## Open Questions

1. **Euler integration compatibility with Phase 1**
   - What we know: Julia uses a simple forward Euler integrator with specific state update ordering (x, s, f, v, q updated sequentially per region per timestep). Phase 1 uses `torchdiffeq.odeint` with various solvers.
   - What's unclear: Whether `torchdiffeq.odeint(..., method='euler')` with the same step size produces identical results to the Julia Euler integrator, given that torchdiffeq's Euler may update all states simultaneously rather than sequentially.
   - Recommendation: Write a dedicated rDCM Euler integrator matching the Julia code exactly for BOLD generation and HRF computation. This ensures test values match. The Phase 1 adaptive integrators can still be used for comparison/validation.

2. **Confound regressor handling**
   - What we know: Julia adds a constant confound regressor (column of ones). The confound is included in the design matrix and its weight is estimated but trimmed from the final output.
   - What's unclear: The confound code in Julia is partially commented out, suggesting it may be in flux.
   - Recommendation: Implement a configurable confound system. Default to a constant confound for task data. Support additional confounds (motion parameters, etc.) as optional input.

3. **Bilinear/nonlinear DCM support**
   - What we know: The Julia implementation supports Linear, BiLinear, and NonLinear DCM for BOLD generation, but rDCM inversion itself only supports linear models.
   - What's unclear: Whether Phase 3 should support bilinear B matrices in the design matrix.
   - Recommendation: Implement only linear rDCM (A + C) for Phase 3, matching the inversion capability. Note in docstrings that bilinear extension exists but is not implemented.

4. **Test reference values from Julia**
   - What we know: The Julia test files contain exact reference values for posterior means, covariances, free energies, and z indicators that can be used for cross-validation.
   - What's unclear: The Julia tests use `load_example_DCM()` which loads a MATLAB .mat file artifact. We would need to either recreate this data or find the artifact.
   - Recommendation: Create self-contained test cases with known small networks (3-5 regions) where reference values are computed analytically or verified against Julia. For larger validation, attempt to load the Julia example DCM artifact.

## Sources

### Primary (HIGH confidence)
- Julia `RegressionDynamicCausalModeling.jl` source code (v0.2.1) -- authoritative reference, all key files read directly via GitHub API:
  - `src/create_regressors.jl` -- design matrix construction
  - `src/rigid_inversion.jl` -- rigid VB update equations and free energy
  - `src/sparse_inversion.jl` -- sparse VB with binary indicators
  - `src/get_priors.jl` -- prior specifications for rigid and sparse
  - `src/generate_BOLD.jl` -- BOLD simulation and HRF generation
  - `src/utils/dcm_euler_integration.jl` -- Euler integration of DCM
  - `src/structs.jl` -- type hierarchy and data structures
  - `src/constructors.jl` -- default parameter values
  - `test/test_model_inversion.jl` -- reference values for rigid and sparse inversion
  - `test/test_regressors.jl` -- reference values for regressor creation
  - `test/test_simulate_dcm.jl` -- reference values for BOLD generation

### Secondary (MEDIUM confidence)
- MATLAB tapas/rDCM toolbox (https://github.com/translationalneuromodeling/tapas/tree/master/rDCM) -- 26 MATLAB files, cross-referenced with Julia for algorithm verification
  - `tapas_rdcm_ridge.m` -- rigid inversion (MATLAB version)
  - `tapas_rdcm_sparse.m` -- sparse inversion (MATLAB version)
  - `tapas_rdcm_create_regressors.m` -- regressor creation (MATLAB version)
  - `tapas_rdcm_get_prior.m` -- prior specification
- Frassle et al. (2017) "Regression DCM for fMRI" NeuroImage 155:406-421 [REF-020] -- original paper defining rDCM equations
- Frassle et al. (2018) -- sparse rDCM extension with ARD/binary indicators

### Tertiary (LOW confidence)
- Frassle et al. (2021) "Regression dynamic causal modeling for resting-state fMRI" (PMC8046067) -- resting-state extension, equations partially available

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- PyTorch FFT, linalg, and special functions verified from torch documentation and existing Phase 1-2 code
- Architecture: HIGH -- direct reading of Julia source code, line-by-line analysis of all key functions
- Pitfalls: HIGH -- identified from Julia/MATLAB code comparison (e.g., QF*0.5 bug), edge cases in FFT handling, and convergence criteria differences
- Mathematical formulas: HIGH -- extracted directly from Julia implementation which implements [REF-020] equations
- Prior specifications: HIGH -- exact values read from Julia `get_priors.jl`
- Test reference values: HIGH -- exact numerical values available from Julia test files

**Research date:** 2026-03-26
**Valid until:** 2026-04-26 (stable domain, well-established mathematics, Julia v0.2.1 is current)
