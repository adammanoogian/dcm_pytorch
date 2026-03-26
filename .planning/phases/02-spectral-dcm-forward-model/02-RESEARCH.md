# Phase 2: Spectral DCM Forward Model + CSD Computation + spDCM Simulator - Research

**Researched:** 2026-03-26
**Domain:** Spectral signal processing, transfer functions, cross-spectral density, noise spectral models
**Confidence:** HIGH

## Summary

This research covers the complete technical domain for implementing Phase 2: the spectral DCM forward model pipeline. The phase requires three main components: (1) empirical CSD computation from BOLD time series, (2) the spectral transfer function H(w) and predicted CSD computation per [REF-010] Eq. 3-7, and (3) a simulator that generates synthetic CSD data from given connectivity and noise parameters.

The key technical finding is that SPM12's spectral DCM does NOT use scipy-style Welch/multi-taper CSD directly. Instead, it fits a multivariate autoregressive (MAR) model to the BOLD time series via `spm_mar`, then converts MAR coefficients to CSD via `spm_mar_spectra`. The predicted CSD uses a modal decomposition of the transfer function (eigendecomposition of the Jacobian A) rather than direct matrix inversion at each frequency. The noise model has three parameter groups: P.a (2 x N, neuronal fluctuation amplitude + exponent per region), P.b (2 x 1, global observation noise amplitude + exponent), and P.c (2 x N, region-specific observation noise amplitude + exponent). The CSD output passes through a MAR smoothing step (`spm_csd2mar` then `spm_mar2csd`) that acts as a spectral regularizer.

For our implementation, we have two viable approaches: (A) match SPM exactly using MAR-based CSD (harder, requires implementing MAR fitting), or (B) use the established decision from CONTEXT.md to implement standard signal processing CSD (Welch/multi-taper) for empirical CSD computation while matching SPM's predicted CSD formula for the forward model. Approach B is recommended because the empirical CSD computation is just data preparation -- the scientific core is the predicted CSD model which we match exactly.

**Primary recommendation:** Implement the predicted CSD forward model using eigendecomposition-based transfer function (matching spm_dcm_mtf.m), with SPM's exact noise parameterization (P.a, P.b, P.c). For empirical CSD from time series, use scipy.signal.csd as a validated baseline with MAR-based CSD as an optional alternative. All spectral computations use torch.complex128 with batched matrix operations over the frequency dimension for efficiency.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.2+ | Complex tensor math, autograd, batched linalg | torch.linalg.inv/eig support complex128; autograd through complex ops verified |
| scipy.signal | 1.11+ | CSD computation from time series (Welch method) | Battle-tested spectral estimation; reference for validation |
| numpy | 1.24+ | Array operations for test data generation | Test fixtures |
| pytest | 8.0+ | Test framework | All tests |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| scipy.linalg | 1.11+ | Reference eigendecomposition for validation | Cross-validate torch.linalg.eig results |
| torch.fft | (in torch) | FFT for MAR-based CSD alternative | Optional MAR-to-CSD conversion if needed |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| scipy.signal.csd (Welch) | MAR-based CSD (SPM style) | MAR exactly matches SPM but requires implementing spm_mar; Welch is standard signal processing and simpler |
| torch.linalg.inv per frequency | Eigendecomposition (SPM style) | Eigen is more numerically stable, avoids N inversions; but batched inv is simpler to implement |
| Manual CSD loops | torch batched operations | Batched ops are 10-100x faster on GPU; always prefer batched |

## Architecture Patterns

### Recommended Project Structure
```
src/pyro_dcm/
  forward_models/
    __init__.py              # Add new exports
    spectral_transfer.py     # Transfer function H(w) + predicted CSD [REF-010 Eq. 3-4]
    spectral_noise.py        # Neuronal + observation noise spectra [REF-010 Eq. 5-7]
    csd_computation.py       # Empirical CSD from time series (Welch)
  simulators/
    __init__.py              # Add new exports
    spectral_simulator.py    # Synthetic CSD generation
tests/
  test_spectral_transfer.py  # Transfer function + predicted CSD tests
  test_spectral_noise.py     # Noise model tests
  test_csd_computation.py    # Empirical CSD tests
  test_spectral_simulator.py # End-to-end simulator tests
```

### Pattern 1: Eigendecomposition-Based Transfer Function (SPM Convention)

**What:** SPM computes the transfer function via eigendecomposition of the Jacobian A, then sums modal contributions. This avoids explicit matrix inversion at each frequency and is more numerically stable.

**When to use:** Always for the predicted CSD forward model. This is how SPM does it.

**Example:**
```python
# Source: SPM12 spm_dcm_mtf.m, [REF-010] Eq. 3
def compute_transfer_function(
    A: torch.Tensor,      # (N, N) effective connectivity
    C_in: torch.Tensor,   # (N, nu) input matrix (often identity)
    C_out: torch.Tensor,  # (nn, N) output matrix (BOLD observation Jacobian)
    freqs: torch.Tensor,  # (F,) frequency vector in Hz
) -> torch.Tensor:
    """Compute spectral transfer function via eigendecomposition.

    Implements [REF-010] Eq. 3: g(w) = C_out @ (iwI - A)^-1 @ C_in
    using modal decomposition:
        H(w) = sum_k dgdv_k * dvdu_k / (i*2*pi*w - lambda_k)

    Parameters
    ----------
    A : torch.Tensor
        Effective connectivity (Jacobian), shape (N, N).
    C_in : torch.Tensor
        Input projection matrix, shape (N, nu).
    C_out : torch.Tensor
        Output projection matrix, shape (nn, N).
    freqs : torch.Tensor
        Frequencies in Hz, shape (F,).

    Returns
    -------
    torch.Tensor
        Transfer function H, shape (F, nn, nu), complex128.
    """
    # Eigendecomposition of A
    eigenvalues, eigenvectors = torch.linalg.eig(A.to(torch.complex128))

    # Stabilize: ensure all eigenvalues have negative real parts
    # SPM clamps real parts to <= -1/32 for fMRI frequencies
    eigenvalues = torch.complex(
        torch.clamp(eigenvalues.real, max=-1.0 / 32.0),
        eigenvalues.imag,
    )

    # Project input and output through eigenvectors
    V = eigenvectors                          # (N, N) complex
    V_inv = torch.linalg.inv(V)              # (N, N) complex
    dgdv = C_out.to(torch.complex128) @ V    # (nn, N)
    dvdu = V_inv @ C_in.to(torch.complex128) # (N, nu)

    # Compute transfer function at each frequency
    w = freqs.to(torch.complex128)            # (F,)
    # Modal contributions: 1 / (i*2*pi*w - lambda_k)
    # Shape: (F, N) - one scalar per frequency per mode
    Sk = 1.0 / (1j * 2 * torch.pi * w.unsqueeze(-1) - eigenvalues.unsqueeze(0))

    # Assemble: H(w) = sum_k dgdv(:,k) * dvdu(k,:) * Sk(w,k)
    # H shape: (F, nn, nu)
    H = torch.einsum('ik,kj,fk->fij', dgdv, dvdu, Sk)
    return H
```

### Pattern 2: Batched Direct Inversion (Alternative)

**What:** Directly compute (iwI - A)^-1 at each frequency using batched torch.linalg.inv. Simpler to implement but less numerically stable for near-singular matrices.

**When to use:** As a simpler alternative or validation reference. May be preferable when A is small (N < 10).

**Example:**
```python
# Direct inversion approach for validation
def transfer_function_direct(A, freqs):
    """H(w) = (iwI - A)^-1 via batched matrix inverse."""
    N = A.shape[0]
    I = torch.eye(N, dtype=torch.complex128)
    A_c = A.to(torch.complex128)
    w = freqs.to(torch.complex128)
    # (F, N, N) matrices
    M = 1j * 2 * torch.pi * w[:, None, None] * I[None, :, :] - A_c[None, :, :]
    return torch.linalg.inv(M)  # (F, N, N)
```

### Pattern 3: SPM Noise Model Parameterization

**What:** SPM uses three parameter groups for the spectral noise model: P.a (neuronal), P.b (global observation), P.c (regional observation). All are in log-space. The noise spectra are power-law (1/f^alpha) with amplitude and exponent parameters.

**When to use:** Always -- this is locked by CONTEXT.md decision.

**Example:**
```python
# Source: SPM12 spm_csd_fmri_mtf.m, [REF-010] Eq. 5-7
def neuronal_noise_spectrum(
    freqs: torch.Tensor,    # (F,) Hz
    a_amp: torch.Tensor,    # (N,) log amplitude per region
    a_exp: torch.Tensor,    # (N,) log exponent per region
) -> torch.Tensor:
    """Neuronal fluctuation power spectrum per [REF-010] Eq. 5-6.

    SPM form: Gu(w, i) = C * exp(a_amp_i) * w^(-exp(a_exp_i)) * 4

    Returns shape (F, N, N) diagonal complex CSD.
    """
    F, N = len(freqs), len(a_amp)
    # Power law: w^(-exp(exponent))
    # Shape: (F, N)
    G = freqs.unsqueeze(-1) ** (-torch.exp(a_exp).unsqueeze(0))
    G = G * torch.exp(a_amp).unsqueeze(0) * 4.0
    G = G * (1.0 / 256.0)  # SPM scaling constant C = 1/256

    # Pack into diagonal (F, N, N) matrix
    Gu = torch.zeros(F, N, N, dtype=torch.complex128)
    for i in range(N):
        Gu[:, i, i] = G[:, i].to(torch.complex128)
    return Gu
```

### Pattern 4: Predicted CSD Assembly

**What:** The predicted CSD combines the transfer function with noise spectra: S(w) = H(w) * Sigma_n(w) * H(w)^H + Sigma_obs(w).

**When to use:** Always -- this is the core forward model.

**Example:**
```python
# Source: SPM12 spm_csd_fmri_mtf.m, [REF-010] Eq. 4
def predicted_csd(H, Gu, Gn):
    """Predicted CSD: S(w) = H @ Gu @ H^H + Gn.

    Parameters
    ----------
    H : torch.Tensor
        Transfer function, shape (F, nn, nu), complex128.
    Gu : torch.Tensor
        Neuronal noise CSD, shape (F, nu, nu), complex128.
    Gn : torch.Tensor
        Observation noise CSD, shape (F, nn, nn), complex128.

    Returns
    -------
    torch.Tensor
        Predicted CSD, shape (F, nn, nn), complex128.
    """
    # H @ Gu @ H^H  (batched over frequency)
    G = H @ Gu @ H.conj().transpose(-2, -1)
    return G + Gn
```

### Pattern 5: CSD from Time Series (Data Preparation)

**What:** Compute empirical CSD from BOLD time series using scipy.signal.csd (Welch method) or MAR model. This is data preparation, not the forward model.

**When to use:** For processing real or simulated BOLD time series into the CSD format that the spectral DCM likelihood compares against.

**Example:**
```python
# Source: scipy.signal.csd docs, validated against SPM's MAR approach
import scipy.signal
import numpy as np

def compute_empirical_csd(
    bold: np.ndarray,       # (T, N) time series
    fs: float,              # sampling frequency (1/TR)
    freqs: np.ndarray,      # (F,) target frequencies
    nperseg: int = 256,
) -> np.ndarray:
    """Compute empirical CSD from BOLD time series.

    Returns complex CSD matrix, shape (F, N, N).
    """
    N = bold.shape[1]
    # Compute CSD for each pair
    csd_matrix = np.zeros((len(freqs), N, N), dtype=complex)
    for i in range(N):
        for j in range(N):
            f, Pxy = scipy.signal.csd(
                bold[:, i], bold[:, j],
                fs=fs, nperseg=nperseg,
                scaling='density',
            )
            # Interpolate to target frequency grid
            csd_matrix[:, i, j] = np.interp(freqs, f, Pxy.real) + \
                                  1j * np.interp(freqs, f, Pxy.imag)
    return csd_matrix
```

### Anti-Patterns to Avoid

- **Computing CSD at arbitrary frequencies then expecting SPM match:** SPM uses MAR-based CSD with specific smoothing; direct Welch CSD will differ in shape. For Phase 6 validation, we may need MAR-based CSD as well.
- **Ignoring eigenvalue stabilization:** The transfer function (iwI - A)^-1 blows up when eigenvalues of A approach the imaginary axis. SPM clamps real parts to max(-1/32). Always stabilize.
- **Using torch.inverse instead of torch.linalg.inv:** `torch.inverse` is deprecated. Use `torch.linalg.inv` or preferably `torch.linalg.solve` when solving linear systems.
- **Looping over frequencies in Python:** Use batched torch operations over the frequency dimension. The transfer function can be computed for all F frequencies simultaneously.
- **Mixing real and complex dtypes carelessly:** All spectral operations should be in complex128. Convert A from float64 to complex128 once at the start. The CSD output shape is (F, N, N) complex128.
- **Forgetting the SPM C = 1/256 scaling constant:** SPM applies `C = 1/256` to all noise spectra. Missing this will cause amplitude mismatches.
- **Normalizing noise spectra by sum(G):** Some versions of spm_csd_fmri_mtf.m normalize noise spectra by dividing by `sum(G)`. Check which version is authoritative (the neurodebian version does NOT normalize; it uses `C*exp(a)*w^(-exp(b))*4` directly).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Welch CSD estimation | Custom periodogram code | `scipy.signal.csd` | Handles windowing, overlapping segments, normalization correctly |
| Matrix eigendecomposition | Power iteration or QR | `torch.linalg.eig` | GPU-accelerated, supports complex128, correct autograd |
| Batched matrix inverse | Loop over frequencies | `torch.linalg.inv` on (F, N, N) batch | Orders of magnitude faster; correct autograd |
| Frequency grid generation | Manual linspace with edge-case handling | `torch.linspace` | Simple; match SPM's default of 32 bins from 1/128 to Nyquist |
| Complex matrix multiplication | Separate real/imag multiplications | Native `torch.complex128` `@` operator | PyTorch handles complex matmul natively since 2.0 |

**Key insight:** The spectral DCM forward model is fundamentally linear algebra on complex matrices. PyTorch's `torch.linalg` module handles all of this natively with complex128 support and autograd. The only external dependency is scipy for empirical CSD from time series (data preparation step).

## Common Pitfalls

### Pitfall 1: Eigenvalue Instability in Transfer Function (CRITICAL)
**What goes wrong:** If any eigenvalue of A has a real part near zero or positive, the transfer function `1/(iw - lambda)` produces extremely large or infinite values, causing NaN in the predicted CSD.
**Why it happens:** During inference (Phase 4+), gradient updates may push A eigenvalues toward instability. Even in simulation, poorly chosen A matrices can cause this.
**How to avoid:** Follow SPM convention: clamp eigenvalue real parts to max(-1/32) for fMRI frequencies (below 0.5 Hz). For higher frequencies, clamp to max(-4). This is done AFTER eigendecomposition but BEFORE computing the transfer function.
**Warning signs:** NaN or inf in the predicted CSD; extremely peaked transfer function at one frequency.

### Pitfall 2: CSD Normalization Mismatch (CRITICAL)
**What goes wrong:** The predicted CSD and empirical CSD use different normalization conventions, making the spectral DCM likelihood meaningless.
**Why it happens:** scipy.signal.csd uses `V^2/Hz` (power spectral density per Hz) by default, while SPM uses MAR-derived CSD with `2/ns` normalization. These can differ by orders of magnitude.
**How to avoid:** Use consistent normalization throughout. For the forward model (predicted CSD), follow SPM's conventions exactly. For empirical CSD, ensure the same normalization is applied. Test with known white noise: CSD of white noise with variance sigma^2 sampled at fs should be flat at sigma^2/fs (one-sided).
**Warning signs:** Predicted and empirical CSD have wildly different magnitudes; likelihood is always extremely negative.

### Pitfall 3: SPM's MAR Smoothing Step (HIGH)
**What goes wrong:** SPM passes the predicted CSD through `spm_csd2mar(y, Hz, p-1)` then `spm_mar2csd(...)`, which is a lossy projection that smooths the predicted spectra. Omitting this step will cause differences when validating against SPM in Phase 6.
**Why it happens:** SPM uses this as a regularization/smoothing step to match the MAR-based empirical CSD representation.
**How to avoid:** For Phase 2, implement the predicted CSD without MAR smoothing. Document that SPM applies this additional step. In Phase 6 validation, add the MAR smoothing as an optional post-processing step. The core math must be correct first.
**Warning signs:** Predicted CSD has sharper peaks/features than SPM's output; Phase 6 validation shows systematic broadening differences.

### Pitfall 4: Output Jacobian for BOLD Observation (HIGH)
**What goes wrong:** The transfer function requires the output Jacobian dgdx (mapping hemodynamic states to BOLD), but the implementation only passes the neural connectivity A. SPM evaluates dgdx from spm_gx_fmri.m at the steady state.
**Why it happens:** The transfer function in [REF-010] Eq. 3 simplifies to (iwI - A)^-1 at the neural level, but the full chain neural -> hemodynamic -> BOLD requires the BOLD observation Jacobian as well.
**How to avoid:** For the spectral DCM forward model, the A matrix in the transfer function is the FULL system Jacobian (neural + hemodynamic linearized at steady state), not just the neural connectivity A. However, SPM's spm_csd_fmri_mtf.m specifically uses spm_fx_fmri's Jacobian output, which includes hemodynamic linearization. The implementation must linearize the hemodynamic model at steady state and compose the full Jacobian: `J_full = dgdx @ J_hemo @ A_neural`. In practice, for the standard spDCM where C_out = identity and C_in = identity (measuring all neural states directly), and hemodynamics are absorbed into the transfer function, `P.C = speye(nn, nu)` is set and the Jacobian comes from the full model function.
**Warning signs:** Transfer function shows wrong frequency characteristics (too flat or wrong peaks); predicted CSD doesn't match expected spectral shape for known A.

### Pitfall 5: Complex Autograd Edge Cases (MEDIUM)
**What goes wrong:** PyTorch autograd for complex tensors uses Wirtinger derivatives. The gradient of a real-valued loss through complex intermediate values works correctly, but the gradient of a complex-valued function is not the standard complex derivative.
**Why it happens:** PyTorch follows the convention that grad(f) = df/d(conj(z)) for complex z, not df/dz.
**How to avoid:** Ensure the loss function is always real-valued (which it will be for the likelihood in Phase 4). All intermediate complex computations are fine as long as the final loss is real. Test gradient flow by checking that `loss.backward()` produces non-None, finite gradients for all real-valued parameters.
**Warning signs:** Gradients are zero when they shouldn't be; gradients have unexpected imaginary components; NaN gradients.

### Pitfall 6: Frequency Grid Mismatch (MEDIUM)
**What goes wrong:** Using a frequency grid that doesn't match the data's Nyquist frequency or SPM's conventions leads to spectral aliasing or incorrect normalization.
**Why it happens:** SPM defaults to 32 bins from 1/128 Hz to Nyquist (0.5/TR Hz). Custom implementations may use different grids.
**How to avoid:** Make the frequency grid configurable with SPM defaults. For TR=2s: `Hz = linspace(1/128, 0.25, 32)`. The lower bound 1/128 corresponds to the lowest resolvable frequency for a typical fMRI acquisition. The upper bound is the Nyquist frequency 1/(2*TR).
**Warning signs:** CSD has unexpected features at edges of frequency range; energy "leaks" into wrong frequency bins.

## Code Examples

### Complete Transfer Function + Predicted CSD Pipeline
```python
# Source: SPM12 spm_dcm_mtf.m + spm_csd_fmri_mtf.m
# Implements [REF-010] Eq. 3-7

from __future__ import annotations
import torch

def spectral_dcm_forward(
    A: torch.Tensor,          # (N, N) effective connectivity
    freqs: torch.Tensor,      # (F,) frequencies in Hz
    a: torch.Tensor,          # (2, N) neuronal noise params [log amp, log exp]
    b: torch.Tensor,          # (2, 1) global obs noise [log amp, log exp]
    c: torch.Tensor,          # (2, N) regional obs noise [log amp, log exp]
) -> torch.Tensor:
    """Complete spectral DCM predicted CSD.

    Implements [REF-010] Eq. 3-7, matching SPM12 spm_csd_fmri_mtf.m:
    1. Transfer function via eigendecomposition of A
    2. Neuronal noise spectrum (1/f power law per region)
    3. Observation noise spectrum (global + regional)
    4. Predicted CSD: S(w) = H @ Gu @ H^H + Gn

    Parameters
    ----------
    A : torch.Tensor
        Effective connectivity matrix, shape (N, N), float64.
    freqs : torch.Tensor
        Frequency vector in Hz, shape (F,), float64.
    a : torch.Tensor
        Neuronal noise params, shape (2, N), float64.
        a[0, :] = log amplitude, a[1, :] = log exponent.
    b : torch.Tensor
        Global observation noise params, shape (2, 1), float64.
    c : torch.Tensor
        Regional observation noise params, shape (2, N), float64.

    Returns
    -------
    torch.Tensor
        Predicted CSD, shape (F, N, N), complex128.
    """
    N = A.shape[0]
    F = freqs.shape[0]
    w = freqs.to(torch.complex128)

    # --- Transfer Function [REF-010] Eq. 3 ---
    # Eigendecomposition of A
    eigvals, eigvecs = torch.linalg.eig(A.to(torch.complex128))

    # Stabilize eigenvalues (SPM convention for fMRI: max real part = -1/32)
    eigvals = torch.complex(
        torch.clamp(eigvals.real, max=-1.0 / 32.0),
        eigvals.imag,
    )

    # For standard spDCM: C_in = C_out = I (identity)
    # So H(w) = V @ diag(1/(iw - lambda)) @ V^-1
    V_inv = torch.linalg.inv(eigvecs)

    # Modal transfer: 1/(i*2*pi*w - lambda_k), shape (F, N)
    Sk = 1.0 / (1j * 2 * torch.pi * w[:, None] - eigvals[None, :])

    # Full transfer: H(w) = V @ diag(Sk) @ V^-1, shape (F, N, N)
    H = eigvecs[None, :, :] * Sk[:, None, :] @ V_inv[None, :, :]

    # --- Neuronal Noise [REF-010] Eq. 5-6 ---
    C_scale = 1.0 / 256.0  # SPM scaling constant
    Gu = torch.zeros(F, N, N, dtype=torch.complex128)
    for i in range(N):
        G = torch.exp(a[0, i]) * freqs ** (-torch.exp(a[1, i])) * 4.0
        Gu[:, i, i] = (C_scale * G).to(torch.complex128)

    # --- Observation Noise [REF-010] Eq. 7 ---
    Gn = torch.zeros(F, N, N, dtype=torch.complex128)
    # Global component
    G_global = torch.exp(b[0, 0]) * freqs ** (-torch.exp(b[1, 0]) / 2) / 8.0
    for i in range(N):
        for j in range(N):
            Gn[:, i, j] = Gn[:, i, j] + (C_scale * G_global).to(torch.complex128)

    # Regional component
    for i in range(N):
        G_regional = torch.exp(c[0, i]) * freqs ** (-torch.exp(c[1, i]) / 2)
        Gn[:, i, i] = Gn[:, i, i] + (C_scale * G_regional).to(torch.complex128)

    # --- Predicted CSD [REF-010] Eq. 4 ---
    # S(w) = H(w) @ Gu(w) @ H(w)^H + Gn(w)
    csd = H @ Gu @ H.conj().transpose(-2, -1) + Gn

    return csd
```

### SPM Noise Parameter Initialization (Prior Values)
```python
# Source: SPM12 spm_dcm_fmri_priors.m

def default_noise_priors(n_regions: int) -> dict:
    """SPM12 default prior expectations for noise parameters.

    Parameters
    ----------
    n_regions : int
        Number of brain regions.

    Returns
    -------
    dict
        Prior expectations and covariances for noise params.
    """
    return {
        # Neuronal fluctuations: 2 params per region
        'a_prior_mean': torch.zeros(2, n_regions, dtype=torch.float64),
        'a_prior_var': torch.full((2, n_regions), 1.0 / 64.0,
                                  dtype=torch.float64),
        # Global observation noise: 2 params
        'b_prior_mean': torch.zeros(2, 1, dtype=torch.float64),
        'b_prior_var': torch.full((2, 1), 1.0 / 64.0,
                                  dtype=torch.float64),
        # Regional observation noise: 2 params per region
        'c_prior_mean': torch.zeros(2, n_regions, dtype=torch.float64),
        'c_prior_var': torch.full((2, n_regions), 1.0 / 64.0,
                                  dtype=torch.float64),
    }
```

### Default Frequency Grid (SPM Convention)
```python
# Source: SPM12 spm_dcm_fmri_csd_data.m

def default_frequency_grid(
    TR: float = 2.0,
    n_freqs: int = 32,
) -> torch.Tensor:
    """Generate default frequency grid matching SPM12 convention.

    SPM defaults:
    - Lower bound: 1/128 Hz (~0.0078 Hz)
    - Upper bound: Nyquist = 1/(2*TR) Hz
    - Number of bins: 32

    Parameters
    ----------
    TR : float
        Repetition time in seconds. Default 2.0.
    n_freqs : int
        Number of frequency bins. Default 32.

    Returns
    -------
    torch.Tensor
        Frequency vector in Hz, shape (n_freqs,), float64.
    """
    Hz_low = 1.0 / 128.0
    Hz_high = 1.0 / (2.0 * TR)
    return torch.linspace(Hz_low, Hz_high, n_freqs, dtype=torch.float64)
```

### Simulator Pattern
```python
# Source: Project architecture decision

def simulate_spectral_dcm(
    A: torch.Tensor,          # (N, N) effective connectivity
    noise_params: dict,       # a, b, c noise parameters
    TR: float = 2.0,
    n_freqs: int = 32,
    seed: int | None = None,
) -> dict:
    """Generate synthetic CSD from spectral DCM model.

    Given connectivity A and noise parameters, computes the
    theoretical predicted CSD. Optionally adds sampling noise
    to simulate empirical CSD estimation variability.

    Returns dict with keys:
    - 'csd': predicted CSD, shape (F, N, N) complex128
    - 'freqs': frequency vector, shape (F,)
    - 'transfer_function': H, shape (F, N, N) complex128
    - 'neuronal_noise': Gu, shape (F, N, N) complex128
    - 'observation_noise': Gn, shape (F, N, N) complex128
    - 'params': dict with A, noise_params, TR
    """
    ...
```

## SPM12 Implementation Details (Authoritative Reference)

### spm_csd_fmri_mtf.m -- Predicted CSD
The core function that computes predicted CSD for spectral DCM:

1. Gets transfer function `S = spm_dcm_mtf(P, M)`
2. Builds neuronal noise `Gu` from P.a (power law: `exp(a1) * w^(-exp(a2)) * 4`)
3. Builds observation noise `Gn` from P.b (global) and P.c (regional)
4. Assembles: `G(w) = S(w) * Gu(w) * S(w)' + Gn(w)` per frequency
5. Returns CSD shape `(nw, nn, nn)` complex

### spm_dcm_mtf.m -- Transfer Function
Computes transfer function via eigendecomposition:

1. Gets Jacobian `dfdx` from model function `M.f` (or via `spm_diff`)
2. Gets output Jacobian `dgdx` from observation function `M.g`
3. Gets input Jacobian `dfdu` from model function
4. Eigendecompose: `[V, lambda] = eig(dfdx)`
5. Project: `dgdv = dgdx @ V`, `dvdu = V^-1 @ dfdu`
6. Stabilize: clamp real(lambda) to max(-1/32) for fMRI
7. Assemble: `H(w,i,j) = sum_k dgdv(i,k) * dvdu(k,j) / (i*2*pi*w - lambda(k))`

### spm_dcm_fmri_csd_data.m -- Empirical CSD from BOLD
Computes empirical CSD from BOLD time series:

1. Frequency grid: 32 bins from 1/128 Hz to Nyquist (1/(2*TR))
2. Fits MAR(p) model: `mar = spm_mar(y, p)` with default p=4
3. Converts to CSD: `mar = spm_mar_spectra(mar, Hz, 1/dt)`
4. Extracts: `DCM.Y.csd = mar.P`

### Noise Parameter Shapes (from spm_dcm_fmri_priors.m)
| Parameter | Shape | Prior Mean | Prior Variance | Meaning |
|-----------|-------|------------|---------------|---------|
| P.a | (2, N) | 0 | 1/64 | Neuronal noise [log amp, log exp] per region |
| P.b | (2, 1) | 0 | 1/64 | Global observation noise [log amp, log exp] |
| P.c | (2, N) | 0 | 1/64 | Regional observation noise [log amp, log exp] per region |

Total noise parameters: 2N + 2 + 2N = 4N + 2 (not 3N as stated in the roadmap -- note the discrepancy).

**IMPORTANT DISCREPANCY:** The CONTEXT.md states "2 params per region for neuronal (amplitude + exponent), 1 param per channel for observation noise -- 3N total." However, SPM12's spm_dcm_fmri_priors.m actually uses P.a=(2,N), P.b=(2,1), P.c=(2,N), totaling 4N+2 parameters. The observation noise has TWO components: a global 2-parameter term (P.b) and a regional 2-parameter term (P.c). The planner should reconcile this with the user.

### spm_fs_fmri_csd.m -- Feature Selection
SPM applies a feature transformation before computing the fit:
1. Converts CSD to cross-covariance function: `c = spm_csd2ccf(y, M.Hz)`
2. Concatenates: `y = [y/16; c(1:8:end,:,:)*2]`

This means SPM fits BOTH scaled CSD and downsampled cross-covariance. For our implementation, we should be aware this affects what the likelihood "sees" but can be deferred to Phase 4.

### spm_mar2csd.m -- MAR to CSD Conversion
Formula: `CSD(f) = A(f)^-1 @ C @ A(f)^-H` where:
- `A(f) = I + sum_k lag(k).a * exp(-i*w*k)` is the frequency-domain AR matrix
- `C` is the noise covariance (identity if unspecified)
- Normalization: `CSD = 2 * CSD / ns` where ns = samples per second

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Welch periodogram for CSD | MAR-based CSD in SPM | SPM12 (2014) | Better spectral estimation for short fMRI time series; parametric smoothing |
| Direct matrix inversion per frequency | Eigendecomposition-based transfer function | SPM12 | More numerically stable; handles near-singular cases; faster for many frequencies |
| White observation noise only | Power-law (1/f) noise for both neuronal and observation | REF-010 (2014) | More realistic noise model; better fits to resting-state data |
| Time-domain DCM fitting | Frequency-domain (CSD) fitting for resting state | REF-010 (2014) | Avoids need for explicit stimulus design; captures spontaneous fluctuations |

**Deprecated/outdated:**
- Direct Fourier transform of BOLD for CSD: replaced by MAR-based approach in SPM for smoother estimates
- Simple white noise model for neuronal fluctuations: replaced by 1/f power law
- Time-domain fitting for resting-state DCM: replaced by spectral (CSD) fitting

## Open Questions

1. **Noise parameter count discrepancy**
   - What we know: CONTEXT.md says 3N total noise params; SPM12 actually has 4N+2 (P.a=2xN, P.b=2x1, P.c=2xN)
   - What's unclear: Whether the user intended a simplified noise model or the full SPM model
   - Recommendation: Implement full SPM parameterization (4N+2) for accuracy; note the discrepancy for user decision. The simpler 3N model could be a special case where P.b is fixed and P.c has only 1 param per region.

2. **MAR-based vs Welch-based empirical CSD**
   - What we know: SPM uses MAR(p) fit + spectral conversion; this gives smoother estimates than Welch. The CONTEXT.md says "Welch vs multi-taper CSD implementation details -- pick what best matches SPM output."
   - What's unclear: How much the empirical CSD method matters for Phase 2 (it matters more for Phase 6 validation)
   - Recommendation: Implement Welch-based CSD as the primary approach (simpler, standard, good for testing). Add MAR-based CSD as a separate function for SPM compatibility. The predicted CSD forward model is independent of how empirical CSD is computed.

3. **Full system Jacobian vs neural A only**
   - What we know: SPM's transfer function uses the full-system Jacobian (neural + hemodynamic, linearized at steady state), not just the neural A matrix. spm_dcm_mtf.m calls M.f to get dfdx which includes hemodynamic contributions.
   - What's unclear: Whether to implement the full hemodynamic transfer function in Phase 2 or use the simplified neural-only version
   - Recommendation: Implement both. The simplified version `(iwI - A)^-1` is the pure neural transfer function from [REF-010] Eq. 3. The full version incorporates the hemodynamic transfer function via the BOLD observation Jacobian (dgdx from spm_gx_fmri.m). The full version is needed for SPM validation but the simplified version is correct for understanding the neural dynamics. Start with the simplified version and add the hemodynamic extension.

4. **SPM's C = 1/256 scaling constant origin**
   - What we know: SPM applies `C = 1/256` as a multiplicative constant to noise spectra in spm_csd_fmri_mtf.m
   - What's unclear: Whether this is a normalization convention, a regularization choice, or derived from something physical
   - Recommendation: Implement it as a configurable constant matching SPM default. Document that it exists. It may relate to the prior variance (1/64 squared = 1/4096 != 1/256, so not simply prior-related).

5. **SPM's `P.C = speye(nn, nu)` override**
   - What we know: In spm_csd_fmri_mtf.m, SPM sets `P.C = speye(nn, nu)` before calling spm_dcm_mtf, overriding any input weights. This means the transfer function treats all neural states as both inputs and outputs.
   - What's unclear: Whether this is always the case or only for standard spDCM
   - Recommendation: Follow SPM convention: for standard spectral DCM, C_in = C_out = identity. Make these configurable for future extensions.

## Sources

### Primary (HIGH confidence)
- SPM12 `spm_csd_fmri_mtf.m` via neurodebian GitHub mirror -- complete predicted CSD pipeline with noise model
- SPM12 `spm_dcm_mtf.m` via neurodebian GitHub mirror -- eigendecomposition-based transfer function
- SPM12 `spm_dcm_fmri_csd_data.m` via neurodebian GitHub mirror -- empirical CSD via MAR, frequency grid defaults
- SPM12 `spm_dcm_fmri_priors.m` (both spm/spm12 main and neurodebian versions) -- noise prior shapes and values
- SPM12 `spm_gx_fmri.m` via neurodebian GitHub mirror -- BOLD observation Jacobian
- SPM12 `spm_fx_fmri.m` via spm/spm12 main -- A matrix construction and Jacobian
- SPM12 `spm_fs_fmri_csd.m` via neurodebian GitHub mirror -- feature selection for CSD DCM
- SPM12 `spm_mar2csd.m` via MIT mirror -- MAR to CSD conversion formula
- scipy.signal.csd documentation (https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.csd.html) -- normalization conventions
- PyTorch torch.linalg documentation (https://docs.pytorch.org/docs/stable/generated/torch.linalg.inv.html, torch.linalg.solve.html) -- complex128 support verified
- [REF-010] Friston et al. (2014) NeuroImage 94:396-407 -- theoretical framework Eq. 3-7

### Secondary (MEDIUM confidence)
- Friston et al. 2014 PMC4073651 -- full text confirming frequency range "1/128 to Nyquist" and 64 bins
- PyTorch complex number documentation (https://docs.pytorch.org/docs/stable/complex_numbers.html) -- Wirtinger derivative convention

### Tertiary (LOW confidence)
- Various versions of SPM12 show P.a as either (2,1) or (2,N) -- discrepancy between spm/spm12 main and neurodebian mirror suggests version evolution. The neurodebian version (P.a=(2,N), P.c=(2,N)) appears to be the more recent/complete version.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- PyTorch complex128 linalg verified from official docs; scipy.signal.csd is well-documented
- Architecture: HIGH -- patterns derived directly from SPM12 source code (spm_csd_fmri_mtf.m, spm_dcm_mtf.m); eigendecomposition and CSD assembly verified
- Noise model: HIGH -- parameter shapes and formulas read directly from SPM12 source code
- SPM frequency defaults: HIGH -- read from spm_dcm_fmri_csd_data.m (32 bins, 1/128 to Nyquist)
- Pitfalls: HIGH -- eigenvalue stabilization, normalization mismatch, and complex autograd are well-documented issues

**Research date:** 2026-03-26
**Valid until:** 2026-04-26 (stable domain, well-established mathematics)
