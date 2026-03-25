---
purpose: "Ground every equation and algorithm to specific published sources"
rule: "NO code is written without a traceable reference. If a function implements
       an equation, the docstring MUST cite [REF-XXX] and the equation number."
updated: "2026-03-24"
---

# References

## How to Use This File

Every module, function, and equation in the codebase must trace to an entry here.
The format is:

```python
def balloon_ode(state, t, params):
    """Balloon-Windkessel hemodynamic model.

    Implements [REF-002] Eq. 2-5 (Stephan et al. 2007).
    See also [REF-001] Eq. 3-6 for original formulation.
    """
```

If you cannot cite a reference for a computation, STOP and find one before coding.

---

## Core DCM Theory

### REF-001: Friston, Harrison & Penny (2003)
**Title:** Dynamic causal modelling
**Journal:** NeuroImage, 19(4), 1273-1302
**DOI:** 10.1016/S1053-8119(03)00202-7
**Used for:**
- Bilinear neural state equation: dx/dt = Ax + Σ_j u_j B^(j) x + Cu (Eq. 1)
- Original DCM framework definition
- Bayesian model inversion via EM/VL (Eq. 6-12)
- Free energy approximation for model comparison (Eq. 13-15)
**Equations needed:** 1 (state equation), 6-15 (inference)

### REF-002: Stephan et al. (2007)
**Title:** Comparing hemodynamic models with DCM
**Journal:** NeuroImage, 38(3), 387-401
**DOI:** 10.1016/j.neuroimage.2007.07.040
**Used for:**
- Balloon-Windkessel model equations (Eq. 2-5):
  - ds/dt = u - κs - γ(f-1)          [vasodilatory signal]
  - df/dt = s                         [blood flow]
  - dτv/dt = f - v^(1/α)             [blood volume]
  - dτq/dt = f E(f,E₀)/E₀ - v^(1/α) q/v  [deoxyhemoglobin]
- BOLD signal equation (Eq. 6): y = V₀[k₁(1-q) + k₂(1-q/v) + k₃(1-v)]
- Parameter priors and physiological ranges (Table 1)
**Equations needed:** 2-6, Table 1 for prior values

### REF-003: Marreiros, Kiebel & Friston (2008)
**Title:** Dynamic causal modelling for fMRI: A two-state model
**Journal:** NeuroImage, 39(1), 269-278
**DOI:** 10.1016/j.neuroimage.2007.08.019
**Used for:**
- Two-state neuronal model extension (inhibitory + excitatory populations)
- Only needed if extending beyond single-state bilinear model
**Equations needed:** 1-4 (two-state dynamics)

---

## Spectral DCM

### REF-010: Friston, Kahan, Biswal & Razi (2014)
**Title:** A DCM for resting state fMRI
**Journal:** NeuroImage, 94, 396-407
**DOI:** 10.1016/j.neuroimage.2013.12.009
**Used for:**
- Spectral DCM formulation for resting-state data
- Transfer function: g(ω,θ) = (iωI - A)⁻¹ (Eq. 3)
- Predicted cross-spectral density: S(ω) = g(ω) Σ_n(ω) g(ω)* + Σ_ε(ω) (Eq. 4)
- Neuronal fluctuation spectrum: 1/f^α power law (Eq. 5-6)
- Observation noise spectrum model (Eq. 7)
- CSD likelihood formulation (Eq. 8-10)
**Equations needed:** 3-10

### REF-011: Razi et al. (2015)
**Title:** Construct validation of a DCM for resting state fMRI
**Journal:** NeuroImage, 106, 1-14
**DOI:** 10.1016/j.neuroimage.2014.11.027
**Used for:**
- Empirical validation of spectral DCM
- Benchmark results for comparison
- Details on CSD estimation from BOLD (multi-taper specifics)

### REF-012: Friston, Bastos, Litvak et al. (2012)
**Title:** DCM for complex-valued data: Cross-spectra, coherence and phase-delays
**Journal:** NeuroImage, 59(1), 439-455
**DOI:** 10.1016/j.neuroimage.2011.07.048
**Used for:**
- General CSD-based DCM framework
- Complex-valued likelihood for spectral data
- Phase-delay modeling
**Equations needed:** 1-8 (CSD generative model)

---

## Regression DCM

### REF-020: Frässle et al. (2017)
**Title:** A generative model of whole-brain effective connectivity
**Journal:** NeuroImage, 145, 270-275
**DOI:** 10.1016/j.neuroimage.2016.11.047
**Used for:**
- Regression DCM formulation (Eq. 1-8):
  - Convolution model in time domain (Eq. 1-2)
  - DFT to frequency domain (Eq. 3-4)
  - Linear regression form per region: y_j = X_j θ_j + ε_j (Eq. 5-6)
  - Analytic posterior: p(θ_j|y_j) = N(μ_j, Σ_j) (Eq. 11-14)
  - Free energy in closed form (Eq. 15)
- ARD priors for sparse connectivity (Eq. 9-10)
**Equations needed:** ALL (1-15)

### REF-021: Frässle et al. (2018)
**Title:** Regression DCM for fMRI
**Journal:** NeuroImage, 155, 406-421
**DOI:** 10.1016/j.neuroimage.2017.02.090
**Used for:**
- Extended rDCM with sparsity constraints
- Variational Bayesian inference details
- Empirical benchmarks on simulated + real data
- Conjugate prior specifications (Eq. 5-12)
**Equations needed:** 5-12 (VB updates), benchmark metrics

### REF-022: Frässle et al. (2021)
**Title:** Whole-brain estimates of directed connectivity for human connectomics
**Journal:** NeuroImage, 225, 117491
**DOI:** 10.1016/j.neuroimage.2020.117491
**Used for:**
- Scalability of rDCM to whole-brain (>200 regions)
- Practical implementation details
- tapas/rDCM toolbox as validation reference

---

## Hemodynamic Modeling

### REF-030: Buxton, Wong & Frank (1998)
**Title:** Dynamics of blood flow and oxygenation changes during brain activation:
         The balloon model
**Journal:** Magnetic Resonance in Medicine, 39(6), 855-864
**DOI:** 10.1002/mrm.1910390602
**Used for:**
- Original Balloon model derivation
- Physiological interpretation of parameters
- Steady-state solutions for validation

### REF-031: Friston et al. (2000)
**Title:** Nonlinear responses in fMRI: The Balloon model, Volterra kernels, and
         other hemodynamics
**Journal:** NeuroImage, 12(4), 466-477
**DOI:** 10.1006/nimg.2000.0630
**Used for:**
- Volterra kernel expansion of hemodynamic response
- Nonlinear BOLD response characterization
- Connection to linear HRF as first-order approximation

---

## Bayesian Inference & Variational Methods

### REF-040: Friston et al. (2007)
**Title:** Variational free energy and the Laplace approximation
**Journal:** NeuroImage, 34(1), 220-234
**DOI:** 10.1016/j.neuroimage.2006.08.035
**Used for:**
- Variational Laplace (VL) algorithm used in SPM
- Free energy formulation: F = ⟨log p(y,θ|m)⟩_q - ⟨log q(θ)⟩_q (Eq. 1-3)
- Laplace approximation to posterior (Eq. 6-10)
- This is what we're replacing with SVI — needed for validation comparison
**Equations needed:** 1-10 (for understanding SPM's approach)

### REF-041: Blei, Kucukelbir & McAuliffe (2017)
**Title:** Variational inference: A review for statisticians
**Journal:** JASA, 112(518), 859-877
**DOI:** 10.1080/01621459.2017.1285773
**Used for:**
- General SVI framework (ELBO, reparameterization trick)
- Mean-field vs structured variational families
- Convergence diagnostics

### REF-042: Papamakarios, Nalisnick, Rezende, Mohamed & Lakshminarayanan (2021)
**Title:** Normalizing flows for probabilistic modeling and inference
**Journal:** JMLR, 22(57), 1-64
**URL:** https://jmlr.org/papers/v22/19-1028.html
**Used for:**
- Normalizing flow theory for amortized guides
- MAF, IAF, Neural Spline Flow architectures
- Density estimation and sampling

### REF-043: Cranmer, Brehmer & Louppe (2020)
**Title:** The frontier of simulation-based inference
**Journal:** PNAS, 117(48), 30055-30062
**DOI:** 10.1073/pnas.1912789117
**Used for:**
- Amortized inference framework (SNPE, SNLE, SNRE)
- Simulation-based inference paradigm
- Connection to our amortized DCM guide approach

---

## Neural Dynamics & Identifiability

### REF-050: Singh et al. (2020) — MINDy
**Title:** Estimation of brain network models with mesoscale individualized
         neurodynamic models (MINDy)
**Journal:** Network Neuroscience, 4(4), 1080-1104
**DOI:** 10.1162/netn_a_00154
**Used for:**
- Closest existing work: neural ODE for connectivity with explicit ROI-aligned states
- Key difference: MINDy lacks full Bayesian inference (point estimates only)
- Architectural reference for neural dynamics modeling

### REF-051: Nozari et al. (2024)
**Title:** Macroscopic resting-state brain dynamics are best described by linear models
**Journal:** Nature Biomedical Engineering, 8, 68-84
**DOI:** 10.1038/s41551-023-01117-y
**Used for:**
- Evidence that linear models suffice for macroscopic BOLD dynamics
- Justification for starting with bilinear (not nonlinear neural ODE) model
- Empirical benchmark: linear vs nonlinear prediction accuracy

### REF-052: Durstewitz et al. (2023)
**Title:** Reconstructing computational system dynamics from neural data with
         recurrent neural networks
**Journal:** Nature Reviews Neuroscience, 24, 693-710
**DOI:** 10.1038/s41583-023-00740-7
**Used for:**
- Rotational degeneracy / identifiability problem in neural state spaces
- Review of state-space model approaches
- Relevance to our v0.2 neural ODE extension

---

## Software References

### REF-060: Pyro
**Title:** Pyro: Deep Universal Probabilistic Programming
**Authors:** Bingham et al.
**Journal:** JMLR, 20(28), 1-6 (2019)
**URL:** https://pyro.ai
**Used for:** Primary probabilistic programming framework

### REF-061: torchdiffeq
**Title:** Neural Ordinary Differential Equations
**Authors:** Chen et al.
**Conference:** NeurIPS 2018
**URL:** https://github.com/rtqichen/torchdiffeq
**Used for:** ODE integration in PyTorch

### REF-062: Zuko
**Title:** Zuko: Normalizing flows in PyTorch
**URL:** https://github.com/probabilists/zuko
**Used for:** Normalizing flow architectures for amortized guides

### REF-063: NumPyro
**Title:** NumPyro
**URL:** https://github.com/pyro-ppl/numpyro
**Used for:** NUTS validation of posteriors (JAX backend)

### REF-064: SPM12
**Title:** Statistical Parametric Mapping
**URL:** https://www.fil.ion.ucl.ac.uk/spm/software/spm12/
**Used for:** Reference implementation for validation (MATLAB)

### REF-065: tapas/rDCM
**Title:** Translational Algorithms for Psychiatry — Advancing Science (TAPAS)
**URL:** https://github.com/translationalneuromodeling/tapas
**Used for:** Reference implementation of regression DCM for validation

---

## Equation Quick-Reference by Module

| Module | Primary Reference | Key Equations |
|--------|------------------|---------------|
| `neural_state.py` | REF-001 | Eq. 1 (bilinear state equation) |
| `balloon_model.py` | REF-002 | Eq. 2-5 (hemodynamic ODEs) |
| `bold_signal.py` | REF-002 | Eq. 6 (BOLD observation) |
| `csd_computation.py` | REF-010 | Eq. 3-4 (transfer function, predicted CSD) |
| `spectral_noise.py` | REF-010 | Eq. 5-7 (neuronal + observation noise spectra) |
| `rdcm_likelihood.py` | REF-020 | Eq. 4-8 (frequency-domain regression) |
| `rdcm_posterior.py` | REF-020 | Eq. 11-15 (analytic posterior, free energy) |
| `pyro_task_dcm.py` | REF-001, REF-040 | Generative model + ELBO |
| `pyro_spectral_dcm.py` | REF-010, REF-040 | Spectral generative model + ELBO |
| `pyro_rdcm.py` | REF-020, REF-040 | rDCM generative model + ELBO |
| `guide_meanfield.py` | REF-041 | Mean-field Gaussian (Laplace baseline) |
| `guide_amortized.py` | REF-042, REF-043 | Normalizing flow amortized guide |
