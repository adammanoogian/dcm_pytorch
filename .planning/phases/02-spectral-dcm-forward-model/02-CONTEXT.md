# Phase 2: Spectral DCM Forward Model + CSD Computation + spDCM Simulator - Context

**Gathered:** 2026-03-26
**Status:** Ready for planning

<domain>
## Phase Boundary

Build the spectral DCM observation pipeline: cross-spectral density (CSD) computation from time series, the spectral transfer function H(w) mapping neural dynamics to predicted CSD, endogenous fluctuation and observation noise spectral models, and a simulator for synthetic CSD data. No probabilistic inference — that's Phase 4.

</domain>

<decisions>
## Implementation Decisions

### CSD Normalization Convention
- One-sided CSD (positive frequencies only) with power spectral density normalization (per Hz), matching SPM's spm_csd_mtf output format exactly
- Frequency range: configurable with SPM defaults [0.01, 0.5] Hz — user can override for different TR/acquisitions
- This is mandatory for Phase 6 SPM cross-validation

### Complex Tensor Representation
- Use PyTorch native complex tensors (torch.complex128) throughout the spectral DCM modules
- CSD shape: (F, N, N) complex — natural matrix form matching SPM convention
- Convert to real representation (stacked real+imag) only at the Pyro model boundary in Phase 4
- All matrix operations (inversion, multiplication) use native complex math

### Noise Spectral Models
- Match SPM12 convention: 2 params per region for neuronal fluctuations (amplitude + exponent), 1 param per channel for observation noise — total 3N free noise parameters
- Endogenous neuronal fluctuations: match SPM12's spm_csd_fmri_mtf parameterization (includes Lorentzian-like terms, not pure 1/f^alpha)
- These noise model choices match what Phase 6 validation will compare against

### General Principle
- **When in doubt, follow SPM convention.** Same principle as Phase 1. Match spm_csd_mtf.m and spm_dcm_csd.m behavior.

### Claude's Discretion
- Welch vs multi-taper CSD implementation details — pick what best matches SPM output
- Internal frequency grid resolution — match SPM's default binning
- Test tolerance thresholds for CSD comparison

</decisions>

<specifics>
## Specific Ideas

- Reference implementation: SPM12 spm_csd_mtf.m (predicted CSD), spm_csd_fmri_mtf.m (noise model)
- Every function must cite [REF-010] (Friston et al. resting-state DCM) and equation number from .planning/REFERENCES.md
- Transfer function: H(w) = (iwI - A)^-1 per [REF-010] Eq. 3
- Predicted CSD: S(w) = H(w) Sigma_neuronal(w) H(w)^H + Sigma_observation(w) per [REF-010] Eq. 4
- CSD computation builds on Phase 1 infrastructure but operates in frequency domain

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 02-spectral-dcm-forward-model*
*Context gathered: 2026-03-26*
