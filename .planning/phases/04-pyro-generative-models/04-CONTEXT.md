# Phase 4: Pyro Generative Models + Baseline Guides - Context

**Gathered:** 2026-03-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Wire the three DCM forward models (task, spectral, regression) into Pyro generative models with proper plate structure and priors, implement mean-field Gaussian variational guides for each variant, and build an SVI training loop with diagnostics. This is the probabilistic programming integration layer. No parameter recovery testing — that's Phase 5.

</domain>

<decisions>
## Implementation Decisions

### rDCM Pyro Integration — Dual Path
- Keep Phase 3's analytic VB (rigid_inversion, sparse_inversion) as the PRIMARY inference method for rDCM
- Add a Pyro generative model for rDCM that enables:
  - ELBO-based model comparison across all 3 DCM variants (uniform metric)
  - Amortized guide training in Phase 7
- The Pyro rDCM model wraps the frequency-domain likelihood, does NOT replace the analytic VB
- For routine rDCM inference, users call the Phase 3 analytic functions directly

### A Matrix Parameterization in Pyro
- Sample A_free from Normal prior: `A_free = pyro.sample('A_free', dist.Normal(prior_mean, prior_std))`
- Apply deterministic transform: `A = pyro.deterministic('A', parameterize_A(A_free))`
- This reuses the existing parameterize_A function from Phase 1 (diagonal = -exp(free)/2)
- Prior on A_free: N(0, 1/64) matching SPM12 (off-diagonal), N(0, 1/64) for diagonal free params

### Hemodynamic Parameters — Fixed at SPM Defaults
- Do NOT sample hemodynamic parameters (kappa, gamma, tau, alpha, E0) in the Pyro model
- Fix at SPM12 code defaults: kappa=0.64, gamma=0.32, tau=2.0, alpha=0.32, E0=0.40
- This reduces the parameter space to A and C only (+ noise), making Phase 5 recovery tests cleaner
- Hemodynamic parameter inference can be added later if needed

### General Principle
- **SPM convention for all prior specifications** — match SPM12 spm_dcm_fmri_priors.m
- C matrix prior: N(0, 1) for present connections
- Noise precision prior: suitable weakly informative prior

### Claude's Discretion
- Pyro plate structure design (time plate, region plate, frequency plate)
- Mean-field guide implementation (AutoNormal vs manual diagonal Gaussian)
- SVI runner: learning rate schedule, convergence criteria, gradient clipping
- How to handle the ODE integration within Pyro's trace (forward vs adjoint)
- Spectral DCM likelihood formulation (Gaussian on vec(CSD) or other)

</decisions>

<specifics>
## Specific Ideas

- Task DCM Pyro model: sample A_free, C -> parameterize_A -> CoupledDCMSystem -> integrate_ode -> bold_signal -> Gaussian likelihood on BOLD
- Spectral DCM Pyro model: sample A_free, noise params (a, b, c) -> spectral_dcm_forward -> Gaussian likelihood on CSD
- Regression DCM Pyro model: sample theta (A, C columns) per region -> frequency-domain Gaussian likelihood
- SVI runner should track ELBO trace, support multiple particles, and include gradient clipping
- All three models must support structural masking (which A connections exist)

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 04-pyro-generative-models*
*Context gathered: 2026-03-27*
