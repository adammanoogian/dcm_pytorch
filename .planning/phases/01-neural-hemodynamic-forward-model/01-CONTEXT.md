# Phase 1: Neural & Hemodynamic Forward Model + Task-DCM Simulator - Context

**Gathered:** 2026-03-25
**Status:** Ready for planning

<domain>
## Phase Boundary

Build the complete task-based DCM forward model pipeline: neural state equation (dx/dt = Ax + Cu), Balloon-Windkessel hemodynamic ODE, BOLD signal observation equation, ODE integration via torchdiffeq, and a simulator that generates realistic synthetic BOLD time series. No probabilistic inference yet — that's Phase 4.

</domain>

<decisions>
## Implementation Decisions

### A Matrix Stability & Parameterization
- SPM convention: A = A_free - (1/2*tau)*I, where A_free is unconstrained and diagonal decay ensures stability (negative real eigenvalues)
- Prior on A_free off-diagonals: N(0, 1/64) matching SPM12 spm_dcm_fmri_priors.m
- All other prior conventions (hemodynamic params, C matrix, noise) follow SPM12 defaults

### Module Organization
- Coupled ODE system: neural state + Balloon-Windkessel solved as one system [x; s; f; v; q], matching SPM's spm_fx_fmri.m approach
- But code organized as separate composable modules: neural_state.py, balloon_model.py, bold_signal.py
- Each module defines its own state derivatives; a combiner assembles the full ODE right-hand side
- Each module testable in isolation before integration
- bold_signal.py is purely algebraic (not an ODE), applied as observation function after integration

### General Principle
- **When in doubt, follow SPM convention.** Every implementation choice defaults to matching SPM12 behavior unless there's an explicit reason to diverge.
- This includes: parameter defaults, prior specifications, scaling constants, integration settings

### Claude's Discretion
- ODE solver choice within torchdiffeq (RK4 vs Dopri5 vs adjoint) — pick what's numerically stable
- Simulator SNR levels and default network configurations — follow typical fMRI literature values
- Internal numerical safeguards (log-transforms, clipping) — whatever prevents NaN without distorting results

</decisions>

<specifics>
## Specific Ideas

- Reference implementation: SPM12 spm_fx_fmri.m (neural + hemodynamic), spm_gx_fmri.m (BOLD observation)
- Every function must cite [REF-XXX] and equation number from .planning/REFERENCES.md
- Balloon model parameters: {kappa, gamma, tau, alpha, E0} per Stephan et al. (2007) [REF-002]
- Neural state equation per Friston et al. (2003) [REF-001] Eq. 1
- BOLD signal constants (k1, k2, k3, V0) per [REF-002] Eq. 6

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 01-neural-hemodynamic-forward-model*
*Context gathered: 2026-03-25*
