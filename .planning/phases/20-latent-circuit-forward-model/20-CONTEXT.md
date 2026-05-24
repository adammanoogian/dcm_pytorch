# Phase 20: Direct Observation Forward Model, Simulator & Synthetic Validation - Context

**Gathered:** 2026-05-24
**Status:** Ready for planning

<domain>
## Phase Boundary

Build the DCM model for fitting bilinear dynamics to N-dimensional neural state
trajectories directly (no hemodynamic convolution), validate parameter recovery on
synthetic bilinear ground truth. No RNN, no PCA, no real data — this is the "does the
math work?" phase proving the model recovers known parameters when correctly specified.

</domain>

<decisions>
## Implementation Decisions

### Forward model: extend, don't fork
- Make hemodynamic stage OPTIONAL in existing `CoupledDCMSystem`, not a separate
  `LatentCircuitSystem`. Something like `CoupledDCMSystem(hemodynamic=False)` that
  skips balloon-Windkessel and returns neural states directly as observations.
- When `hemodynamic=False`: state is N-dimensional (neural only), observation is
  `y = x + noise` (identity C_obs), no TR downsampling.
- When `hemodynamic=True`: existing behavior preserved exactly (5N state, BOLD output).
- This way, switching to real fMRI data later just flips the flag. Same code path,
  same tests, no fork drift.
- Zero edits to `neural_state.py` or `NeuralStateEquation` — reuse as-is.

### Pyro model: fork and simplify, keep modular
- New `latent_circuit_dcm_model.py` — copies the A/B/C sampling pattern from
  `task_dcm_model.py` but uses the hemodynamic-off path. Keep modular, no shared
  base class or fragile abstractions.
- New prior constants: `LC_A_PRIOR_VARIANCE`, `LC_B_PRIOR_VARIANCE` — separate from
  task DCM's `A_PRIOR_VARIANCE` and `B_PRIOR_VARIANCE`. Calibrated empirically.
- `C_obs` fixed at identity for v0.6.0. No learned projection.

### Prior recalibration strategy
- **Empirical sweep on synthetic data.** Generate synthetic bilinear trajectories at
  known A/B scales, sweep prior_variance x init_scale jointly (learned from v0.3.0
  RECOV-04 lesson where this interaction caused systematic B shrinkage).
- **Fixed constants** — calibrate once, use everywhere. No per-dataset adaptive scaling.
- **Run on cluster (M3)** — always route to cluster, even if fast.

### Multi-start SVI
- **Extend `run_svi()` in `guides.py`** with `n_restarts=1` kwarg. Default 1 preserves
  backward compat for all existing callers. When n_restarts > 1: clear param store,
  re-init, run SVI, repeat N times, select best by final ELBO.
- **10 restarts** for v0.6.0 (minimum viable; L&E uses 100 but we have Bayesian
  regularization).
- **Save all restarts' results** — enables ensemble analysis, convergence diagnostics,
  ELBO landscape plots. Return best but store all.

### Synthetic validation: generic bilinear, not CDDM
- **N=4 generic stable bilinear system** — same philosophy as v0.3.0's 3-region recovery.
  No task interpretation at this stage (CDDM structure belongs in Phase 22).
- **10 seeds** for acceptance gate (same as v0.3.0).
- **Thresholds empirically determined** — run the sweep, see what the method achieves,
  set thresholds at a documented level. The bilinear ground truth is correctly specified
  here (no model misspecification), so recovery should be better than v0.3.0 BOLD.

### Recovery metrics: full Bayesian + trajectory
- **Primary: Bayesian parameter-level metrics** (same as RECOV-01 through RECOV-07):
  - Posterior shrinkage: std_post / std_prior per parameter
  - CI coverage at multiple levels: 50%, 75%, 90%, 95%
  - Sign recovery on non-null elements (>= 80%)
  - Coverage-of-zero on null elements (>= 85%)
  - A RMSE, B RMSE
- **Additional (latent-circuit-specific):**
  - Trajectory R-squared on held-out trials (>= 0.95 for correctly specified model)
  - ELBO model selection: correctly selects true N from candidates
- **Reuse existing infrastructure**: `compute_acceptance_gates()` from
  `benchmarks/bilinear_metrics.py`, fixture pipeline from `generate_fixtures.py`,
  forest plot / acceptance-gate table from `benchmarks/plotting.py`.
- Do NOT rebuild recovery infrastructure — extend what's already validated.

### ELBO model selection test
- **Claude's Discretion:** candidate N set (e.g., {2,3,4,5,6} or {2,4,6,8}).
  Pick whatever makes the clearest figure.

### Claude's Discretion
- Exact noise model (scalar noise_prec vs per-dimension)
- Observation time grid details (match simulator output directly)
- Stimulus design for synthetic data (block, event, or epoch)
- How to structure the CoupledDCMSystem hemodynamic toggle internally

</decisions>

<specifics>
## Specific Ideas

- "Shouldn't we just use the DCM system but turn off the neural state equations?
  That way we can turn it back on when we need to put this on real data."
  → Make hemodynamic stage optional, not a fork. The existing CoupledDCMSystem
  should gain a flag to bypass balloon-Windkessel.

- Recovery validation must use the same Bayesian metrics (CI coverage, shrinkage,
  sign recovery) as v0.3.0 RECOV, not just RMSE and trajectory R-squared. The
  Bayesian parameter-level metrics are primary; trajectory/ELBO metrics are
  additional.

- Reuse existing v0.3.0 benchmark infrastructure (bilinear_metrics.py, fixture
  pipeline, plotting) rather than rebuilding. Phase 20 extends, doesn't recreate.

- Phase 16.1 RECOV-04 lesson: prior_variance x init_scale interaction must be
  swept jointly, not sequentially.

</specifics>

<deferred>
## Deferred Ideas

- Neural ODE DCM (Approach 2) — captured as todo, v0.7.0+ milestone
- Learned C_obs projection — v0.6.1 candidate (LC-OBS-01)
- Perturbation validation — Phase 22 / v0.6.1 scope
- Dale's law constraints on CT-RNN — v0.6.1
- CDDM-specific circuit structure (2 sensory + context + decision) — belongs in
  Phase 22 when RNNs are trained on CDDM, not in Phase 20's generic validation

</deferred>

---

*Phase: 20-latent-circuit-forward-model*
*Context gathered: 2026-05-24*
