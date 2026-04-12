# Phase 10: Guide Variants - Context

**Gathered:** 2026-04-12
**Status:** Ready for planning

<domain>
## Phase Boundary

Extend the `create_guide` factory to return working Pyro guides for 6 guide types
(AutoDelta, AutoNormal, AutoLowRankMultivariateNormal, AutoMultivariateNormal,
AutoIAFNormal, AutoLaplaceApproximation) and plumb 3 ELBO objectives (Trace_ELBO,
TraceMeanField_ELBO, RenyiELBO) through BenchmarkConfig and `run_svi` without
breaking v0.1.0 behavior. Existing runners gain `guide_type` parameterization.

</domain>

<decisions>
## Implementation Decisions

### create_guide API shape
- Guide type specified via **string key** (e.g., `guide_type='auto_lowrank_mvn'`), serializable to BenchmarkConfig/JSON/CLI
- Guide-specific hyperparams via **kwargs passthrough**: `create_guide(model, guide_type='auto_iaf', num_transforms=2, hidden_dims=[20])`
- **Deprecate `init_scale` as a named parameter** -- it moves into kwargs. All existing v0.1.0 call sites must be migrated. `guide_type` defaults to `'auto_normal'` so omitting it still returns AutoNormal
- `extract_posterior_params` becomes **sample-based**: draw N samples from guide, return dict of means/stds per site. Guide-agnostic, works for every Pyro Auto* guide including AutoIAF

### Guide hyperparameter defaults
- AutoLowRankMultivariateNormal: **rank=2** (fixed, not scaling with latent dim)
- AutoIAFNormal: **2 transforms, hidden_dims=[20]** (minimal -- IAF is already expensive)
- init_scale: **0.01 for all 6 guide types** (ODE blow-up risk is guide-independent)
- AutoLaplaceApproximation: **1000 MAP pre-fit steps** (fixed default)

### ELBO plumbing
- `elbo_type` flows as **string in BenchmarkConfig** (e.g., `'trace_elbo'`, `'tracemeanfield_elbo'`, `'renyi_elbo'`). `run_svi` gains an `elbo_type` param, maps string to Pyro class internally
- RenyiELBO alpha: **Claude's discretion** -- pick based on what Phase 11 calibration needs
- Test coverage: **full 6x3 matrix** (18 convergence tests). Every guide x every ELBO combination must be verified
- TraceMeanField_ELBO + non-mean-field guide: **raise ValueError**. Incompatible pairs fail loud at `run_svi` or `create_guide` level, not silently

### Failure mode policy
- Known-bad combinations (AutoMVN at N=10): **raise at `create_guide`** before allocating. Blocklist checked proactively
- Blocklist definition: **Claude's discretion** -- hardcoded dict vs BenchmarkConfig.excluded_combos
- Error messages: **include suggested alternative** (e.g., "Use 'auto_lowrank_mvn' instead")
- NaN ELBO during SVI sweeps: **Claude's discretion** -- keep raising or return sentinel

### Claude's Discretion
- RenyiELBO alpha value (0.5 specified in success criteria, may be configurable)
- Blocklist storage location (create_guide dict vs BenchmarkConfig)
- NaN ELBO handling pattern (raise vs sentinel return)
- Number of posterior samples for extract_posterior_params summary
- String key naming convention for guide types

</decisions>

<specifics>
## Specific Ideas

- kwargs passthrough pattern mirrors the existing BenchmarkConfig factory kwargs passthrough from Phase 9
- The full 18-test matrix (6 guides x 3 ELBOs) ensures no broken combinations ship -- Phase 11 runs real benchmarks on this foundation
- Sample-based extract_posterior_params aligns with how calibration benchmarks already consume posterior output (means + stds for coverage computation)

</specifics>

<deferred>
## Deferred Ideas

None -- discussion stayed within phase scope

</deferred>

---

*Phase: 10-guide-variants*
*Context gathered: 2026-04-12*
