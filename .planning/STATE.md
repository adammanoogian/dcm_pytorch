# State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-24)

**Core value:** A matrix (effective connectivity) remains explicit and interpretable with full posterior uncertainty
**Current focus:** v0.6.0 Latent Circuit DCM -- distill trained RNN latent dynamics into bilinear DCM circuits with posterior uncertainty

## Current Position

**Milestone:** v0.6.0 Latent Circuit DCM (defined 2026-05-24)
**Phase:** 20 of 25 (Direct Observation Forward Model, Simulator & Synthetic Validation)
**Plan:** 02 of 05
**Status:** In progress (plans 01-02 complete)
**Last activity:** 2026-05-24 -- Completed 20-02-PLAN.md (multi-start SVI). Extended run_svi() with n_restarts and guide_factory parameters. MODEL-05 requirement satisfied.

**Prior milestones in flight:**
- v0.5.0: Phase 18 COMPLETE; Phase 19 pending
- v0.3.0: Phase 16.1 pending (RECOV-04 B-RMSE diagnostic)

Progress: v0.1.0 [==========] 100% | v0.2.0 [==========] 100% | v0.3.0 [=========-] 16.1 pending | v0.4.0 [==========] Phase 17 complete | v0.5.0 [==--------] Phase 18 complete; 19 pending | v0.6.0 [==--------] Phase 20: 2/5 plans complete

## Decisions

- **v0.6.0 phase structure = 6 phases (20-25).** Derived from 10 requirement categories clustered into 6 delivery boundaries. Phase 20 (14 reqs) is the scientific core; Phases 20 and 21 can run in parallel.
- **C_obs fixed at identity for v0.6.0.** Addresses pitfall LC5 (rotation ambiguity). Learned C_obs deferred to v0.7.0+.
- **Multi-start SVI (>=10 restarts) non-optional.** Addresses pitfall LC11; L&E uses 100. Implemented in Plan 20-02.
- **n_restarts=1 preserves exact backward-compatible code path.** No structural changes to single-restart behavior (D20-02-01).
- **guide_factory required for n_restarts>1.** Safe re-initialization needs fresh guide instances (D20-02-02).
- **NaN restarts get inf penalty and are skipped.** One bad init doesn't abort entire multi-start (D20-02-03).
- **Prior recalibration mandatory.** LC_A_PRIOR_VARIANCE separate from BOLD priors. Addresses pitfall LC4.
- Prior v0.3.0/v0.4.0/v0.5.0 decisions: see earlier STATE.md history in git log.

## Blockers

None currently.

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 001 | Cluster sbatch infrastructure for Phase 16 | 2026-04-19 | 6bade20 | [001](./quick/001-cluster-sbatch-phase-16-acceptance/) |
| 002 | Structure-audit migration waves 1-5 | 2026-04-29 | 33e03e1 | [002](./quick/002-structure-audit-migration-waves-1-5/) |

### Pending Todos

1 pending — see `.planning/todos/pending/`.
- Neural ODE DCM (Approach 2) — separate milestone v0.7.0+ after v0.6.0 informs whether bilinear suffices

## Key Risks

- **Bilinear misspecification (LC1).** Bilinear DCM is a first-order approximation of nonlinear RNN dynamics. Mitigated by linearization quality diagnostic and L&E nonlinear comparison.
- **Prior scale mismatch (LC4).** BOLD-calibrated priors wrong for RNN hidden states. Mitigated by mandatory recalibration on 5+ synthetic RNNs in Phase 20.
- **Rotational degeneracy (LC2).** PCA basis is arbitrary. Mitigated by Procrustes alignment and perturbation validation.
- **PCA discards task-relevant dynamics (LC3).** Mitigated by output-R-squared gate (>= 0.90) in Phase 21.
- **Multi-start convergence (LC11).** ELBO landscape has local optima. Mitigated by >=10 random restarts.

## Session Continuity

Last session: 2026-05-24T16:29Z
Stopped at: Completed 20-02-PLAN.md (multi-start SVI)
Next: Execute 20-03-PLAN.md (latent circuit DCM Pyro model)
Resume file: None
