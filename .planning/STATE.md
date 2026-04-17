# State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-17)

**Core value:** A matrix (effective connectivity) remains explicit and interpretable with full posterior uncertainty
**Current focus:** v0.3.0 Bilinear DCM Extension -- defining requirements

## Current Position

**Milestone:** v0.3.0 Bilinear DCM Extension (started 2026-04-17)
**Phase:** Not started (defining requirements)
**Plan:** --
**Status:** Milestone initialized; research agents to be spawned next, then REQUIREMENTS.md, then ROADMAP.md
**Last activity:** 2026-04-17 -- milestone v0.3.0 initialized from GSD_pyro_dcm.yaml (Option A: bilinear-only scope)

Progress: v0.1.0 [██████████] 100% | v0.2.0 [██████████] 100% | v0.3.0 [░░░░░░░░░░] 0% (defining)

## Decisions

- **v0.3.0 scope: bilinear-only.** DCM.5 (PEB-lite group GLM) and DCM.V3 (4-node HEART2ADAPT
  circuit) are HEART2ADAPT-specific despite the YAML framing and are deferred. DCM.V2 (SPM12
  cross-validation) deferred pending MATLAB access. See PROJECT.md `Current Milestone`.
- **Research: ON.** User opted to run the full 4-agent project-research pass despite the YAML
  citing Friston 2003 + SPM12 directly -- value is in surfacing B-matrix-specific pitfalls
  (identifiability under sparse modulatory events, prior scale interactions, etc.) that are
  not in-codebase today.

See `.planning/milestones/v0.2.0-ROADMAP.md` and `.planning/milestones/v0.1.0-ROADMAP.md` for prior milestones.

## Blockers

None currently.

## Key Risks

- **Identifiability** of B-matrix elements under sparse or low-amplitude modulatory inputs
  (research pass should address this).
- **Numerical stability:** `A_eff(t) = A + Σ u_j·B_j` can become unstable if `B` is large or
  `u_mod` is sustained; the YAML calls for an eigenvalue monitor in the forward pass.
- **Runtime:** Bilinear forward model adds per-timestep cost proportional to number of
  modulators. Benchmark against the ~235s/500-step linear-DCM baseline.

## Accumulated Context

### Roadmap Evolution

- 2026-04-17: v0.3.0 milestone started. Bilinear DCM extension scoped from
  `C:\Users\aman0087\Downloads\GSD_pyro_dcm.yaml`. HEART2ADAPT-specific tasks
  (DCM.5, DCM.V3) deferred; SPM12 cross-val (DCM.V2) deferred pending MATLAB.

## Session Continuity

Last session: 2026-04-17
Stopped at: PROJECT.md + STATE.md updated; about to spawn 4 research agents.
Resume file: None

---
*Last updated: 2026-04-17 after v0.3.0 milestone initialization*
