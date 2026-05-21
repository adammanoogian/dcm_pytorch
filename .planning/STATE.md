# State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-21)

**Core value:** A matrix (effective connectivity) remains explicit and interpretable with full posterior uncertainty
**Current focus:** v0.5.0 MNE-Python Integration -- test/validate IO loaders, build end-to-end pipeline demos.

## Current Position

**Milestone:** v0.5.0 MNE-Python Integration (started 2026-05-21)
**Phase:** 18 of 19 (MNE/BIDS IO Test Suite)
**Plan:** 0 of TBD in current phase
**Status:** Ready to plan
**Last activity:** 2026-05-21 -- Roadmap created. 2 phases (18-19), 18/18 requirements mapped.

Progress: v0.1.0 [##########] 100% | v0.2.0 [##########] 100% | v0.3.0 [##########] 100% | v0.4.0 [##########] 100% | v0.5.0 [..........] 0%

## Decisions

- **MNE-Python as optional dep.** `pip install pyro-dcm[mne]` keeps core lightweight; IO module in `src/pyro_dcm/io/`.
- **VL as bilinear inference engine** (v0.3.0). SVI+AutoNormal cannot recover B; VL (Gauss-Newton) recovers B with RMSE=0.017.
- **v0.3.0 + v0.4.0 closed 2026-05-21.** 27/27 v0.3.0 reqs complete. VIZ-01..10 complete. All code committed.

See `.planning/milestones/v0.1.0-ROADMAP.md`, `v0.2.0-ROADMAP.md` for prior milestone decisions.

## Blockers

None currently.

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 001 | Cluster sbatch infrastructure for Phase 16 acceptance-gate test | 2026-04-19 | 6bade20 | [001-cluster-sbatch-phase-16-acceptance](./quick/001-cluster-sbatch-phase-16-acceptance/) |

### Pending Todos

0 pending -- see `.planning/todos/pending/`.

## Key Risks

- **MNE optional dependency isolation:** Tests must skip cleanly when MNE is absent (`pytest.importorskip` + `@pytest.mark.mne`). Failure to isolate would break CI for non-MNE users.
- **CSD frequency grid mismatch (P1):** `csd_multitaper` returns FFT-locked frequencies regardless of `fmin`/`fmax`. Must be caught by sine-injection round-trip test (TEST-09).
- **Channel picks inconsistency (P3):** `Epochs.get_data(picks="eeg")` silently excludes bad channels; IO loaders must handle this correctly.

## Session Continuity

Last session: 2026-05-21 (v0.5.0 roadmap creation)
Stopped at: Roadmap created with 2 phases (18-19). Next: `/gsd:plan-phase 18` to derive plans for MNE/BIDS IO test suite.
Resume file: None
