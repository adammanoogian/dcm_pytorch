---
phase: 19-pipeline-demos
plan: 01
subsystem: pipeline-demos
tags: [mne, spectral-dcm, svi, pyro, csd, epochs_to_csd, demo]

# Dependency graph
requires:
  - phase: 18-mne-bids-io-test-suite
    provides: epochs_to_csd IO bridge (pyro_dcm.io.mne_loader)
  - phase: 04-pyro-generative-models
    provides: spectral_dcm_model, create_guide, run_svi, extract_posterior_params
  - phase: 02-spectral-dcm-forward-model
    provides: simulate_spectral_dcm, make_stable_A_spectral, parameterize_A
provides:
  - Self-contained spectral DCM demo script (scripts/demo_spectral_dcm.py)
  - Demonstrates MNE Epochs -> epochs_to_csd -> spectral_dcm_model -> SVI -> posterior A
  - A-RMSE recovery metric printed alongside posterior A matrix
  - MNE import guard with pip install hint on ImportError
affects:
  - 19-02 (next pipeline demo -- rDCM or task-DCM variant)
  - docs/02_pipeline_guide (consumer quickstart docs may reference this script)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "MNE import guard pattern: try/except at top-level with sys.exit(1)"
    - "Separate observed_csd sources: simulate for recovery metrics, epochs_to_csd for IO demo"

key-files:
  created:
    - scripts/demo_spectral_dcm.py
  modified: []

key-decisions:
  - "Use simulate_spectral_dcm CSD for SVI fitting (not MNE noise epochs) so A-RMSE is meaningful"
  - "epochs_to_csd call retained as a pure IO bridge demo with printed shapes"
  - "300 SVI steps at lr=0.01 -- fast enough for laptop (~30s), recovery A-RMSE ~0.037"

patterns-established:
  - "Demo script structure: numbered sections with comment headers, MNE guard at top, main() + guard"

# Metrics
duration: 4min
completed: 2026-05-24
---

# Phase 19 Plan 01: Spectral DCM Pipeline Demo Summary

**Executable demo script (scripts/demo_spectral_dcm.py) bridging MNE Epochs to spectral DCM via epochs_to_csd, fitting with SVI, and printing posterior A matrix with A-RMSE recovery metric**

## Performance

- **Duration:** 4 min
- **Started:** 2026-05-24T20:32:26Z
- **Completed:** 2026-05-24T20:36:33Z
- **Tasks:** 1/1
- **Files modified:** 1

## Accomplishments

- Created `scripts/demo_spectral_dcm.py` as a self-contained spectral DCM pipeline demo
- MNE import guard at top with user-friendly error and `pip install pyro-dcm[mne]` hint
- Six clearly numbered sections covering the full pipeline from ground truth to recovery metrics
- epochs_to_csd bridge demonstrated with printed shapes; SVI fit uses simulated CSD for valid A-RMSE
- A-RMSE of 0.0369 with 300 SVI steps (~30s runtime) on 3-region synthetic circuit

## Task Commits

Each task was committed atomically:

1. **Task 1: Create spectral DCM pipeline demo script** - `cc92511` (feat)

**Plan metadata:** (pending docs commit)

## Files Created/Modified

- `scripts/demo_spectral_dcm.py` - End-to-end spectral DCM pipeline demo with MNE IO bridge

## Decisions Made

- Use `simulate_spectral_dcm` CSD as the fitted observation, not MNE noise epochs, so A-RMSE is meaningful. The `epochs_to_csd` call is a pure IO bridge demo (shapes printed). The script comments this clearly.
- 300 SVI steps at lr=0.01: fast on laptop (~30s), achieves A-RMSE 0.037 on tiny 3-region circuit.
- `extract_posterior_params` returns `posterior["A_free"]["mean"]` (raw free parameters); `parameterize_A` is applied manually to get A, matching the transform used inside `spectral_dcm_model`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed f-string without placeholder**

- **Found during:** Task 1 verification (`ruff check`)
- **Issue:** `print(f"\nepochs_to_csd output:")` was an f-string with no placeholders
- **Fix:** Changed to `print("\nepochs_to_csd output:")` (removed extraneous `f` prefix)
- **Files modified:** scripts/demo_spectral_dcm.py
- **Verification:** `ruff check scripts/demo_spectral_dcm.py` reports "All checks passed!"
- **Committed in:** cc92511 (part of Task 1 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - lint bug)
**Impact on plan:** Trivial fix caught by ruff. No scope creep.

## Issues Encountered

None - script executed cleanly on first run (after lint fix).

## User Setup Required

None - no external service configuration required. MNE-Python must be installed (`pip install pyro-dcm[mne]`), but this is a user-facing prerequisite, not a setup step.

## Next Phase Readiness

- Phase 19 Plan 01 complete. `scripts/demo_spectral_dcm.py` is functional and ruff-clean.
- Phase 19 Plan 02 can proceed (next pipeline demo script).
- No blockers.

---

*Phase: 19-pipeline-demos*
*Completed: 2026-05-24*
