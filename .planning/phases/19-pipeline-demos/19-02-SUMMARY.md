---
phase: 19-pipeline-demos
plan: 02
subsystem: demos
tags: [mne, mne-python, task-dcm, svi, bilinear, epochs_to_timeseries, pipeline, demo]

# Dependency graph
requires:
  - phase: 18-mne-bids-io-test-suite
    provides: epochs_to_timeseries MNE IO bridge
  - phase: 15-bilinear-task-dcm
    provides: task_dcm_model with bilinear B path, parameterize_B
provides:
  - scripts/demo_task_dcm.py: end-to-end task DCM pipeline demo with MNE IO bridge and bilinear B
affects:
  - 20-documentation
  - consumer projects (dcm_hgf_mixed_models and similar)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "MNE Epochs pipeline: EpochsArray -> epochs_to_timeseries -> task_dcm_model SVI fit"
    - "t_eval constructed from DURATION and DT_MODEL (not from simulation output times)"
    - "MNE import guard: try/except ImportError + sys.exit(1) with pip install hint"
    - "partial(task_dcm_model, **model_kwargs) pattern for extract_posterior_params with bilinear kwargs"

key-files:
  created:
    - scripts/demo_task_dcm.py
  modified: []

key-decisions:
  - "DT_MODEL=0.5 for SVI integration step, DT_SIM=0.01 for forward simulation only - keeps SVI efficient"
  - "t_eval from arange(0, DURATION+DT_MODEL/2, DT_MODEL) not from ts_result times - ensures model dt contract"
  - "Single modulator (J=1) gating 0->1 edge only - simpler than demo_bilinear_consumer's two-edge mask"
  - "b_mask[0, 1, 0]=1.0 only (not 0,2,1 also) - reduced mask to make B recovery metric cleaner for demo"

patterns-established:
  - "Numbered section headers (# --- N. Section name ---) for demo script readability"
  - "EpochsArray wrapping: bold (T,N) -> bold.T[np.newaxis] -> (1,N,T) for mne.EpochsArray"
  - "Recovery metric trio: RMSE + sign match + non-zero element comparison"

# Metrics
duration: 12min
completed: 2026-05-24
---

# Phase 19 Plan 02: Task DCM Pipeline Demo Summary

**Self-contained MNE-Python task DCM demo (PIPE-02): synthetic EpochsArray -> epochs_to_timeseries -> bilinear task_dcm_model SVI -> posterior A + B recovery metrics**

## Performance

- **Duration:** ~12 min
- **Started:** 2026-05-24T20:29:00Z
- **Completed:** 2026-05-24T20:41:48Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments
- Created `scripts/demo_task_dcm.py` as a self-contained PIPE-02 demo
- Demonstrates full MNE IO bridge: EpochsArray wrapping -> epochs_to_timeseries -> (T, N) tensor
- Fits task_dcm_model with bilinear B matrices (b_masks + stim_mod) via SVI in ~2-5 min on CPU
- Prints A-RMSE, B-RMSE, B sign recovery (1.00 on demo run), A_true/A_est, B_true/B_est
- MNE import guard with clear `pip install pyro-dcm[mne]` error and sys.exit(1)
- Ruff check passes with zero lint errors

## Task Commits

Each task was committed atomically:

1. **Task 1: Create task DCM pipeline demo script** - `f6f07b1` (feat)

**Plan metadata:** (see docs commit below)

## Files Created/Modified
- `scripts/demo_task_dcm.py` - End-to-end task DCM demo with MNE IO bridge, bilinear B, SVI, recovery metrics

## Decisions Made
- `DT_MODEL=0.5` for SVI (coarser step for efficiency), `DT_SIM=0.01` for forward sim only - separating concerns prevents anti-pattern of using fine dt in model
- `t_eval` constructed from `DURATION` and `DT_MODEL` via `torch.arange`, not from `ts_result["times_fine"]` - honoring the model contract that t_eval spacing must equal dt
- Single-edge B mask (0->1 only, J=1) rather than two-edge mask from demo_bilinear_consumer - simpler B recovery metric display for demo purpose
- `partial(task_dcm_model, **model_kwargs)` for `extract_posterior_params` - required to thread bilinear kwargs through posterior sampling

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- PIPE-02 complete; `scripts/demo_task_dcm.py` is executable and copy-pasteable
- B recovery is poor at 100 steps / 100s (by design - demo is local-friendly, not publication-quality)
- Ready for Phase 20 documentation integration; the script can serve as the canonical MNE workflow example

---
*Phase: 19-pipeline-demos*
*Completed: 2026-05-24*
