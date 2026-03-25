---
phase: 01-neural-hemodynamic-forward-model
plan: 03
subsystem: simulators
tags: [torch, simulator, bold, fmri, block-design, snr, synthetic-data]

# Dependency graph
requires:
  - 01-01 (NeuralStateEquation, BalloonWindkessel, bold_signal)
  - 01-02 (CoupledDCMSystem, PiecewiseConstantInput, integrate_ode, make_initial_state)
provides:
  - "simulate_task_dcm: end-to-end task-DCM data simulator generating noisy BOLD at TR"
  - "make_block_stimulus: block-design fMRI stimulus convenience function"
  - "make_random_stable_A: random stable connectivity matrix generator"
  - "18 validation tests covering all Phase 1 success criteria"
affects:
  - phase-5 (parameter recovery uses simulator to generate ground-truth data)
  - phase-6 (SPM validation uses simulator for synthetic reference datasets)
  - phase-7 (amortized guide training uses simulator for training data generation)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Simulator returns comprehensive dict with clean/noisy BOLD, neural states, hemodynamic states, and ground truth params"
    - "Per-region SNR: noise_std = signal_std / SNR, Gaussian noise"
    - "TR downsampling via integer index stride: indices = arange(0, T_fine, round(TR/dt))"
    - "Accepts both dict stimulus and PiecewiseConstantInput directly"

key-files:
  created:
    - src/pyro_dcm/simulators/task_simulator.py
    - tests/test_task_simulator.py
  modified:
    - src/pyro_dcm/simulators/__init__.py

key-decisions:
  - "Per-region noise scaling: SNR = std(clean_bold_region) / std(noise_region)"
  - "A matrix accepted as parameterized form (not A_free) for direct control in simulations"
  - "make_random_stable_A generates both excitatory and inhibitory off-diagonal connections with equal probability"

patterns-established:
  - "Simulator output dict is the standard data format for all downstream testing"
  - "Block stimulus dict with 'times' and 'values' keys is the standard stimulus format"
  - "C=0.25 Hz driving strength produces ~4% peak BOLD (physiologically realistic)"

# Metrics
duration: 7min
completed: 2026-03-25
---

# Phase 1 Plan 3: Task-DCM Data Simulator Summary

**End-to-end task-DCM simulator generating N-region synthetic BOLD with controllable SNR, TR downsampling, and complete intermediate state output, validated by 18 tests covering 500s stability, BOLD range, SNR accuracy, and multi-region configurations**

## Performance

- **Duration:** 7 min
- **Started:** 2026-03-25T21:51:33Z
- **Completed:** 2026-03-25T21:58:18Z
- **Tasks:** 2/2
- **Files created:** 2
- **Files modified:** 1

## Accomplishments

- simulate_task_dcm wrapping CoupledDCMSystem + integrate_ode + bold_signal into a single function that produces synthetic fMRI data from (A, C, stimulus, hemo_params, SNR)
- Per-region Gaussian noise with controllable SNR (measured SNR within 20% of requested in tests), TR downsampling via integer stride, and comprehensive output dict with all intermediate states
- make_block_stimulus for standard block-design paradigms and make_random_stable_A for random connectivity with guaranteed stability
- 18 validation tests verifying all 5 Phase 1 success criteria: 500s stability (SC#1), stable neural trajectories (SC#2), BOLD 0.5-5% range (SC#3), end-to-end simulator (SC#5)

## Task Commits

1. **Task 1: Task-DCM data simulator** - `0e0b1c1` (feat)
2. **Task 2: Simulator validation tests** - `9ed2e0a` (test)

## Files Created/Modified

- `src/pyro_dcm/simulators/task_simulator.py` - simulate_task_dcm, make_block_stimulus, make_random_stable_A
- `tests/test_task_simulator.py` - 18 tests: output structure, numerics, reproducibility, multi-region, convenience functions, neural dynamics
- `src/pyro_dcm/simulators/__init__.py` - Added exports for all simulator functions

## Decisions Made

- Per-region noise scaling: each region gets independent Gaussian noise scaled by its own BOLD signal standard deviation divided by the requested SNR. This ensures the SNR is accurate for each region regardless of its BOLD amplitude.
- A matrix accepted in parameterized form (with negative diagonal), not A_free. This gives the simulator user direct control over the actual connectivity values. parameterize_A() can be called beforehand if needed.
- make_random_stable_A generates both positive (excitatory) and negative (inhibitory) off-diagonal connections with equal probability, controlled by density and strength_range parameters.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 1 is now COMPLETE: all three plans (01-01, 01-02, 01-03) are done
- The simulator is ready for Phase 5 parameter recovery testing (generate ground truth, recover A/C via inference)
- The simulator is ready for Phase 6 SPM validation (generate synthetic data, compare with SPM12 output)
- The simulator is ready for Phase 7 amortized guide training (batch data generation)
- All 5 Phase 1 success criteria are validated by tests:
  - SC#1: 500s simulation stable (test_simulator_500s, test_simulator_no_nan)
  - SC#2: Neural trajectories bounded for stable A (test_neural_state_stable_trajectory)
  - SC#3: BOLD 0.5-5% range (test_simulator_bold_range, test_simulator_500s)
  - SC#4: Multiple solvers supported (tested in Plan 02 tests)
  - SC#5: End-to-end simulator from (A, C, u(t), hemo_params, SNR) to BOLD (all simulator tests)
- No blockers for Phase 2

---
*Phase: 01-neural-hemodynamic-forward-model*
*Completed: 2026-03-25*
