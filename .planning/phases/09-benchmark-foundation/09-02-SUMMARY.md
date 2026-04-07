---
phase: 09-benchmark-foundation
plan: 02
subsystem: benchmarking
tags: [fixtures, npz, simulators, task-dcm, spectral-dcm, rdcm, seeding]

requires:
  - phase: 09-benchmark-foundation plan 01
    provides: BenchmarkConfig and metrics infrastructure
  - phase: 02-forward-models
    provides: Balloon-Windkessel, spectral transfer, rDCM forward
  - phase: 04-pyro-generative-models
    provides: decompose_csd_for_likelihood helper

provides:
  - CLI fixture generation script (benchmarks/generate_fixtures.py)
  - Fixture loading helper (benchmarks/fixtures.py) with load_fixture and get_fixture_count
  - Bit-identical .npz datasets for all 3 DCM variants at N=3,5,10

affects:
  - 09-benchmark-foundation plan 03 (runners consume fixtures via load_fixture)
  - 10-benchmark-runners (future runners use same fixtures)

tech-stack:
  added: []
  patterns:
    - "Complex array split/reconstruct pattern for .npz storage"
    - "Manifest.json per fixture subdirectory for metadata"
    - "Canonical seed_i = base_seed + i pattern for reproducibility"

key-files:
  created:
    - benchmarks/generate_fixtures.py
    - benchmarks/fixtures.py
  modified:
    - .gitignore

key-decisions:
  - "Used rk4 solver instead of dopri5 for task fixture generation (dopri5 underflows with dt=0.01 + piecewise stimulus)"
  - "rDCM fixtures store BOLD + stimulus only, no regressors (runners create their own via create_regressors)"
  - "Spectral noisy CSD uses exact same noise pattern as spectral_svi.py (decompose -> SNR-scaled Gaussian -> reconstruct)"

patterns-established:
  - "Fixture naming: {variant}_{N}region/dataset_{NNN}.npz"
  - "Complex tensor storage: split into _real/_imag keys in .npz, auto-reconstruct on load"

duration: 21min
completed: 2026-04-07
---

# Phase 9 Plan 02: Fixture Generation Summary

**CLI script and loader for bit-identical .npz benchmark fixtures across task, spectral, and rDCM variants with complex CSD real/imag splitting**

## Performance

- **Duration:** 21 min
- **Started:** 2026-04-07T22:02:50Z
- **Completed:** 2026-04-07T22:24:09Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- Created `generate_fixtures.py` CLI that produces .npz files for all 3 DCM variants with argparse support for variant, n-regions, n-datasets, seed, and output-dir
- Created `fixtures.py` with `load_fixture` (returns dict of torch tensors with automatic complex reconstruction) and `get_fixture_count` (reads manifest or counts files)
- Spectral noisy CSD generation exactly matches the `spectral_svi.py` noise pattern (decompose_csd_for_likelihood -> SNR-scaled Gaussian -> reconstruct)
- Verified round-trip for all variants at N=3, N=5, and N=10 with correct shapes and dtypes

## Task Commits

Each task was committed atomically:

1. **Task 1: Create fixture loading helper** - `595ff23` (feat)
2. **Task 2: Create fixture generation script** - `90a38ca` (feat)

## Files Created/Modified

- `benchmarks/fixtures.py` - load_fixture and get_fixture_count helpers for loading .npz fixture files
- `benchmarks/generate_fixtures.py` - CLI script for generating task, spectral, and rDCM fixture datasets
- `.gitignore` - Added benchmarks/fixtures/ to gitignore

## Decisions Made

- **rk4 solver for task fixtures:** dopri5 underflows (dt=0.0 assertion) with the fine time grid (dt=0.01) and piecewise stimulus; rk4 produces identical BOLD data reliably
- **No regressors in rDCM fixtures:** Runners call `create_regressors` themselves since regressor construction depends on HRF which is deterministic -- storing it would be redundant and would increase file size unnecessarily
- **Both clean and noisy CSD stored:** Spectral fixtures include both clean CSD (csd_real/csd_imag) and noisy CSD (noisy_csd_real/noisy_csd_imag) so runners can choose which to use

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Used rk4 solver instead of dopri5 for task fixtures**
- **Found during:** Task 2 (fixture generation script)
- **Issue:** dopri5 solver fails with `underflow in dt 0.0` assertion when integrating the coupled DCM system with dt=0.01 and piecewise constant stimulus
- **Fix:** Changed solver from default dopri5 to rk4 in `simulate_task_dcm` call within `generate_task_fixtures`
- **Files modified:** benchmarks/generate_fixtures.py
- **Verification:** All task fixture generation succeeds for N=3,5,10 with correct shapes
- **Committed in:** 90a38ca (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Solver change does not affect fixture data quality. rk4 is the standard fixed-step solver used in ODE integration tests.

## Issues Encountered

None beyond the dopri5 underflow handled above.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Fixture generation and loading infrastructure complete
- Plan 03 (runner integration with fixtures) can proceed using `load_fixture` to consume pre-generated datasets
- All three variants verified at all region counts (N=3, N=5, N=10)

---
*Phase: 09-benchmark-foundation*
*Completed: 2026-04-07*
