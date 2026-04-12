---
phase: 10-guide-variants
plan: 01
subsystem: inference
tags: [pyro, autoguide, variational-inference, factory-pattern]

# Dependency graph
requires:
  - phase: 04-pyro-generative-models
    provides: "create_guide factory with AutoNormal, run_svi, extract_posterior_params"
provides:
  - "Extended create_guide factory with 6 guide types via GUIDE_REGISTRY"
  - "MEAN_FIELD_GUIDES set for TraceMeanField_ELBO compatibility checks"
  - "_MAX_REGIONS blocklist preventing memory explosion at high N"
affects: [10-02-PLAN, 10-03-PLAN, benchmarks]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "String-keyed registry dict for guide type dispatch"
    - "init_scale asymmetry: only passed to guides that accept it"
    - "N-based blocklist for memory-unsafe guide types"

key-files:
  created:
    - "tests/test_guide_factory.py"
  modified:
    - "src/pyro_dcm/models/guides.py"
    - "src/pyro_dcm/models/__init__.py"

key-decisions:
  - "hidden_dim for AutoIAFNormal wrapped in list (Pyro expects list[int], not int)"
  - "auto_mvn blocked at n_regions >= 8 (max allowed = 7) per P6 risk"

patterns-established:
  - "GUIDE_REGISTRY: dict mapping string keys to AutoGuide classes"
  - "_INIT_SCALE_GUIDES: set of guide types that accept init_scale"
  - "MEAN_FIELD_GUIDES: set of guide types compatible with TraceMeanField_ELBO"

# Metrics
duration: 18min
completed: 2026-04-12
---

# Phase 10 Plan 01: Guide Factory Summary

**6-type create_guide factory with GUIDE_REGISTRY, init_scale asymmetry, and N-based blocklist for AutoMVN**

## Performance

- **Duration:** 18 min
- **Started:** 2026-04-12T17:04:21Z
- **Completed:** 2026-04-12T17:22:46Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Extended create_guide to support 6 Pyro AutoGuide types via string-keyed registry
- init_scale asymmetry: only passed to AutoNormal, AutoLowRankMVN, AutoMVN
- N-based blocklist blocks auto_mvn at n_regions >= 8 with helpful suggestion
- 24 unit tests covering instantiation, backward compat, blocklist, kwargs, SVI smoke
- Full backward compatibility: all existing call sites unchanged

## Task Commits

Each task was committed atomically:

1. **Task 1: Extend create_guide factory with 6 guide types and blocklist** - `ef42d22` (feat)
2. **Task 2: Add unit tests for guide factory** - `902e069` (test)

## Files Created/Modified
- `src/pyro_dcm/models/guides.py` - Extended create_guide factory with GUIDE_REGISTRY, blocklist, init_scale asymmetry
- `src/pyro_dcm/models/__init__.py` - Export GUIDE_REGISTRY and MEAN_FIELD_GUIDES
- `tests/test_guide_factory.py` - 24 tests for all 6 guide types, blocklist, kwargs, SVI smoke

## Decisions Made
- hidden_dim for AutoIAFNormal defaults to `[20]` (list) not `20` (int), because Pyro's AutoRegressiveNN iterates over hidden_dims
- Integer hidden_dim values are auto-wrapped in a list for user convenience

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] AutoIAFNormal hidden_dim must be list, not int**
- **Found during:** Task 2 (SVI smoke test for auto_iaf)
- **Issue:** Pyro's AutoIAFNormal passes hidden_dim to AutoRegressiveNN which iterates over it; an int causes TypeError
- **Fix:** Changed default from `20` to `[20]` and added `isinstance(hidden_dim, int)` guard to wrap in list
- **Files modified:** src/pyro_dcm/models/guides.py
- **Verification:** SVI smoke test passes for auto_iaf
- **Committed in:** 902e069 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Essential fix for AutoIAFNormal to function. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- GUIDE_REGISTRY and MEAN_FIELD_GUIDES exported and ready for Plan 10-02 ELBO plumbing
- create_guide accepts guide_type kwarg ready for Plan 10-03 runner integration
- All backward compatibility preserved

---
*Phase: 10-guide-variants*
*Completed: 2026-04-12*
