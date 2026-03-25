---
phase: 01-neural-hemodynamic-forward-model
plan: 01
subsystem: forward-models
tags: [torch, ode, balloon-windkessel, bold, spm12, hemodynamics]

# Dependency graph
requires: []
provides:
  - "pyro_dcm package scaffolding (src layout, pyproject.toml)"
  - "NeuralStateEquation class: dx/dt = Ax + Cu with parameterize_A"
  - "BalloonWindkessel class: log-space hemodynamic ODE (ds, dlnf, dlnv, dlnq)"
  - "bold_signal function: algebraic BOLD observation equation"
  - "21 unit tests covering correctness, stability, and numerical safety"
affects:
  - 01-02 (coupled ODE system uses all three modules)
  - 01-03 (simulator uses all three modules)
  - phase-4 (Pyro generative models wrap these forward models)

# Tech tracking
tech-stack:
  added: [torch, torchdiffeq, pyro-ppl, scipy, numpy, pytest, ruff, mypy, hatchling]
  patterns:
    - "SPM12 code defaults for hemodynamic parameters (not paper values)"
    - "Log-space hemodynamic states (lnf, lnv, lnq) for positivity"
    - "A matrix parameterization: diag = -exp(free)/2 for stability"
    - "from __future__ import annotations in all modules"
    - "NumPy-style docstrings citing [REF-XXX] Eq. N"

key-files:
  created:
    - pyproject.toml
    - src/pyro_dcm/__init__.py
    - src/pyro_dcm/forward_models/__init__.py
    - src/pyro_dcm/forward_models/neural_state.py
    - src/pyro_dcm/forward_models/balloon_model.py
    - src/pyro_dcm/forward_models/bold_signal.py
    - src/pyro_dcm/utils/__init__.py
    - src/pyro_dcm/simulators/__init__.py
    - tests/__init__.py
    - tests/conftest.py
    - tests/test_neural_state.py
    - tests/test_balloon.py
    - tests/test_bold_signal.py
  modified: []

key-decisions:
  - "SPM12 code hemodynamic defaults: kappa=0.64, gamma=0.32, tau=2.0, alpha=0.32, E0=0.40"
  - "BOLD constants: simplified Buxton form k1=7*E0, k2=2, k3=2*E0-0.2, V0=0.02"
  - "Log-space clamping: lnf >= -14, f >= 1e-6 for oxygen extraction safety"

patterns-established:
  - "Forward model modules are plain classes/functions, not nn.Module (nn.Module deferred to coupled system in Plan 02)"
  - "conftest.py provides SPM12-convention fixtures for hemodynamic params, A/C matrices"
  - "float64 for all numerical precision tests"

# Metrics
duration: 13min
completed: 2026-03-25
---

# Phase 1 Plan 1: Project Scaffolding + Forward Model Modules Summary

**Bilinear neural state equation, log-space Balloon-Windkessel ODE, and BOLD observation equation with SPM12-convention parameterization and 21 passing unit tests**

## Performance

- **Duration:** 13 min
- **Started:** 2026-03-25T21:17:53Z
- **Completed:** 2026-03-25T21:31:12Z
- **Tasks:** 3/3
- **Files created:** 13

## Accomplishments

- Package scaffolding with src/pyro_dcm layout, pyproject.toml, and dev tooling (ruff, mypy, pytest)
- Neural state equation with SPM12 A matrix parameterization guaranteeing negative self-connections via -exp(diag)/2
- Balloon-Windkessel hemodynamic ODE in log-space matching SPM12 spm_fx_fmri.m conventions (kappa=0.64, gamma=0.32, tau=2.0, alpha=0.32, E0=0.40)
- Algebraic BOLD signal equation producing zero at steady state and realistic percent signal change for physiological states
- 21 unit tests verifying mathematical correctness, numerical stability, shape handling, and differentiability

## Task Commits

1. **Task 1: Project scaffolding and package structure** - `436fb43` (feat)
2. **Task 2: Forward model modules** - `53b0323` (feat)
3. **Task 3: Unit tests for all three modules** - `69234de` (test)

## Files Created/Modified

- `pyproject.toml` - Package config: hatchling build, torch/pyro/torchdiffeq deps, ruff/mypy config
- `src/pyro_dcm/__init__.py` - Package root with __version__ = "0.1.0"
- `src/pyro_dcm/forward_models/neural_state.py` - parameterize_A + NeuralStateEquation class [REF-001 Eq. 1]
- `src/pyro_dcm/forward_models/balloon_model.py` - BalloonWindkessel class with log-space derivatives [REF-002 Eq. 2-5]
- `src/pyro_dcm/forward_models/bold_signal.py` - bold_signal function [REF-002 Eq. 6]
- `tests/conftest.py` - SPM12-convention fixtures (hemo_params, test_A, test_C, device, dtype)
- `tests/test_neural_state.py` - 8 tests for A parameterization and neural state derivatives
- `tests/test_balloon.py` - 7 tests for hemodynamic ODE correctness and numerical safety
- `tests/test_bold_signal.py` - 6 tests for BOLD output range, differentiability, and batch shapes

## Decisions Made

- Used SPM12 code hemodynamic defaults (kappa=0.64, gamma=0.32, tau=2.0, alpha=0.32, E0=0.40), not Stephan 2007 paper values. Discrepancy documented in balloon_model.py comments.
- BOLD equation uses simplified Buxton form with E0=0.40 default (SPM12 code), not 0.34 (Stephan 2007 paper).
- Log-space clamping at lnf >= -14 and f >= 1e-6 before oxygen extraction to prevent NaN from 1/f divergence.
- Forward model modules implemented as plain classes/functions (not nn.Module), as specified in plan. nn.Module wrapping deferred to coupled ODE system in Plan 02.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Created README.md for hatchling build**
- **Found during:** Task 1 (package installation)
- **Issue:** pyproject.toml references readme = "README.md" but file did not exist, causing hatchling metadata generation to fail
- **Fix:** Created minimal README.md
- **Files modified:** README.md
- **Verification:** pip install -e succeeded after fix
- **Committed in:** 436fb43 (Task 1 commit)

**2. [Rule 3 - Blocking] Installed dependencies to shorter path to work around Windows long path limitation**
- **Found during:** Task 1 (dependency installation)
- **Issue:** PyTorch installation failed due to Windows long path limit on default site-packages path
- **Fix:** Installed packages to C:/Users/aman0087/torch_pkg and used PYTHONPATH for resolution
- **Verification:** All imports succeed, all tests pass
- **Note:** This is an environment-specific workaround, not a code change

---

**Total deviations:** 2 auto-fixed (2 blocking)
**Impact on plan:** Both blocking issues resolved to enable development. No scope creep.

## Issues Encountered

- Windows long path limitation prevented PyTorch installation to default site-packages. Resolved by installing to a shorter path and using PYTHONPATH. This is an environment-specific issue that does not affect the code.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- All three forward model modules are tested and ready for integration into the coupled ODE system (Plan 02)
- BalloonWindkessel.derivatives returns (ds, dlnf, dlnv, dlnq) matching the state layout needed for the coupled system [x, s, lnf, lnv, lnq]
- NeuralStateEquation.derivatives returns dx matching the neural state portion
- bold_signal takes linear-space v, q (from exp(lnv), exp(lnq)) matching the post-integration observation step
- No blockers for Plan 02

---
*Phase: 01-neural-hemodynamic-forward-model*
*Completed: 2026-03-25*
