---
phase: 07-amortized-neural-inference-guides
plan: 01
subsystem: inference
tags: [zuko, normalizing-flows, summary-networks, parameter-packing, amortized-inference, nsf]

# Dependency graph
requires:
  - phase: 04-pyro-generative-models
    provides: task_dcm_model and spectral_dcm_model Pyro sample site definitions
  - phase: 01-neural-hemodynamic-forward-model
    provides: parameterize_A transform and simulators
  - phase: 02-spectral-dcm-forward-model
    provides: spectral_dcm_forward and noise models
provides:
  - BoldSummaryNet (1D-CNN) for BOLD time series compression
  - CsdSummaryNet (MLP) for complex CSD compression
  - TaskDCMPacker with log-space noise_prec contract
  - SpectralDCMPacker with log-space csd_noise_scale contract
  - invert_A_to_A_free formula (log(-2*A_ii) for diagonal)
  - generate_training_data.py CLI for task and spectral data
affects: [07-02-PLAN.md, 07-03-PLAN.md]

# Tech tracking
tech-stack:
  added: [zuko>=1.2]
  patterns: [summary-network-embedding, parameter-packing-standardization, log-space-contract]

key-files:
  created:
    - src/pyro_dcm/guides/__init__.py
    - src/pyro_dcm/guides/summary_networks.py
    - src/pyro_dcm/guides/parameter_packing.py
    - scripts/generate_training_data.py
    - tests/test_summary_networks.py
    - tests/test_parameter_packing.py
  modified:
    - pyproject.toml
    - src/pyro_dcm/guides/__init__.py

key-decisions:
  - "Log-space contract: noise_prec and csd_noise_scale stored as log in packed vectors"
  - "1D-CNN for BOLD (temporal patterns), MLP for CSD (compact frequency-domain)"
  - "Standardization to zero mean/unit variance for NSF spline domain [-5, 5]"
  - "Training data filenames use requested count, not valid count"
  - "csd_noise_scale defaults to 1.0 (HalfCauchy prior mode, absent from simulator)"

patterns-established:
  - "Log-space contract: positive params go through log() in pack, exp() in unpack caller"
  - "Summary net batch handling: unbatched input auto-unsqueezed, output auto-squeezed"
  - "A_free inversion: log(-2*A_ii) for diagonal, identity for off-diagonal"

# Metrics
duration: 49min
completed: 2026-03-29
---

# Phase 7 Plan 1: Amortized Inference Shared Infrastructure Summary

**Zuko NSF dependency, BOLD/CSD summary networks, parameter packing with log-space contracts and [-5,5] standardization, training data CLI**

## Performance

- **Duration:** 49 min
- **Started:** 2026-03-29T07:50:46Z
- **Completed:** 2026-03-29T08:39:57Z
- **Tasks:** 2/2
- **Files created:** 6
- **Files modified:** 1

## Accomplishments
- Added Zuko normalizing flow library as project dependency
- BoldSummaryNet (1D-CNN + AdaptiveAvgPool1d) handles variable-length BOLD with gradient flow
- CsdSummaryNet (MLP on real/imag decomposition) handles complex128 CSD matrices
- TaskDCMPacker and SpectralDCMPacker with exact log-space contracts for positive params
- Standardization achieves >99% of values in [-5, 5] NSF spline domain
- Training data generator produces cached .pt files with embedded metadata for both DCM variants
- 19 unit tests all passing

## Task Commits

Each task was committed atomically:

1. **Task 1: Add Zuko dependency, create guides package with summary networks** - `63b8d30` (feat)
2. **Task 2: Parameter packing utilities and training data generation script** - `6f8ad32` (feat)

## Files Created/Modified
- `pyproject.toml` - Added zuko>=1.2 dependency
- `src/pyro_dcm/guides/__init__.py` - Package exports: BoldSummaryNet, CsdSummaryNet, TaskDCMPacker, SpectralDCMPacker
- `src/pyro_dcm/guides/summary_networks.py` - BoldSummaryNet (1D-CNN) and CsdSummaryNet (MLP) nn.Modules
- `src/pyro_dcm/guides/parameter_packing.py` - TaskDCMPacker and SpectralDCMPacker with standardization
- `scripts/generate_training_data.py` - CLI for generating task/spectral training datasets
- `tests/test_summary_networks.py` - 10 tests for shape, gradient flow, variable length, complex decomposition
- `tests/test_parameter_packing.py` - 9 tests for round-trips, batch dims, log-space, standardization, A_free inversion

## Decisions Made
- **Log-space contract for positive params**: noise_prec and csd_noise_scale stored as log values in packed vector; wrapper model calls .exp(). This ensures NSF spline flow operates on unconstrained reals.
- **1D-CNN for BOLD, MLP for CSD**: BOLD has temporal structure suited to convolutions; CSD is already a compact frequency-domain summary where MLP suffices.
- **Standardization to [-5, 5]**: NSF spline bins operate on this range; values outside are identity-mapped (07-RESEARCH.md Pitfall 3).
- **csd_noise_scale = 1.0 default**: Absent from spectral simulator (noiseless predicted CSD). 1.0 is the HalfCauchy(1.0) prior mode.
- **Training data filenames use requested count**: Predictable for downstream scripts even when some simulations are filtered.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Torch installation broken on Windows**
- **Found during:** Task 1 (dependency installation)
- **Issue:** Windows long path limitation prevented pip from installing torch to standard site-packages
- **Fix:** Installed torch to C:/Users/aman0087/tinstall, created .pth file for Python path resolution
- **Files modified:** None (system-level fix, not committed)
- **Verification:** All imports work: torch, pyro, zuko, pyro_dcm

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Environment fix needed for any execution; no code impact.

## Issues Encountered
- Some task DCM simulations produce NaN BOLD (2/5 in small test run) due to random A matrices causing ODE divergence. This is expected and handled by NaN filtering in the generation script (07-RESEARCH.md Pitfall 5).

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Summary networks, packers, and data generator are ready for 07-02 (task DCM amortized guide)
- The log-space contract is documented and tested; 07-02 wrapper model must call .exp() on noise_prec
- Training data .pt files include all metadata (stimulus, masks, t_eval) needed by the training loop
- Zuko + Pyro bridge (ZukoToPyro) verified working

---
*Phase: 07-amortized-neural-inference-guides*
*Completed: 2026-03-29*
