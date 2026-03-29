---
phase: 07-amortized-neural-inference-guides
plan: 02
subsystem: inference
tags: [zuko, normalizing-flows, nsf, amortized-inference, pyro, svi, wrapper-model]

# Dependency graph
requires:
  - phase: 07-amortized-neural-inference-guides
    provides: BoldSummaryNet, CsdSummaryNet, TaskDCMPacker, SpectralDCMPacker (plan 01)
  - phase: 04-pyro-generative-models
    provides: task_dcm_model and spectral_dcm_model Pyro sample site definitions
  - phase: 01-neural-hemodynamic-forward-model
    provides: parameterize_A, CoupledDCMSystem, BalloonWindkessel, bold_signal
  - phase: 02-spectral-dcm-forward-model
    provides: spectral_dcm_forward, decompose_csd_for_likelihood
provides:
  - AmortizedFlowGuide (Zuko NSF + summary net + Pyro guide)
  - amortized_task_dcm_model wrapper (single _latent site)
  - amortized_spectral_dcm_model wrapper (single _latent site)
  - NaN-protected ODE forward model for amortized training
  - train_amortized_guide.py end-to-end training CLI
  - rDCM deferral documentation in amortized_wrappers.__doc__
affects: [07-03-PLAN.md]

# Tech tracking
tech-stack:
  added: []
  patterns: [wrapper-model-single-latent-site, nan-detach-protection, coarse-dt-svi]

key-files:
  created:
    - src/pyro_dcm/guides/amortized_flow.py
    - src/pyro_dcm/models/amortized_wrappers.py
    - scripts/train_amortized_guide.py
    - tests/test_amortized_task_dcm.py
  modified:
    - src/pyro_dcm/guides/__init__.py
    - src/pyro_dcm/models/__init__.py

key-decisions:
  - "Wrapper model pattern: single _latent site in both model and guide for Pyro ELBO compatibility"
  - "NaN detach protection: ODE divergence produces zero-gradient penalty, not NaN gradients"
  - "Coarse dt=0.5 for SVI training (metadata dt=0.01 is for simulation fidelity)"
  - "num_particles=1 with sequential eval (parameterize_A does not support batch dims)"
  - "Dict-to-PiecewiseConstantInput auto-conversion in wrapper model"

patterns-established:
  - "Wrapper model pattern: sample single packed _latent, deterministically unpack to named params"
  - "NaN detach: if torch.isnan(predicted).any(): predicted = torch.zeros_like(predicted).detach()"
  - "Training script loads all metadata from .pt file (no CLI args for n_regions, stimulus, etc.)"

# Metrics
duration: 39min
completed: 2026-03-29
---

# Phase 7 Plan 2: AmortizedFlowGuide and Task DCM Amortized Inference Summary

**Conditional NSF flow guide wrapping Zuko + BoldSummaryNet into Pyro-compatible guide, wrapper models with single packed latent site, NaN-protected ODE forward pass, training script and 6 tests**

## Performance

- **Duration:** 39 min
- **Started:** 2026-03-29T08:46:32Z
- **Completed:** 2026-03-29T09:25:41Z
- **Tasks:** 2/2
- **Files created:** 4
- **Files modified:** 2

## Accomplishments
- AmortizedFlowGuide wraps Zuko NSF + BoldSummaryNet/CsdSummaryNet into a Pyro-compatible guide that samples a single `_latent` site
- Wrapper models (amortized_task_dcm_model, amortized_spectral_dcm_model) sample a single packed latent vector and deterministically unpack, solving the Pyro site-matching problem
- NaN protection via detach-and-replace ensures ODE divergence produces zero-gradient penalties instead of corrupting flow parameters
- SVI converges on task DCM data: ELBO decreases from ~7000 to ~85 in 30 steps
- Posterior sampling via single forward pass produces 1000 samples in < 1 second
- Training script loads all metadata (stimulus, masks, t_eval, TR, dt) from .pt file

## Task Commits

Each task was committed atomically:

1. **Task 1: AmortizedFlowGuide class and wrapper models** - `6e86521` (feat)
2. **Task 2: Training script and task DCM amortized inference tests** - `c93f63f` (feat)

## Files Created/Modified
- `src/pyro_dcm/guides/amortized_flow.py` - AmortizedFlowGuide class with Zuko NSF + summary net
- `src/pyro_dcm/models/amortized_wrappers.py` - Wrapper models with single _latent site, NaN protection, rDCM deferral doc
- `scripts/train_amortized_guide.py` - End-to-end CLI for training amortized guides (task/spectral)
- `tests/test_amortized_task_dcm.py` - 6 tests: construction, trace, site matching, SVI convergence, sampling, speed
- `src/pyro_dcm/guides/__init__.py` - Added AmortizedFlowGuide export
- `src/pyro_dcm/models/__init__.py` - Added amortized wrapper model exports

## Decisions Made
- **Wrapper model pattern (single _latent site):** Both model and guide sample exactly one `_latent` site. The model uses N(0,I) prior in standardized space; the guide uses the conditional NSF flow. This is the cleanest Pyro pattern from 07-RESEARCH.md that avoids Delta distribution workarounds and keeps automatic ELBO working.
- **NaN detach protection:** When ODE integration produces NaN BOLD (from extreme parameter samples), the wrapper model replaces predicted_bold with zeros and detaches from the computation graph. This produces a large finite likelihood penalty with zero gradient, preventing NaN from corrupting optimizer state. Without this, a single NaN permanently corrupts all subsequent SVI steps.
- **Coarse dt=0.5 for SVI:** The training data uses dt=0.01 for high-fidelity simulation. SVI training uses dt=0.5 for the ODE integration, which is 50x faster. The coarser dt is sufficient for learning the guide -- the exact BOLD shape matters less than the gradient direction.
- **Sequential particles (no vectorize_particles):** `parameterize_A` uses boolean mask indexing that doesn't support batch dimensions. Using num_particles=1 avoids the issue. For production training with num_particles > 1, sequential evaluation is used.
- **Dict-to-PiecewiseConstantInput auto-conversion:** The .pt metadata stores stimulus as a dict (from make_block_stimulus). The wrapper model auto-converts to PiecewiseConstantInput when a dict is received.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] NaN gradient corruption from ODE divergence**
- **Found during:** Task 2 (SVI convergence test)
- **Issue:** Untrained flow samples extreme parameter combinations causing ODE divergence. A single NaN loss corrupts the entire optimizer state (Adam moment buffers), making all subsequent steps NaN.
- **Fix:** Added detach-and-replace in `_run_task_forward_model`: when `predicted_bold` contains NaN, replace with `torch.zeros_like(predicted_bold).detach()`. This produces a large finite penalty with zero gradient.
- **Files modified:** `src/pyro_dcm/models/amortized_wrappers.py`
- **Verification:** SVI converges: 30 steps go from loss=7001 to loss=85 with zero NaN losses.
- **Committed in:** `c93f63f` (Task 2 commit)

**2. [Rule 1 - Bug] Dict stimulus not callable for ODE integration**
- **Found during:** Task 2 (wrapper model trace test)
- **Issue:** `make_block_stimulus` returns a dict, but `CoupledDCMSystem` expects a callable `PiecewiseConstantInput`.
- **Fix:** Added auto-conversion in `_run_task_forward_model`: if stimulus is a dict, construct `PiecewiseConstantInput(stimulus["times"], stimulus["values"])`.
- **Files modified:** `src/pyro_dcm/models/amortized_wrappers.py`
- **Verification:** Wrapper model trace test passes with dict stimulus from metadata.
- **Committed in:** `c93f63f` (Task 2 commit)

**3. [Rule 1 - Bug] vectorize_particles incompatible with parameterize_A**
- **Found during:** Task 2 (SVI convergence test)
- **Issue:** `parameterize_A` uses `torch.eye(N, dtype=torch.bool)` for boolean mask indexing, which fails with batched tensors from `vectorize_particles=True`.
- **Fix:** Changed tests and training script to use `num_particles=1` (or `vectorize_particles=False` for `num_particles > 1`).
- **Files modified:** `tests/test_amortized_task_dcm.py`, `scripts/train_amortized_guide.py`
- **Verification:** All 6 tests pass.
- **Committed in:** `c93f63f` (Task 2 commit)

---

**Total deviations:** 3 auto-fixed (3 bugs)
**Impact on plan:** All fixes necessary for correct operation. The NaN detach protection is the critical addition -- without it, amortized task DCM training is impossible due to inevitable ODE divergence from random initial flow samples.

## Issues Encountered
- The coarse dt=1.0 initially tried for speed was too aggressive -- even stable A matrices can cause numerical issues with rk4 at that step size when combined with realistic parameter ranges. Changed to dt=0.5 which is reliable.
- The packer's standardization statistics for diagonal A_free elements have near-zero std (1e-6) because all training A matrices share the same self-inhibition structure. This is correct behavior -- the diagonal is effectively fixed at zero (A_free=0 maps to A_ii=-0.5).

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- AmortizedFlowGuide is ready for 07-03 (evaluation and comparison tests)
- The wrapper model NaN protection is essential -- 07-03 must use it (or import from amortized_wrappers)
- Training script supports both task and spectral variants
- All module-level docstrings include rDCM deferral documentation (verified by test)
- The `amortized_wrappers.__doc__` check required by 07-03 is satisfied

---
*Phase: 07-amortized-neural-inference-guides*
*Completed: 2026-03-29*
