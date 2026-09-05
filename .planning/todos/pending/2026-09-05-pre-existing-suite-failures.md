---
created: 2026-09-05T22:30
title: 12 pre-existing test failures (8 numerical NaN + 4 stale API), not migration-caused
area: testing / numerical stability
priority: medium
files:
  - tests/test_vl_forward_model_protocol.py
  - tests/test_spectral_dcm_recovery.py
  - tests/test_amortized_spectral_dcm.py
  - tests/test_spectral_transfer.py
  - tests/test_task_simulator.py
  - tests/test_elbo_model_comparison.py
  - tests/test_hybrid_vae_dcm_model.py
---

## Problem

First full core-suite run after the M3 -> DCCN migration (job 55199993, DCCN,
`-m "not slow and not spm and not tapas and not mne and not foundation and not
latent and not vl"`) gave **18 failed, 747 passed, 2 skipped, 144 deselected**.

Triaged into three groups. **None is caused by the migration.**

### A. Missing optional dependency -- RESOLVED (6 failures)

`ModuleNotFoundError: No module named 'sklearn'` in `test_meeg_extractor.py` (4)
and `test_crossmodal_comparison.py` (2). Only `.[benchmark,dev]` had been
installed. Fixed by installing `scikit-learn>=1.3` on both the workstation and
the cluster; all 6 now pass.

Note these tests need scikit-learn but are **not** marked `latent` (the extra
that declares it), so `-m "not latent"` does not deselect them. Either mark them
or move `scikit-learn` into the core dependencies.

### B. Stale test against a changed API -- OPEN (4 failures)

`tests/test_vl_forward_model_protocol.py::test_task_*` fail with:

```
TypeError: make_block_stimulus() missing 1 required positional argument: 'rest_duration'
```

The tests call `make_block_stimulus(n_blocks=2, block_duration=5.0)`, but the
signature (`task_simulator.py:446`) requires `rest_duration` positionally.
**Reproduces on the workstation as well** (5 failed, 6 passed locally), so it is
environment-independent -- the tests were simply never updated. A 5th test in
the same file, `test_task_dcm_vl_recovery`, fails the same way.

Fix is probably one line per call site (supply a `rest_duration`), but confirm
the intended block design before picking a value rather than guessing.

### C. Numerical / NaN failures -- OPEN (8 failures)

```
tests/test_amortized_spectral_dcm.py::TestSpectralSVIConvergence::test_spectral_svi_convergence
tests/test_amortized_spectral_dcm.py::TestSpectralPosteriorSampling::test_spectral_posterior_sampling
tests/test_elbo_model_comparison.py::TestModelComparison::test_spectral_dcm_correct_model_wins
tests/test_hybrid_vae_dcm_model.py::TestSVI::test_svi_smoke_elbo_decreases
tests/test_spectral_dcm_recovery.py::TestSpectralDCMRecovery::test_spectral_dcm_rmse_below_threshold
tests/test_spectral_dcm_recovery.py::TestSpectralDCMRecovery::test_spectral_dcm_coverage_calibrated
tests/test_spectral_transfer.py::TestHemodynamicModel::test_hemodynamic_transfer_lowpass
tests/test_task_simulator.py::TestSimulatorOutputStructure::test_simulator_output_keys
```

Dominant causes:
- `RuntimeError: torch.linalg.eig: input tensor should not contain infs or NaNs` (4)
- `ValueError: Expected parameter loc ... to satisfy the constraint Real()`, with
  an all-NaN 22-element `loc` tensor, inside a Pyro `Normal` (2)
- one `'simulation_diverged'` key assertion

**Ruled out: the torch version.** The project was validated on torch 2.10.0 (50
occurrences in the archived M3 logs); `pyproject.toml` pins only `torch>=2.0`,
so the fresh install resolved to 2.14.0. A dedicated `.venv-t210` with
`torch==2.10.0+cpu` was built on the cluster and the same 8 tests were re-run
(job 55200039): **identical 8 failures, 55 passed**. So this is not a torch
regression and not cluster-specific -- it is pre-existing numerical fragility in
the spectral/SVI path.

This is consistent with things already known about that path: spectral DCM
identifiability limits ([[reference-spectral-dcm-identifiability]]) and the VL
overconfidence / eig-clamp boundary issues documented in Phases 30-31. The
`.venv-t210` comparison venv is still on the cluster and can be deleted.

## Why this matters

`main` currently has 12 permanently-red tests. That is corrosive: it trains
everyone to ignore the suite, and it means a genuine regression in the spectral
path would be invisible. Either fix them, or mark them `xfail` with a reason
linking to this todo so the signal is honest.

## Next step

Group B first (cheap, mechanical). Group C needs a real look at whether the
NaNs come from the eig clamp near the stability boundary, and whether the
affected tests are asserting recovery under conditions Phase 30 already
classified as identifiability limits.
