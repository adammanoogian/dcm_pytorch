# Status of the SVI inference path

**Decision (2026-09-06): SVI is a documented baseline, not the supported
inference path. Variational Laplace is.**

## Why

Pyro-DCM originally inferred with stochastic variational inference (SVI) over
mean-field Gaussian guides. In v0.3.0 the 3-region bilinear recovery benchmark
failed its RECOV-04 acceptance gate under SVI: **B-RMSE 0.3467**, far outside
tolerance. Rather than tune the guide, the failure was diagnosed to the
first-order mean-field approximation itself and answered by writing a
**Variational Laplace** engine matching SPM12 `spm_nlsi_GN`. On the same
problem VL recovered **B-RMSE 0.0170** — proving the forward model correct and
isolating the fault to SVI.

That engine became the backbone of everything since: v0.6.0 latent circuits,
the v0.7.0 recovery matrix, and the v0.8.0 CMC/ERP stack. The recovery matrix,
BMR validation and SPM12 cross-validation are all VL.

## What this means in practice

| | Path |
|---|---|
| Supported inference | **`method="vl"`** — `run_variational_laplace`, `run_variational_laplace_generic` |
| Baseline / comparison only | `method="svi"`, `method="amortized"` — `run_svi` |
| Registered runners | keys of `benchmarks.runners.RUNNER_REGISTRY` |

New work should use VL. `scripts/demo_spectral_dcm.py`,
`scripts/22_perturbation_experiment.py`, `scripts/22_run_full_validation.py`
and `scripts/24_fit_dcm_tribe.py` already do.

SVI is **not deleted**. The SVI-vs-VL contrast is a real result of this project
and `scripts/compare_sbi_svi.py` depends on it. It is kept, marked, and no
longer treated as a quality gate.

## The marked tests

Tests exercising the superseded path carry `@pytest.mark.svi_legacy` and a
non-strict `xfail`:

```
tests/test_hybrid_vae_dcm_model.py::TestSVI::test_svi_smoke_elbo_decreases
tests/test_spectral_dcm_recovery.py::TestSpectralDCMRecovery::test_spectral_dcm_rmse_below_threshold
tests/test_spectral_dcm_recovery.py::TestSpectralDCMRecovery::test_spectral_dcm_coverage_calibrated
tests/test_spm_spectral_dcm_validation.py::TestSpectralDCMvsSPM  (3 tests)
tests/test_spm_task_dcm_validation.py::TestTaskDCMvsSPM          (3 tests)
```

Failure mode is uniform: `ValueError: Expected parameter loc ... to satisfy the
constraint Real()` with an all-NaN `loc`, i.e. the guide diverging — surfaced by
Pyro as `UserWarning: Encountered NaN: loss`.

The `xfail` is **non-strict on purpose.** `test_svi_smoke_elbo_decreases`
XPASSES on Windows and FAILS on Linux (DCCN) with the same seed. SVI divergence
here is platform-dependent, so a strict xfail would itself be flaky. Run them
deliberately with:

```bash
pytest -m svi_legacy -rX      # -rX reports XPASSes
```

Two of the six SPM-gated ones additionally needed real infrastructure fixes
before they could even reach the SVI failure (an `rk4` solver switch and a
`float64` cast on `DCM.n`/`DCM.v`); those are fixed, and MATLAB/SPM12 now runs
the task DCM to completion. What remains is the SVI divergence.

## What would reopen this

- A structured (non-mean-field) guide recovering `B` at VL-comparable accuracy.
- Amortized inference needing SVI as its training objective at production
  quality — note the hybrid VAE-DCM already reached A-RMSE 0.076 and masked
  sign recovery 0.77, so this is not hopeless, just not the default.

Related: `.planning/todos/pending/2026-09-05-pre-existing-suite-failures.md`.
