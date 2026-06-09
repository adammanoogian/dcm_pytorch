---
phase: 26-sbi-spectral-dcm
plan: 01
subsystem: inference
tags: [sbi, npe, spectral-dcm, simulator, prior, embedding, sbc, diagnostics]
backfilled: 2026-06-09
dependency-graph:
  requires: []
  provides:
    - make_spectral_dcm_simulator
    - make_spectral_dcm_prior
    - train_npe
    - build_sbi_posterior
    - CSDEmbeddingNet
    - run_sbc_validation
  affects: ["26-02"]
tech-stack:
  added: ["sbi"]
  patterns: ["simulator-closure", "import-error-guard", "nan-zeros-fallback"]
key-files:
  created:
    - src/pyro_dcm/inference/sbi_spectral.py
    - src/pyro_dcm/inference/sbi_embedding.py
    - src/pyro_dcm/inference/sbi_diagnostics.py
    - tests/test_sbi_spectral.py
  modified:
    - src/pyro_dcm/inference/__init__.py
    - pyproject.toml
decisions:
  - id: "26-01-D1"
    description: "Simulator parameterizes only free A entries (n_free,) selected by a_mask; noise spectra passed as fixed noise_params dict rather than packed into theta as the plan's [A_free, noise_a, noise_b, noise_c] layout"
  - id: "26-01-D2"
    description: "NaN/unstable A handled by explicit eigenvalue check (real eig > 0) plus try/except RuntimeError, returning torch.zeros(output_dim) fallback (no warnings.warn)"
  - id: "26-01-D3"
    description: "CSDEmbeddingNet originally shipped with BatchNorm1d (commit f96bf71); later changed to LayerNorm so sbi batch_size=1 shape-probing works"
metrics:
  completed: "2026-05-27"
  commit: f96bf71
---

# Phase 26 Plan 01: SBI Spectral DCM Core Infrastructure Summary

Reusable simulation-based inference (SBI) modules for spectral DCM: a simulator
closure bridging the existing spectral forward model to sbi's 1D-tensor interface,
an SPM12-matched Normal prior over free A entries, an MLP CSD embedding network,
NPE training/posterior wrappers, and SBC calibration diagnostics. These modules are
consumed by the Plan 02 training pipeline.

## What Was Done

All work landed in a single commit `f96bf71`
(`feat(26-01): SBI spectral DCM infrastructure with simulator, prior, embedding, and diagnostics`).

### Task 1: SBI core modules — simulator, prior, embedding, diagnostics

Created `src/pyro_dcm/inference/sbi_spectral.py` (312 lines) with:

- **`make_spectral_dcm_simulator(n_regions, freqs, a_mask, noise_params=None, eig_clamp=-1/32)`**
  — returns a closure mapping a 1D float64 `theta` of free A entries to a flattened
  CSD vector `(2*F*N*N,)` (real parts then imag parts). Internally maps `theta` to
  `A_free` via the mask, applies `parameterize_A` for a negative diagonal, checks
  eigenvalue stability, calls `spectral_dcm_forward`, and returns
  `torch.zeros(output_dim)` on unstable/NaN/RuntimeError simulations (sbi failed-sim
  convention).
- **`make_spectral_dcm_prior(n_regions, a_mask, prior_variance=1/64)`** — independent
  Normal prior over the free A entries matching SPM12's N(0, 1/64); `prior_std =
  prior_variance ** 0.5`.
- **`train_npe(...)`** and **`build_sbi_posterior(...)`** — thin wrappers over the
  `sbi` NPE API, guarded by `ImportError` so importing the package without the `sbi`
  extra fails with a clear message.

Created `src/pyro_dcm/inference/sbi_embedding.py` (69 lines):

- **`CSDEmbeddingNet(input_dim, embed_dim=64, hidden_dim=128)`** — `nn.Module`
  compressing the `2*F*N*N` CSD vector to a compact summary. Shipped in `f96bf71`
  with `BatchNorm1d`; later switched to `LayerNorm` (current tree) so sbi's
  batch_size=1 shape probing works.

Created `src/pyro_dcm/inference/sbi_diagnostics.py` (153 lines as committed in
`f96bf71`) with:

- **`run_sbc_validation(...)`** — rank-based Simulation-Based Calibration (Talts et
  al. 2018).
- **`compare_sbi_svi_posteriors(...)`** — per-parameter SBI-vs-SVI posterior comparison.

Updated `src/pyro_dcm/inference/__init__.py` to re-export the SBI public API and
`pyproject.toml` to add the `sbi` optional-dependency extra.

### Task 2: Unit and integration tests

Created `tests/test_sbi_spectral.py` (213 lines) with **8 test functions** covering
simulator output shape/determinism/NaN-safety/forward-model agreement, prior
shape/SPM12-match, embedding shape/gradients, and an NPE training smoke path.
Per the commit message, 7 pass and 1 skips when `sbi` is not installed.

## Decisions Made

- **[26-01-D1] Free-A-only theta layout.** The simulator parameterizes only the free
  A entries selected by `a_mask` (length `n_free`), with noise spectra passed as a
  fixed `noise_params` dict, rather than the plan's packed
  `[A_free, noise_a, noise_b, noise_c]` theta of length `N*N + 4*N + 2`.
- **[26-01-D2] Explicit stability check for NaN handling.** Failed simulations
  (positive real eigenvalue, RuntimeError, or NaN CSD) return
  `torch.zeros(output_dim)`; no `warnings.warn` is emitted.
- **[26-01-D3] BatchNorm → LayerNorm in embedding.** Originally `BatchNorm1d`; changed
  to `LayerNorm` so sbi's batch_size=1 network shape-probing does not fail.

## Deviations from Plan

- **Simulator signature/layout** differs from the plan (see 26-01-D1); the plan's
  full `[A_free, noise_a, noise_b, noise_c]` parameter vector was not adopted.
- **`pyproject.toml` pins `sbi>=0.22`**, not the planned `sbi>=0.26`.
- **8 tests** were written, not the planned 10; the
  `make_stable_A_spectral`-based forward-match and the explicit diagnostics-output
  tests were folded down.
- **Diagnostics module** ships `run_sbc_validation` (hand-rolled rank-based SBC) and
  `compare_sbi_svi_posteriors`; the plan's `sbi.diagnostics.run_sbc` path was not used.

## Key Files

- `src/pyro_dcm/inference/sbi_spectral.py` — simulator closure, prior, NPE/posterior
  wrappers (`make_spectral_dcm_simulator`, `make_spectral_dcm_prior`, `train_npe`,
  `build_sbi_posterior`).
- `src/pyro_dcm/inference/sbi_embedding.py` — `CSDEmbeddingNet`.
- `src/pyro_dcm/inference/sbi_diagnostics.py` — `run_sbc_validation` (and originally
  `compare_sbi_svi_posteriors`). NOTE: this file was committed in `f96bf71` but was
  later rewritten/trimmed to the current 80-line `run_sbc_validation`-only version,
  which was committed in `c2ccf6f` (2026-06-09); the
  `(posterior, simulator, prior, n_trials=200, n_posterior_samples=1000)` signature in
  the current tree differs from the plan's `(posterior, prior, simulator, n_sbc=200, ...)`.
- `tests/test_sbi_spectral.py` — 8 SBI tests.
- `src/pyro_dcm/inference/__init__.py` — SBI re-exports were added in `f96bf71`; the
  current tree no longer re-exports the SBI API (later VL-focused edits left only the
  VL/forward-model exports).
- `pyproject.toml` — `sbi` optional extra (`sbi>=0.22`).

## Next Phase Readiness

The simulator, prior, embedding, and diagnostics primitives are in place for Phase
26-02 (SBI training pipeline / NPE sweep) to consume.
