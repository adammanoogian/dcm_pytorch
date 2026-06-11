---
phase: 32
plan: 01
subsystem: spm12-cross-validation
tags: [csd-injection, spm12, free-energy-matching, pitfall-s4, validation-bridge]
requires:
  - "29-03 (C-order CSD contract, tests/test_csd_corder_roundtrip.py)"
  - "validation/export_to_mat.py spectral exporter conventions"
provides:
  - "export_spectral_dcm_csd_for_spm() — injects a (F,N,N) complex CSD into the SPM DCM struct"
  - "run_spm_spectral_dcm_csd_injected.m — SPM batch fitting the injected CSD, not a MAR recompute"
  - "tests/test_csd_injection_roundtrip.py — S4 transpose guard for the injected CSD path"
affects:
  - "32-03 (strict-5%-matched-F cross-validation: consumes the injected-CSD bridge)"
tech-stack:
  added: []
  patterns:
    - "Same-data free-energy matching: inject identical CSD into both VL and SPM engines"
    - "C-order (F,N,N) .mat round-trip with no transpose (pitfall S4)"
key-files:
  created:
    - "validation/matlab_scripts/run_spm_spectral_dcm_csd_injected.m"
    - "tests/test_csd_injection_roundtrip.py"
  modified:
    - "validation/export_to_mat.py"
decisions:
  - "[32-01-D1] Injected CSD wrapped as a one-element cell {squeeze(csd)} only if not already a cell."
  - "[32-01-D2] BOLD-less export synthesizes a (len(freqs)*8, N) float64 zeros DCM.Y.y placeholder."
  - "[32-01-D3] mypy ndarray type-arg + scipy import-untyped errors are pre-existing file conventions, not gated."
duration: "~25m"
completed: 2026-06-11
---

# Phase 32 Plan 01: Same-CSD Injection Bridge Summary

**One-liner:** A Python-computed analytic `(F, N, N)` complex CSD now injects element-identical into the SPM12 `DCM.Y.csd`/`DCM.Y.Hz` struct (C-order, no transpose — S4 guarded), and a forked MATLAB batch makes `spm_dcm_fmri_csd` fit THAT injected CSD instead of recomputing it from BOLD via MAR — the precondition for a strict 5%-matched-nat free-energy comparison in Plan 32-03.

## What Was Built

### Task 1 — `export_spectral_dcm_csd_for_spm()` (commit a4582a3)
Added to `validation/export_to_mat.py`, modeled on `export_spectral_dcm_for_spm` (the BOLD-only exporter, left untouched). It:
- Casts `observed_csd` to `np.complex128` and `freqs` to `np.float64` (pitfall N3).
- Writes `DCM.Y.csd = observed_csd` (bare 3-D complex array) and `DCM.Y.Hz = freqs.reshape(-1, 1)`.
- Keeps the spectral struct conventions: `induced=1`, `analysis='CSD'`, `nograph=1`, `order=8`, constant microtime `U.u`.
- Optional `bold_data`; when None synthesizes a shape-valid `(len(freqs)*8, N)` float64 zeros `DCM.Y.y` placeholder with consistent `DCM.v`/`DCM.n`.
- Docstring documents the C-order `(F,N,N)` layout, the 1-based MATLAB ↔ 0-based Python index correspondence, the no-transpose contract, and cites pitfall S4.

### Task 2 — `run_spm_spectral_dcm_csd_injected.m` (commit 158bc22)
Forked from `run_spm_spectral_dcm.m` (untouched). Same SPM12 addpath, `spm('defaults','FMRI')`, `DCM_INPUT_PATH`/`DCM_OUTPUT_PATH` env I/O, `induced=1`/`analysis='CSD'` forcing. The single behavioral change: after `load`, it verifies `isfield(DCM.Y,'csd')` and `isfield(DCM.Y,'Hz')` (loud failure naming the missing field), wraps `DCM.Y.csd` as a one-element cell `{squeeze(...)}` only if bare, makes `DCM.Y.Hz` a column vector, then runs `spm_dcm_fmri_csd` on the injected CSD (which skips the internal `spm_dcm_fmri_csd_data` MAR estimation when `DCM.Y.csd` is populated). Prints the `Ep.A(1,2)` vs `Ep.A(2,1)` S4 asymmetry readout and saves the identical `results.Ep_A`/`results.Cp`/`results.F`(+`Hz`/`transit`/`decay`) block that `load_spm_results` already parses.

### Task 3 — `tests/test_csd_injection_roundtrip.py` (commit d72a514)
Two module-level `@pytest.mark.vl` tests (no MATLAB needed, 0.54s):
- `test_injected_csd_roundtrips_asymmetric`: exports a deterministic asymmetric `(F=4,N=2,N=2)` complex CSD, reloads via `scipy.io.loadmat`, asserts `[w,i,j]` preserved on real+imag, and CRITICALLY `loaded[0,0,1] != loaded[0,1,0]` so a transpose bug fails.
- `test_injected_freqs_roundtrip`: asserts `DCM.Y.Hz` reloads equal to `freqs` (float64).

## Verification

- **Gate (Task 1, first step):** `pytest tests/test_csd_corder_roundtrip.py -m vl -q` → **2 passed** BEFORE any build (VLSPM-03 C-order contract green).
- **Combined:** `pytest tests/test_csd_corder_roundtrip.py tests/test_csd_injection_roundtrip.py -m vl -q` → **4 passed** (2.97s).
- **ruff:** `ruff check validation/export_to_mat.py tests/test_csd_injection_roundtrip.py` → **clean** (both changed files).
- **mypy:** new errors are exclusively the pre-existing file-convention classes — bare `np.ndarray` `[type-arg]` (used by every function in `export_to_mat.py`) and `scipy` `[import-untyped]` (the repo ships no scipy stubs). No new error class introduced.

## Deviations from Plan

### Auto-fixed / Auto-resolved

**1. [Rule 3 — Blocking] MATLAB parse-check could not run (licensing).**
- **Found during:** Task 2 verify. MATLAB R2022a IS installed (`C:\Program Files\MATLAB\R2022a\bin\matlab.exe`), but `matlab -batch` failed with a license checkout error (`-15,10032`), so the optional `mtree` parse sanity check could not execute.
- **Resolution:** Fell back to the plan's explicitly-allowed grep verification — confirmed `DCM.Y.csd`, `DCM.Y.Hz`, `spm_dcm_fmri_csd`, and the `results.Ep_A`/`results.Cp`/`results.F` save block all present (22 token occurrences). Full SPM estimation is Plan 32-03 by design; not attempted here.

**2. [Doc-only] mypy `ndarray` type-arg / scipy import-untyped not gated.**
- The new function and test add 5 `[type-arg]` errors and `scipy [import-untyped]`, identical in class to the 15 baseline errors already in `export_to_mat.py` and matching the repo-wide bare-`np.ndarray` convention. Honored the existing file style rather than introducing a divergent typing scheme on one function (decision 32-01-D3).

## Decisions Made

- **[32-01-D1] Injected CSD cell-wrapping is conditional.** `DCM.Y.csd` is wrapped `{squeeze(csd)}` only if `~iscell`, because `savemat` writes a bare numeric `(Nf,n,n)` array while `spm_dcm_fmri_csd` expects a one-element cell block. Idempotent for an already-cell input.
- **[32-01-D2] BOLD-less placeholder shape is `(len(freqs)*8, N)`.** Once `DCM.Y.csd` is injected the `y` values are unused, but `DCM.v`/`DCM.n` must stay valid; the `*8` matches the spectral `order` so the struct is internally consistent.
- **[32-01-D3] Pre-existing mypy/scipy typing errors are not in scope.** Scoped to the plan's declared files following their established convention; no repo-wide typing change.

## Next Phase Readiness

The same-CSD bridge is layout-safe and S4-guarded. Plan 32-03 (strict-5%-matched-F cross-validation) can now: (1) call `export_spectral_dcm_csd_for_spm` with the exact CSD the VL engine fits, (2) run `run_spm_spectral_dcm_csd_injected.m` under MATLAB+SPM12, (3) compare VL F vs SPM F on identical data. **Carry-forward:** the MATLAB script is unexecuted (license/by-design); its first live run is Plan 32-03 and must confirm `DCM.Y.csd`-populated actually bypasses `spm_dcm_fmri_csd_data` in this SPM12 build, plus that the `Ep.A(1,2)`/`Ep.A(2,1)` readout matches the injected asymmetric ground truth (0.15 / 0.10).

## Commits

| Commit  | Task | Description |
| ------- | ---- | ----------- |
| a4582a3 | 1    | feat(32-01): inject precomputed CSD into SPM DCM struct |
| 158bc22 | 2    | feat(32-01): SPM batch fitting injected CSD instead of MAR recompute |
| d72a514 | 3    | test(32-01): injected-CSD round-trip transpose guard |
