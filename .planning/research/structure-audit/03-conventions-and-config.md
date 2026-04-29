# Conventions & Config Audit

## Verdict
The repo is excellent on docstrings, type hints, and `from __future__ import annotations` (effectively 100% conformant) but lacks a top-level `config.py`, scatters relative-string and absolute Windows paths through `benchmarks/` and `validation/`, and has accumulated unnumbered `debug_*`/`diagnose_*` scripts that don't follow the `{stage}_{verb}_{noun}.py` pipeline pattern.

## 1. `from __future__ import annotations` adoption
- `src/pyro_dcm/`: 28/28 files conformant (100%) — every package module incl. all `__init__.py`.
- `tests/`: 46/46 files conformant (100%, includes `conftest.py` and `__init__.py`).
- `scripts/`: 7/7 conformant (100%).
- `benchmarks/`: 19/19 conformant (100%, includes `runners/__init__.py`).
- `validation/`: 4/4 *.py modules conformant; `validation/__init__.py` is a pure docstring package marker (no executable code) — acceptable.
- Worst offenders: **none**. This is the strongest area of compliance in the project.

## 2. Type hint syntax (legacy `Optional`/`List`/etc.)
- Total occurrences across whole repo: **0**.
- Confirmed via two patterns: `\b(Optional|List|Dict|Tuple|Union|Set|FrozenSet)\[` (zero hits) and `from typing import ...Optional|List|Dict|Tuple|Union` (zero hits).
- `from typing import` is still used (e.g., `models/guides.py:19` imports `Any, Callable`) — these are legal in 3.10+ native syntax, no action needed.
- Top offenders: none.

## 3. NumPy-style docstrings (spot-checks)
| Module | Style | Notes |
|---|---|---|
| `src/pyro_dcm/forward_models/neural_state.py` | NumPy | `Parameters`, `Returns`, `Notes`, `Examples` headings; shapes documented as `(N, N)`. |
| `src/pyro_dcm/forward_models/balloon_model.py` | NumPy | Module + class + method all NumPy; SPM12 references in `References` block. |
| `src/pyro_dcm/forward_models/spectral_transfer.py` | NumPy | `Parameters`/`Returns`/`Examples`; cites [REF-010] Eq. 3. |
| `src/pyro_dcm/models/task_dcm_model.py` | NumPy | Module-level `References` heading; private helpers have one-line docstrings (allowed by standard). |
| `src/pyro_dcm/models/guides.py` | NumPy | Module + registry attribute docs both NumPy-style. |
| `src/pyro_dcm/utils/ode_integrator.py` | NumPy | Class `Parameters` + `Examples`; method docstrings continue NumPy style. |
- Repo-wide check: zero `^\s*(Args:|Returns:)\s*$` Google-style headings found anywhere; 93 `Parameters` headings across 22 src modules. Pass.

## 4. Pipeline script naming
- Conformant (or acceptable non-pipeline): `train_amortized_guide.py`, `generate_training_data.py`, `demo_bilinear_consumer.py` — these are `{verb}_{noun}.py` analysis/utility scripts, not numbered pipeline stages, which the standard explicitly allows.
- Non-conformant scripts in `scripts/`:
  - `debug_phase16_nan_seeds.py`
  - `debug_phase16_fixture_check.py`
  - `debug_phase16_pool_smoke.py`
  - `diagnose_phase16_init_scale.py` (untracked)
  These start with `debug_`/`diagnose_` and embed a phase tag (`phase16`) into the filename. They aren't pipeline stages so `{stage}_{verb}_{noun}.py` doesn't apply, but they also aren't simple `{verb}_{noun}.py` either — `debug` and `diagnose` are verbs, but `phase16_init_scale` is a contextual qualifier rather than a noun. Suggest moving these to `scripts/debug/` (or deleting once their phases close) and renaming to e.g. `diagnose_init_scale.py`.
- `benchmarks/`: a flat package, not pipeline scripts — naming is acceptable (`run_all_benchmarks.py`, `calibration_sweep.py`, `generate_fixtures.py`, `plotting.py` etc.). `benchmarks/runners/spm_reference.py` etc. are dispatcher modules, fine.

## 5. Config centralization
- Top-level `config.py`: **no** (only `benchmarks/config.py` exists, scoped to benchmark dataclasses).
- `configs/` directory holds JSON model fixtures (`heart2adapt_dcm_config.json`), not Path constants.
- Top files with hardcoded path strings (relative or absolute):
  1. `validation/run_rdcm_validation.py:65` — `tapas_path = "C:/Users/aman0087/Documents/Github/tapas/rDCM"` (absolute Windows path baked in)
  2. `validation/run_rdcm_validation.py:301` — same path repeated in error message
  3. `validation/run_validation.py:102` — same `C:/Users/aman0087/...tapas/rDCM` in error string
  4. `benchmarks/config.py:63` — `output_dir: str = "benchmarks/results"` default
  5. `benchmarks/calibration_sweep.py:341,698` — `default="benchmarks/results"` and same as fn arg
  6. `benchmarks/calibration_analysis.py:102,103,239,392,495` — `benchmarks/results/calibration_results.json` and `benchmarks/figures` literals (5 sites in one file)
  7. `benchmarks/plotting.py:1962,1984` — `benchmarks/results/benchmark_results.json` literal
  8. `benchmarks/fixtures.py:39,96` — `fixtures_dir: str = "benchmarks/fixtures"` (twice)
  9. `benchmarks/generate_fixtures.py:573` — `default="benchmarks/fixtures"`
  10. `benchmarks/run_all_benchmarks.py:273` — `default="benchmarks/results"`
- Also: `benchmarks/runners/spm_reference.py:20` (`Path("validation/VALIDATION_REPORT.md")`), `tests/test_circuit_viz.py:93` (`Path("configs/heart2adapt_dcm_config.json")`), `scripts/generate_training_data.py:383` (`default="data/training/"`).
- **No occurrences** of `data/`, `figures/`, `/home/` used as path roots in `src/pyro_dcm/` itself — the library is clean. The drift is concentrated in `benchmarks/` and `validation/`.

## 6. Fitted-attribute trailing underscore
- `src/pyro_dcm/models/`: classes are mostly Pyro generative-function modules — no `fit()`/`update()` post-state to track. Pass by absence.
- `src/pyro_dcm/guides/parameter_packing.py:204-205, 390-391`: uses `self.mean_` and `self.std_` correctly after `fit_standardization()`. Pass.
- No `inference/` package exists yet (CLAUDE.md mentions one — see audit doc 01).
- No obvious violations found.

## 7. Tensor shape comments (spot-check)
- `forward_models/neural_state.py`: shapes documented inside docstrings (`shape ``(N, N)``, ``(J, N, N)```) rather than as `# shape:` comments. Consistent with CLAUDE.md tensor-shape table (`A: (N, N)`).
- `forward_models/balloon_model.py`: per-region shapes given as `shape ``(N,)``` in `Parameters`. Consistent with table convention.
- `forward_models/spectral_transfer.py`: docstring says `shape ``(n_freqs,)```. CLAUDE.md table uses `F` (frequency bins) — minor naming drift (`n_freqs` vs `F`) but the meaning is unambiguous.
- The CLAUDE.md-prescribed inline `# shape: (n_regions, n_timepoints)` comment style is **not** the dominant convention in the repo; instead shapes live inside NumPy `Parameters`/`Returns` blocks, which is arguably better. CLAUDE.md should be updated to reflect actual practice rather than the other way around.

## Top 5 fixes ranked by effort vs impact
1. **[LOW effort, HIGH impact]** Create top-level `config.py` with `PROJECT_ROOT = Path(__file__).parent`, `BENCHMARK_RESULTS_DIR`, `BENCHMARK_FIGURES_DIR`, `FIXTURES_DIR`, `VALIDATION_REPORT_PATH`, and `TAPAS_RDCM_PATH = Path(os.environ.get("TAPAS_RDCM", "..."))`. Removes ~15 hardcoded literals across `benchmarks/` and `validation/`.
2. **[LOW effort, HIGH impact]** Replace the absolute `C:/Users/aman0087/...tapas/rDCM` path in `validation/run_rdcm_validation.py:65,301` and `validation/run_validation.py:102` with an env-var read or config import — currently un-runnable on any other machine including Monash M3.
3. **[LOW effort, MEDIUM impact]** Move `scripts/debug_phase16_*.py` and `scripts/diagnose_phase16_init_scale.py` into `scripts/debug/` (or delete once Phase 16.1 closes); add a `.gitignore` rule for `scripts/debug/_scratch_*.py` to prevent future drift.
4. **[LOW effort, LOW impact]** Update `dcm_pytorch/CLAUDE.md` "Tensor Shape Conventions" section to acknowledge that shapes are documented in NumPy `Parameters` blocks, not inline `# shape:` comments — codify the actual practice.
5. **[MEDIUM effort, MEDIUM impact]** Add a ruff rule (`I002` `from __future__ import annotations` required) and a CI test to enforce the now-100% adoption rate so it can't regress as new modules are added.
