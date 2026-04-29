# Quick Task 002 — Structure-Audit Migration (Waves 1-5) — SUMMARY

Executed 2026-04-29. Five atomic-commit waves landed in chronological order on `main`.
Each wave is independently `git revert`-able.

## Per-wave outcome

| Wave | Commit | One-line outcome |
|------|--------|-------------------|
| 1 | `815eb5c` | Janitorial cleanup of repo-root orphans (stackdump, VERIFICATION.md, HANDOFF_viz.md, HTML template). |
| 2 | `16a3312` | docs/ reorganized into the 5-subfolder layout (`00..04` + `legacy/`) per `DATA_ANALYSIS_PROJECT_TEMPLATE.md`; `guide_selection.md` moved to methods reference. |
| 3 | `fdbb056` | `CLAUDE.md` "Directory Structure (src/ layout)" synced to disk; tensor-shape convention bullet updated. |
| 4 | `61c9185` | New top-level `config.py` (PROJECT_ROOT, BENCHMARK_*_DIR, TAPAS_RDCM_PATH); validation scripts now import `TAPAS_RDCM_PATH` from it. |
| 5 | `33e03e1` | `scripts/debug/` created (4 phase-16 scripts moved); top-level `models/` -> `checkpoints/` rename + literal updates + pyproject.toml comment. |

## Disk-shape deltas (before -> after)

| Location | Before | After |
|----------|--------|-------|
| Repo root | `bash.exe.stackdump`, `VERIFICATION.md`, `models/` (gitignored) | (none) — replaced by `checkpoints/`, `config.py` |
| `docs/` | `02_pipeline_guide/`, `03_methods_reference/`, `04_scientific_reports/`, `dcm_circuit_explorer_template.html`, `HANDOFF_viz.md` | `00_current_todos/`, `01_project_protocol/`, `02_pipeline_guide/`, `03_methods_reference/`, `04_scientific_reports/`, `legacy/`, `README.md` |
| `src/pyro_dcm/utils/` | `__init__.py`, `circuit_viz.py`, `ode_integrator.py` | + `templates/dcm_circuit_explorer_template.html` (relocated runtime asset) |
| `scripts/` | `debug_phase16_*.py` (×3) at top level + untracked `diagnose_phase16_init_scale.py` | `scripts/debug/` (4 files; diagnose_ now tracked) |
| `models/test/task_final.pt` (gitignored artifact) | present | renamed -> `checkpoints/test/task_final.pt` (still gitignored) |

## Path literals migrated

| File | Before | After | Wave |
|------|--------|-------|------|
| `src/pyro_dcm/utils/circuit_viz.py:4` | `docs/dcm_circuit_explorer_template.html` | `src/pyro_dcm/utils/templates/dcm_circuit_explorer_template.html` | 1 |
| `docs/00_current_todos/HANDOFF_viz.md:8,15,268,297,301` (5 hits) | `docs/dcm_circuit_explorer_template.html` (or with `http://localhost:8080/` prefix) | `src/pyro_dcm/utils/templates/dcm_circuit_explorer_template.html` (and the localhost variant) | 1 |
| `.gitignore:24` | `bash.exe.stackdump` | `*.stackdump` | 1 |
| `.gitignore:28` | `/models/` | `/checkpoints/` | 5 |
| `validation/run_rdcm_validation.py:65` | `tapas_path = "C:/Users/aman0087/Documents/Github/tapas/rDCM"` | `tapas_path = str(TAPAS_RDCM_PATH)` | 4 |
| `validation/run_rdcm_validation.py:301` | `"C:/Users/aman0087/Documents/Github/tapas/rDCM. "` | `f"{TAPAS_RDCM_PATH}. "` | 4 |
| `validation/run_validation.py:102` | `"'C:/Users/aman0087/Documents/Github/tapas/rDCM'"` | `f"'{TAPAS_RDCM_PATH}'"` | 4 |
| `benchmarks/runners/task_amortized.py:217` | `["models/task_final.pt", "models/task_ci.pt"]` | `["checkpoints/task_final.pt", "checkpoints/task_ci.pt"]` | 5 |
| `benchmarks/runners/spectral_amortized.py:238-239` | `"models/spectral_final.pt"`, `"models/spectral_ci.pt"` | `"checkpoints/spectral_final.pt"`, `"checkpoints/spectral_ci.pt"` | 5 |
| `scripts/train_amortized_guide.py:19,27,124,125` (4 hits) | `models/task/`, `models/spectral/`, `models/`, `models/` | `checkpoints/task/`, `checkpoints/spectral/`, `checkpoints/`, `checkpoints/` | 5 |
| `pyproject.toml:5-9` | "top-level `models/` directory ... that breaks `from pyro_dcm.models import …`" | "top-level `models/` directory (now renamed to `checkpoints/`) ... Kept explicit for forward stability." | 5 |

## CLAUDE.md directory-tree edits (Wave 3)

| Subtree | Edit |
|---------|------|
| `connectivity/` block | DROPPED (dir does not exist on disk) |
| `inference/` block | DROPPED (dir does not exist on disk) |
| `forward_models/rdcm_likelihood.py` | RENAMED to `rdcm_forward.py` |
| `forward_models/` | ADDED `coupled_system.py`, `rdcm_posterior.py`, `spectral_noise.py` |
| `guides/meanfield.py` | DROPPED (file does not exist) |
| `guides/` | ADDED `parameter_packing.py`, `summary_networks.py` |
| `utils/spectral_utils.py`, `utils/diagnostics.py` | DROPPED (files do not exist) |
| `utils/` | ADDED `circuit_viz.py`, `templates/` |
| Tensor-shape convention bullet | Updated wording: "NumPy-style ``Parameters`` blocks with explicit shape annotations" |

## Plan deviations (Rule 3 — blocking, fixed automatically)

1. **`scripts/train_amortized_guide.py` had 4 hits of `models/` literals** that the audit report
   (`01-src-layout-and-claude-md.md`) and the plan claimed it did NOT. After the
   `models/ -> checkpoints/` rename in Wave 5 these would have been stale documentation. They
   were updated in the same Wave 5 commit (CLI defaults + docstring usage examples).
   - Lines 19, 27 (docstring usage examples)
   - Lines 124, 125 (argparse `--output-dir` default + help text)
   - Documented in the Wave 5 commit message.

No Rule-1 (bug), Rule-2 (missing critical), or Rule-4 (architectural) deviations.

## Authentication gates

None encountered.

## Verification snapshot

### `git log --oneline -n 5`

```
33e03e1 chore(002): wave 5 — scripts/debug/ + rename top-level models/ -> checkpoints/
61c9185 feat(002): wave 4 — top-level config.py for TAPAS_RDCM_PATH centralization
fdbb056 docs(002): wave 3 — sync CLAUDE.md directory tree to disk
16a3312 docs(002): wave 2 — reorganize docs/ into 5-subfolder layout
815eb5c chore(002): wave 1 — janitorial cleanup of repo-root orphans
```

### `pytest tests/ -m "not slow" --collect-only -q | tail -3` (after Wave 5)

```
tests/test_validation_export.py::TestCompareModelRanking::test_pairwise_details

484/521 tests collected (37 deselected) in 32.35s
```

(Identical collection count after Waves 1, 4, and 5; no import failures.)

### `grep -rn 'C:/Users/aman0087/Documents/Github/tapas/rDCM' --include='*.py' .`

```
config.py:44:        "C:/Users/aman0087/Documents/Github/tapas/rDCM",
```

(Only the env-var default fallback inside `config.py` remains; all `validation/` literals
are gone. Override on a different machine via `TAPAS_RDCM_PATH=...`.)

### `python -c "import pyro_dcm; import pyro_dcm.utils.circuit_viz; from config import TAPAS_RDCM_PATH; print('OVERALL_OK')"`

```
OVERALL_OK
```

### `python -c "from importlib.resources import files; p = files('pyro_dcm.utils') / 'templates' / 'dcm_circuit_explorer_template.html'; print(p.is_file())"`

```
True
```

(Template asset is reachable via the installed package path.)

### `git status --short` (post Wave 5, executor-local)

```
?? .claude/
?? .planning/quick/002-structure-audit-migration-waves-1-5/
?? .planning/research/structure-audit/
```

(Three untracked entries: `.claude/` is the local user-instructions dir,
`.planning/quick/...` and `.planning/research/...` are this task's planning artifacts —
the orchestrator commits these after the executor returns. Working tree is otherwise clean.)

## Deferred future work

(Verbatim from the plan's `<deferred_future_work>` block.)

1. **Mass-centralize `benchmarks/` paths** through `config.py::BENCHMARK_*_DIR` constants.
   Audit `03-conventions-and-config.md` estimated ~15 hardcoded `benchmarks/results`,
   `benchmarks/figures`, `benchmarks/fixtures` literals across `benchmarks/runners/*.py`,
   `benchmarks/generate_fixtures.py`, etc. Out of scope for this migration because of the
   call-site count and risk of touching active CLI surfaces.

2. **MATLAB / planning-doc absolute paths.** `validation/matlab_scripts/run_tapas_rdcm.m:29`,
   `.planning/phases/06-*/*` MATLAB-batch examples, `.planning/STATE.md` historical refs --
   intentionally left as-is.

3. **`benchmarks/config.py`** -- a locally scoped `config.py` already exists under
   `benchmarks/`. A future cleanup could fold it into the top-level `config.py` or rename
   to `benchmarks/_paths.py` to disambiguate.

4. **`docs/01_project_protocol/`** is currently a directory with only a placeholder README.
   Future work: migrate the relevant subset of `CLAUDE.md` "Coding Conventions" into a
   first-class `01_project_protocol/coding_standards.md` once the protocol stabilizes.

5. **Test surfaced during Phase 17** (`tests/test_task_simulator.py::TestSimulatorOutputStructure::test_simulator_output_keys`)
   pre-dates this plan and is tracked separately as a Phase-16 follow-up. NOT this plan's
   problem.
