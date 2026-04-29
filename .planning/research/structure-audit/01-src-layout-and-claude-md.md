# src/ Layout & CLAUDE.md Compliance Audit

## Verdict
DRIFT — `src/` layout is structurally correct and `pyproject.toml` is configured for it, but the package is missing two CLAUDE.md-promised submodules (`connectivity/`, `inference/`), the top-level `models/` directory is a confusing (though non-importable) shadow of `src/pyro_dcm/models/`, and several orphan files litter the repo root.

## Conforms
- All Python package code lives under `src/pyro_dcm/` — no importable Python modules at repo root.
- `pyproject.toml` explicitly targets `src/pyro_dcm` for the wheel build (hatchling), and the inline comment shows the maintainer is aware of the top-level `models/` collision risk.
- `tests/`, `scripts/`, `docs/`, `benchmarks/`, `validation/`, `figures/`, `data/`, `configs/`, `cluster/` all sit outside `src/` as the SRC_LAYOUT_PATTERN expects.
- `src/pyro_dcm/__init__.py` exists; every existing submodule has an `__init__.py`.

## Drift (ranked by blast radius)

### 1. HIGH — Two CLAUDE.md submodules do not exist on disk
- **What:** `src/pyro_dcm/connectivity/` and `src/pyro_dcm/inference/` are absent.
- **Standard says:** CLAUDE.md "Directory Structure" enumerates `connectivity/` (with `static_a.py`, `structural_mask.py`) and `inference/` (with `svi_runner.py`, `nuts_validator.py`, `model_comparison.py`).
- **Actual:** Neither directory exists; no SVI runner, NUTS validator, or model-comparison module is present in the package. Static-A / structural-mask logic, if it lives anywhere, is not in the documented home.
- **Fix:** Either create the submodules and migrate/implement the missing files, or delete these sections from CLAUDE.md so the spec matches reality.

### 2. MEDIUM — Top-level `models/` shadows `src/pyro_dcm/models/`
- **What:** `C:\Users\aman0087\Documents\Github\dcm_pytorch\models\test\task_final.pt` (only file under top-level `models/`).
- **Standard says:** SRC_LAYOUT_PATTERN — "Project root stays clean — no confusion between package name and project directory."
- **Actual:** The directory contains only a `.pt` checkpoint (not Python), so it does not violate import isolation, but the name collision is real enough that `pyproject.toml` carries a defensive comment about hatchling auto-discovery shipping wheels missing `src/pyro_dcm/models/`.
- **Fix:** Rename top-level `models/` to `checkpoints/` (or move under `data/` or `benchmarks/results/`) to remove the foot-gun.

### 3. MEDIUM — `models/guides.py` exists in two homes
- **What:** `src/pyro_dcm/models/guides.py` (17 KB) and `src/pyro_dcm/guides/` (a separate sibling package with `amortized_flow.py`, `parameter_packing.py`, `summary_networks.py`).
- **Standard says:** CLAUDE.md lists `guides.py` under `models/` AND a separate `guides/` package with `meanfield.py` + `amortized_flow.py`.
- **Actual:** Both coexist, but `guides/meanfield.py` (CLAUDE.md-listed) is missing; instead `guides/parameter_packing.py` and `guides/summary_networks.py` (not in CLAUDE.md) live there.
- **Fix:** Decide one home for guide code; reconcile CLAUDE.md.

### 4. LOW — Repo-root orphans
- **What:** `bash.exe.stackdump`, `VERIFICATION.md`.
- **Standard says:** SRC_LAYOUT_PATTERN keeps project root minimal; CLAUDE.md routes reports to `docs/04_scientific_reports/` and planning artifacts to `.planning/`.
- **Actual:** `bash.exe.stackdump` is a Cygwin crash artifact (10 lines of frame addresses). `VERIFICATION.md` is a v0.1.0-foundation integration report dated 2026-03-31, sitting next to `README.md`.
- **Fix:** Delete `bash.exe.stackdump` and add `*.stackdump` to `.gitignore`. Move `VERIFICATION.md` to `docs/04_scientific_reports/` or `.planning/`.

## Aspirational vs real (CLAUDE.md submodules)

| Submodule | In CLAUDE.md | Exists on disk | Notes |
|---|---|---|---|
| `forward_models/` | yes | yes | Has extra files vs spec |
| `models/` | yes | yes | Has extra `guides.py`, `amortized_wrappers.py` |
| `guides/` | yes | yes | `meanfield.py` missing; 2 unlisted files present |
| `connectivity/` | yes | NO | Entire submodule absent |
| `simulators/` | yes | yes | Matches spec |
| `inference/` | yes | NO | Entire submodule absent |
| `utils/` | yes | yes | `spectral_utils.py`, `diagnostics.py` missing; `circuit_viz.py` unlisted |

## Files in `src/pyro_dcm/` not described in CLAUDE.md
- `forward_models/coupled_system.py`
- `forward_models/rdcm_forward.py` (CLAUDE.md lists `rdcm_likelihood.py` — likely renamed)
- `forward_models/rdcm_posterior.py`
- `forward_models/spectral_noise.py`
- `guides/parameter_packing.py`
- `guides/summary_networks.py`
- `utils/circuit_viz.py`

CLAUDE.md-listed but missing: `guides/meanfield.py`, `utils/spectral_utils.py`, `utils/diagnostics.py`, all of `connectivity/`, all of `inference/`, `forward_models/rdcm_likelihood.py` (likely renamed to `rdcm_forward.py`).

## Top-level orphans
- `bash.exe.stackdump` — delete; add `*.stackdump` to `.gitignore`.
- `VERIFICATION.md` — move to `docs/04_scientific_reports/` (matches CLAUDE.md doc taxonomy).
- `models/` (top-level, contains only `test/task_final.pt`) — rename to `checkpoints/` or relocate the `.pt` under `data/` / `benchmarks/results/`; the defensive comment in `pyproject.toml` becomes unnecessary once renamed.
