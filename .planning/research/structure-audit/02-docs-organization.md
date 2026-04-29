# docs/ Organization Audit

## Verdict
Partially compliant: 3 of 5 mandated subfolders exist, no `legacy/`, no navigation hub `README.md`, no per-subfolder `README.md` indexes, and 2 root-level orphan files violate the "no root clutter" rule.

## Subfolder presence
| Standard subfolder | Present | Has README.md | Notes |
|---|---|---|---|
| `00_current_todos/` | No | n/a | Missing entirely. Active todos likely live in `.planning/` instead. |
| `01_project_protocol/` | No | n/a | Missing entirely. No environment-setup or folder-structure docs in `docs/`. |
| `02_pipeline_guide/` | Yes | No | 3 markdown files, no index. |
| `03_methods_reference/` | Yes | No | 4 files (markdown, tex, bib), no index. |
| `04_scientific_reports/` | Yes | No | 1 file, no index. |
| `legacy/` | No | n/a | Missing. No archive location for retired docs. |
| `docs/README.md` (hub) | No | n/a | No navigation hub. New visitors have no entry point. |

## Root-of-docs orphans
| File | Recommended destination | Why |
|---|---|---|
| `HANDOFF_viz.md` | `00_current_todos/` (after creating it) | Handoff/active-work doc describing an unfinished `circuit_viz.py` task — matches "active feature development notes" per template. |
| `dcm_circuit_explorer_template.html` | Out of `docs/` entirely (e.g., `src/pyro_dcm/utils/templates/` or `assets/`) | Not documentation — it is a runtime HTML template consumed by code per `HANDOFF_viz.md`. Storing executable/runtime assets under `docs/` is a category error. |

## Per-subfolder content review

### `02_pipeline_guide/`
- `quickstart.md` (8 KB) — fits purpose.
- `consumer_bilinear_quickstart.md` (13 KB) — fits purpose.
- `guide_selection.md` (13 KB) — name suggests SVI guide selection (AutoNormal vs AutoLowRankMVN etc.), which is closer to a methods/how-to-choose reference than a pipeline-execution guide. Possibly misfiled — verify whether content is "how to run" vs "which method to pick"; if the latter, move to `03_methods_reference/`.
- Missing `README.md` index.

### `03_methods_reference/`
- `equations.md` (9 KB) — fits purpose.
- `methods.md` (19 KB) — fits purpose.
- `methods.tex` (9 KB) — manuscript fragment; arguably belongs in `04_scientific_reports/` or a separate `manuscripts/` area, but acceptable here as authoritative methods source.
- `references.bib` (5 KB) — bibliography, fits.
- Possible duplication risk between `equations.md` and `methods.md`/`methods.tex` (all describe equations); confirm scope split.
- Missing `README.md` index.

### `04_scientific_reports/`
- `benchmark_report.md` (23 KB, 575 lines) — fits purpose (results doc).
- Near-empty subfolder (single file). Fine for current project stage but should grow with publication artefacts; no figures/ or PDFs colocated.
- Missing `README.md` index.

### Missing subfolders impact
- No `00_current_todos/` means active work (e.g., `HANDOFF_viz.md`) has no proper home and gets dumped at root.
- No `01_project_protocol/` means env-setup info likely lives only in top-level `README.md` / `CLAUDE.md`; new contributors lack a curated setup index in `docs/`.
- No `legacy/` means there is no archive path; retired docs would have to be deleted rather than preserved.

## Broken/outdated references
- N/a — `docs/README.md` does not exist, so no hub links to validate.
- `HANDOFF_viz.md` references `src/pyro_dcm/utils/circuit_viz.py` (planned), `Approach Avoid Anxiety/dcm_circuit_explorer.html`, and `configs/heart2adapt_dcm_config.json`. These are outside the audit scope but should be sanity-checked when the file is relocated.

## Migration plan (ranked by effort)
1. **[LOW EFFORT]** Create empty `docs/00_current_todos/`, `docs/01_project_protocol/`, and `docs/legacy/` directories with stub `README.md` files describing each subfolder's purpose.
2. **[LOW EFFORT]** Move `docs/HANDOFF_viz.md` into `docs/00_current_todos/HANDOFF_viz.md` (preserves active-work-item semantics).
3. **[LOW EFFORT]** Move `docs/dcm_circuit_explorer_template.html` out of `docs/` to a code/assets directory (e.g., `src/pyro_dcm/utils/templates/`); update any references in `HANDOFF_viz.md` and consumer code.
4. **[LOW EFFORT]** Add `README.md` index files to `02_pipeline_guide/`, `03_methods_reference/`, and `04_scientific_reports/` listing each file with a one-line description.
5. **[LOW EFFORT]** Author top-level `docs/README.md` as navigation hub linking to all 6 subfolders.
6. **[MEDIUM EFFORT]** Inspect `02_pipeline_guide/guide_selection.md`; if it is a method-selection rationale rather than execution steps, move it to `03_methods_reference/`.
7. **[MEDIUM EFFORT]** Populate `01_project_protocol/` with `ENVIRONMENT_SETUP.md` and `FOLDER_STRUCTURE.md` extracted from top-level `README.md`/`CLAUDE.md`.
8. **[LOW EFFORT]** Reconcile potential overlap between `equations.md` and `methods.md`/`methods.tex` in `03_methods_reference/` (decide canonical source, cross-link the others).
