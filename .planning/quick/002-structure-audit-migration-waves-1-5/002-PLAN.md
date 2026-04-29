---
quick_id: 002
slug: structure-audit-migration-waves-1-5
type: execute
wave: 1
depends_on: []
autonomous: true
files_modified:
  # Wave 1
  - bash.exe.stackdump                              # deleted
  - .gitignore                                      # *.stackdump glob + checkpoints/ rename
  - VERIFICATION.md                                 # moved -> docs/04_scientific_reports/
  - docs/HANDOFF_viz.md                             # moved -> docs/00_current_todos/
  - docs/dcm_circuit_explorer_template.html         # moved -> src/pyro_dcm/utils/templates/
  - src/pyro_dcm/utils/templates/dcm_circuit_explorer_template.html  # new location
  - src/pyro_dcm/utils/circuit_viz.py               # docstring path update
  # Wave 2
  - docs/README.md                                  # new navigation hub
  - docs/00_current_todos/README.md                 # new index
  - docs/01_project_protocol/README.md              # new index
  - docs/02_pipeline_guide/README.md                # new index
  - docs/03_methods_reference/README.md             # new index
  - docs/04_scientific_reports/README.md            # new index
  - docs/legacy/README.md                           # new placeholder
  - docs/02_pipeline_guide/guide_selection.md       # moved -> 03_methods_reference/
  # Wave 3
  - CLAUDE.md                                       # directory tree + tensor-shape bullet
  # Wave 4
  - config.py                                       # NEW top-level path centralizer
  - validation/run_rdcm_validation.py               # TAPAS_RDCM_PATH import (lines ~65, ~301)
  - validation/run_validation.py                    # TAPAS_RDCM_PATH import (line ~102)
  # Wave 5
  - scripts/debug/debug_phase16_fixture_check.py    # moved
  - scripts/debug/debug_phase16_nan_seeds.py        # moved
  - scripts/debug/debug_phase16_pool_smoke.py       # moved
  - scripts/debug/diagnose_phase16_init_scale.py    # moved (was untracked)
  - models/test/task_final.pt                       # renamed -> checkpoints/test/task_final.pt
  - benchmarks/runners/task_amortized.py            # path literal update
  - benchmarks/runners/spectral_amortized.py        # path literal update
  - pyproject.toml                                  # update defensive comment about models/

must_haves:
  truths:
    - "Repo root contains no stray crash dumps (bash.exe.stackdump deleted; *.stackdump in .gitignore)"
    - "All HTML/template runtime assets live under src/, not docs/"
    - "docs/ has the 5-subfolder layout from DATA_ANALYSIS_PROJECT_TEMPLATE.md (00..04 + legacy + top-level README)"
    - "CLAUDE.md 'Directory Structure' section matches what `ls src/pyro_dcm/**` actually shows"
    - "validation/run_rdcm_validation.py and validation/run_validation.py have NO baked-in absolute path to C:/Users/aman0087/...tapas/rDCM (env-var override available via TAPAS_RDCM_PATH)"
    - "scripts/ has only top-level pipeline/training scripts; phase-specific debug & diagnostic scripts live under scripts/debug/"
    - "Top-level checkpoints/ holds runtime artifacts (rename of models/); pyproject.toml comment reflects the rename"
    - "Each wave is its own atomic commit (5 commits total) so any wave can be `git revert`-ed independently"
  artifacts:
    - path: "config.py"
      provides: "PROJECT_ROOT + benchmark dir + TAPAS_RDCM_PATH constants"
      contains: "TAPAS_RDCM_PATH"
    - path: "docs/README.md"
      provides: "Navigation hub linking 00..04 subfolders + legacy/"
    - path: "src/pyro_dcm/utils/templates/dcm_circuit_explorer_template.html"
      provides: "Runtime HTML template (relocated from docs/)"
    - path: "checkpoints/test/task_final.pt"
      provides: "Renamed checkpoint artifact (was models/test/task_final.pt)"
    - path: "scripts/debug/diagnose_phase16_init_scale.py"
      provides: "Phase-16.1 diagnostic relocated under scripts/debug/"
  key_links:
    - from: "validation/run_rdcm_validation.py"
      to: "config.py::TAPAS_RDCM_PATH"
      via: "from config import TAPAS_RDCM_PATH"
      pattern: "from config import TAPAS_RDCM_PATH"
    - from: "validation/run_validation.py"
      to: "config.py::TAPAS_RDCM_PATH"
      via: "from config import TAPAS_RDCM_PATH"
      pattern: "from config import TAPAS_RDCM_PATH"
    - from: "src/pyro_dcm/utils/circuit_viz.py docstring"
      to: "src/pyro_dcm/utils/templates/dcm_circuit_explorer_template.html"
      via: "Updated path string in module docstring"
      pattern: "src/pyro_dcm/utils/templates/dcm_circuit_explorer_template.html"
    - from: "benchmarks/runners/task_amortized.py"
      to: "checkpoints/test/task_final.pt"
      via: "guide_paths literal updated from models/ -> checkpoints/"
      pattern: "checkpoints/.*task_final\\.pt"
---

<objective>
Execute the structure-audit migration in 5 atomic-commit waves so the repository's on-disk
layout matches both `DATA_ANALYSIS_PROJECT_TEMPLATE.md` (5-subfolder docs/ + config.py
centralization) and the documentation in `CLAUDE.md` (which has drifted from disk).

Purpose: Bring repo hygiene to v0.4.0-ship quality. The audit reports already enumerated
every concrete change; this plan is purely execution. Five atomic commits give clean
revert points if any single wave breaks something.

Output:
- Janitorial cleanup (Wave 1) -- 1 commit
- docs/ 5-subfolder reorg (Wave 2) -- 1 commit
- CLAUDE.md sync to disk (Wave 3) -- 1 commit
- top-level config.py + TAPAS_RDCM_PATH centralization (Wave 4) -- 1 commit
- scripts/ cleanup + models/->checkpoints/ rename (Wave 5) -- 1 commit
</objective>

<execution_context>
@C:\Users\aman0087\.claude\get-shit-done\workflows\execute-plan.md
</execution_context>

<context>
@.planning/STATE.md
@.planning/research/structure-audit/01-src-layout-and-claude-md.md
@.planning/research/structure-audit/02-docs-organization.md
@.planning/research/structure-audit/03-conventions-and-config.md

# Files audited and edited (only those needed to land the migration):
@CLAUDE.md
@.gitignore
@pyproject.toml
@docs/HANDOFF_viz.md
@src/pyro_dcm/utils/circuit_viz.py
@validation/run_rdcm_validation.py
@validation/run_validation.py
@benchmarks/runners/task_amortized.py
@benchmarks/runners/spectral_amortized.py

# /gsd:quick scope notes:
# - Skip ROADMAP.md updates (orchestrator handles "Quick Tasks Completed" in STATE.md after return).
# - No tests are rewritten. The only test-relevant changes are Wave 1 (HTML template move) and
#   Wave 5 (checkpoint-path literals). Verification step in each wave catches import breakage.
</context>

<scope_constraints>
- ATOMIC COMMITS: Each wave = exactly one `git commit`. If a wave's verification fails,
  STOP, fix, then commit; do NOT bundle the fix into a later wave.
- DEFERRED (do NOT do in this plan):
  - Mass-rewrite of `benchmarks/results | benchmarks/figures | benchmarks/fixtures` literals
    across `benchmarks/`. Audit Section 03 flags this as ~15 call sites; out of scope here.
    Capture in SUMMARY under "Deferred future work."
  - Renaming any Python module inside `src/pyro_dcm/`.
  - Touching test files (none of these waves require test changes).
- TESTS: Do not rewrite tests. Run `pytest tests/ -m "not slow" --collect-only` after Waves
  1 and 5 to confirm imports still resolve.
</scope_constraints>

<tasks>

<task type="auto">
  <name>Task 1: Wave 1 -- janitorial cleanup of repo-root orphans</name>
  <files>
    bash.exe.stackdump (delete)
    .gitignore
    VERIFICATION.md (move -> docs/04_scientific_reports/VERIFICATION.md)
    docs/HANDOFF_viz.md (move -> docs/00_current_todos/HANDOFF_viz.md)
    docs/dcm_circuit_explorer_template.html (move -> src/pyro_dcm/utils/templates/dcm_circuit_explorer_template.html)
    src/pyro_dcm/utils/circuit_viz.py (docstring path update)
    docs/00_current_todos/ (new directory)
    src/pyro_dcm/utils/templates/ (new directory)
  </files>
  <action>
1. **Delete the Cygwin crash dump.**
   - `rm bash.exe.stackdump` (file is at repo root, untracked, already in .gitignore line "bash.exe.stackdump").
2. **Harden .gitignore against future stackdumps.**
   - Open `.gitignore`. Find the line `bash.exe.stackdump` under the `# OS` block.
   - Replace it with `*.stackdump` (glob covers the existing case + any future `name.stackdump`).
3. **Move VERIFICATION.md out of repo root.**
   - `mkdir -p docs/04_scientific_reports/` (dir already exists per `ls docs/`; mkdir is no-op).
   - `git mv VERIFICATION.md docs/04_scientific_reports/VERIFICATION.md`
   - Rationale: it is a v0.1.0 integration report, not a project-root pinned doc.
4. **Create docs/00_current_todos/ and move HANDOFF_viz.md into it.**
   - `mkdir -p docs/00_current_todos/`
   - `git mv docs/HANDOFF_viz.md docs/00_current_todos/HANDOFF_viz.md`
5. **Move HTML template out of docs/ into src/.**
   - `mkdir -p src/pyro_dcm/utils/templates/`
   - `git mv docs/dcm_circuit_explorer_template.html src/pyro_dcm/utils/templates/dcm_circuit_explorer_template.html`
   - DO NOT add `__init__.py` under `templates/` -- it is a static asset directory, not a Python
     subpackage. (pyproject.toml uses `packages = ["src/pyro_dcm"]` which already covers static
     children; verify wheel build is unchanged in the verification step.)
6. **Update the docstring reference in `src/pyro_dcm/utils/circuit_viz.py`.**
   - Open `src/pyro_dcm/utils/circuit_viz.py`. Locate the module docstring at the top
     (around line 4): `schema consumed by ``docs/dcm_circuit_explorer_template.html``. See`.
   - Replace `docs/dcm_circuit_explorer_template.html` with
     `src/pyro_dcm/utils/templates/dcm_circuit_explorer_template.html`.
7. **Update the references in `docs/00_current_todos/HANDOFF_viz.md` (the file just moved).**
   - Open `docs/00_current_todos/HANDOFF_viz.md`. Update path references at lines ~8, ~15,
     ~268, ~297, ~301 (line numbers may shift -- search for `dcm_circuit_explorer_template.html`
     and `docs/dcm_circuit_explorer_template.html`).
   - Replace `docs/dcm_circuit_explorer_template.html` with
     `src/pyro_dcm/utils/templates/dcm_circuit_explorer_template.html` everywhere it appears
     in this file.
   - The `http://localhost:8080/docs/dcm_circuit_explorer_template.html` line should become
     `http://localhost:8080/src/pyro_dcm/utils/templates/dcm_circuit_explorer_template.html`.
8. **Do NOT update `.planning/` references** (`.planning/STATE.md`, `.planning/phases/17-*`).
   These are historical planning artifacts; mutating them rewrites history. Leave alone.

Reference: audit `02-docs-organization.md` Section 1 + Section "Specific Recommendations" #3.
  </action>
  <verify>
- `ls bash.exe.stackdump` -- expect "No such file or directory".
- `git ls-files --error-unmatch docs/04_scientific_reports/VERIFICATION.md docs/00_current_todos/HANDOFF_viz.md src/pyro_dcm/utils/templates/dcm_circuit_explorer_template.html` -- all 3 must be tracked.
- `git ls-files VERIFICATION.md docs/HANDOFF_viz.md docs/dcm_circuit_explorer_template.html` -- expect empty (all moved).
- `grep -n "stackdump" .gitignore` -- expect a `*.stackdump` line.
- `python -c "import pyro_dcm.utils.circuit_viz"` -- imports cleanly (verifies no import path broke).
- `pytest tests/ -m "not slow" --collect-only -q` -- collection succeeds (no errors).
- `python -c "from importlib.resources import files; p = files('pyro_dcm.utils') / 'templates' / 'dcm_circuit_explorer_template.html'; print(p.is_file())"` -- prints `True` (asset accessible via package path).
  </verify>
  <done>
Wave 1 commit lands: stackdump deleted, gitignore hardened, VERIFICATION.md / HANDOFF_viz.md /
HTML template all in their correct new homes, all references updated, all tests still collect.

**Atomic commit:**
```
chore(002): wave 1 — janitorial cleanup of repo-root orphans

- delete bash.exe.stackdump (Cygwin crash artifact)
- harden .gitignore: *.stackdump glob
- move VERIFICATION.md -> docs/04_scientific_reports/
- create docs/00_current_todos/, move HANDOFF_viz.md into it
- move dcm_circuit_explorer_template.html out of docs/ into
  src/pyro_dcm/utils/templates/ (it is a runtime asset, not docs)
- update path references in circuit_viz.py docstring + HANDOFF_viz.md
```
  </done>
</task>

<task type="auto">
  <name>Task 2: Wave 2 -- docs/ 5-subfolder reorg per DATA_ANALYSIS_PROJECT_TEMPLATE.md</name>
  <files>
    docs/README.md (new)
    docs/00_current_todos/README.md (new)
    docs/01_project_protocol/ (new directory)
    docs/01_project_protocol/README.md (new)
    docs/02_pipeline_guide/README.md (new)
    docs/03_methods_reference/README.md (new)
    docs/04_scientific_reports/README.md (new)
    docs/legacy/ (new directory)
    docs/legacy/README.md (new)
    docs/02_pipeline_guide/guide_selection.md (move -> docs/03_methods_reference/guide_selection.md)
  </files>
  <action>
1. **Move `guide_selection.md` from pipeline-guide to methods-reference.**
   - `git mv docs/02_pipeline_guide/guide_selection.md docs/03_methods_reference/guide_selection.md`
   - Rationale (audit `02-docs-organization.md`): file is a methods-selection rationale
     (when to use which Pyro guide), NOT execution steps. It belongs under methods.
2. **Create `docs/01_project_protocol/` (was missing on disk).**
   - `mkdir -p docs/01_project_protocol/`
3. **Create `docs/legacy/` (audit recommends a parking spot for superseded docs).**
   - `mkdir -p docs/legacy/`
4. **Write `docs/README.md` -- the navigation hub.**
   - Single-purpose top-level docs index. Content (verbatim, write this file):
   ```markdown
   # docs/

   Project documentation organized per `DATA_ANALYSIS_PROJECT_TEMPLATE.md`.

   ## Subfolders

   - **[00_current_todos/](00_current_todos/)** — Active handoffs and short-lived TODO docs.
     Move things OUT of here as they become stable references or get superseded.
   - **[01_project_protocol/](01_project_protocol/)** — Stable, repo-wide protocols
     (coding standards pointers, contribution flow, release checklist).
   - **[02_pipeline_guide/](02_pipeline_guide/)** — How to run the DCM pipeline end-to-end
     (quickstart, consumer usage of bilinear API).
   - **[03_methods_reference/](03_methods_reference/)** — Mathematical methods reference
     (equations, methods.tex/.md, references.bib, guide-selection rationale).
   - **[04_scientific_reports/](04_scientific_reports/)** — Versioned scientific reports
     (benchmark runs, validation reports, milestone integration reports).
   - **[legacy/](legacy/)** — Superseded documents kept for historical reference.

   See also: `.planning/` (planning artifacts, NOT under docs/) and
   `src/pyro_dcm/utils/templates/` (runtime HTML assets, NOT under docs/).
   ```
5. **Write per-subfolder `README.md` files (one short paragraph each).**
   - `docs/00_current_todos/README.md`:
     ```markdown
     # 00_current_todos/

     Active handoffs, short-lived TODO docs, and in-flight specs that have not yet
     stabilized into permanent references. Files here are expected to move OUT
     (to `01_project_protocol/`, `02_pipeline_guide/`, `03_methods_reference/`, or
     `legacy/`) as the work they describe ships or gets superseded.

     Currently here:
     - `HANDOFF_viz.md` — v0.4.0 Phase 17 circuit-visualization handoff spec.
     ```
   - `docs/01_project_protocol/README.md`:
     ```markdown
     # 01_project_protocol/

     Stable, repo-wide protocols. Coding standards, contribution flow, and release
     checklists live here once they are no longer churning.

     Pointers (until protocol docs land here):
     - Coding standards: `CLAUDE.md` ("Coding Conventions" section).
     - Project rules: `CLAUDE.md` ("Critical Rules" section, especially "No Placeholders"
       and "Every Equation Must Be Cited").
     ```
   - `docs/02_pipeline_guide/README.md`:
     ```markdown
     # 02_pipeline_guide/

     How to run the DCM pipeline end-to-end as a consumer of `pyro_dcm`.

     - `quickstart.md` — Minimal task-DCM fit walkthrough.
     - `consumer_bilinear_quickstart.md` — Bilinear (B-matrix) extension walkthrough.

     Methods *rationale* (e.g., which Pyro guide to choose for which inference
     problem) lives next door under `03_methods_reference/guide_selection.md`.
     ```
   - `docs/03_methods_reference/README.md`:
     ```markdown
     # 03_methods_reference/

     Mathematical methods reference for DCM as implemented in `pyro_dcm`.

     - `equations.md` — Forward model + likelihood equations indexed by REF-id.
     - `methods.md` / `methods.tex` — Methods write-up sources.
     - `references.bib` — BibTeX bibliography.
     - `guide_selection.md` — Rationale for SVI guide family selection
       (AutoNormal vs AutoLowRankMVN vs amortized flow vs ...).
     ```
   - `docs/04_scientific_reports/README.md`:
     ```markdown
     # 04_scientific_reports/

     Versioned scientific reports: benchmark runs, validation reports, milestone
     integration reports.

     - `benchmark_report.md` — Latest cross-runner benchmark report.
     - `VERIFICATION.md` — v0.1.0 integration report (cross-validation against SPM12).
     ```
   - `docs/legacy/README.md`:
     ```markdown
     # legacy/

     Superseded documents kept for historical reference. Nothing here is current.
     Migrate items in here only when actively replaced by a successor doc in 00..04.
     ```
6. **Do NOT touch existing files inside the subfolders** beyond the `guide_selection.md` move.
   The audit explicitly limited Wave 2 to layout, not content rewriting.

Reference: `02-docs-organization.md` Sections 1, 2, "Specific Recommendations" #1, #2.
`DATA_ANALYSIS_PROJECT_TEMPLATE.md` line ~745 has the canonical 5-subfolder template.
  </action>
  <verify>
- `ls docs/` -- expect: `00_current_todos  01_project_protocol  02_pipeline_guide  03_methods_reference  04_scientific_reports  legacy  README.md`.
- `ls docs/00_current_todos docs/01_project_protocol docs/02_pipeline_guide docs/03_methods_reference docs/04_scientific_reports docs/legacy` -- each contains a `README.md`.
- `ls docs/02_pipeline_guide/guide_selection.md` -- expect "No such file" (moved).
- `ls docs/03_methods_reference/guide_selection.md` -- expect file exists.
- `git status --porcelain | grep -c '^A'` -- expect at least 7 new files (1 hub + 6 subfolder READMEs); the 7th `A` is the renamed file via `git mv` showing as A+D pair.
  </verify>
  <done>
Wave 2 commit lands: docs/ has the canonical 5-subfolder layout (`00..04` + `legacy/`),
each subfolder + the docs/ root has a README, and `guide_selection.md` is in the right place.

**Atomic commit:**
```
docs(002): wave 2 — reorganize docs/ into 5-subfolder layout

Adopt the DATA_ANALYSIS_PROJECT_TEMPLATE.md docs/ layout:
- 00_current_todos/, 01_project_protocol/, 02_pipeline_guide/,
  03_methods_reference/, 04_scientific_reports/, legacy/.
- Per-subfolder README.md as a one-paragraph index.
- Top-level docs/README.md as the navigation hub.
- Move guide_selection.md from 02_pipeline_guide/ to 03_methods_reference/
  (it is methods-selection rationale, not execution steps).
```
  </done>
</task>

<task type="auto">
  <name>Task 3: Wave 3 -- sync CLAUDE.md "Directory Structure" to actual disk state</name>
  <files>
    CLAUDE.md
  </files>
  <action>
The "Directory Structure (src/ layout)" section in `CLAUDE.md` (lines ~67–135) has drifted
from disk. Update it to match `ls src/pyro_dcm/**` exactly. Also update one bullet under
"Coding Conventions".

**Reference disk state (verified `2026-04-29`):**

`src/pyro_dcm/forward_models/` actual contents:
  `__init__.py  balloon_model.py  bold_signal.py  coupled_system.py  csd_computation.py
   neural_state.py  rdcm_forward.py  rdcm_posterior.py  spectral_noise.py  spectral_transfer.py`

`src/pyro_dcm/guides/` actual contents:
  `__init__.py  amortized_flow.py  parameter_packing.py  summary_networks.py`

`src/pyro_dcm/utils/` actual contents:
  `__init__.py  circuit_viz.py  ode_integrator.py  templates/` (templates/ added in Wave 1)

`src/pyro_dcm/models/` actual contents:
  `__init__.py  amortized_wrappers.py  guides.py  rdcm_model.py  spectral_dcm_model.py
   task_dcm_model.py`

`src/pyro_dcm/simulators/` -- unchanged from CLAUDE.md, no edits.

**Edits to make:**

1. **DROP entire `connectivity/` block.** Currently shows `connectivity/__init__.py`,
   `connectivity/static_a.py`, `connectivity/structural_mask.py`. Remove all 4 lines
   (the dir line and 3 children).

2. **DROP entire `inference/` block.** Currently shows `inference/__init__.py`,
   `inference/svi_runner.py`, `inference/nuts_validator.py`, `inference/model_comparison.py`.
   Remove all 5 lines.

3. **`forward_models/` block edits:**
   - Rename `rdcm_likelihood.py` line to `rdcm_forward.py` (keep the `[REF-020]` annotation).
   - ADD: `coupled_system.py        # Joint neural+hemodynamic ODE` (no ref needed if uncertain;
     reasonable: `# coupled neural + hemodynamic system  [REF-002]`).
   - ADD: `rdcm_posterior.py       # rDCM analytic VB posterior  [REF-020]`
   - ADD: `spectral_noise.py       # Innovation/measurement noise spectra  [REF-010]`

4. **`guides/` block edits:**
   - DROP: `meanfield.py         # Baseline Gaussian guide` (file does not exist; baseline guide
     is provided by `models/guides.py::AutoNormal` factory).
   - ADD: `parameter_packing.py # Parameter packing for amortized guide`
   - ADD: `summary_networks.py  # Summary networks (CNN/MLP) for amortized guide`

5. **`utils/` block edits:**
   - DROP: `spectral_utils.py    # FFT, CSD, frequency grids` (does not exist; FFT helpers live in
     `forward_models/csd_computation.py`).
   - DROP: `diagnostics.py       # Convergence checks, posterior plots` (does not exist).
   - ADD: `circuit_viz.py       # CircuitViz JSON serializer for circuit-explorer template`
   - ADD: `templates/           # Static HTML/JS assets (dcm_circuit_explorer_template.html)`

6. **`models/` block edits:**
   - The block currently lists `task_dcm_model.py`, `spectral_dcm_model.py`, `rdcm_model.py`,
     `guides.py`, `amortized_wrappers.py`. All 5 exist on disk. Keep as-is. Verify the
     `task_dcm_model.py` annotation includes `[v0.3.0: + bilinear B path]` (it already does).

7. **Update tensor-shape convention bullet under "Coding Conventions" (~line 167):**
   - Current: `- **Tensor shapes**: Documented in docstrings as `# shape: (n_regions, n_timepoints)``
   - Replace with: `- **Tensor shapes**: Documented in NumPy-style ``Parameters`` blocks with
     explicit shape annotations (e.g., ``A : torch.Tensor, shape (N, N)``)`

8. **Verify no other content changes.** Critical Rules section, Tech Stack, Tensor Shape
   Conventions table, and "When Stuck" section are factually accurate -- DO NOT touch them.

Reference: audit `01-src-layout-and-claude-md.md` Section "What CLAUDE.md says vs what
actually exists" + Section "Specific Recommendations" #1.
  </action>
  <verify>
- For every line in CLAUDE.md's `src/pyro_dcm/**` tree, `ls` the path and confirm it exists.
  Specifically:
  - `ls src/pyro_dcm/forward_models/{rdcm_forward,coupled_system,rdcm_posterior,spectral_noise}.py` -- all 4 must exist.
  - `ls src/pyro_dcm/guides/{parameter_packing,summary_networks}.py` -- both must exist.
  - `ls src/pyro_dcm/utils/{circuit_viz.py,templates}` -- both must exist.
  - `ls src/pyro_dcm/connectivity src/pyro_dcm/inference 2>&1` -- BOTH must report "No such file or directory".
  - `ls src/pyro_dcm/guides/meanfield.py src/pyro_dcm/utils/spectral_utils.py src/pyro_dcm/utils/diagnostics.py 2>&1` -- ALL 3 must report "No such file".
- `grep -c "connectivity/" CLAUDE.md` -- expect 0 (block deleted).
- `grep -c "inference/" CLAUDE.md` -- expect 0 (block deleted).
- `grep -c "rdcm_likelihood.py" CLAUDE.md` -- expect 0 (renamed to rdcm_forward.py).
- `grep -n "Tensor shapes" CLAUDE.md` -- expect the new "Parameters blocks" wording.
  </verify>
  <done>
Wave 3 commit lands: CLAUDE.md "Directory Structure" section is byte-for-byte consistent
with disk; tensor-shape bullet matches actual practice.

**Atomic commit:**
```
docs(002): wave 3 — sync CLAUDE.md directory tree to disk

CLAUDE.md "Directory Structure (src/ layout)" had drifted from reality.
Update so every listed path exists and every existing path is listed:
- drop fictional connectivity/ and inference/ subpackages
- forward_models/: rename rdcm_likelihood.py -> rdcm_forward.py;
  add coupled_system.py, rdcm_posterior.py, spectral_noise.py
- guides/: drop meanfield.py; add parameter_packing.py, summary_networks.py
- utils/: drop spectral_utils.py + diagnostics.py;
  add circuit_viz.py + templates/
- "Tensor shapes" convention bullet: NumPy Parameters blocks (matches practice)
```
  </done>
</task>

<task type="auto">
  <name>Task 4: Wave 4 -- top-level config.py + TAPAS_RDCM_PATH centralization</name>
  <files>
    config.py (new)
    validation/run_rdcm_validation.py
    validation/run_validation.py
  </files>
  <action>
1. **Create top-level `config.py`** at the repo root with this exact content:
   ```python
   """Repo-wide path constants.

   Single source of truth for paths used across `benchmarks/`, `validation/`, and
   `scripts/`. Intentionally minimal in this revision: only the cluster-blocking
   absolute path (`TAPAS_RDCM_PATH`) is centralized.

   The broader migration of `benchmarks/results`, `benchmarks/figures`, and
   `benchmarks/fixtures` literals (~15 call sites across `benchmarks/`) is
   deferred -- see SUMMARY "Deferred future work."

   Constants
   ---------
   PROJECT_ROOT : pathlib.Path
       Absolute path to the repository root (parent of this file).
   BENCHMARK_RESULTS_DIR : pathlib.Path
       Default directory for benchmark JSON / CSV outputs.
       Note: most `benchmarks/` callers still hardcode this literal; this
       constant is the migration target, not yet the call-site source of truth.
   BENCHMARK_FIGURES_DIR : pathlib.Path
       Default directory for benchmark figure outputs.
   BENCHMARK_FIXTURES_DIR : pathlib.Path
       Default directory for benchmark `.npz` fixture caches.
   TAPAS_RDCM_PATH : pathlib.Path
       Path to the local clone of `tapas/rDCM` (MATLAB rDCM toolbox), used by
       `validation/run_rdcm_validation.py` and `validation/run_validation.py`.
       Override via the ``TAPAS_RDCM_PATH`` environment variable when running
       on a different machine (e.g., Monash M3 cluster).
   """

   from __future__ import annotations

   import os
   from pathlib import Path

   PROJECT_ROOT: Path = Path(__file__).resolve().parent

   BENCHMARK_RESULTS_DIR: Path = PROJECT_ROOT / "benchmarks" / "results"
   BENCHMARK_FIGURES_DIR: Path = PROJECT_ROOT / "benchmarks" / "figures"
   BENCHMARK_FIXTURES_DIR: Path = PROJECT_ROOT / "benchmarks" / "fixtures"

   TAPAS_RDCM_PATH: Path = Path(
       os.environ.get(
           "TAPAS_RDCM_PATH",
           "C:/Users/aman0087/Documents/Github/tapas/rDCM",
       )
   )
   ```

2. **Replace the hardcoded literal in `validation/run_rdcm_validation.py`.**
   - Open the file. Find the two occurrences (audit cited lines 65 and 301; re-verify
     with `grep -n 'tapas/rDCM' validation/run_rdcm_validation.py` since lines may have shifted).
   - At line ~65 (assignment): replace
     ```python
     tapas_path = "C:/Users/aman0087/Documents/Github/tapas/rDCM"
     ```
     with
     ```python
     tapas_path = str(TAPAS_RDCM_PATH)
     ```
     (`str(...)` so callers expecting a `str` keep working; downstream MATLAB-batch shellouts
     concatenate this into a string command, so a Path would break unless converted.)
   - At line ~301 (string used in an error message): replace the literal
     `"C:/Users/aman0087/Documents/Github/tapas/rDCM. "` with
     `f"{TAPAS_RDCM_PATH}. "` (preserve the trailing period + space).
   - Add the import at the top of the module (after `from __future__ import annotations`,
     before any sibling imports):
     ```python
     from config import TAPAS_RDCM_PATH
     ```
     If a `from __future__ import annotations` line is missing, add it as the file's first
     non-docstring statement (project convention -- see CLAUDE.md "Python Conventions").

3. **Same change in `validation/run_validation.py:102`** (audit `03-conventions-and-config.md`
   flagged this third hit; re-verify line with `grep -n 'tapas/rDCM' validation/run_validation.py`).
   - The literal there is inside an error-message string:
     `"'C:/Users/aman0087/Documents/Github/tapas/rDCM'"` (with embedded single quotes).
   - Replace with `f"'{TAPAS_RDCM_PATH}'"` to keep the quoted appearance.
   - Add `from config import TAPAS_RDCM_PATH` at top alongside other imports.

4. **DO NOT modify** the absolute paths in `validation/matlab_scripts/run_tapas_rdcm.m`
   (line 29) or in `.planning/phases/06-*/*` MATLAB-batch examples. These are MATLAB
   files / planning documents; out of scope for a Python config.py wave. Note this in
   the SUMMARY's "Deferred" section.

5. **DO NOT mass-rewrite `benchmarks/` literals** (results/, figures/, fixtures/).
   Audit Section 03 estimates ~15 call sites and the existing `benchmarks/config.py`
   is locally scoped. Capture this as deferred future work in the SUMMARY.

Reference: audit `03-conventions-and-config.md` Sections 1, 2, "Specific Recommendations" #1, #2.
  </action>
  <verify>
- `python -c "from config import TAPAS_RDCM_PATH, PROJECT_ROOT, BENCHMARK_RESULTS_DIR, BENCHMARK_FIGURES_DIR, BENCHMARK_FIXTURES_DIR; print(TAPAS_RDCM_PATH); print(PROJECT_ROOT)"` -- prints the default `tapas/rDCM` path and the repo root with no traceback.
- `TAPAS_RDCM_PATH=/tmp/foo python -c "from config import TAPAS_RDCM_PATH; print(TAPAS_RDCM_PATH)"` -- prints `/tmp/foo` (env-var override works).
- `grep -n "C:/Users/aman0087/Documents/Github/tapas/rDCM" validation/run_rdcm_validation.py validation/run_validation.py` -- expect ZERO hits.
- `python -c "import ast, sys; ast.parse(open('validation/run_rdcm_validation.py').read()); ast.parse(open('validation/run_validation.py').read()); ast.parse(open('config.py').read()); print('OK')"` -- prints `OK` (all 3 files parse).
- `python -c "import validation.run_rdcm_validation"` -- imports cleanly OR errors only on missing MATLAB / pyvista (expected; what matters is that `from config import TAPAS_RDCM_PATH` resolves).
- `pytest tests/ -m "not slow" --collect-only -q 2>&1 | tail -5` -- collection succeeds.
  </verify>
  <done>
Wave 4 commit lands: `config.py` exists with `TAPAS_RDCM_PATH` (env-overridable) +
benchmark-dir constants; both validation scripts import from it; the absolute
`C:/Users/aman0087/...` literal is gone from Python source; broader benchmarks/
literal centralization explicitly deferred.

**Atomic commit:**
```
feat(002): wave 4 — top-level config.py for TAPAS_RDCM_PATH centralization

- create config.py at repo root with PROJECT_ROOT, BENCHMARK_*_DIR, and
  TAPAS_RDCM_PATH (env-overridable via TAPAS_RDCM_PATH env var)
- replace baked-in C:/Users/aman0087/Documents/Github/tapas/rDCM in
  validation/run_rdcm_validation.py (2 hits) and
  validation/run_validation.py (1 hit) with `from config import TAPAS_RDCM_PATH`
- DEFERRED: mass-rewrite of benchmarks/results|figures|fixtures literals
  across benchmarks/ (~15 call sites; tracked for follow-up)
```
  </done>
</task>

<task type="auto">
  <name>Task 5: Wave 5 -- scripts/ cleanup + models/ -> checkpoints/ rename</name>
  <files>
    scripts/debug/debug_phase16_fixture_check.py (moved from scripts/)
    scripts/debug/debug_phase16_nan_seeds.py (moved from scripts/)
    scripts/debug/debug_phase16_pool_smoke.py (moved from scripts/)
    scripts/debug/diagnose_phase16_init_scale.py (moved from scripts/, was untracked)
    models/test/task_final.pt (renamed -> checkpoints/test/task_final.pt)
    benchmarks/runners/task_amortized.py (path literal: models/ -> checkpoints/)
    benchmarks/runners/spectral_amortized.py (path literals: models/ -> checkpoints/)
    pyproject.toml (defensive comment update)
    .gitignore (rename `/models/` -> `/checkpoints/`)
  </files>
  <action>
1. **Create `scripts/debug/` and move debug + diagnostic scripts.**
   - `mkdir -p scripts/debug/`
   - Tracked debug scripts (use `git mv` to preserve history):
     - `git mv scripts/debug_phase16_fixture_check.py scripts/debug/debug_phase16_fixture_check.py`
     - `git mv scripts/debug_phase16_nan_seeds.py scripts/debug/debug_phase16_nan_seeds.py`
     - `git mv scripts/debug_phase16_pool_smoke.py scripts/debug/debug_phase16_pool_smoke.py`
   - **Untracked** diagnostic (currently shows in `git status` as `?? scripts/diagnose_phase16_init_scale.py`):
     - `mv scripts/diagnose_phase16_init_scale.py scripts/debug/diagnose_phase16_init_scale.py`
     - Then `git add scripts/debug/diagnose_phase16_init_scale.py` (first time tracking).
   - Verify nothing else under `scripts/` matches `debug_phase*` or `diagnose_phase*`:
     `ls scripts/ | grep -E '^(debug|diagnose)_'` -- must report empty.

2. **Rename top-level `models/` -> `checkpoints/`.**
   - The `models/` directory contains only `models/test/task_final.pt` (verified `ls models/test/`).
   - The directory is currently gitignored (`.gitignore` has `/models/`), so `models/test/task_final.pt`
     is NOT tracked; a plain `mv` works (no `git mv` needed for the file). The directory itself
     also has nothing tracked under it.
   - `mv models/ checkpoints/` (ensures the working-tree artifact moves with its parent).
   - Update `.gitignore`: replace the line `/models/` with `/checkpoints/` so the renamed
     directory remains gitignored (these are runtime artifacts, never committed).

3. **Update path literals in benchmark runners.**
   - `benchmarks/runners/task_amortized.py:217`:
     - Current: `guide_paths = ["models/task_final.pt", "models/task_ci.pt"]`
     - Change to: `guide_paths = ["checkpoints/task_final.pt", "checkpoints/task_ci.pt"]`
   - `benchmarks/runners/spectral_amortized.py:238-239`:
     - Current literals: `"models/spectral_final.pt"` and `"models/spectral_ci.pt"`
     - Change to: `"checkpoints/spectral_final.pt"` and `"checkpoints/spectral_ci.pt"`
   - **Re-verify** with `grep -n '"models/' benchmarks/runners/*.py` -- expect ZERO hits after edits.

4. **Update the defensive comment in `pyproject.toml`.**
   - Currently the comment block (lines 4-8) reads:
     ```
     # Explicit package target — required because the project has both a top-level
     # `models/` directory (generated artifacts, NOT a Python package) and
     # `src/pyro_dcm/models/` (the real subpackage). Without this, hatchling's
     # auto-discovery ships a wheel missing `src/pyro_dcm/models/`, which breaks
     # `from pyro_dcm.models import …` in `src/pyro_dcm/__init__.py`.
     ```
   - The collision concern is **gone** after the rename. Replace with:
     ```
     # Explicit package target. Historically required because of a top-level
     # `models/` directory (now renamed to `checkpoints/`) that confused
     # hatchling's auto-discovery into shipping a wheel missing
     # `src/pyro_dcm/models/`. Kept explicit for forward stability.
     ```

5. **DO NOT touch** any non-Python references to `models/` (e.g., `.planning/` SUMMARYs that
   talk about the historical layout, or REF docs). These are historical artifacts.

6. **DO NOT touch** the `scripts/train_amortized_guide.py` -- it does not contain a hardcoded
   `models/` literal (verified by grep; the only `models/` mention in scripts/ comes from a
   `.planning/phases/07-*/07-02-PLAN.md` example command line, which is a planning doc).

Reference: audit `01-src-layout-and-claude-md.md` Sections 2, 3, "Specific Recommendations"
#2 + #4. The hatchling comment edit is `01-...md` Section 3 final paragraph.
  </action>
  <verify>
- `ls scripts/debug/` -- expect 4 files: `debug_phase16_fixture_check.py`,
  `debug_phase16_nan_seeds.py`, `debug_phase16_pool_smoke.py`, `diagnose_phase16_init_scale.py`.
- `ls scripts/ | grep -E '^(debug|diagnose)_'` -- expect EMPTY.
- `ls models/ 2>&1` -- expect "No such file or directory".
- `ls checkpoints/test/task_final.pt` -- expect file exists.
- `grep -n 'models/' .gitignore` -- expect ZERO hits (the `/models/` line is gone).
- `grep -n 'checkpoints/' .gitignore` -- expect the new `/checkpoints/` line.
- `grep -rn '"models/' benchmarks/runners/` -- expect ZERO hits.
- `grep -n '"checkpoints/' benchmarks/runners/task_amortized.py benchmarks/runners/spectral_amortized.py` -- expect 1 hit + 2 hits respectively.
- `grep -n 'top-level' pyproject.toml` -- expect the new comment block mentioning the rename.
- `python -c "import benchmarks.runners.task_amortized; import benchmarks.runners.spectral_amortized; print('OK')"` -- prints `OK` (no syntax / import error). If these modules require a heavy `benchmarks/__init__.py` setup not on the cwd path, fall back to `python -c "import ast; ast.parse(open('benchmarks/runners/task_amortized.py').read()); ast.parse(open('benchmarks/runners/spectral_amortized.py').read()); print('OK')"`.
- `pytest tests/ -m "not slow" --collect-only -q 2>&1 | tail -5` -- collection succeeds, no import errors.
- `python -m build --wheel --outdir /tmp/wave5_wheel_check 2>&1 | tail -10` (only if `build` is installed; OPTIONAL): wheel build succeeds and the comment-edited pyproject.toml still produces a wheel containing `pyro_dcm/models/__init__.py`. Skip if `build` not available.
  </verify>
  <done>
Wave 5 commit lands: `scripts/debug/` exists with all 4 phase-16 debug + diagnostic scripts;
top-level `models/` is gone, replaced by `checkpoints/`; benchmark runners point at the new
location; pyproject.toml comment reflects the rename; .gitignore updated.

**Atomic commit:**
```
chore(002): wave 5 — scripts/debug/ + rename top-level models/ -> checkpoints/

- create scripts/debug/, move debug_phase16_*.py + diagnose_phase16_init_scale.py
  into it (the diagnose_ script was previously untracked; now tracked)
- rename top-level models/ -> checkpoints/ (only contained the gitignored
  models/test/task_final.pt artifact); update .gitignore /models/ -> /checkpoints/
- update path literals in benchmarks/runners/task_amortized.py and
  spectral_amortized.py from models/*.pt to checkpoints/*.pt
- update pyproject.toml defensive comment to reflect the rename
  (the historical name collision with src/pyro_dcm/models/ is resolved)
```
  </done>
</task>

</tasks>

<verification>
After all 5 waves are committed:

1. **Disk shape sanity:**
   - `ls` repo root: NO `bash.exe.stackdump`, NO `VERIFICATION.md`, NO `models/`,
     YES `checkpoints/`, YES `config.py`.
   - `ls docs/`: 6 entries (`00_..04_*` + `legacy/`) plus `README.md`.
   - `ls src/pyro_dcm/utils/templates/dcm_circuit_explorer_template.html` exists.
   - `ls scripts/debug/` has the 4 phase-16 debug + diagnostic scripts.

2. **Import + collection sanity (single check, runs all tests in collect-only mode):**
   - `pytest tests/ -m "not slow" --collect-only -q 2>&1 | tail -5` -- expect "collected NNN items" with no errors.
   - `python -c "import pyro_dcm; import pyro_dcm.utils.circuit_viz; from config import TAPAS_RDCM_PATH; print('OK')"` -- prints `OK`.

3. **No baked-in absolute paths in Python source:**
   - `grep -rn 'C:/Users/aman0087/Documents/Github/tapas/rDCM' --include='*.py' .` -- expect ZERO hits.

4. **CLAUDE.md ↔ disk consistency:**
   - For every path mentioned in CLAUDE.md's "Directory Structure" tree, `ls` it and confirm
     it exists on disk. (Spot-checked in Task 3's verification.)

5. **Git log shape:**
   - `git log --oneline -n 5` -- expect 5 new commits, one per wave, in order:
     `chore(002): wave 1 ...`, `docs(002): wave 2 ...`, `docs(002): wave 3 ...`,
     `feat(002): wave 4 ...`, `chore(002): wave 5 ...`.
   - `git log --oneline -n 5 | wc -l` == 5.

6. **No regression in working tree:**
   - `git status` should report clean (nothing untracked, nothing staged) after Wave 5 commit.
</verification>

<success_criteria>
- 5 atomic commits land in chronological wave order, each independently revert-able.
- Every audit-recommended on-disk change in waves 1–5 is applied; deferred items are
  explicitly listed in the SUMMARY.
- `pytest tests/ -m "not slow" --collect-only` succeeds at the end of waves 1, 4, and 5
  (the waves that touch Python files).
- No hardcoded `C:/Users/aman0087/...tapas/rDCM` literal remains in any `.py` file.
- CLAUDE.md "Directory Structure" lists each existing `src/pyro_dcm/**` subpackage and
  contains no fictional ones.
- `models/` directory at repo root no longer exists; `checkpoints/` holds the renamed
  artifact and is gitignored.
- `git status` is clean after Wave 5.
</success_criteria>

<deferred_future_work>
Capture in the SUMMARY (not done in this plan):

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
</deferred_future_work>

<output>
After all 5 wave commits, write SUMMARY to:
`.planning/quick/002-structure-audit-migration-waves-1-5/002-SUMMARY.md`

SUMMARY structure:
- One line per wave (commit SHA + one-sentence outcome).
- "Disk before vs after" snapshot showing the 5 directory-shape deltas.
- "Path literals migrated" table (file:line, old, new).
- "Deferred future work" -- copy verbatim from the plan's `<deferred_future_work>` block.
- "Verification snapshot" -- captured outputs of the verification commands listed under
  `<verification>` (paste-in result of `pytest --collect-only`, `git log --oneline -n 5`,
  `grep -rn 'C:/Users/aman0087/Documents/Github/tapas/rDCM' --include='*.py' .`).

NOTE (/gsd:quick scope): Do NOT update ROADMAP.md. STATE.md "Quick Tasks Completed" is
updated by the orchestrator after the executor returns, not by the executor itself.
</output>
