---
phase: 27-publication-artifacts
plan: 02
subsystem: publication-figures
tags: [matplotlib, figures, publication, pipeline-schematic, v0.6.0]

dependency-graph:
  requires: [27-01]
  provides: [publication-figure-script, pipeline-schematic-figure]
  affects: [27-03]

tech-stack:
  added: []
  patterns: [modular-figure-functions, cli-figure-generation, FileNotFoundError-gating]

file-tracking:
  key-files:
    created:
      - scripts/generate_publication_figures.py
    modified: []

decisions:
  - id: 27-02-D1
    summary: "Figures gitignored -- script is source of truth"
    detail: "figures/*.png and figures/*.pdf are in .gitignore. Generated figures exist on disk but are not committed. The script is the reproducible artifact."

metrics:
  duration: ~12 min
  completed: 2026-05-28
---

# Phase 27 Plan 02: Publication Figure Generation Summary

Modular publication figure generation script with 7 figure functions covering
the full v0.6.0 DCM interpretability story, plus CLI entry point.

## Completed Tasks

| Task | Name | Commit | Key Files |
|------|------|--------|-----------|
| 1 | Create publication figure generation script | 558d568 | scripts/generate_publication_figures.py |
| 2 | Generate pipeline schematic figure | (gitignored) | figures/pipeline_schematic.{png,pdf} |

## What Was Built

### scripts/generate_publication_figures.py (1011 lines)

Seven modular figure functions, each returning a `matplotlib.figure.Figure`:

1. **fig_pipeline_schematic** -- Pipeline diagram using matplotlib patches/arrows.
   Color-coded stages: blue (data), green (training), orange (DCM inference).
   No data dependencies; generates immediately.

2. **fig_synthetic_recovery** -- Phase 20 parameter recovery. 2x2 grid:
   A/B scatter plots + RMSE violin distributions.

3. **fig_connectivity_matrices** -- Phase 22 effective connectivity. 1x3 grid:
   posterior mean A, std A, significant connections with hatching.

4. **fig_bmr_model_comparison** -- Phase 23 BMR results. Bar chart of
   relative log evidence + BMR vs brute-force ELBO scatter.

5. **fig_foundation_model_comparison** -- Phase 24 cross-modality comparison.
   Grouped bar chart of DCM metrics across model types.

6. **fig_hybrid_vae_dcm** -- Phase 25 VAE-DCM. ELBO training curve +
   VAE vs SVI A-matrix agreement scatter.

7. **fig_sbi_calibration** -- Phase 26 SBI calibration. SBC rank histogram +
   expected vs observed coverage plot.

### CLI Interface

```
python scripts/generate_publication_figures.py
    --results-dir benchmarks/results  # NPZ file directory
    --output-dir figures/             # output directory
    --figures pipeline_schematic      # comma-separated or 'all'
    --formats png,pdf                 # output formats
```

### Pipeline Schematic Figure

Generated on disk at `figures/pipeline_schematic.{png,pdf}`:
- PNG: 151 KB, 300 dpi
- PDF: 42 KB, vector
- Shows 6-stage pipeline: Neural Data -> Train Model -> Extract Latents ->
  PCA + R2 Gate -> Fit DCM -> Posterior A, Bj
- Color legend for stage categories

## Deviations from Plan

### Noted Behaviors

**1. [Note] Generated figures are gitignored**

- `.gitignore` lines 37-38 exclude `figures/*.png` and `figures/*.pdf`
- This is correct project convention (regenerable artifacts)
- The script is the source of truth; figures regenerate on demand
- Task 2 does not have a separate git commit since the outputs are gitignored

## Decisions Made

| ID | Decision | Rationale |
|----|----------|-----------|
| 27-02-D1 | Figures gitignored; script is source of truth | Standard project convention for regenerable artifacts |

## Verification Results

| Check | Result |
|-------|--------|
| ruff check passes | PASS |
| Script imports cleanly | PASS |
| 7 figure functions defined with docstrings | PASS |
| Pipeline schematic generates (PNG + PDF) | PASS |
| PNG > 10KB (151 KB) | PASS |
| PDF > 5KB (42 KB) | PASS |
| Data-dependent figures raise FileNotFoundError | PASS |
| --figures all produces 1 generated + 6 skipped | PASS |
| Uses benchmarks/plotting._apply_style | PASS |

## Next Phase Readiness

Phase 27-03 (manuscript LaTeX integration) can proceed. The figure script
is ready to generate all remaining figures once upstream phases produce
their NPZ result artifacts.
