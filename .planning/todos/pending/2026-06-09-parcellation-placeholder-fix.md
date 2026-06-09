---
created: 2026-06-09T23:30
title: Replace Schaefer parcellation placeholder (No-Placeholders violation)
area: foundation-models
priority: high
files:
  - src/pyro_dcm/foundation/parcellation.py:142-165
  - .planning/phases/24-foundation-model-use-cases/24-01-SUMMARY.md
---

## Problem

`src/pyro_dcm/foundation/parcellation.py:146` is a literal placeholder that **violates the
project's "No Placeholders -- Ever" critical rule** (CLAUDE.md). `parcellate_vertices_to_rois`
fetches the real Schaefer-2018 atlas *labels* but then assigns vertices to ROIs by **naive
equal-size contiguous blocks** (`_FSAVERAGE5_TOTAL_VERTICES // n_rois`) instead of the atlas's
actual vertex-to-parcel mapping. The ROI timeseries it averages are therefore over the wrong
vertices -- scientifically invalid for any real Phase 24 foundation-model (TRIBE / M-EEG)
analysis. Surfaced during the 2026-06-09 doc-debt backfill (see 24-01-SUMMARY anomalies).

## Solution (sketch)

Use nilearn's real surface atlas mapping:
- `fetch_atlas_surf_destrieux` / `fetch_atlas_schaefer_2018` with surface annotation, or project
  the volumetric Schaefer atlas to fsaverage5 via `nilearn.surface` and build a genuine
  vertex -> parcel-label lookup, then average per parcel.
- Validate against a known parcellation (e.g. ROI count, label names, and a vertex-count-per-ROI
  distribution that is NOT uniform).
- Add a unit test asserting non-contiguous, atlas-correct vertex assignment.

Until fixed, Phase 24 foundation-model connectivity results are not trustworthy. This blocks any
real-data Phase 24 claim.
