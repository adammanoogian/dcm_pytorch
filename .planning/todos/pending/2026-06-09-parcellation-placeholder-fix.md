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

## Status: PLACEHOLDER REMOVED 2026-06-09 (runtime validation pending nilearn)

`parcellation.py` rewritten (commit pending):
- `aggregate_vertices_by_labels(ts, vertex_labels, n_rois)` — real label-based averaging,
  pure numpy, fully unit-tested (incl. a regression test that non-contiguous labels are
  honoured, NOT contiguous blocks; empty parcels → NaN + warning, never fabricated).
- `load_schaefer_fsaverage5_labels(n_rois)` — real nilearn `vol_to_surf` nearest-neighbour
  projection of the volumetric Schaefer atlas onto fsaverage5 (sampling white→pial). **Fails
  loudly** (ImportError / RuntimeError) if nilearn or the atlas data is unavailable — the
  silent contiguous-block fallback is gone.
- `parcellate_vertices_to_rois` now loads real labels + aggregates; identical signature.

**Remaining (the only open item):** runtime-validate the nilearn vol_to_surf path. nilearn is
NOT installed in either the laptop or the `actinf-py-scripts` cluster env, so the projection
code is correct-against-the-API but unexecuted. A `@pytest.mark.slow` integration test
(`test_real_schaefer_labels_nonuniform`, skipped when nilearn absent) asserts the real parcel
sizes are non-uniform once `pip install '.[foundation]'` is done. Verify the exact nilearn API
(`fetch_surf_fsaverage` keys `pial_left/white_left`, `vol_to_surf(inner_mesh=...)`) against the
installed nilearn version at that point.

## Audit disposition (2026-06-10) — VIOLATION RESOLVED; runtime validation → v0.7.0

The **No-Placeholders-Ever violation is RESOLVED** — `parcellation.py` is now real code (audit
confirmed the rewrite: `aggregate_vertices_by_labels` + real nilearn `vol_to_surf`, no silent
fallback). This does **not** block v0.6.0 closure: under the scope-cut, parcellation is only
*exercised* by real foundation-model runs (Phase 24), which are **deferred to v0.7.0**.

Priority lowered high → **low/deferred**. The remaining runtime validation is a **v0.7.0
prerequisite** (it gates the first real Phase-24 run) and should be done as the opening step of
the v0.7.0 foundation-model phase, when `[foundation]`/nilearn is installed on the cluster.
