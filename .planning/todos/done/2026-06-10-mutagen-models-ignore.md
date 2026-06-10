---
created: 2026-06-10T21:30
title: Mutagen `models/` ignore silently excludes src/pyro_dcm/models/ from M3 sync
area: infrastructure / cluster
priority: high
files:
  - (mutagen session config — set at session creation, not a repo file)
  - .planning/v0.6.0-AUDIT.md
---

## Problem

The `dcm-pytorch` Mutagen session's ignore list (created by `m3-init-project`) contains
unanchored Mutagen-syntax patterns: `data/`, **`models/`**, `checkpoints/`, `wandb/`.
Mutagen matches these at ANY depth, so `models/` also matches **`src/pyro_dcm/models/`** —
the entire DCM model source package is silently excluded from M3 sync. M3's copies were
frozen at May 29 2026; `mutagen sync list` reports "Watching for changes" with no conflict,
so the staleness is invisible.

Surfaced 2026-06-10 when an M3 eval job failed `ImportError: cannot import name
'masked_sign_recovery'` despite the local file having it (commit fbddc0e). Past-result
impact was nil (only that June-9 metric helper was in the stale window; VL/training runs
used current code — the VL engine lives in the SYNCED `src/pyro_dcm/inference/`).

## Fix

**Stopgap (done 2026-06-10):** `scp src/pyro_dcm/models/*.py m3:.../src/pyro_dcm/models/`.
Safe because the path is mutagen-ignored, so a direct write does NOT corrupt sync direction.
Repeat after any local edit to `src/pyro_dcm/models/` until the real fix lands.

**Real fix (user-owned — recreates the session):**
1. `mutagen sync terminate dcm-pytorch`
2. Recreate with ANCHORED ignores so only project-root dirs are excluded, e.g.
   `--ignore=/data/ --ignore=/models/ --ignore=/checkpoints/ --ignore=/wandb/`
   (leading slash = root-anchored). There is no top-level `models/` dir in this repo, so
   anchoring loses nothing and lets `src/pyro_dcm/models/` sync.
3. Let the initial scan reconcile; verify `ssh m3 grep -c 'def masked_sign_recovery'
   .../src/pyro_dcm/models/hybrid_vae_dcm.py` returns 1.
4. Check sister projects (actinf_physics, hgf-analysis, nn4psych) for the same unanchored
   `models/` ignore if any have a `src/.../models/` package.

## Guardrail

Before trusting any M3 run that imports from `pyro_dcm.models`, verify the M3 file matches
local (grep/mtime). See memory `reference-mutagen-models-ignore-footgun`.

## ✅ RESOLVED 2026-06-10
Recreated the `dcm-pytorch` Mutagen session with `models/` -> `/models/` (root-anchored), leaving data/checkpoints/wandb unanchored. `src/pyro_dcm/models/` now syncs (657 vs 649 files; 0 conflicts; live edit propagated). New session id sync_rfd9rQ8kkxqRaMTkzQ4qYnWyPUtopXaXGGIY2n3HvpH.
