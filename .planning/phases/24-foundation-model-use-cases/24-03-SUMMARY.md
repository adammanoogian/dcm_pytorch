---
phase: 24-foundation-model-use-cases
plan: 03
subsystem: foundation-meeg
tags: [labram, brainomni, meeg, eeg, meg, foundation-model, pca, dcm]
dependency-graph:
  requires: [24-01]
  provides: [LaBraMExtractor, BrainOmniExtractor, meeg-pipeline-scripts]
  affects: [24-04]
tech-stack:
  added: [braindecode, brainomni, huggingface_hub]
  patterns: [forward-hook-extraction, pca-reduction, argparse-pipeline]
key-files:
  created:
    - src/pyro_dcm/foundation/labram_extractor.py
    - src/pyro_dcm/foundation/brainomni_extractor.py
    - scripts/24_extract_meeg_latents.py
    - scripts/24_fit_dcm_meeg.py
    - cluster/sbatch/24_meeg_extract.slurm
    - tests/test_meeg_extractor.py
  modified:
    - src/pyro_dcm/foundation/__init__.py
decisions:
  - id: 24-03-D1
    summary: "LaBraM primary, BrainOmni secondary for M/EEG"
    detail: "LaBraM has cleaner braindecode API with return_features; BrainOmni uses hooks exclusively"
  - id: 24-03-D2
    summary: "PCA reduction shared across both extractors"
    detail: "Both LaBraM and BrainOmni use sklearn PCA to reduce from embed_dim to n_components DCM dimensions"
  - id: 24-03-D3
    summary: "Spectral DCM default for M/EEG latents, latent-circuit as alternative"
    detail: "Pipeline scripts support both DCM variants; spectral DCM is default because patch-level dynamics have limited temporal resolution"
metrics:
  duration: ~58 min
  completed: 2026-05-28
---

# Phase 24 Plan 03: M/EEG Foundation Model Pipeline Summary

LaBraM and BrainOmni extractors with braindecode return_features and forward-hook APIs, PCA reduction to DCM space, pipeline scripts for latent extraction and DCM fitting, GPU cluster sbatch.

## What Was Built

### LaBraMExtractor (`src/pyro_dcm/foundation/labram_extractor.py`)

- Subclass of `BaseExtractor` wrapping braindecode's `Labram` model (ICLR 2024)
- `load_model()`: loads pretrained weights via `Labram.from_pretrained()` or local checkpoint
- `extract_features()`: uses LaBraM's built-in `return_features=True` API to get patch-level embeddings `(batch, n_patches, embed_dim)` and `[CLS]` token
- `extract_latents()`: delegates to `extract_features` by default; supports hook-based extraction when `layer_names` specified
- `reduce_to_dcm_space()`: PCA reduction from `embed_dim` (200) to `n_components` (default 4)
- `chs_info` parameter stored for future channel-ordering validation (Pitfall 5)
- Import guarded: `try/except ImportError` for `braindecode` and `sklearn`

### BrainOmniExtractor (`src/pyro_dcm/foundation/brainomni_extractor.py`)

- Subclass of `BaseExtractor` for BrainOmni (NeurIPS 2025) EEG+MEG model
- Modality-aware (`"eeg"` or `"meg"`) with input validation
- `load_model()`: loads via local checkpoint or `huggingface_hub.hf_hub_download`
- `extract_latents()`: uses forward hooks exclusively (no return_features API); auto-discovers encoder blocks when `layer_names=None`
- `reduce_to_dcm_space()`: PCA reduction from selected layer, handles 2-D and 3-D activations
- Import guarded: `try/except ImportError` for `brainomni` and `huggingface_hub`

### Pipeline Scripts

- `scripts/24_extract_meeg_latents.py`: argparse CLI for loading .fif epochs, running LaBraM or BrainOmni extraction, PCA reduction, saving `.npz`
- `scripts/24_fit_dcm_meeg.py`: argparse CLI for loading latent dynamics, computing CSD, fitting spectral or latent-circuit DCM, saving posterior A matrix
- `cluster/sbatch/24_meeg_extract.slurm`: GPU sbatch with `braindecode` install, runs extraction + DCM fitting sequentially

### Test Coverage

17 tests in `tests/test_meeg_extractor.py`:
- LaBraMExtractor: abstract contract, feature shape, PCA reduction, not-loaded error
- BrainOmniExtractor: abstract contract, modality validation, hook-based extraction, PCA reduction for 2-D/3-D activations, missing layer error
- All tests use mocked dependencies (no GPU or downloads required)

## Decisions Made

| ID | Decision | Rationale |
|----|----------|-----------|
| 24-03-D1 | LaBraM primary, BrainOmni secondary | LaBraM has cleaner braindecode API with `return_features`; BrainOmni requires hooks exclusively |
| 24-03-D2 | Shared PCA reduction pattern | Both extractors use `sklearn.decomposition.PCA`; consistent API (`reduce_to_dcm_space`) |
| 24-03-D3 | Spectral DCM default for M/EEG | Patch-level dynamics have limited temporal samples; spectral DCM more appropriate than latent-circuit for short sequences |

## Deviations from Plan

None -- plan executed exactly as written.

## Verification Results

| Check | Result |
|-------|--------|
| `pytest tests/test_meeg_extractor.py -v` | 17/17 passed |
| `24_extract_meeg_latents.py --help` | Parses correctly |
| `24_fit_dcm_meeg.py --help` | Parses correctly |
| `ruff check` on all 4 source files | All checks passed |
| Foundation `__init__.py` exports | `LaBraMExtractor`, `BrainOmniExtractor` importable |

## Commits

| Hash | Description |
|------|-------------|
| `ba0a02e` | feat(24-03): add LaBraM and BrainOmni M/EEG extractors with unit tests |
| `1da868f` | feat(24-03): add M/EEG pipeline scripts and cluster sbatch |

## Next Phase Readiness

- Phase 24-04 (cross-modal comparison) can proceed: both fMRI (TRIBE) and M/EEG extractors are complete
- Cam-CAN data access required for actual cross-modal A-matrix comparison
- No blockers identified
