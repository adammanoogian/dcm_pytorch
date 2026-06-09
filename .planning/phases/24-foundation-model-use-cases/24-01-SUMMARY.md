---
phase: 24-foundation-model-use-cases
plan: 01
backfilled: 2026-06-09
subsystem: foundation-model-extraction
tags: [base-extractor, parcellation, forward-hooks, schaefer, pyproject-extras]
requires: []
provides:
  - BaseExtractor ABC defining the latent-extraction contract for any pretrained model
  - parcellate_vertices_to_rois utility for fsaverage5 vertex-to-ROI aggregation
  - foundation optional-dependencies group in pyproject.toml
affects: ["24-02", "24-03", "24-04"]
tech-stack:
  added:
    - nilearn>=0.10.3
    - huggingface_hub>=0.20
    - braindecode>=1.3.0
  patterns:
    - "ABC with abstract load_model/extract_latents + concrete forward-hook utility"
    - "try/except ImportError guard for optional nilearn dependency"
    - "lru_cache on Schaefer label fetch to avoid repeated atlas downloads"
key-files:
  created:
    - src/pyro_dcm/foundation/__init__.py
    - src/pyro_dcm/foundation/base_extractor.py
    - src/pyro_dcm/foundation/parcellation.py
    - tests/test_foundation_base.py
  modified:
    - pyproject.toml
decisions:
  - id: "24-01-D1"
    summary: "extract_layer_activations is a concrete shared hook utility on the base class, not abstract"
  - id: "24-01-D2"
    summary: "nilearn guarded with try/except ImportError + install hint, foundation deps in optional extras only"
  - id: "24-01-D3"
    summary: "Schaefer surface assignment implemented as equal-size contiguous fallback, not full surface projection"
metrics:
  duration: "~2 minutes (two feat commits at 22:25 and 22:27)"
  completed: "2026-05-27"
---

# Phase 24 Plan 01: Foundation Extractor Infrastructure Summary

Shared foundation model extractor infrastructure: BaseExtractor ABC, vertex-to-ROI parcellation utility, and pyproject.toml [foundation] extras that Plans 02-04 build on.

## One-liner

BaseExtractor ABC defines the latent-extraction contract with a concrete forward-hook activation utility, plus parcellate_vertices_to_rois for fsaverage5 Schaefer aggregation, installable via a new [foundation] optional-dependencies group.

## What Was Built

### BaseExtractor ABC (`src/pyro_dcm/foundation/base_extractor.py`)

- `BaseExtractor(ABC)` with `__init__(model_name, device=None)` storing `model_name` and resolving `device` (defaults to CPU)
- Abstract `load_model(checkpoint_path=None) -> None`: loads weights from checkpoint or hub
- Abstract `extract_latents(input_data, layer_names=None) -> dict[str, torch.Tensor]`: returns layer-name -> activation mapping
- Concrete `extract_layer_activations(model, input_tensor, layer_names) -> dict[str, torch.Tensor]`: shared forward-hook utility -- validates each requested name against `model.named_modules()` (raises `ValueError` with available names if missing), registers temporary `register_forward_hook` hooks, runs a single `torch.no_grad()` forward pass, then removes all hooks in a `finally` block
- NumPy-style docstrings, `from __future__ import annotations`

### Parcellation Utility (`src/pyro_dcm/foundation/parcellation.py`)

- `parcellate_vertices_to_rois(vertex_timeseries, n_rois=100, atlas_name="schaefer") -> tuple[np.ndarray, list[str]]`: aggregates `(T, 20484)` fsaverage5 vertex timeseries into `(T, n_rois)` ROI timeseries plus `roi_names`
- `_fetch_schaefer_labels(n_rois)` helper decorated with `functools.lru_cache(maxsize=4)` to cache atlas labels and avoid repeated downloads
- Validates 2-D input and exactly 20484 columns (`_FSAVERAGE5_TOTAL_VERTICES = 2 * 10242`), raising `ValueError` otherwise; raises `ValueError` for unsupported `atlas_name`
- nilearn import (`fetch_atlas_schaefer_2018`) guarded with `try/except ImportError` and an install hint

### pyproject.toml Extras

- New `[project.optional-dependencies]` group `foundation = ["nilearn>=0.10.3", "huggingface_hub>=0.20", "braindecode>=1.3.0"]`
- New pytest marker `foundation: marks tests requiring foundation model deps (nilearn, huggingface_hub, braindecode)`
- Same commit also added an `sbi = ["sbi>=0.22"]` extras group (not in the plan; see Deviations)

### Unit Tests (`tests/test_foundation_base.py`)

5 tests across two classes:

- `TestBaseExtractor::test_base_extractor_is_abstract`: `BaseExtractor` cannot be instantiated directly
- `TestBaseExtractor::test_extract_layer_activations_hook_pattern`: 3-layer `nn.Sequential`, requests subset of layers, asserts exact keys and activation shapes
- `TestBaseExtractor::test_extract_layer_activations_removes_hooks`: confirms no residual hooks after extraction
- `TestParcellation::test_parcellate_vertices_mock`: mocks the atlas fetch, asserts `(T, n_rois)` output shape
- `TestParcellation::test_parcellate_vertices_wrong_shape`: non-20484 input raises `ValueError`

## Decisions Made

| ID | Decision | Rationale |
|----|----------|-----------|
| 24-01-D1 | extract_layer_activations concrete on base class | Shared forward-hook pattern reused by all subclass extractors |
| 24-01-D2 | nilearn guarded + foundation deps in extras only | Keeps base install lightweight; foundation deps are optional |
| 24-01-D3 | Equal-size contiguous Schaefer fallback | Avoids a full nilearn surface-projection pipeline while preserving the API contract |

## Deviations from Plan

- **Schaefer assignment is a simplified placeholder.** The plan specified averaging vertices "belonging to that ROI" via true Schaefer label assignment. The implemented `parcellate_vertices_to_rois` instead uses an equal-size contiguous parcellation (vertices split into `20484 // n_rois` contiguous blocks), self-described in the code as "a simplified placeholder that maintains the correct API contract; real analyses should use nilearn's surface projection." `_fetch_schaefer_labels` only caches label *names*, not vertex-to-label assignment.
- **`sbi = ["sbi>=0.22"]` extras added.** The feat commit added an `sbi` optional-dependencies group alongside `foundation`; this was not part of the 24-01 plan scope.
- **Two identical feat commits.** Both `0280d31` and `6d823a7` carry the same message and content (~2 minutes apart), likely a branch/rebase artifact.

## Verification Results

| Check | Status |
|-------|--------|
| Package import (BaseExtractor, parcellate_vertices_to_rois) | Re-exported in `__init__.py` |
| pytest tests/test_foundation_base.py | 5 tests present (3 base + 2 parcellation) |
| pyproject.toml [foundation] extras | Added (nilearn, huggingface_hub, braindecode) |
| pytest marker `foundation` registered | Yes |

## Next Phase Readiness

- **24-02** (TRIBE v2 fMRI): subclasses `BaseExtractor`, reuses `parcellate_vertices_to_rois` -- ready
- **24-03** (M/EEG extractors): same `BaseExtractor` hook pattern -- ready
- **24-04** (cross-modal comparison): builds on 24-02 + 24-03 outputs
- **Open item:** parcellation contiguous-block fallback should be replaced with true nilearn surface projection before quantitative ROI analyses
