---
phase: 24-foundation-model-use-cases
plan: 02
subsystem: foundation-model-extraction
tags: [tribe-v2, fmri, spectral-dcm, parcellation, slurm]
requires: ["24-01"]
provides:
  - TRIBEExtractor class wrapping Meta TRIBE v2 for vertex prediction + parcellation
  - Pipeline scripts for TRIBE v2 extraction and spectral DCM fitting
  - A100 GPU cluster sbatch for TRIBE v2 pipeline
affects: ["24-04"]
tech-stack:
  added: []
  patterns:
    - "try/except ImportError guard for optional tribev2 dependency"
    - "BaseExtractor subclass with vertex-to-ROI and hook extraction paths"
    - "Pipeline scripts with argparse + lazy imports for heavy dependencies"
key-files:
  created:
    - src/pyro_dcm/foundation/tribe_extractor.py
    - tests/test_tribe_extractor.py
    - scripts/24_extract_tribe_latents.py
    - scripts/24_fit_dcm_tribe.py
    - cluster/sbatch/24_tribe_extract.slurm
  modified: []
decisions:
  - id: "24-02-D1"
    summary: "TRIBE v2 import guarded with try/except ImportError, not added to pyproject.toml"
  - id: "24-02-D2"
    summary: "Pipeline scripts use lazy imports after argparse to keep --help fast"
  - id: "24-02-D3"
    summary: "Spectral DCM fitting uses compute_empirical_csd with fs=1.0 Hz for TRIBE v2 1-second TR"
metrics:
  duration: "~66 minutes (including branch resolution issues)"
  completed: "2026-05-28"
---

# Phase 24 Plan 02: TRIBE v2 fMRI Pipeline Summary

TRIBE v2 fMRI extraction pipeline with TRIBEExtractor class, pipeline scripts, and A100 cluster sbatch.

## One-liner

TRIBEExtractor wraps Meta TRIBE v2 model for vertex-wise fMRI prediction + Schaefer parcellation to ROI timeseries, with spectral DCM fitting via multi-start SVI.

## What Was Built

### TRIBEExtractor Class (`src/pyro_dcm/foundation/tribe_extractor.py`)

- Subclasses `BaseExtractor` with `model_name="tribe_v2"`
- `load_model()`: imports `tribev2.demo_utils.TribeModel` with ImportError guard and install hint
- `predict_vertex_timeseries(video_path, audio_path, events_df)`: returns `np.ndarray` shape `(T, 20484)` on fsaverage5
- `extract_roi_timeseries(vertex_timeseries)`: delegates to `parcellate_vertices_to_rois` for Schaefer atlas
- `extract_latents(input_data, layer_names)`: implements ABC; default path returns ROI timeseries, hook path uses `extract_layer_activations` for transformer internals
- `model_: Any = None` follows fitted attribute convention

### Pipeline Scripts

1. **`scripts/24_extract_tribe_latents.py`**: stimulus video -> TRIBE v2 -> ROI timeseries `.npz`
   - argparse: `--video-path`, `--output-dir`, `--n-rois`, `--cache-folder`
   - Saves `roi_timeseries`, `roi_names`, `n_rois`, `vertex_shape`, `stimulus_path`

2. **`scripts/24_fit_dcm_tribe.py`**: ROI timeseries -> empirical CSD -> spectral DCM -> posterior A
   - argparse: `--input-npz`, `--output-dir`, `--n-regions`, `--roi-indices`, `--num-steps`, `--n-restarts`, `--lr`, `--n-freqs`, `--seed`
   - Uses `compute_empirical_csd` with `fs=1.0` Hz (TRIBE v2 TR)
   - Multi-start SVI via `run_svi` with `guide_factory` for `create_guide` fresh initialization
   - Saves `A_mean`, `A_std`, `A_free_mean`, `A_free_std`, `final_loss`, `roi_names_selected`

### Cluster Sbatch (`cluster/sbatch/24_tribe_extract.slurm`)

- `--gres=gpu:A100:1`, `--mem=64G`, `--time=02:00:00`, `--partition=gpu`
- Pitfall mitigations: numpy pin (`>=1.26.4,<2.1.0`), `HF_HUB_DOWNLOAD_TIMEOUT=300`, `--no-deps` tribev2 install
- Configurable via environment: `STIMULUS_PATH`, `N_ROIS`, `N_REGIONS`, `ROI_INDICES`, `NUM_STEPS`, `N_RESTARTS`
- Runs both extraction and DCM fitting sequentially

### Unit Tests (`tests/test_tribe_extractor.py`)

4 tests, all passing:
1. `test_tribe_extractor_load_model_import_error`: verifies ImportError with install hint when tribev2 missing
2. `test_tribe_extractor_predict_vertex_shape`: mocked TribeModel returns correct (50, 20484) shape
3. `test_tribe_extractor_extract_roi_shape`: mocked parcellation returns correct (50, 100) shape
4. `test_tribe_extractor_not_loaded_error`: ValueError when predict called before load_model

## Decisions Made

| ID | Decision | Rationale |
|----|----------|-----------|
| 24-02-D1 | tribev2 import guarded, not in pyproject.toml | Optional GPU dependency; requires A100; install via git URL |
| 24-02-D2 | Lazy imports after argparse in pipeline scripts | Heavy torch/pyro imports slow; --help should be fast |
| 24-02-D3 | compute_empirical_csd with fs=1.0 Hz | TRIBE v2 outputs at 1 Hz (fMRI TR); Nyquist at 0.5 Hz |

## Deviations from Plan

None -- plan executed as written.

## Verification Results

| Check | Status |
|-------|--------|
| pytest tests/test_tribe_extractor.py -v | 4/4 PASSED |
| python scripts/24_extract_tribe_latents.py --help | Parses correctly |
| python scripts/24_fit_dcm_tribe.py --help | Parses correctly |
| ruff check (all source files) | All checks passed |
| cluster/sbatch/24_tribe_extract.slurm | Valid SLURM with A100 GPU, numpy pin, tribev2 install |

## Next Phase Readiness

- **24-03** (M/EEG extractors): can proceed independently; same BaseExtractor pattern
- **24-04** (cross-modal comparison): needs both 24-02 and 24-03 outputs
- **Cluster execution**: requires HuggingFace token with LLaMA 3.2 license accepted before running
