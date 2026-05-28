# Phase 24: Foundation Model Use Cases (TRIBE + M/EEG) - Research

**Researched:** 2026-05-26
**Domain:** Brain foundation model latent extraction + DCM interpretability
**Confidence:** MEDIUM (models verified as available; latent extraction for DCM is novel -- no established pipeline exists)

## Summary

This phase applies DCM as an interpretability tool to pretrained brain foundation models,
extracting latent temporal dynamics from (1) Meta TRIBE v2 for fMRI and (2) M/EEG
foundation models (LaBraM, MEG-GPT, BrainOmni). The key challenge is that these
foundation models were designed for prediction/classification, NOT for producing
time-varying latent trajectories suitable for DCM fitting. Each model requires a
different strategy for temporal dynamics extraction.

**TRIBE v2** (facebookresearch/tribev2) is publicly available under CC-BY-NC-4.0, outputs
vertex-wise fMRI predictions at 1 Hz on fsaverage5 (~20k vertices), and internally uses
an 8-layer, 8-head transformer with D_model=1152. The critical insight: TRIBE v2's
output IS already time-varying fMRI predictions, so we can either (a) parcellate its
vertex-wise output into ROI timeseries and fit spectral DCM directly, or (b) extract
intermediate transformer activations via PyTorch forward hooks for richer dynamics.
Requires 40 GB VRAM minimum (A100).

**For M/EEG**, three viable models exist with released code/weights: (1) **LaBraM** (ICLR
2024 spotlight, MIT license, integrated into braindecode, pretrained weights on
HuggingFace), (2) **MEG-GPT** (Oxford, MIT license, TensorFlow-based via osl-foundation,
pretrained on Cam-CAN N=612), and (3) **BrainOmni** (NeurIPS 2025, unified EEG+MEG,
HuggingFace checkpoints). The recommended primary M/EEG model is **BrainOmni** for its
dual-modality support and released PyTorch-compatible checkpoints, with **LaBraM** as the
EEG-specific fallback given its braindecode integration and explicit `return_features`
API.

**Cross-modal comparison** is feasible via the Cam-CAN dataset, which contains overlapping
sensorimotor tasks recorded in both fMRI and MEG on ~700 subjects. TRIBE v2 can generate
predicted fMRI for the same stimuli that MEG foundation models process, enabling
comparison of DCM-derived A matrices across modalities.

**Primary recommendation:** Use TRIBE v2 vertex-wise output parcellated to ROIs for fMRI
DCM; use BrainOmni (primary) or LaBraM (fallback) with PyTorch forward hooks for M/EEG
latent extraction; compare on Cam-CAN sensorimotor task.

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| tribev2 | latest (2026-03) | TRIBE v2 fMRI brain encoding model | Meta's official open-source release; CC-BY-NC-4.0 |
| braindecode | >=1.3.0 | LaBraM EEG model wrapper with pretrained weight loading | Official integration; `Labram.from_pretrained()` API |
| BrainOmni | latest (2025-05) | Unified EEG+MEG foundation model | NeurIPS 2025; only model with both EEG+MEG in one architecture |
| nilearn | >=0.10.3 | fMRI parcellation (fsaverage5 vertices to ROIs) | Standard for cortical parcellation in Python |
| mne | >=1.6 | M/EEG preprocessing, source localization | Already in project (Phase 18/19) |
| torch | >=2.0 | PyTorch hooks for intermediate layer extraction | Already in project |
| huggingface_hub | >=0.20 | Model/weight downloads for TRIBE v2, BrainOmni, LaBraM | Standard model distribution hub |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| osl-foundation | latest | MEG-GPT model (TensorFlow-based) | Only if MEG-GPT is chosen over BrainOmni |
| transformers | >=4.35 | HuggingFace transformer utilities | May be needed for TRIBE v2 LLaMA dependency |
| nibabel | >=5.0 | NIfTI/surface file I/O for fsaverage5 | TRIBE v2 cortical surface output |
| scipy.signal | -- | CSD computation from parcellated timeseries | Spectral DCM path |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| BrainOmni (M/EEG) | LaBraM (EEG only) | LaBraM has better braindecode API but EEG-only; BrainOmni handles both EEG+MEG |
| BrainOmni (M/EEG) | MEG-GPT | MEG-GPT is TensorFlow-based (ecosystem friction); requires Cam-CAN source-localized data |
| TRIBE v2 (fMRI) | BrainLM | BrainLM operates on existing fMRI, not stimulus->fMRI prediction; different use case |
| BrainOmni (M/EEG) | Brain-OF | Brain-OF handles fMRI+EEG+MEG but code/weights availability unconfirmed as of 2026-05 |

**Installation:**

```bash
# TRIBE v2 (requires A100 40GB+ GPU)
pip install 'numpy>=1.26.4,<2.1.0'
pip install 'tribev2[plotting] @ git+https://github.com/facebookresearch/tribev2.git'
pip install 'nilearn>=0.10.3'

# LaBraM via braindecode
pip install braindecode>=1.3.0

# BrainOmni
pip install -r requirements.txt  # from BrainOmni repo clone

# MEG-GPT (TensorFlow - only if needed)
pip install osl-foundation  # via conda/mamba recommended
```

## Architecture Patterns

### Recommended Project Structure

```
src/pyro_dcm/
├── foundation/                    # NEW: Foundation model wrappers
│   ├── __init__.py
│   ├── base_extractor.py          # Abstract base for latent extraction
│   ├── tribe_extractor.py         # TRIBE v2 fMRI latent/output extraction
│   ├── labram_extractor.py        # LaBraM EEG latent extraction
│   ├── brainomni_extractor.py     # BrainOmni EEG+MEG latent extraction
│   └── parcellation.py            # fsaverage5 vertex->ROI parcellation
├── models/
│   └── (existing models unchanged)
scripts/
├── 24_extract_tribe_latents.py    # Pipeline: stimulus -> TRIBE v2 -> ROI timeseries
├── 24_extract_meeg_latents.py     # Pipeline: M/EEG -> foundation model -> latent dynamics
├── 24_fit_dcm_foundation.py       # Pipeline: latent dynamics -> DCM fit
└── 24_compare_crossmodal.py       # Pipeline: cross-modal A-matrix comparison
```

### Pattern 1: Vertex-to-ROI Parcellation (TRIBE v2 Output Path)

**What:** TRIBE v2 outputs vertex-wise fMRI predictions at shape (T, 20484) on
fsaverage5. Parcellate to ROI timeseries (T, N) using a standard atlas, then fit
spectral DCM to the ROI timeseries.

**When to use:** Primary path for fMRI foundation model use case. This is the
simplest approach -- no hook extraction needed.

**Example:**

```python
# Source: Verified from TRIBE v2 DataCamp tutorial + nilearn docs
from tribev2.demo_utils import TribeModel
from nilearn import datasets, surface
import numpy as np

# 1. Load model and predict
model = TribeModel.from_pretrained('facebook/tribev2', cache_folder='./cache')
events = model.get_events_dataframe(video_path='stimulus.mp4')
preds, segments = model.predict(events=events)
preds = np.asarray(preds)  # shape: (T_seconds, 20484)

# 2. Parcellate to ROIs using Schaefer atlas on fsaverage5
atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100, resolution_mm=2)
# Map vertex predictions to ROI averages
# Left hemisphere: preds[:, :10242], Right: preds[:, 10242:]
# Use nilearn.surface utilities for parcellation

# 3. Result: roi_timeseries shape (T, N_rois) -- feed to spectral DCM
```

### Pattern 2: Forward Hook Extraction (M/EEG and Optional TRIBE v2 Deep Path)

**What:** Register PyTorch forward hooks on intermediate transformer layers to
capture time-varying hidden state activations during inference. These activations
serve as the "latent dynamics" that DCM fits to.

**When to use:** For M/EEG foundation models (LaBraM, BrainOmni) where the goal
is to characterize dynamics within the model's learned representation space.

**Example:**

```python
# Source: PyTorch docs + braindecode Labram API
import torch
from braindecode.models import Labram

model = Labram(
    n_times=1600, n_chans=64, n_outputs=4,
    sfreq=200, neural_tokenizer=True,
)
# Load pretrained weights
model = Labram.from_pretrained("braindecode/labram-pretrained")
model.eval()

# Option A: Use built-in feature extraction
with torch.no_grad():
    features = model(eeg_input, return_features=True)
    # features dict contains 'features' and 'cls_token'
    latent_dynamics = features['features']  # (batch, n_patches, embed_dim)

# Option B: Hook intermediate transformer layers for richer dynamics
activations = {}
def hook_fn(name):
    def hook(module, input, output):
        activations[name] = output.detach()
    return hook

# Register hooks on transformer blocks
for i, block in enumerate(model.encoder.blocks):
    block.register_forward_hook(hook_fn(f'block_{i}'))

with torch.no_grad():
    _ = model(eeg_input)
    # activations['block_5'] etc. now contain (batch, seq_len, embed_dim)
```

### Pattern 3: Cross-Modal Comparison via Cam-CAN

**What:** Use Cam-CAN sensorimotor task data (available in both fMRI and MEG) as
the common ground truth. Extract latent dynamics from TRIBE v2 (fMRI path) and
BrainOmni/LaBraM (M/EEG path) for overlapping ROIs, fit DCM to each, compare
posterior A matrices.

**When to use:** Success criterion 3 (cross-modal connectivity comparison).

**Example:**

```python
# Pseudocode for cross-modal comparison
# 1. Define common ROIs (e.g., motor cortex, auditory cortex, visual cortex)
common_rois = ['M1', 'A1', 'V1', 'SMA']  # Example sensorimotor ROIs

# 2. fMRI path: TRIBE v2 predictions for sensorimotor stimulus
#    -> parcellate to common_rois -> spectral DCM -> A_fmri posterior
A_fmri_mean, A_fmri_std = fit_spectral_dcm(tribe_roi_timeseries)

# 3. MEG path: BrainOmni features for same stimulus on source-localized data
#    -> extract latent dynamics at ROI level -> DCM -> A_meg posterior
A_meg_mean, A_meg_std = fit_latent_circuit_dcm(brainomni_latents)

# 4. Compare: correlation of A matrices, overlap of credible intervals
cross_modal_r = np.corrcoef(A_fmri_mean.flatten(), A_meg_mean.flatten())[0, 1]
```

### Anti-Patterns to Avoid

- **Treating foundation model embeddings as static vectors:** DCM needs TIME-VARYING
  dynamics, not single CLS tokens. Always extract patch-level or layer-level temporal
  sequences, never just the final classification embedding.

- **Mixing temporal resolutions without resampling:** TRIBE v2 operates at 1 Hz (fMRI
  TR), M/EEG models operate at 200-250 Hz. Latent dynamics must be resampled to
  match the DCM model's expected temporal resolution before fitting.

- **Assuming model outputs are neural activity:** Foundation model predictions/latents
  are LEARNED REPRESENTATIONS, not ground truth neural signals. The DCM A matrix
  describes effective connectivity WITHIN the model's representation space, which may
  or may not map to true neural connectivity. Document this distinction clearly.

- **Training foundation models from scratch:** All models in this phase are PRETRAINED.
  No training is needed. Only inference + latent extraction + DCM fitting.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Cortical parcellation | Custom vertex-to-ROI averaging | `nilearn.datasets.fetch_atlas_schaefer_2018` + surface utilities | Handles hemisphere indexing, atlas alignment, missing vertices |
| Intermediate activation extraction | Manual gradient tracking | `register_forward_hook` (PyTorch) | Hook API is battle-tested; handles autograd correctly |
| fMRI prediction from stimuli | Custom encoding model | TRIBE v2 `model.predict()` | 700-subject training; 1115 hours of data behind it |
| EEG preprocessing for foundation models | Custom bandpass/notch pipeline | MNE-Python (`mne.io.Raw.filter`) + braindecode preprocessing | Already validated in Phase 18; LaBraM requires specific 0.1-75 Hz bandpass |
| Source localization for MEG | Custom beamformer | MNE-Python `mne.beamformer.make_lcmv` | MEG-GPT uses LCMV beamformer; standard in MNE |
| Cross-modal ROI alignment | Manual atlas mapping | `nilearn` with common atlas (Schaefer/Desikan-Killiany) | Consistent parcellation across modalities |

**Key insight:** This phase is an INTEGRATION task, not an implementation task. The
foundation models, DCM fitting, and IO infrastructure all exist. The novel contribution
is the pipeline that connects them: stimulus -> foundation model -> latent dynamics ->
DCM -> interpretable connectivity.

## Common Pitfalls

### Pitfall 1: TRIBE v2 Requires Gated LLaMA Access

**What goes wrong:** TRIBE v2 loads LLaMA 3.2-3B as its text encoder. LLaMA is a gated
model on HuggingFace requiring license acceptance.
**Why it happens:** The model.predict() call loads all three encoders (LLaMA, V-JEPA2,
Wav2Vec-BERT) regardless of which modality the input is.
**How to avoid:** Accept the LLaMA 3.2 license on HuggingFace before running. Set
`HF_HUB_DOWNLOAD_TIMEOUT=300`. Pre-download with `snapshot_download()`.
**Warning signs:** `TimeoutError` or `GatedRepoError` during `TribeModel.from_pretrained()`.

### Pitfall 2: NumPy Version Conflict with TRIBE v2

**What goes wrong:** TRIBE v2's dependency `neuralset` was compiled against NumPy <2.1.
NumPy 2.x causes silent failures or import errors.
**Why it happens:** Package was built before NumPy 2.0 became default.
**How to avoid:** Pin `numpy>=1.26.4,<2.1.0` and install BEFORE tribev2.
**Warning signs:** Import errors mentioning `numpy.core` or ABI incompatibility.

### Pitfall 3: GPU Memory Exhaustion with TRIBE v2

**What goes wrong:** Full trimodal pipeline requires 28-32 GB VRAM. T4 (16 GB) and
V100 (32 GB with overhead) may fail.
**Why it happens:** Three frozen encoders loaded simultaneously: LLaMA 3.2-3B (~7 GB),
V-JEPA2-Giant (~14 GB), Wav2Vec-BERT 2.0 (~1 GB), plus transformer weights.
**How to avoid:** Use A100 40 GB or higher. On M3 cluster, request `--gres=gpu:A100:1`
or equivalent. Consider audio-only mode on L4 (24 GB) if video not needed.
**Warning signs:** CUDA OOM during `model.predict()`.

### Pitfall 4: Temporal Resolution Mismatch Between Modalities

**What goes wrong:** TRIBE v2 outputs at 1 Hz (fMRI TR), M/EEG models operate at
200-250 Hz natively. Comparing A matrices across modalities with different temporal
bases is scientifically questionable.
**Why it happens:** fMRI and M/EEG have fundamentally different temporal resolutions.
**How to avoid:** For cross-modal comparison, use spectral DCM (frequency domain) where
the frequency resolution can be matched across modalities. Or downsample M/EEG
latent dynamics to a common temporal basis before fitting temporal DCM.
**Warning signs:** A matrix magnitudes differ by orders of magnitude between modalities.

### Pitfall 5: LaBraM Channel Ordering Sensitivity

**What goes wrong:** LaBraM requires specific EEG channel ordering. Passing channels
in wrong order produces garbage features.
**Why it happens:** The model was pretrained with a specific channel montage; spatial
embeddings are position-dependent.
**How to avoid:** Always provide `chs_info` with correct channel names/positions when
instantiating via braindecode. Use `Labram(chs_info=epochs.info['chs'], ...)`.
**Warning signs:** Feature values near zero or constant across channels.

### Pitfall 6: MEG-GPT Is TensorFlow-Based

**What goes wrong:** osl-foundation (MEG-GPT) requires TensorFlow 2.11, which conflicts
with PyTorch GPU memory management in the same process.
**Why it happens:** MEG-GPT was developed at Oxford using TensorFlow; PyTorch port
is still under development.
**How to avoid:** If using MEG-GPT, run extraction in a separate process/environment.
Or prefer BrainOmni (PyTorch-native) to avoid ecosystem friction.
**Warning signs:** GPU memory conflicts, import errors between TF and PyTorch.

### Pitfall 7: Foundation Model Latents Are Not Neural Activity

**What goes wrong:** Interpreting DCM A matrices from foundation model latents as if
they describe true neural effective connectivity.
**Why it happens:** Easy to conflate "connectivity in model representation space" with
"connectivity in the brain."
**How to avoid:** Frame results as "effective connectivity within the foundation model's
learned representations" explicitly. Compare to DCM fit on raw data (Phase 22) to
assess what the model's representations add.
**Warning signs:** Claiming DCM recovers "neural circuits" from model embeddings.

## Code Examples

### TRIBE v2: Full Pipeline from Stimulus to ROI Timeseries

```python
# Source: DataCamp TRIBE v2 tutorial + nilearn docs (verified 2026-05-26)
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

# --- Step 1: Load TRIBE v2 ---
from tribev2.demo_utils import TribeModel

model = TribeModel.from_pretrained(
    "facebook/tribev2",
    cache_folder="./tribe_cache",
)

# --- Step 2: Predict fMRI response for a stimulus ---
events = model.get_events_dataframe(video_path="sensorimotor_stimulus.mp4")
preds, segments = model.predict(events=events)
vertex_timeseries = np.asarray(preds)  # (T_seconds, 20484) at 1 Hz

# --- Step 3: Parcellate to ROIs ---
from nilearn import datasets

# Schaefer 100-ROI atlas on fsaverage5
atlas = datasets.fetch_atlas_schaefer_2018(
    n_rois=100, resolution_mm=2
)
# vertex_timeseries[:, :10242] = left hemi
# vertex_timeseries[:, 10242:] = right hemi
# Use atlas labels to average vertices within each ROI
# Result: roi_timeseries shape (T, N_rois)
```

### LaBraM: Feature Extraction via Braindecode

```python
# Source: braindecode Labram API docs (verified 2026-05-26)
from __future__ import annotations

import torch
from braindecode.models import Labram

# Load pretrained LaBraM
model = Labram(
    n_times=1600,       # 8 seconds at 200 Hz
    n_chans=64,
    n_outputs=4,        # task classes (ignored for feature extraction)
    sfreq=200,
    patch_size=200,     # 1-second patches
    embed_dim=200,
    num_layers=12,
    num_heads=10,
    neural_tokenizer=True,
)
model = Labram.from_pretrained("braindecode/labram-pretrained")
model.eval()

# Extract features (patch-level temporal dynamics)
eeg_input = torch.randn(1, 64, 1600)  # (batch, channels, timepoints)
with torch.no_grad():
    out = model(eeg_input, return_features=True)
    features = out["features"]    # (batch, n_patches, embed_dim)
    cls_token = out["cls_token"]  # (batch, embed_dim) -- DO NOT use for DCM

# features has shape (1, 8, 200) for 8 one-second patches
# Each patch embedding is a 200-dim latent state -- these are the dynamics
# Reduce to N DCM dimensions via PCA (reuse Phase 21 DIM pipeline)
```

### PyTorch Forward Hook Pattern for Any Foundation Model

```python
# Source: PyTorch docs register_forward_hook + TorchLens paper
from __future__ import annotations

import torch
from torch import nn


def extract_layer_activations(
    model: nn.Module,
    input_tensor: torch.Tensor,
    layer_names: list[str],
) -> dict[str, torch.Tensor]:
    """Extract intermediate activations from named layers.

    Parameters
    ----------
    model : nn.Module
        Pretrained foundation model.
    input_tensor : torch.Tensor
        Model input.
    layer_names : list[str]
        Names of layers to hook (from model.named_modules()).

    Returns
    -------
    dict[str, torch.Tensor]
        Layer name -> activation tensor.
    """
    activations: dict[str, torch.Tensor] = {}
    hooks: list[torch.utils.hooks.RemovableHook] = []

    for name, module in model.named_modules():
        if name in layer_names:

            def _hook(mod, inp, out, name=name):
                activations[name] = out.detach().cpu()

            hooks.append(module.register_forward_hook(_hook))

    with torch.no_grad():
        _ = model(input_tensor)

    for h in hooks:
        h.remove()

    return activations
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Hand-crafted brain encoding models | Foundation models (TRIBE v2, LaBraM, BrainOmni) | 2024-2026 | Pre-trained on 1000s of hours; transfer to new tasks/subjects |
| EEG-only or MEG-only models | Unified EEG+MEG models (BrainOmni) | NeurIPS 2025 | Single model handles both modalities |
| Stimulus->voxel encoding (ridge regression) | Deep multimodal encoding (TRIBE v2) | March 2026 | 70x resolution increase over prior models |
| DCM on raw fMRI/MEG only | DCM on foundation model latents (this phase) | Novel | Characterize learned representations via interpretable connectivity |
| Separate EEG and MEG foundation models | Brain-OF (omnifunctional fMRI+EEG+MEG) | Feb 2026 | Early-stage; code availability uncertain |

**Deprecated/outdated:**
- Single-modality brain encoding (superseded by multimodal TRIBE v2)
- GPT2MEG (early MEG autoregressive; superseded by MEG-GPT and BrainOmni)

## Open Questions

1. **How to define "temporal dynamics" from TRIBE v2's output for DCM fitting?**
   - What we know: Output is (T, 20484) at 1 Hz on fsaverage5. Can parcellate to ROI
     timeseries. Internal transformer has 8 layers with 100s context window.
   - What's unclear: Is the parcellated vertex output rich enough for spectral DCM, or
     do we need intermediate transformer layer activations? The vertex output is already
     a prediction of fMRI, which has hemodynamic convolution baked in -- so the
     direct-observation (latent circuit) DCM model may not be appropriate here.
   - Recommendation: Start with the simpler path (vertex output -> ROI -> spectral DCM).
     Only add hook extraction if spectral DCM on vertex output is uninformative.

2. **Which DCM variant to use for each modality path?**
   - What we know: TRIBE v2 output resembles fMRI (hemodynamic convolution baked in) ->
     spectral DCM is natural. M/EEG foundation model latents are learned representations
     (no hemodynamics) -> latent circuit (direct observation) DCM is natural.
   - What's unclear: Whether spectral DCM on M/EEG latent dynamics makes sense
     (depends on whether the model preserves spectral structure).
   - Recommendation: TRIBE v2 -> spectral DCM. M/EEG models -> try both spectral and
     latent circuit DCM; compare. Phase 22 will establish the M/EEG DCM approach first.

3. **Cross-modal A-matrix comparison: what metric?**
   - What we know: A matrices will have different scales (fMRI 1 Hz vs M/EEG ~200 Hz
     native dynamics). Can compare topology (sign pattern, relative magnitudes).
   - What's unclear: Whether frequency-domain comparison (spectral DCM on both) or
     topology comparison (binarized sign matrix correlation) is more informative.
   - Recommendation: Report both Pearson correlation of A values AND sign-pattern
     agreement (Cohen's kappa). Use spectral DCM for both modalities to make scales
     comparable.

4. **BrainOmni vs LaBraM: which to prioritize?**
   - What we know: BrainOmni handles both EEG+MEG (ideal for cross-modal); LaBraM has
     cleaner braindecode API with explicit `return_features=True`. BrainOmni NeurIPS 2025
     checkpoints on HuggingFace. LaBraM ICLR 2024 spotlight, more citations/maturity.
   - What's unclear: BrainOmni's exact Python API for feature extraction (need to
     inspect demo.ipynb). Whether BrainOmni's tokenizer preserves temporal structure
     needed for DCM.
   - Recommendation: Try BrainOmni first (dual modality). Fall back to LaBraM for
     EEG-only if BrainOmni's API is too immature. MEG-GPT as third option (but
     TensorFlow friction).

5. **Cam-CAN data access for cross-modal comparison**
   - What we know: Cam-CAN has ~700 subjects with overlapping sensorimotor task in both
     fMRI and MEG. Data access requires application to Cam-CAN.
   - What's unclear: Whether TRIBE v2 was trained on Cam-CAN fMRI (if so, comparing
     TRIBE v2 predictions on Cam-CAN stimuli to MEG-GPT trained on Cam-CAN MEG is a
     valid within-dataset comparison).
   - Recommendation: Check TRIBE v2's training data manifest. If Cam-CAN fMRI is
     included, use Cam-CAN sensorimotor stimulus set. Otherwise, use a public
     audio-visual stimulus that both modalities can process.

6. **Compute requirements for full pipeline**
   - What we know: TRIBE v2 requires A100 40 GB. LaBraM and BrainOmni are smaller
     (fit on consumer GPUs). DCM fitting (SVI) runs on CPU (established in Phase 20).
   - What's unclear: Total wall-time for end-to-end pipeline: stimulus processing +
     latent extraction + parcellation + DCM fitting across multiple subjects/seeds.
   - Recommendation: Route TRIBE v2 inference to M3 cluster GPU nodes. LaBraM/BrainOmni
     inference may fit on laptop GPU but route to cluster for safety. DCM fitting on
     cluster CPU nodes (established practice from Phase 20).

## Sources

### Primary (HIGH confidence)

- **TRIBE v2 GitHub** (https://github.com/facebookresearch/tribev2) -- official Meta
  repository, CC-BY-NC-4.0 license, verified available 2026-05-26
- **TRIBE v2 DataCamp tutorial** (https://www.datacamp.com/tutorial/tribe-v2-tutorial) --
  step-by-step code, GPU requirements (A100 40GB), output format (T, 20484), NumPy
  version constraint
- **LaBraM GitHub** (https://github.com/935963004/LaBraM) -- ICLR 2024 spotlight,
  MIT license, pretrained weights via HuggingFace
- **Braindecode Labram API** (https://braindecode.org/dev/generated/braindecode.models.Labram.html)
  -- `Labram` class API, `return_features=True`, architecture (12 layers, 10 heads, embed_dim=200)
- **BrainOmni GitHub** (https://github.com/OpenTSLab/BrainOmni) -- NeurIPS 2025,
  unified EEG+MEG, HuggingFace checkpoints (tiny, base, tokenizer)
- **MEG-GPT / osl-foundation** (https://github.com/OHBA-analysis/osl-foundation) --
  TensorFlow implementation, MIT license, Cam-CAN pretrained (N=612), HuggingFace weights
- **Cam-CAN dataset** (https://pmc.ncbi.nlm.nih.gov/articles/PMC5182075/) --
  ~700 subjects, overlapping fMRI+MEG sensorimotor task

### Secondary (MEDIUM confidence)

- **Meta AI blog** (https://ai.meta.com/blog/tribe-v2-brain-predictive-foundation-model/)
  -- architecture overview (8-layer, 8-head transformer, D_model=1152, 100s context)
- **TRIBE paper arxiv:2507.22229** -- original TRIBE architecture details (D_model =
  3x384 = 1152)
- **MEG-GPT paper arxiv:2510.18080** -- 52 brain regions, LCMV beamformer, Perceiver AR,
  K*=61 tokens, ~400 GPU hours training
- **Brain foundation models survey** (arxiv:2503.00580) -- taxonomy of BFMs, BrainLM,
  LaBraM, NeuroLM comparison
- **BrainOmni paper** (arxiv:2505.18185) -- BrainTokenizer architecture, 1997h EEG +
  656h MEG pretraining data
- **TorchLens** (Nature Scientific Reports, 2023) -- PyTorch hidden activation extraction
  patterns

### Tertiary (LOW confidence)

- **Brain-OF** (arxiv:2602.23410) -- omnifunctional fMRI+EEG+MEG model; code/weights
  availability UNCONFIRMED
- **LaBraM brain state trajectory** (https://github.com/MarksonChen/labram-brain-state-trajectory)
  -- "interpreting pretrained EEG model as dynamic system approximator"; limited
  documentation, relationship to DCM unclear
- **BrainLM** (bioRxiv 2023, HuggingFace vandijklab/brainlm) -- fMRI foundation model;
  operates on existing fMRI data (different use case from TRIBE v2 stimulus encoding)
- **Brain-Semantoks** (arxiv:2512.11582) -- semantic tokenizer for brain dynamics;
  early-stage, no code verified

## Metadata

**Confidence breakdown:**
- Standard stack: MEDIUM -- models verified as available with code+weights; integration
  patterns (hook extraction, parcellation) are standard PyTorch; but no existing
  end-to-end pipeline for "foundation model -> DCM" exists to validate against
- Architecture: MEDIUM -- patterns are well-established (hooks, parcellation, DCM
  fitting) but their COMPOSITION for this use case is novel and untested
- Pitfalls: HIGH -- GPU requirements, NumPy conflicts, channel ordering, TF/PyTorch
  ecosystem friction all verified from official sources and tutorials
- Cross-modal comparison: LOW -- feasibility depends on Cam-CAN data access and
  whether TRIBE v2 predictions are meaningful on Cam-CAN stimuli; scientifically novel
  territory

**Research date:** 2026-05-26
**Valid until:** 2026-07-26 (60 days -- brain foundation models are a fast-moving field;
  new models/versions may appear; check for BrainOmni v2, Brain-OF code release,
  MEG-GPT PyTorch port)
