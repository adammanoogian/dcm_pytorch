# Phase 22: DCM Interpretability for Neural Data Models - Research

**Researched:** 2026-05-26
**Domain:** Deep learning on M/EEG source-reconstructed data + DCM fitting to latent dynamics
**Confidence:** MEDIUM (novel integration of established components; no single prior work does exactly this)

## Summary

Phase 22 trains a temporal deep learning model on real MEG data (Cam-CAN sensorimotor
task), extracts its learned latent dynamics, and fits spectral or bilinear DCM to
characterize effective connectivity inside the model's representations. This is the core
scientific contribution of v0.6.0.

The research identified five major technical pillars: (1) MEG data acquisition and
preprocessing via Cam-CAN, (2) source reconstruction to ROI timeseries via MNE-Python
LCMV beamformer + parcellation, (3) temporal model architecture for learning latent
representations, (4) latent dynamics extraction and CSD computation, and (5) DCM fitting
using existing Pyro-DCM infrastructure. The existing codebase already provides robust
spectral DCM (`spectral_dcm_model`), bilinear task DCM (`latent_circuit_dcm_model`),
MNE IO loaders (`mne_loader.py`), and CSD computation (`csd_computation.py`).

The recommended approach is a **two-path analysis**: (a) fit spectral DCM directly to
CSD of source-reconstructed MEG ROI timeseries (baseline), and (b) train an LSTM
autoencoder on the same ROI timeseries, extract latent trajectories, compute CSD of
latent dynamics, and fit spectral DCM to the latent CSD. Comparing (a) vs (b) is
Success Criterion 5. Bilinear DCM on task-epoched latent trajectories is a secondary
analysis for task-modulated connectivity.

**Primary recommendation:** Use an LSTM autoencoder (not a transformer or VAE) as the
temporal model -- it is the simplest architecture that learns temporal latent
representations, is well-understood, trains in hours on a single GPU, and avoids the
complexity of foundation model fine-tuning. Reserve MEG-GPT/LaBraM for Phase 24.

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| MNE-Python | >=1.6 | MEG preprocessing, source reconstruction, parcellation | Already in project deps; industry standard for M/EEG analysis |
| PyTorch | >=2.0 | LSTM autoencoder training | Already in project deps |
| Pyro | >=1.9 | SVI for spectral/bilinear DCM | Already in project deps |
| scipy | >=1.10 | Welch CSD computation on latent timeseries | Already in project deps |
| scikit-learn | >=1.3 | PCA for optional latent dim reduction, metrics | Already in optional deps |
| numpy | >=1.24 | Array operations | Already in project deps |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| mne-camcan | latest | Convenience IO for Cam-CAN FIF files | Optional; can use raw MNE instead |
| matplotlib | >=3.7 | Diagnostic plots, connectivity matrices, CSD spectra | Visualization only |
| h5py | >=3.9 | Storage for preprocessed ROI timeseries | If .npz is insufficient for large data |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| LSTM autoencoder | Temporal VAE (LFADS-style) | VAE has principled latent space + uncertainty, but more complex to implement and tune; LSTM-AE sufficient for reconstruction |
| LSTM autoencoder | Transformer autoencoder | Higher capacity but needs more data and compute; overkill for 52-ROI timeseries |
| LSTM autoencoder | DyNeMo (osl-dynamics) | TensorFlow dependency, heavy preprocessing pipeline; not PyTorch-native |
| LSTM autoencoder | MEG-GPT (pretrained) | TensorFlow, requires osl-dynamics toolchain; better for Phase 24 |
| LCMV beamformer | MNE minimum-norm estimate | LCMV is volumetric and standard for MEG; MNE provides both |
| Welch CSD | Multitaper CSD (mne.time_frequency) | Welch matches existing `csd_computation.py`; multitaper available via `epochs_to_csd` |
| Desikan-Killiany atlas | Schaefer parcellation | DK is coarser (68 regions) but well-validated; Schaefer (100+) gives finer granularity but more parameters for DCM |

**Installation (new deps only):**
```bash
pip install mne>=1.6  # already in pyproject.toml [mne] extra
# No new dependencies needed beyond existing [mne] + [latent] extras
```

## Architecture Patterns

### Recommended Project Structure

New code for Phase 22 lives in existing package structure:

```
src/pyro_dcm/
  neural_data_models/           # NEW subpackage
    __init__.py
    lstm_autoencoder.py         # LSTM-AE architecture (nn.Module)
    trainer.py                  # Training loop + checkpointing
    latent_extraction.py        # Extract latent trajectories from trained model
  io/
    camcan_loader.py            # NEW: Cam-CAN specific data loading + preprocessing
    mne_loader.py               # EXISTING: extended with source_reconstruct_to_roi()
  forward_models/
    csd_computation.py          # EXISTING: reuse compute_empirical_csd for latent CSD
    spectral_transfer.py        # EXISTING: reuse for spectral DCM forward model
  models/
    spectral_dcm_model.py       # EXISTING: fits CSD of latent dynamics
    latent_circuit_dcm_model.py # EXISTING: fits latent trajectories directly
scripts/
  22_preprocess_camcan.py       # Pipeline step 1: raw MEG -> source ROI timeseries
  22_train_autoencoder.py       # Pipeline step 2: train LSTM-AE on ROI timeseries
  22_extract_latent_dynamics.py # Pipeline step 3: extract latent trajectories
  22_fit_spectral_dcm.py        # Pipeline step 4: fit spDCM to raw and latent CSD
  22_compare_connectivity.py    # Pipeline step 5: compare DCM results
cluster/sbatch/
  22_train_autoencoder.slurm    # M3 sbatch for GPU training
  22_fit_dcm.slurm              # M3 sbatch for DCM fitting
tests/
  test_lstm_autoencoder.py      # Unit tests for model architecture
  test_camcan_pipeline.py       # Integration tests (synthetic data, no download)
```

### Pattern 1: LSTM Autoencoder for MEG ROI Timeseries

**What:** Encoder-decoder LSTM that compresses multivariate ROI timeseries (T, N_roi)
into a lower-dimensional latent trajectory (T, N_latent), then reconstructs the original.

**When to use:** When you want to learn a compressed temporal representation of
multi-channel neural data that preserves temporal dynamics.

**Architecture:**
```python
class MEGAutoencoder(nn.Module):
    """LSTM autoencoder for MEG source-localized ROI timeseries.

    Parameters
    ----------
    n_roi : int
        Number of input ROIs (e.g., 52 for MEG-GPT parcellation, or
        6-10 selected task-relevant ROIs).
    n_latent : int
        Latent dimension (bottleneck size). Typical: 4-16.
    hidden_size : int
        LSTM hidden units. Typical: 64-128.
    n_layers : int
        Number of LSTM layers. Typical: 1-2.
    """
    def __init__(self, n_roi, n_latent, hidden_size=64, n_layers=1):
        super().__init__()
        self.encoder_lstm = nn.LSTM(n_roi, hidden_size, n_layers, batch_first=True)
        self.encoder_fc = nn.Linear(hidden_size, n_latent)
        self.decoder_fc = nn.Linear(n_latent, hidden_size)
        self.decoder_lstm = nn.LSTM(hidden_size, n_roi, n_layers, batch_first=True)

    def encode(self, x):
        # x: (batch, T, N_roi) -> latent: (batch, T, N_latent)
        h, _ = self.encoder_lstm(x)
        latent = self.encoder_fc(h)  # (batch, T, N_latent)
        return latent

    def decode(self, latent):
        h = self.decoder_fc(latent)
        out, _ = self.decoder_lstm(h)
        return out  # (batch, T, N_roi)

    def forward(self, x):
        latent = self.encode(x)
        recon = self.decode(latent)
        return recon, latent
```

**Key design decisions:**
- The encoder produces a latent trajectory at **every timestep** (not just the final
  hidden state), because we need continuous latent dynamics for CSD computation.
- The bottleneck `n_latent` forces compression. Start with N_latent = N_roi // 4
  (e.g., 4 latent dims for 16 ROIs) and sweep {2, 4, 8, 16}.
- Training loss: MSE reconstruction + optional L1 sparsity on latent.

### Pattern 2: Source Reconstruction Pipeline

**What:** Complete MNE-Python pipeline from raw Cam-CAN MEG to parcellated ROI timeseries.

**Pipeline steps:**
```python
# 1. Load raw MEG data
raw = mne.io.read_raw_fif(camcan_fif_path, preload=True)

# 2. Preprocessing (already done by Cam-CAN: tSSS + MaxFilter)
# Additional: bandpass 1-45 Hz, resample to 250 Hz
raw.filter(1.0, 45.0)
raw.resample(250)

# 3. Epoch for sensorimotor task
events = mne.find_events(raw, stim_channel="STI101")
epochs = mne.Epochs(raw, events, tmin=-0.2, tmax=0.8, preload=True)

# 4. Source reconstruction via LCMV beamformer
fwd = mne.make_forward_solution(epochs.info, trans, src, bem)
data_cov = mne.compute_covariance(epochs)
noise_cov = mne.compute_covariance(epochs, tmax=0.0)
filters = mne.beamformer.make_lcmv(epochs.info, fwd, data_cov,
                                    noise_cov=noise_cov,
                                    reg=0.05, weight_norm="unit-noise-gain",
                                    pick_ori="max-power")
stc = mne.beamformer.apply_lcmv_epochs(epochs, filters)

# 5. Extract ROI timeseries
labels = mne.read_labels_from_annot("fsaverage", "aparc", subjects_dir=...)
# Select task-relevant ROIs:
roi_names = [
    "superiortemporal-lh",     # auditory cortex L
    "superiortemporal-rh",     # auditory cortex R
    "precentral-lh",           # motor cortex L
    "precentral-rh",           # motor cortex R
    "lateraloccipital-lh",     # visual cortex L
    "lateraloccipital-rh",     # visual cortex R
]
selected_labels = [l for l in labels if l.name in roi_names]
roi_ts = mne.extract_label_time_course(stc, selected_labels, src, mode="mean_flip")
# roi_ts shape: (n_epochs, N_roi, T)
```

### Pattern 3: CSD of Latent Dynamics

**What:** Compute cross-spectral density from latent trajectory timeseries for spectral
DCM fitting.

**Key difference from BOLD CSD:** MEG latent dynamics are at 250 Hz (not 0.5 Hz fMRI),
so the frequency range is 1-45 Hz (not 1/128 to 0.25 Hz). The existing
`compute_empirical_csd` function handles this via the `fs` and `freqs` parameters.

```python
# latent_ts shape: (T, N_latent) at 250 Hz
freqs = np.linspace(1.0, 45.0, 64)  # 1-45 Hz, 64 bins
csd = compute_empirical_csd(latent_ts, fs=250.0, freqs=freqs)
# csd shape: (64, N_latent, N_latent) complex128
```

**Critical adaptation for spectral DCM:** The existing `spectral_dcm_forward` uses
`default_frequency_grid(TR=2.0)` which gives fMRI frequencies (1/128 to 0.25 Hz). For
MEG latent dynamics, pass `freqs` directly (1-45 Hz). The transfer function math is
identical; only the frequency grid changes. The eigenvalue stabilization clamp
(`max(-1/32)`) may need adjustment for electrophysiology-scale dynamics.

### Pattern 4: Spectral DCM on Latent CSD vs Raw CSD

**What:** Fit the same `spectral_dcm_model` to both raw ROI CSD and latent dynamics
CSD, then compare posterior A matrices.

```python
# Path A: Direct fit to raw ROI CSD
raw_csd = epochs_to_csd(epochs, picks=roi_names, fmin=1.0, fmax=45.0)
# Fit spectral DCM
result_raw = run_svi(spectral_dcm_model, guide,
    model_args=(raw_csd["csd"], raw_csd["freqs"], a_mask))

# Path B: Fit to latent dynamics CSD
latent_csd = bold_to_csd_torch(latent_ts, fs=250.0, freqs_torch)
result_latent = run_svi(spectral_dcm_model, guide,
    model_args=(latent_csd, freqs_torch, a_mask_latent))

# Compare: posterior A matrices, ELBO, connectivity patterns
```

### Anti-Patterns to Avoid

- **Training the LSTM-AE on sensor-space data.** Source reconstruction MUST happen
  before the autoencoder. Sensor-space mixing makes latent dynamics uninterpretable.
- **Using the full 52-region parcellation for DCM.** 52x52 A matrix = 2704 free
  parameters; DCM cannot identify this many. Select 4-8 task-relevant ROIs.
- **Using fMRI frequency conventions for MEG CSD.** MEG operates at 1-100+ Hz, not
  0.008-0.25 Hz. The `default_frequency_grid` is for fMRI only.
- **Expecting the eigenvalue clamp (-1/32) to work for MEG.** Electrophysiology
  dynamics are faster; eigenvalue stability threshold needs recalibration for
  higher frequencies (candidate: -1 Hz instead of -1/32 Hz).
- **Skipping the comparison to direct DCM on raw data.** Without Path A (baseline),
  the autoencoder-DCM results have no reference for validation.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Source reconstruction from MEG | Custom beamformer | `mne.beamformer.make_lcmv` + `apply_lcmv` | Validated, handles forward model, noise covariance, regularization |
| ROI extraction from source estimate | Manual atlas masking | `mne.extract_label_time_course` | Handles sign flipping, vertex averaging, label interpolation |
| CSD from timeseries | Custom FFT + averaging | `compute_empirical_csd` (existing) or `mne.time_frequency.csd_multitaper` | Welch/multitaper methods handle windowing, tapering, frequency interpolation |
| MEG preprocessing | Manual artifact rejection | MNE ICA + autoreject or Cam-CAN preprocessed data | Cam-CAN already applied tSSS, MaxFilter, notch; only need bandpass + ICA |
| LSTM autoencoder | Custom temporal model from scratch | `torch.nn.LSTM` + `torch.nn.Linear` | Standard PyTorch LSTM is optimized, cuDNN-accelerated, well-tested |
| Spectral DCM fitting | New Pyro model | `spectral_dcm_model` (existing) | Already validated in Phases 4-8 |
| CSD likelihood decomposition | New complex-to-real handling | `decompose_csd_for_likelihood` (existing) | Already handles complex128 -> float64 for Pyro |

**Key insight:** Most of the heavy lifting is already done by existing Pyro-DCM
infrastructure. Phase 22's main novelties are: (1) real data preprocessing, (2) the
LSTM-AE, and (3) the scientific comparison pipeline. The DCM math is unchanged.

## Common Pitfalls

### Pitfall 1: Cam-CAN Data Access Latency

**What goes wrong:** Cam-CAN requires registration and DUA agreement. Data download
is ~100 GB for MEG + MRI. Processing takes days.
**Why it happens:** Academic data sharing agreements have manual review steps.
**How to avoid:** Register early. Start with a small subset (10-20 subjects). Use
FreeSurfer `fsaverage` template for source reconstruction instead of per-subject
anatomy (faster, sufficient for proof-of-concept).
**Warning signs:** DUA not approved after 2 weeks; data download incomplete.

### Pitfall 2: Source Reconstruction Requires Structural MRI

**What goes wrong:** LCMV beamformer needs a forward model, which needs BEM surfaces
from structural MRI.
**Why it happens:** Head model is subject-specific (shape of skull/brain matters).
**How to avoid:** Use `fsaverage` template anatomy with Cam-CAN's digitized headshape
for coregistration. MEG-GPT used this approach (8mm isotropic grid on template).
Alternatively, use Cam-CAN's T1 MRI (available alongside MEG).
**Warning signs:** Missing trans file, BEM model errors, no FreeSurfer recon.

### Pitfall 3: Frequency Grid Mismatch Between fMRI and MEG DCM

**What goes wrong:** Using `default_frequency_grid(TR=2.0)` gives 0.008-0.25 Hz,
which is meaningless for MEG data at 250 Hz.
**Why it happens:** The existing spectral DCM was built for fMRI.
**How to avoid:** Create a `default_frequency_grid_meg(sfreq=250.0)` that returns
1-45 Hz (or similar). Pass this explicitly to all spectral DCM functions.
**Warning signs:** CSD looks flat or near-zero; spectral DCM fails to converge.

### Pitfall 4: Eigenvalue Stabilization Threshold

**What goes wrong:** The SPM12 eigenvalue clamp `max(Re(lambda), -1/32)` was
calibrated for fMRI BOLD dynamics (~0.008-0.25 Hz). For MEG (1-100 Hz), neural
dynamics are orders of magnitude faster.
**Why it happens:** Faster dynamics require larger (more negative) eigenvalues.
Clamping at -1/32 would prevent the model from expressing any dynamics above ~0.03 Hz.
**How to avoid:** Parameterize the eigenvalue clamp threshold. For MEG source-space
data, use a threshold of -1.0 or remove the clamp entirely (the `parameterize_A`
negative-diagonal constraint already ensures stability).
**Warning signs:** Transfer function H(w) looks wrong; all regions show identical
flat spectra; posterior A has all eigenvalues pinned at -1/32.

### Pitfall 5: Too Many ROIs for DCM

**What goes wrong:** Fitting spectral DCM to 52 ROIs gives a 52x52 A matrix with
2704 free parameters. SVI cannot identify this many parameters from one subject's
CSD data.
**Why it happens:** MEG-GPT uses 52 parcels for reconstruction, but DCM is designed
for small networks (3-8 regions).
**How to avoid:** Select 4-8 task-relevant ROIs a priori (auditory, motor, visual
cortex). Use anatomical/functional knowledge to define the network. Report ROI
selection criteria explicitly.
**Warning signs:** SVI non-convergence; posterior A has many elements near prior mean;
identifiability shrinkage > 0.9.

### Pitfall 6: Latent Dimension Mismatch

**What goes wrong:** If LSTM-AE latent dim does not match DCM region count, the
relationship between latent dynamics and neural ROIs is unclear.
**Why it happens:** The autoencoder learns data-driven dimensions, not anatomically
defined ones.
**How to avoid:** Two strategies: (a) set N_latent = N_roi (autoencoder learns a
denoised version of each ROI), or (b) use arbitrary N_latent and interpret as abstract
latent circuit. Strategy (a) is more interpretable for DCM validation.
**Warning signs:** Latent dimensions do not correspond to any identifiable brain
region; connectivity interpretation becomes circular.

### Pitfall 7: Training Compute Requirements

**What goes wrong:** Training LSTM-AE on 600+ subjects of 8-minute MEG at 250 Hz is
a large dataset. GPU OOM or multi-day training.
**Why it happens:** 600 subjects x 250 Hz x 500s x 52 ROIs = ~4.7 billion samples.
**How to avoid:** Start with 50-100 subjects. Use windowed segments (e.g., 2s windows
with 50% overlap). LSTM-AE with hidden=64, latent=8 should train in 2-4 hours on a
single V100/A100 on M3 cluster. Batch size ~64 windows.
**Warning signs:** OOM on GPU; training loss plateaus early; >24h training time.

### Pitfall 8: Spectral DCM Prior Scale for MEG

**What goes wrong:** The existing `spectral_dcm_model` uses `prior_std = (1/64)**0.5`
for A_free, calibrated for fMRI BOLD connectivity. MEG effective connectivity operates
at different scales.
**Why it happens:** fMRI A matrix values represent slow hemodynamic coupling; MEG A
values represent faster neural coupling.
**How to avoid:** Recalibrate priors for MEG. Start with `LC_A_PRIOR_VARIANCE = 1/16`
(same as latent circuit DCM) and sweep. The noise model parameters (a, b, c) also need
recalibration for electrophysiology spectral shapes (1/f^alpha rather than fMRI noise).
**Warning signs:** Posterior A values pinned near zero (prior too tight) or wildly
large (prior too loose).

### Pitfall 9: Cam-CAN Sensorimotor Task Event Structure

**What goes wrong:** The sensorimotor task has complex event structure (simultaneous
auditory + visual, variable ISI, multiple tone frequencies). Incorrect epoching loses
task structure.
**Why it happens:** The task is designed for aging studies, not simple on/off blocks.
**How to avoid:** Use MNE `find_events` on STI101 trigger channel. Epoch around
stimulus onset. Separate auditory-only, visual-only, and combined conditions. For
bilinear DCM, the stimulus condition is the modulator.
**Warning signs:** Events not found; wrong trigger channel; missing conditions.

## Code Examples

### Example 1: Loading Cam-CAN MEG and Extracting ROI Timeseries
```python
# Source: MNE-Python docs (mne.tools/stable)
import mne

# Load preprocessed Cam-CAN data
raw = mne.io.read_raw_fif(
    "sub-CC110033/meg/sub-CC110033_task-smt_meg.fif",
    preload=True,
)
raw.filter(1.0, 45.0)

# Events from trigger channel
events = mne.find_events(raw, stim_channel="STI101")
epochs = mne.Epochs(raw, events, tmin=-0.2, tmax=0.8, preload=True)

# Source reconstruction with template anatomy
src = mne.setup_source_space("fsaverage", spacing="oct6")
fwd = mne.make_forward_solution(epochs.info, trans="fsaverage", src=src, bem=bem)
data_cov = mne.compute_covariance(epochs, tmin=0.0, tmax=0.8)
noise_cov = mne.compute_covariance(epochs, tmin=-0.2, tmax=0.0)
filters = mne.beamformer.make_lcmv(
    epochs.info, fwd, data_cov, noise_cov=noise_cov,
    reg=0.05, weight_norm="unit-noise-gain", pick_ori="max-power",
)
stcs = mne.beamformer.apply_lcmv_epochs(epochs, filters)

# Extract ROI timeseries
labels = mne.read_labels_from_annot("fsaverage", "aparc")
roi_ts = mne.extract_label_time_course(stcs, labels, src, mode="mean_flip")
# roi_ts: list of (N_roi, T) arrays, one per epoch
```

### Example 2: Computing CSD from Latent Dynamics
```python
# Source: existing csd_computation.py (adapted for MEG frequencies)
import numpy as np
from pyro_dcm.forward_models.csd_computation import compute_empirical_csd

# latent_ts: (T, N_latent) at 250 Hz sampling rate
freqs = np.linspace(1.0, 45.0, 64)  # MEG frequency range
csd = compute_empirical_csd(latent_ts, fs=250.0, freqs=freqs)
# csd: (64, N_latent, N_latent) complex128
```

### Example 3: Fitting Spectral DCM with MEG-Adapted Frequency Grid
```python
# Source: existing spectral_dcm_model.py (adapted for MEG)
import torch
from pyro_dcm.models import spectral_dcm_model, create_guide, run_svi

# Convert CSD to torch
csd_torch = torch.tensor(csd, dtype=torch.complex128)
freqs_torch = torch.tensor(freqs, dtype=torch.float64)
a_mask = torch.ones(N_latent, N_latent, dtype=torch.float64)

# NOTE: spectral_dcm_model needs adaptation for MEG prior scales
# and eigenvalue clamp threshold
guide = create_guide(spectral_dcm_model, init_scale=0.01)
result = run_svi(
    spectral_dcm_model, guide,
    model_args=(csd_torch, freqs_torch, a_mask),
    num_steps=2000, lr=0.01,
)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| DCM only on raw neuroimaging | DCM on model latent dynamics (this work) | Novel (2026) | Enables interpretability of deep learning models via established neuroscience framework |
| fMRI-only spectral DCM | MEG source-space spectral DCM | Established (SPM12 DCM for ERP) | Different frequency range (1-100 Hz), different noise model, different dynamics timescale |
| Manual feature extraction from neural data | Learned representations via autoencoders | Mature (LFADS 2018, osl-dynamics 2024) | Autoencoder captures nonlinear temporal structure |
| HMM for MEG state dynamics (osl-dynamics) | DCM on learned latent dynamics | Novel combination (2026) | DCM provides directed connectivity; HMM only provides undirected covariance patterns |
| Transformer/foundation models (MEG-GPT) | Start simple with LSTM-AE | Strategic | Foundation models are TensorFlow/osl-dynamics ecosystem; LSTM-AE is native PyTorch |

**Deprecated/outdated:**
- SPM12 fMRI-only frequency grid for spectral DCM when applied to MEG data
- Eigenvalue clamp at -1/32 Hz for electrophysiology applications

## Open Questions

### 1. Eigenvalue Clamp Threshold for MEG Spectral DCM

- **What we know:** The existing `compute_transfer_function` clamps eigenvalue real
  parts to `max(-1/32)`, following SPM12 convention for fMRI. This is appropriate for
  BOLD dynamics (0.008-0.25 Hz) but too aggressive for MEG (1-100 Hz).
- **What's unclear:** What is the correct clamp for MEG? SPM12's DCM for ERP uses
  neural mass models with different dynamics, not the simple linear transfer function.
  The correct threshold depends on the expected neural time constants.
- **Recommendation:** Make the clamp threshold a parameter (default -1/32 for backward
  compat). For MEG, try -1.0 Hz or no clamp. Validate by checking that predicted CSD
  matches observed CSD spectral shape.

### 2. Noise Model for MEG Spectral DCM

- **What we know:** The existing `spectral_noise.py` implements `1/f^alpha` neuronal
  noise and observation noise parameterized for fMRI. MEG has different noise
  characteristics (higher 1/f slope, line noise, muscle artifacts).
- **What's unclear:** Whether the same parametric noise model works for MEG latent
  dynamics or needs modification.
- **Recommendation:** Start with the existing noise model (same parametric form,
  different prior scales). If CSD fit is poor, consider adding a narrowband noise
  component for alpha/beta peaks.

### 3. How Many Subjects Are Needed

- **What we know:** MEG-GPT trained on 612 subjects (resting state). Our task
  (sensorimotor) may have fewer valid recordings after preprocessing rejection.
- **What's unclear:** How many subjects are needed to train an adequate LSTM-AE for
  this specific task. Prior work on similar architectures suggests 50-100 is sufficient
  for a simple autoencoder.
- **Recommendation:** Start with 50 subjects for proof-of-concept. Scale up if
  reconstruction quality is insufficient.

### 4. Latent Dimension Selection Strategy

- **What we know:** Phase 21 used PCA with variance-explained diagnostic and
  output-R-squared gate for dimension selection. For LSTM-AE, the bottleneck size
  is a hyperparameter, not data-driven.
- **What's unclear:** Whether to match latent dimensions to ROI count (interpretable)
  or use fewer dimensions (more compressed, potentially better denoising).
- **Recommendation:** Primary analysis: N_latent = N_roi (e.g., 6 latent dims for 6
  ROIs). Secondary sweep: N_latent in {2, 4, 6, 8, 12}. Compare CSD reconstruction
  quality and DCM fit quality across settings.

### 5. Cam-CAN DUA Timeline

- **What we know:** Cam-CAN requires registration and Data Usage Agreement via
  camcan-archive.mrc-cbu.cam.ac.uk.
- **What's unclear:** How long the review takes (could be days to weeks).
- **Recommendation:** Register immediately during planning phase. In parallel, develop
  and test the entire pipeline on synthetic MEG data (MNE simulation tools). Switch to
  real data when access is granted.

## Sources

### Primary (HIGH confidence)

- MNE-Python documentation (mne.tools/stable) -- LCMV beamformer tutorial, extract_label_time_course API, CSD computation
- Existing Pyro-DCM codebase -- spectral_dcm_model.py, csd_computation.py, mne_loader.py, latent_circuit_dcm_model.py
- Cam-CAN data repository paper (Taylor et al., 2017, NeuroImage) -- Dataset description, task paradigm details

### Secondary (MEDIUM confidence)

- MEG-GPT (arXiv:2510.18080, Oct 2025) -- LCMV beamformer + 52-region parcellation pipeline on Cam-CAN, tokenizer architecture
- osl-dynamics (eLife 2024) -- DyNeMo architecture, LSTM-based temporal dynamics modeling for MEG
- LFADS (Pandarinath et al., 2018, Nature Methods) -- Sequential VAE for latent neural dynamics
- lfads-torch (arXiv:2309.01230) -- PyTorch implementation of LFADS architecture
- Novelli, Friston & Razi (2024, Network Neuroscience) -- Spectral DCM didactic introduction

### Tertiary (LOW confidence)

- WebSearch results on LSTM autoencoder architectures for timeseries -- general patterns, no MEG-specific validation
- WebSearch results on Cam-CAN sensorimotor connectivity -- functional connectivity studies exist but not DCM on model latents specifically
- osl-foundation GitHub (TensorFlow MEG-GPT code) -- confirmed TensorFlow dependency, not directly usable in PyTorch project

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries are already in the project or well-established
- Architecture (LSTM-AE): MEDIUM -- standard architecture but not yet validated on MEG ROI timeseries specifically in this codebase
- Source reconstruction pipeline: HIGH -- MNE-Python is the gold standard; well-documented
- Spectral DCM adaptation for MEG: MEDIUM -- math is correct but prior/clamp recalibration is untested
- Cam-CAN data access: MEDIUM -- known to be open but DUA timeline uncertain
- Scientific validation (connectivity patterns): MEDIUM -- sensorimotor connectivity is well-studied but specific DCM on latent dynamics is novel
- Pitfalls: HIGH -- based on direct codebase inspection and domain knowledge

**Research date:** 2026-05-26
**Valid until:** 2026-06-26 (30 days; stable domain, no fast-moving dependencies)
