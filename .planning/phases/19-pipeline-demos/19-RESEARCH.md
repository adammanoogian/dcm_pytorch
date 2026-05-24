# Phase 19: End-to-End Pipeline Demos - Research

**Researched:** 2026-05-24
**Domain:** MNE-Python IO integration with Pyro-DCM model fitting (intra-codebase)
**Confidence:** HIGH — all findings come from reading the actual codebase; no external libraries to research

## Summary

Phase 19 requires two self-contained demo scripts that bridge the MNE IO layer
(Phase 18) to the Pyro-DCM fitting infrastructure (Phases 4–16). All the building
blocks exist and work correctly: `epochs_to_csd`, `epochs_to_timeseries`,
`spectral_dcm_model`, `task_dcm_model`, `create_guide`, `run_svi`, and
`extract_posterior_params` are all implemented, tested, and exported from
`pyro_dcm`. The primary work is integration plumbing: constructing synthetic MNE
`EpochsArray` objects, calling the loaders to get tensors, building the correct
model args tuple, and calling the inference stack.

There is a canonical style reference: `scripts/demo_bilinear_consumer.py`
shows exactly the expected structure for a pipeline demo in this codebase. The
tests in `tests/test_mne_loader.py` show the exact fixtures pattern for creating
synthetic MNE data.

**Primary recommendation:** Copy the structural skeleton from
`scripts/demo_bilinear_consumer.py` for each demo. Use the fixture patterns
from `tests/test_mne_loader.py` to create synthetic MNE data. Wire loader output
directly into model args without intermediate transformation.

## Standard Stack

### Core (already available — no new dependencies)

| Symbol | Module | Purpose |
|--------|--------|---------|
| `epochs_to_csd` | `pyro_dcm.io` | Converts `mne.EpochsArray` → `dict['csd', 'freqs', ...]` |
| `epochs_to_timeseries` | `pyro_dcm.io` | Converts `mne.EpochsArray` → `dict['timeseries', 'times', ...]` |
| `spectral_dcm_model` | `pyro_dcm` | Pyro model for spectral DCM (CSD-based) |
| `task_dcm_model` | `pyro_dcm` | Pyro model for task DCM (BOLD-based) |
| `create_guide` | `pyro_dcm` | Guide factory (AutoNormal default) |
| `run_svi` | `pyro_dcm` | SVI runner with ClippedAdam, NaN guard, LR decay |
| `extract_posterior_params` | `pyro_dcm` | Posterior sampling via Predictive |
| `simulate_spectral_dcm` | `pyro_dcm` | Generates synthetic CSD from known A |
| `simulate_task_dcm` | `pyro_dcm` | Generates synthetic BOLD from known A, C |
| `make_stable_A_spectral` | `pyro_dcm` | Stable A matrix generator for spectral DCM |
| `make_random_stable_A` | `pyro_dcm` | Stable A matrix generator for task DCM |
| `make_block_stimulus` | `pyro_dcm` | Block-design stimulus dictionary |
| `parameterize_A` | `pyro_dcm` | A_free → A with stable diagonal transform |
| `PiecewiseConstantInput` | `pyro_dcm` | Wraps stimulus dict for ODE integrator |

### Optional dependency: MNE

MNE is an optional dependency: `pip install pyro-dcm[mne]`. Demo scripts MUST
guard with `try/except ImportError` or clear top-of-file `import mne` with a
user-facing error message if missing.

## Architecture Patterns

### Demo Script Structure (from `scripts/demo_bilinear_consumer.py`)

```
scripts/
├── demo_bilinear_consumer.py   # existing style reference
├── demo_spectral_dcm.py        # PIPE-01 (to create)
└── demo_task_dcm.py            # PIPE-02 (to create)
```

Each demo follows this pattern:
1. Module docstring stating demo purpose, expected runtime, what it produces
2. Imports: `from __future__ import annotations`, then stdlib/third-party/pyro_dcm
3. `main()` function with numbered sections using `# --- N. Section name ---` comments
4. `if __name__ == "__main__": main()` guard

### Pattern 1: Spectral DCM Demo (PIPE-01)

**MNE data creation** (from `tests/test_mne_loader.py` fixture pattern):
```python
import numpy as np
import mne

rng = np.random.default_rng(42)
sfreq = 256.0
n_epochs, n_channels, n_times = 20, 3, int(sfreq * 2.0)  # 2s epochs
info = mne.create_info(ch_names=["EEG1","EEG2","EEG3"], sfreq=sfreq, ch_types="eeg")
data = rng.standard_normal((n_epochs, n_channels, n_times)) * 1e-6
epochs = mne.EpochsArray(data, info)
```

**MNE -> CSD tensor**:
```python
from pyro_dcm.io import epochs_to_csd

result = epochs_to_csd(epochs, fmin=1.0, fmax=50.0, n_freqs=32)
observed_csd = result["csd"]    # (F, N, N) complex128
freqs = result["freqs"]          # (F,) float64
N = observed_csd.shape[1]
```

**Model args for `spectral_dcm_model`** (exact positional order):
```python
a_mask = torch.ones(N, N, dtype=torch.float64)
model_args = (observed_csd, freqs, a_mask)
# Optional: model_args = (observed_csd, freqs, a_mask, N)
```
Note: `spectral_dcm_model` signature is `(observed_csd, freqs, a_mask, N=None)`.

**SVI call**:
```python
guide = create_guide(spectral_dcm_model, init_scale=0.01)
svi_result = run_svi(
    spectral_dcm_model, guide, model_args,
    num_steps=500, lr=0.01,
    clip_norm=10.0, lr_decay_factor=0.1,
)
```

**Posterior extraction**:
```python
posterior = extract_posterior_params(guide, model_args)
A_free_mean = posterior["A_free"]["mean"]     # (N, N)
A_inferred = parameterize_A(A_free_mean)      # (N, N)
# Also available: posterior["A"]["mean"] (deterministic site)
```

### Pattern 2: Task DCM Demo (PIPE-02)

**MNE data creation** (same fixture pattern, slightly different dims):
```python
sfreq = 250.0
n_epochs, n_channels, n_times = 10, 3, int(sfreq * 4.0)  # 4s epochs
info = mne.create_info(ch_names=["ROI1","ROI2","ROI3"], sfreq=sfreq, ch_types="eeg")
data = rng.standard_normal((n_epochs, n_channels, n_times)) * 1e-6
epochs = mne.EpochsArray(data, info)
```

**MNE -> timeseries tensor**:
```python
from pyro_dcm.io import epochs_to_timeseries

result = epochs_to_timeseries(epochs, average=True)
observed_bold = result["timeseries"]   # (T, N) float64
times = result["times"]                 # (T,) float64
N = observed_bold.shape[1]
```

**Construct stimulus and t_eval** (required by task_dcm_model):
```python
from pyro_dcm import make_block_stimulus, PiecewiseConstantInput

stimulus = make_block_stimulus(n_blocks=3, block_duration=15.0, rest_duration=15.0, n_inputs=1)
# stimulus is a dict with 'times' and 'values' keys

TR = 2.0     # must match observed_bold time resolution
dt = 0.5     # ODE step size
T_total = float(times[-1])  # total duration
t_eval = torch.arange(0, T_total, dt, dtype=torch.float64)
```

**Important note on task DCM model_args ordering** (strict positional):
```python
a_mask = torch.ones(N, N, dtype=torch.float64)
c_mask = torch.zeros(N, 1, dtype=torch.float64)
c_mask[0, 0] = 1.0
model_args = (observed_bold, stimulus, a_mask, c_mask, t_eval, TR, dt)
```
`task_dcm_model` signature: `(observed_bold, stimulus, a_mask, c_mask, t_eval, TR, dt=0.5, *, b_masks=None, stim_mod=None)`.

**SVI call**:
```python
guide = create_guide(task_dcm_model, init_scale=0.01, n_regions=N)
svi_result = run_svi(
    task_dcm_model, guide, model_args,
    num_steps=500, lr=0.005,
    clip_norm=10.0, lr_decay_factor=0.01,
)
```

**Posterior extraction**:
```python
posterior = extract_posterior_params(guide, model_args)
A_free_mean = posterior["A_free"]["mean"]   # (N, N)
A_inferred = parameterize_A(A_free_mean)    # (N, N)
C_mean = posterior["C"]["mean"]             # (N, M)
```

### Anti-Patterns to Avoid

- **Don't pass `model_kwargs={}` as positional arg** — `model_kwargs` is a keyword arg to `run_svi`; only needed for bilinear branch (`b_masks`, `stim_mod`). Linear task DCM needs none.
- **Don't import `mne` at top level unconditionally** — wrap in `try/except ImportError` with user message and `sys.exit(1)`.
- **Don't use `dt=0.01` for SVI** — the benchmark runners use `dt_model=0.5` for SVI (vs `dt=0.01` for simulation). Using fine dt in the model blows up runtime.
- **Don't use `average=False` for task DCM** — `observed_bold` must be `(T, N)` not `(n_epochs, T, N)`.
- **Don't use `observed_csd.real` or cast to float** — `spectral_dcm_model` handles the complex→real decomposition internally via `decompose_csd_for_likelihood`.
- **Don't call `pyro.clear_param_store()` manually** — `run_svi` already calls it at the start.
- **Don't construct `t_eval` from `times`** — `times` from `epochs_to_timeseries` is the TR-resolution grid. `t_eval` for the ODE must be at `dt` resolution, not TR resolution. Derive from total duration and `dt`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| CSD computation | custom FFT/welch | `epochs_to_csd` | Handles MNE CSD object, interpolation, Hermitian property |
| Complex→real conversion | manual `.real`, `.imag` | `spectral_dcm_model` internal | Already done inside model via `decompose_csd_for_likelihood` |
| SVI loop | manual `svi.step` loop | `run_svi` | Handles NaN guard, LR decay, ClippedAdam, param store clear |
| Posterior stats | manual guide `.median()` | `extract_posterior_params` | Provides mean/std/samples, handles AutoDelta edge case |
| Stable A matrix | manual construction | `make_stable_A_spectral` / `make_random_stable_A` | Guaranteed eigenvalue stability |

**Key insight:** All inference boilerplate is encapsulated. The demos need zero
custom Pyro code beyond the calls to `create_guide`, `run_svi`, and
`extract_posterior_params`.

## Common Pitfalls

### Pitfall 1: t_eval dimension mismatch in task DCM
**What goes wrong:** `t_eval` and `TR` are inconsistent — e.g., using
`t_eval` from `result["times"]` (TR-resolution) instead of a fine ODE grid.
**Why it happens:** `epochs_to_timeseries` returns `times` at observation
frequency; `task_dcm_model` needs `t_eval` at ODE step `dt`.
**How to avoid:** Always construct `t_eval = torch.arange(0, duration, dt, dtype=torch.float64)` independently from the observed data times.
**Warning signs:** `predicted_bold` shape mismatch error or silent NaN.

### Pitfall 2: MNE `picks=None` includes all channels including bads
**What goes wrong:** `epochs_to_timeseries(epochs, picks=None)` passes all
channels including bad ones (documented in Phase 18 TEST-06).
**How to avoid:** Demo scripts should set explicit picks or note this in comments.

### Pitfall 3: CSD `n_freqs` must match what `spectral_dcm_model` expects
**What goes wrong:** Spectral DCM forward model computes its own frequency grid
internally. The observed CSD `freqs` vector is passed directly to `spectral_dcm_model`
which uses it for the transfer function. No shape mismatch if `freqs` from the
loader is passed as-is.
**How to avoid:** Pass `result["freqs"]` unchanged as the `freqs` model arg.

### Pitfall 4: Using real-world `sfreq` without matching `fmax`
**What goes wrong:** `epochs_to_csd` with `fmax=None` defaults to Nyquist (`sfreq/2`).
For `sfreq=256`, that's 128 Hz — spectral DCM is designed for `fmax ~ 0.25 Hz` (fMRI).
EEG/MEG demos need explicit `fmax` matching the signal bandwidth of interest.
**How to avoid:** Always pass explicit `fmin` and `fmax` to `epochs_to_csd`.
For demonstration purposes use `fmin=1.0, fmax=50.0`.

### Pitfall 5: `run_svi` raises RuntimeError on NaN ELBO
**What goes wrong:** Early SVI steps with poorly-initialized guide can NaN.
**Why it happens:** A_free drawn from N(0, 1/64) is fine, but the guide's
initial uncertainty before any updates can be large.
**How to avoid:** Use `init_scale=0.01` (not the default `0.01` but confirm
it matches what benchmarks use). The task DCM model has a NaN-safe guard
that zero-fills BOLD before the likelihood, preventing most NaN ELBOs.

### Pitfall 6: Demo MNE data amplitude scale matters for CSD
**What goes wrong:** If synthetic data is not scaled to realistic units (e.g.
µV range, `* 1e-6`), CSD magnitudes will be orders of magnitude off.
**Why it matters:** CSD scale affects the HalfCauchy noise prior fit.
**How to avoid:** Follow the fixture pattern: `data * 1e-6`.

## Code Examples

### Minimal Spectral DCM Demo Skeleton
```python
# Source: inferred from benchmarks/runners/spectral_svi.py + tests/test_mne_loader.py

import mne
import numpy as np
import torch
from pyro_dcm import (
    create_guide, extract_posterior_params, parameterize_A, run_svi, spectral_dcm_model,
)
from pyro_dcm.io import epochs_to_csd

# 1. Synthetic MNE Epochs
rng = np.random.default_rng(42)
sfreq, n_epochs, N = 256.0, 20, 3
info = mne.create_info(ch_names=[f"ROI{i}" for i in range(N)], sfreq=sfreq, ch_types="eeg")
data = rng.standard_normal((n_epochs, N, int(sfreq * 2.0))) * 1e-6
epochs = mne.EpochsArray(data, info)

# 2. CSD extraction
csd_result = epochs_to_csd(epochs, fmin=1.0, fmax=50.0, n_freqs=32)
observed_csd = csd_result["csd"]    # (32, 3, 3) complex128
freqs = csd_result["freqs"]          # (32,) float64

# 3. Model args
a_mask = torch.ones(N, N, dtype=torch.float64)
model_args = (observed_csd, freqs, a_mask)

# 4. Guide + SVI
guide = create_guide(spectral_dcm_model, init_scale=0.01)
svi_result = run_svi(spectral_dcm_model, guide, model_args, num_steps=300, lr=0.01)

# 5. Posterior
posterior = extract_posterior_params(guide, model_args)
A_est = parameterize_A(posterior["A_free"]["mean"])   # (3, 3)
```

### Minimal Task DCM Demo Skeleton
```python
# Source: inferred from benchmarks/runners/task_svi.py + tests/test_mne_loader.py

import mne
import numpy as np
import torch
from pyro_dcm import (
    PiecewiseConstantInput, create_guide, extract_posterior_params,
    make_block_stimulus, parameterize_A, run_svi, task_dcm_model,
)
from pyro_dcm.io import epochs_to_timeseries

# 1. Synthetic MNE Epochs
rng = np.random.default_rng(42)
TR, sfreq = 2.0, 0.5   # TR-matched sfreq: 1/TR Hz
N, n_epochs = 3, 1
T_obs = 75  # 75 TRs → 150s
info = mne.create_info(ch_names=[f"ROI{i}" for i in range(N)], sfreq=1.0/TR, ch_types="misc")
data = rng.standard_normal((n_epochs, N, T_obs)) * 0.01
epochs = mne.EpochsArray(data, info)

# 2. Timeseries extraction
ts_result = epochs_to_timeseries(epochs, average=True)
observed_bold = ts_result["timeseries"]   # (T, N) float64
T_total = float(ts_result["times"][-1])

# 3. Stimulus + t_eval
stimulus = make_block_stimulus(n_blocks=3, block_duration=15.0, rest_duration=15.0, n_inputs=1)
dt = 0.5
t_eval = torch.arange(0, T_total + dt, dt, dtype=torch.float64)

# 4. Model args
a_mask = torch.ones(N, N, dtype=torch.float64)
c_mask = torch.zeros(N, 1, dtype=torch.float64); c_mask[0, 0] = 1.0
model_args = (observed_bold, stimulus, a_mask, c_mask, t_eval, TR, dt)

# 5. Guide + SVI
guide = create_guide(task_dcm_model, init_scale=0.01, n_regions=N)
svi_result = run_svi(task_dcm_model, guide, model_args, num_steps=300, lr=0.005)

# 6. Posterior
posterior = extract_posterior_params(guide, model_args)
A_est = parameterize_A(posterior["A_free"]["mean"])   # (3, 3)
C_est = posterior["C"]["mean"]                         # (3, 1)
```

## Open Questions

1. **Task DCM demo: what sfreq for EpochsArray?**
   - What we know: `epochs_to_timeseries` returns times at the epoch's `sfreq`.
     `task_dcm_model` expects `observed_bold` at TR resolution.
   - What's unclear: The simplest approach is to create epochs already at TR
     resolution (`sfreq = 1/TR = 0.5 Hz`). This avoids needing a resampling step.
     Alternatively, create high-sfreq epochs and downsample before calling
     `epochs_to_timeseries`. The benchmark runners do not use MNE at all — they
     simulate at `dt=0.01` and the model runs at `dt=0.5`, so there's no established
     pattern for creating epochs at TR resolution from the codebase.
   - Recommendation: Create `EpochsArray` at `sfreq = 1/TR` directly. This is the
     simplest approach and avoids adding a resampling step to the demo. Document
     this choice clearly in the demo.

2. **Should demo scripts show a recovery quality metric?**
   - What we know: `demo_bilinear_consumer.py` prints A-RMSE and B-RMSE against
     known ground truth (because it uses simulated data with known A_true).
   - What's unclear: PIPE-01 and PIPE-02 demo scripts start from MNE Epochs (no
     known ground truth). They could either (a) simulate A_true first then create
     MNE Epochs from that, or (b) just fit and show posterior.
   - Recommendation: Follow `demo_bilinear_consumer.py` pattern — simulate with
     known ground truth, wrap in MNE Epochs, fit, report recovery quality. This
     makes the demo validate itself end-to-end.

3. **Spectral DCM: simulate CSD via `simulate_spectral_dcm` first or use raw MNE CSD?**
   - What we know: `epochs_to_csd` computes CSD from time-domain EpochsArray data.
     The spectral DCM model expects CSD from fMRI (very low frequencies, ~1/128–0.25 Hz).
     Random EEG-like data will produce noise CSD, not meaningful connectivity structure.
   - Recommendation: For a meaningful demo, simulate a CSD directly via
     `simulate_spectral_dcm(A_true)`, then convert the CSD into synthetic epoch data
     OR just pass the simulated CSD tensor directly to the model (bypassing MNE entirely
     for the CSD step). If the PIPE-01 requirement strictly requires going through
     `epochs_to_csd`, then the synthetic EpochsArray needs to have fMRI-like long
     duration and low sampling rate.
   - This is the most significant open question for PIPE-01.

## Sources

### Primary (HIGH confidence — direct code inspection)
- `src/pyro_dcm/io/mne_loader.py` — all four loader APIs, exact signatures and return dicts
- `src/pyro_dcm/models/spectral_dcm_model.py` — `spectral_dcm_model` signature and model_args contract
- `src/pyro_dcm/models/task_dcm_model.py` — `task_dcm_model` signature and model_args contract
- `src/pyro_dcm/models/guides.py` — `create_guide`, `run_svi`, `extract_posterior_params` full signatures
- `scripts/demo_bilinear_consumer.py` — canonical demo structure pattern
- `tests/test_mne_loader.py` — synthetic MNE fixture construction patterns
- `benchmarks/runners/spectral_svi.py` — exact SVI call pattern for spectral DCM
- `benchmarks/runners/task_svi.py` — exact SVI call pattern for task DCM
- `src/pyro_dcm/__init__.py` — full public API surface

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — reading pyro_dcm public API directly
- Architecture patterns: HIGH — direct code inspection of demo_bilinear_consumer.py and benchmark runners
- Pitfalls: HIGH (for items 1–4) — confirmed in test file comments and model code; MEDIUM (item 5–6) — inferred from model design notes

**Research date:** 2026-05-24
**Valid until:** indefinite — this is intra-codebase research
