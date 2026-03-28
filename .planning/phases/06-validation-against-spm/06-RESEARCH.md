# Phase 6: Validation Against SPM / Reference Implementations - Research

**Researched:** 2026-03-28
**Domain:** Cross-validation of DCM inference against SPM12 (Variational Laplace), tapas/rDCM, and Julia rDCM
**Confidence:** HIGH (SPM12 source verified locally), MEDIUM (tapas API from GitHub), LOW (Julia rDCM -- not installed)

## Summary

This research covers the complete technical domain for Phase 6: cross-validating our Pyro-based DCM implementation against established reference implementations. Three reference implementations are targeted: (1) SPM12 for task-based DCM (`spm_dcm_estimate`) and spectral DCM (`spm_dcm_fmri_csd`), (2) tapas/rDCM MATLAB toolbox (`tapas_rdcm_estimate`) for regression DCM, and (3) Julia `RegressionDynamicCausalModeling.jl` for regression DCM.

The primary technical challenge is constructing the correct MATLAB DCM struct format from Python-generated synthetic data, exporting via `scipy.io.savemat`, running SPM12/tapas estimation via `matlab -batch`, and loading results back into Python for element-wise comparison. SPM12 is installed locally at `C:/Users/aman0087/Documents/Github/spm12/` with MATLAB R2022a available at `C:/Program Files/MATLAB/R2022a/bin/matlab`. tapas is NOT currently installed but can be cloned from GitHub. Julia is NOT installed on the system.

A critical finding is that SPM12's Variational Laplace (VL) and our Pyro SVI use fundamentally different inference algorithms. VL uses a deterministic Gauss-Newton optimization with Laplace (Gaussian) posterior approximation and analytical Hessian-based covariance, while our SVI uses stochastic gradient descent with mean-field Normal guides. Expected discrepancies include: (a) posterior means may differ by 5-15% due to different optimization landscapes and local optima, (b) posterior uncertainties will differ systematically (VL typically underestimates uncertainty due to the Laplace approximation; SVI's mean-field approximation ignores posterior correlations), and (c) free energy (F) and negative ELBO are both lower bounds on log model evidence but computed differently, so absolute values will differ while ranking should agree.

**Primary recommendation:** Use a "generate in Python, export to .mat, run in MATLAB, import results" pipeline. Build MATLAB batch scripts that construct the DCM struct from exported .mat data, call `spm_dcm_estimate`/`spm_dcm_fmri_csd`, and save results. Compare posterior means (Ep.A) element-wise with 10% relative error tolerance. Compare model ranking (not absolute F values) for ELBO validation. For tapas, clone the repository and follow the same export/import pattern. For Julia, pre-generate reference data as .jld2 files (or skip if Julia not available and document as open question).

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| scipy.io | 1.14.1 | `savemat`/`loadmat` for .mat file I/O | Only Python tool that creates MATLAB-compatible v5 .mat files with nested struct support |
| MATLAB R2022a | R2022a | Run SPM12 and tapas via `matlab -batch` | Installed locally, required for SPM12 |
| SPM12 | r7771+ | Reference DCM estimation (VL) | Gold standard for DCM, installed at `C:/Users/aman0087/Documents/Github/spm12/` |
| numpy | 1.24+ | Array conversion between torch tensors and .mat format | scipy.io requires numpy arrays |
| subprocess | stdlib | Call MATLAB from Python | `matlab -batch "script_name"` pattern |
| pytest | 8.0+ | Validation test framework | Existing project convention |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| tapas rDCM | latest | Reference rDCM estimation | Clone from `github.com/translationalneuromodeling/tapas`, add to MATLAB path |
| matplotlib | 3.8+ | Scatter plots of true vs inferred, discrepancy visualizations | Validation report figures |
| json | stdlib | Validation report metadata | Machine-readable validation results |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `matlab -batch` | MATLAB Engine API for Python | Engine API is more Pythonic but requires separate install and version matching; `-batch` is simpler and always works |
| Pre-generated Julia reference data | PyJulia/juliacall | Julia not installed; pre-generating .jld2 is simpler but requires one-time Julia setup |
| scipy.io.savemat v5 | hdf5storage (v7.3) | v5 format is sufficient for our data sizes and has broader MATLAB compatibility |

## Architecture Patterns

### Recommended Project Structure
```
validation/
    export_to_mat.py           # Python -> .mat conversion for all DCM variants
    matlab_scripts/
        run_spm_task_dcm.m     # MATLAB batch: load .mat, build DCM struct, call spm_dcm_estimate
        run_spm_spectral_dcm.m # MATLAB batch: load .mat, build DCM struct, call spm_dcm_fmri_csd
        run_tapas_rdcm.m       # MATLAB batch: load .mat, call tapas_rdcm_estimate
    compare_results.py         # Load SPM results, compare with Pyro results
    run_validation.py          # Orchestrator: generate data, export, call MATLAB, compare
    validation_report.py       # Generate summary report with figures
tests/
    test_validation_export.py  # Test .mat export round-trips correctly
    test_validation_pipeline.py # Integration test for full pipeline
```

### Pattern 1: Python-to-MATLAB DCM Struct Export

**What:** Convert Python simulation outputs to the exact DCM struct format that SPM12 expects.

**When to use:** Every cross-validation scenario.

**Key insight:** `scipy.io.savemat` saves Python dicts as MATLAB structs. Nested dicts become nested structs. All arrays must be numpy `float64`. Scalar values must be wrapped as `np.array([[value]])` (2D) to match MATLAB convention.

**Example -- Task DCM export:**
```python
# Source: Verified against spm_dcm_estimate.m header (local SPM12 install)
import numpy as np
import scipy.io

def export_task_dcm_for_spm(
    bold_data: np.ndarray,      # (T, N) BOLD time series
    stimulus: np.ndarray,       # (T_micro, M) stimulus at microtime resolution
    a_mask: np.ndarray,         # (N, N) binary connectivity mask
    c_mask: np.ndarray,         # (N, M) binary input mask
    TR: float,                  # repetition time in seconds
    u_dt: float,                # stimulus sampling interval in seconds
    output_path: str,
) -> None:
    """Export synthetic data as SPM12-compatible DCM .mat file."""
    N = bold_data.shape[1]
    M = c_mask.shape[1]
    v = bold_data.shape[0]  # number of scans

    DCM = {
        # Connectivity masks (binary: 1=present, 0=absent)
        'a': a_mask.astype(np.float64),
        'b': np.zeros((N, N, 0), dtype=np.float64),  # no modulatory
        'c': c_mask.astype(np.float64),
        'd': np.zeros((N, N, 0), dtype=np.float64),  # no nonlinear

        # Response data
        'Y': {
            'y': bold_data.astype(np.float64),
            'dt': np.array([[TR]]),
            'X0': np.ones((v, 1), dtype=np.float64),  # constant confound
            'name': np.array([[f'R{i+1}' for i in range(N)]], dtype=object),
        },

        # Input data
        'U': {
            'u': stimulus.astype(np.float64),
            'dt': np.array([[u_dt]]),
            'name': np.array([[f'stim{i+1}' for i in range(M)]], dtype=object),
        },

        # Dimensions
        'n': np.array([[N]]),
        'v': np.array([[v]]),

        # Timing
        'TE': np.array([[0.04]]),
        'delays': np.ones((1, N)) * TR / 2,

        # Options
        'options': {
            'nonlinear': np.array([[0]]),
            'two_state': np.array([[0]]),
            'stochastic': np.array([[0]]),
            'centre': np.array([[0]]),
            'induced': np.array([[0]]),
            'nograph': np.array([[1]]),
            'maxit': np.array([[128]]),
        },
    }
    scipy.io.savemat(output_path, {'DCM': DCM})
```

### Pattern 2: MATLAB Batch Script for SPM12 Estimation

**What:** MATLAB script that loads the exported .mat, runs SPM12 estimation, and saves results.

**When to use:** Called via `subprocess.run(['matlab', '-batch', 'run_spm_task_dcm'])`.

**Example -- Task DCM estimation:**
```matlab
% run_spm_task_dcm.m -- called via: matlab -batch "run_spm_task_dcm"
% Source: Verified against spm_dcm_estimate.m (local SPM12)

% Add SPM12 to path
addpath('C:/Users/aman0087/Documents/Github/spm12');
spm('defaults', 'FMRI');

% Load exported DCM struct
load('validation/data/task_dcm_input.mat', 'DCM');

% Ensure required fields exist
if ~isfield(DCM.Y, 'Q')
    DCM.Y.Q = spm_Ce(ones(1, DCM.n) * DCM.v);
end

% Run estimation (deterministic DCM, Variational Laplace)
DCM = spm_dcm_estimate(DCM);

% Save results
results.Ep_A = DCM.Ep.A;           % Posterior mean A matrix
results.Ep_C = DCM.Ep.C;           % Posterior mean C matrix
results.Cp = full(DCM.Cp);         % Full posterior covariance (sparse->full)
results.F = DCM.F;                 % Free energy (log evidence bound)
results.y_predicted = DCM.y;       % Predicted BOLD
results.R = DCM.R;                 % Residuals

save('validation/data/task_dcm_spm_results.mat', 'results');
fprintf('SPM12 task DCM estimation complete. F = %.4f\n', DCM.F);
```

### Pattern 3: Spectral DCM Export and Estimation

**What:** Export CSD-compatible data for `spm_dcm_fmri_csd`. Unlike task DCM, spectral DCM estimates CSD from the raw BOLD time series internally.

**Key insight from spm_dcm_fmri_csd.m (line 213):** SPM computes CSD from BOLD via MAR model internally (`spm_dcm_fmri_csd_data`). We should NOT export our pre-computed CSD. Instead, export BOLD time series and let SPM compute its own CSD. This ensures the comparison is apples-to-apples for the estimation algorithm.

**Example -- Spectral DCM estimation:**
```matlab
% run_spm_spectral_dcm.m
addpath('C:/Users/aman0087/Documents/Github/spm12');
spm('defaults', 'FMRI');

load('validation/data/spectral_dcm_input.mat', 'DCM');

% Force CSD analysis mode
DCM.options.induced = 1;
DCM.options.analysis = 'CSD';

if ~isfield(DCM.Y, 'Q')
    DCM.Y.Q = spm_Ce(ones(1, DCM.n) * DCM.v);
end

% Run spectral DCM estimation
DCM = spm_dcm_fmri_csd(DCM);

% Save results
results.Ep_A = DCM.Ep.A;
results.Ep_transit = DCM.Ep.transit;
results.Ep_decay = DCM.Ep.decay;
results.Cp = full(DCM.Cp);
results.F = DCM.F;
results.Hc = DCM.Hc;              % Predicted CSD
results.Hz = DCM.Hz;              % Frequency vector

save('validation/data/spectral_dcm_spm_results.mat', 'results');
fprintf('SPM12 spectral DCM estimation complete. F = %.4f\n', DCM.F);
```

### Pattern 4: tapas rDCM Estimation

**What:** Export data for tapas rDCM and run estimation.

**Key finding from tapas source (GitHub):** `tapas_rdcm_estimate(DCM, type, options, methods)` takes:
- `DCM` struct with `Y.y`, `Y.dt`, `U.u`, `U.dt`, `a`, `b`, `c`, `d`, `n`
- `type`: `'s'` for simulated data, `'r'` for real data
- `options`: struct with estimation config
- `methods`: `1` for original rDCM, `2` for sparse rDCM

**Example:**
```matlab
% run_tapas_rdcm.m
addpath('C:/Users/aman0087/Documents/Github/spm12');
addpath(genpath('C:/Users/aman0087/Documents/Github/tapas/rDCM'));
spm('defaults', 'FMRI');

load('validation/data/rdcm_input.mat', 'DCM');

% Set options
options = [];
options.filter_str = 0;       % No temporal filtering for synthetic data
options.restrictInputs = 0;   % Don't restrict inputs
options.iter = 100;           % Permutation iterations for sparse

% Run rigid rDCM (methods=1)
[output_rigid, ~] = tapas_rdcm_estimate(DCM, 's', options, 1);

% Run sparse rDCM (methods=2)
[output_sparse, ~] = tapas_rdcm_estimate(DCM, 's', options, 2);

% Extract results
results.rigid.Ep = output_rigid.Ep;
results.rigid.logF = output_rigid.logF;
results.sparse.Ep = output_sparse.Ep;
results.sparse.logF = output_sparse.logF;

save('validation/data/rdcm_tapas_results.mat', 'results');
```

### Pattern 5: Loading SPM Results Back into Python

**What:** Load SPM12/tapas results from .mat files and compare with Pyro posteriors.

**Example:**
```python
# Source: scipy.io.loadmat documentation, verified locally
import scipy.io
import numpy as np

def load_spm_results(mat_path: str) -> dict:
    """Load SPM12 estimation results from .mat file."""
    data = scipy.io.loadmat(mat_path, squeeze_me=False)
    results = data['results']

    # Access nested struct fields -- scipy returns structured numpy arrays
    Ep_A = results['Ep_A'][0, 0].astype(np.float64)
    F = results['F'][0, 0].item()

    return {'Ep_A': Ep_A, 'F': F}

def compare_posterior_means(
    pyro_A: np.ndarray,
    spm_A: np.ndarray,
    tolerance: float = 0.10,
) -> dict:
    """Element-wise comparison of posterior mean A matrices.

    Uses relative error where |true| > threshold, absolute error otherwise.
    """
    abs_diff = np.abs(pyro_A - spm_A)
    abs_spm = np.abs(spm_A)

    # Relative error where SPM values are non-negligible
    threshold = 0.01
    rel_mask = abs_spm > threshold
    rel_error = np.where(rel_mask, abs_diff / abs_spm, abs_diff)

    max_rel_error = rel_error.max()
    mean_rel_error = rel_error.mean()
    within_tolerance = (rel_error < tolerance).all()

    return {
        'max_relative_error': max_rel_error,
        'mean_relative_error': mean_rel_error,
        'within_tolerance': within_tolerance,
        'element_errors': rel_error,
    }
```

### Anti-Patterns to Avoid
- **Comparing absolute free energy values between SPM and Pyro:** SPM's F and Pyro's -ELBO use different approximations. Compare RANKINGS only.
- **Exporting pre-computed CSD to SPM's spectral DCM:** `spm_dcm_fmri_csd` computes CSD from BOLD internally. Export BOLD time series, not CSD.
- **Using scipy.io.savemat without wrapping scalars in 2D arrays:** MATLAB expects scalars as `[[value]]`, not bare floats.
- **Forgetting `DCM.Y.Q`:** SPM12 crashes without error precision components. Use `spm_Ce(ones(1,n)*v)` in MATLAB.
- **Not calling `spm('defaults','FMRI')` before estimation:** SPM12 functions depend on global defaults being set.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| .mat file I/O | Custom binary format | `scipy.io.savemat`/`loadmat` | Handles MATLAB struct nesting, data types, versioning correctly |
| DCM struct construction | Manual struct assembly | Template-based export functions | SPM12 is extremely sensitive to missing/misformatted fields |
| Error precision components | Custom Q matrix | `spm_Ce(ones(1,n)*v)` in MATLAB | SPM's internal format uses sparse block-diagonal precision components |
| CSD from BOLD | Custom CSD estimation | Let SPM's `spm_dcm_fmri_csd_data` handle it | SPM uses MAR model with specific windowing and detrending |
| Model ranking comparison | Custom BIC/AIC | SPM's free energy F and Pyro's -ELBO | Both are ELBO-like bounds; comparison of ranking is the valid test |

**Key insight:** The validation pipeline is fundamentally an I/O and struct-formatting exercise. The math is already done in SPM12/tapas. Getting the data format exactly right is 80% of the work.

## Common Pitfalls

### Pitfall 1: MATLAB Struct Field Ordering and Types
**What goes wrong:** `scipy.io.savemat` saves dicts as structs, but field values must be numpy arrays, not Python scalars or lists. MATLAB expects specific types.
**Why it happens:** Python dicts allow mixed types; MATLAB structs are stricter.
**How to avoid:** Always wrap scalars as `np.array([[value]])`, strings as `np.array([['string']], dtype=object)`, and empty arrays as `np.zeros((N, N, 0))` for unused 3D fields (like `DCM.b` and `DCM.d`).
**Warning signs:** MATLAB error "Subscript indices must be real" or "Undefined function or variable".

### Pitfall 2: SPM12 Input Microtime Resolution
**What goes wrong:** SPM12 expects `DCM.U.u` at microtime resolution (e.g., 16x TR), not at TR resolution. The first 32 samples of U.u are discarded internally (`Sess.U(i).u(33:end,j)` in `spm_dcm_specify_ui.m`).
**Why it happens:** SPM12 uses microtime bins for numerical integration. Inputs at TR resolution are undersampled.
**How to avoid:** Upsample stimulus to microtime resolution: `u_dt = TR / 16` (default microtime factor). Ensure `DCM.U.dt = u_dt`. Pad the beginning with 32 zero samples to match SPM's convention.
**Warning signs:** Estimation converges to wrong parameters or very different from Python results.

### Pitfall 3: SPM12 BOLD Scaling
**What goes wrong:** SPM12 internally scales BOLD data to enforce max change of 4% (task DCM: `spm_dcm_estimate.m` line 151-153) or scales to precision of 4 (spectral DCM: `spm_dcm_fmri_csd.m` line 124-127).
**Why it happens:** SPM12 normalizes data for numerical stability of the VL optimizer.
**How to avoid:** Apply the same scaling in Python before comparison, or compare normalized posterior means. Document the scaling factor.
**Warning signs:** Posterior means off by a constant multiplicative factor.

### Pitfall 4: Variational Laplace vs SVI Systematic Differences
**What goes wrong:** Posterior means differ by more than the 10% tolerance.
**Why it happens:** VL and SVI use fundamentally different optimization: VL uses Gauss-Newton with analytical Hessian and full (non-diagonal) posterior covariance; SVI uses SGD with diagonal (mean-field) covariance. VL finds a single local mode; SVI may find a different local mode. VL's Laplace approximation assumes unimodal Gaussian posterior; SVI's mean-field assumes factorized posterior.
**How to avoid:** Use multiple random seeds, report median discrepancy across scenarios. For connectivity parameters near zero, use absolute error (0.02) rather than relative error. Document which elements exceed tolerance and why. Expected systematic differences:
  - VL posterior means tend to be slightly more regularized (pulled toward prior) due to different prior handling
  - SVI with enough steps and good initialization should converge to similar modes
  - Self-connection diagonal elements may differ more due to different parameterization handling
**Warning signs:** Consistent bias in one direction across all scenarios suggests a bug; random scatter suggests optimization noise (expected).

### Pitfall 5: Free Energy vs ELBO Scale Mismatch
**What goes wrong:** Absolute F values from SPM differ enormously from -ELBO values from Pyro.
**Why it happens:** SPM's F is the negative variational free energy (a lower bound on log model evidence) computed using the full posterior covariance and analytical integration. Pyro's ELBO is a stochastic estimate of the same bound but using the mean-field guide and Monte Carlo gradient estimates. They use different base measures, different entropy terms, and different numbers of data points in the likelihood.
**How to avoid:** Never compare absolute values. Compare RANKINGS: if SPM says Model A > Model B (higher F), Pyro should say Model A > Model B (lower loss / higher ELBO). Use 3+ model comparison scenarios with clearly differentiated models.
**Warning signs:** Rankings disagree on a majority of scenarios = real problem. Rankings disagree on 1/5 scenarios = expected noise.

### Pitfall 6: tapas rDCM Expects SPM-Style DCM Struct
**What goes wrong:** `tapas_rdcm_estimate` crashes or gives wrong results.
**Why it happens:** tapas rDCM wraps SPM functions and expects the same DCM struct format. Some fields are used differently (e.g., `type='s'` for simulated data triggers internal data generation which must be bypassed).
**How to avoid:** Use `type='r'` (real data mode) and pre-populate all fields tapas expects. Or use `type='s'` with the correct `options.SNR` and `options.y_dt` fields. Verify with `tapas_rdcm_tutorial` first.
**Warning signs:** MATLAB error in tapas internal functions.

## Code Examples

### Complete Task DCM Validation Scenario
```python
# Source: Assembled from local SPM12 source and existing pyro_dcm API
import subprocess
import numpy as np
import scipy.io
import torch

from pyro_dcm.simulators.task_simulator import (
    simulate_task_dcm, make_block_stimulus, make_random_stable_A,
)
from pyro_dcm.models import (
    task_dcm_model, create_guide, run_svi, extract_posterior_params,
)
from pyro_dcm.forward_models.neural_state import parameterize_A


def run_task_dcm_validation(seed: int = 42) -> dict:
    """Full cross-validation: Python sim -> SPM12 -> compare."""

    # 1. Generate synthetic data in Python
    N, M = 3, 1
    A_true = make_random_stable_A(N, density=0.5, seed=seed)
    C = torch.zeros(N, M, dtype=torch.float64)
    C[0, 0] = 1.0
    stim = make_block_stimulus(n_blocks=5, block_duration=30, rest_duration=20)

    sim = simulate_task_dcm(A_true, C, stim, duration=250.0, TR=2.0, SNR=5.0, seed=seed)
    bold = sim['bold'].numpy()

    # 2. Upsample stimulus to microtime resolution for SPM
    TR = 2.0
    u_dt = TR / 16  # microtime resolution
    # ... (upsample sim['stimulus'] to microtime grid)

    # 3. Export to .mat for SPM12
    # (use export_task_dcm_for_spm pattern above)

    # 4. Run SPM12 via MATLAB
    subprocess.run(
        ['matlab', '-batch', 'run_spm_task_dcm'],
        cwd='validation/matlab_scripts',
        check=True,
        timeout=600,
    )

    # 5. Run Pyro SVI
    # (use existing run_svi pipeline)

    # 6. Load SPM results and compare
    spm_results = load_spm_results('validation/data/task_dcm_spm_results.mat')

    # 7. Element-wise comparison
    comparison = compare_posterior_means(pyro_A, spm_results['Ep_A'])

    return comparison
```

### Model Comparison Ranking Validation
```python
def validate_model_ranking(scenarios: list[dict]) -> bool:
    """Check that ELBO ranking matches SPM F ranking across scenarios.

    Each scenario has:
    - 'spm_F': float, SPM free energy (higher = better)
    - 'pyro_loss': float, Pyro SVI final loss (lower = better)
    """
    n = len(scenarios)
    rank_matches = 0

    for i in range(n):
        for j in range(i + 1, n):
            spm_prefers_i = scenarios[i]['spm_F'] > scenarios[j]['spm_F']
            pyro_prefers_i = scenarios[i]['pyro_loss'] < scenarios[j]['pyro_loss']
            if spm_prefers_i == pyro_prefers_i:
                rank_matches += 1

    total_pairs = n * (n - 1) // 2
    agreement_rate = rank_matches / total_pairs
    return agreement_rate >= 0.8  # Allow some disagreement
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual DCM specification via GUI | `spm_dcm_specify` with settings struct | SPM12 r7264 | Enables batch scripting for automated validation |
| Comparing absolute free energy values | Comparing model rankings / Bayes factors | Standard since Stephan et al. 2009 | Absolute values depend on approximation; rankings are robust |
| VL-only inference | Multiple inference backends (VL, ADVI, HMC) | 2024-2025 (Friston lab, Stan/Turing ports) | SVI is a valid alternative; discrepancies are expected and documented |
| SPM GUI-based analysis | `matlab -batch` headless execution | SPM12 | Enables CI/CD integration |

**Deprecated/outdated:**
- `DCM.options.nN`: Deprecated in favor of `DCM.options.maxit` (SPM12 spm_dcm_estimate.m line 106-108)
- `DCM.options.nmax`: Deprecated in favor of `DCM.options.maxnodes` (SPM12 spm_dcm_estimate.m line 122-128)

## Open Questions

1. **Julia rDCM availability**
   - What we know: Julia is NOT installed on this system. The `RegressionDynamicCausalModeling.jl` package was the reference for Phase 3 implementation.
   - What's unclear: Whether Julia installation + package setup is feasible within this phase, or whether pre-generated reference data should be used.
   - Recommendation: Make Julia validation OPTIONAL. Focus on tapas rDCM (MATLAB) as the primary rDCM reference. If Julia is installed later, add it as a bonus validation. Document this decision.

2. **tapas rDCM installation**
   - What we know: tapas is not currently installed. It needs to be cloned from `github.com/translationalneuromodeling/tapas` and added to MATLAB path. The repository was archived November 2025.
   - What's unclear: Whether the archived tapas rDCM code works with MATLAB R2022a and SPM12.
   - Recommendation: Clone tapas, test `tapas_rdcm_tutorial()` first. If it works, proceed. If not, document and fall back to SPM12-only validation for task/spectral DCM.

3. **SPM12 BOLD scaling normalization**
   - What we know: SPM12 internally scales BOLD to max 4% change (task DCM) or precision of 4 (spectral DCM). Our simulator does not apply this scaling.
   - What's unclear: Whether this scaling affects posterior means of the A matrix (it should not, as the noise model absorbs the scale), or whether it only affects the noise precision estimate.
   - Recommendation: Test with and without applying SPM-style scaling to our data before export. Document any differences.

4. **Spectral DCM comparison strategy**
   - What we know: Our spectral DCM works directly on CSD, while SPM's `spm_dcm_fmri_csd` computes CSD from BOLD via MAR model. This means SPM has an extra estimation step (CSD from BOLD) that our model does not.
   - What's unclear: How much the MAR-estimated CSD differs from the "true" CSD we generate. This introduces an additional source of discrepancy beyond VL vs SVI.
   - Recommendation: For fair comparison, generate BOLD time series (not CSD) in Python, export to SPM, let SPM estimate CSD internally. Our Pyro model should also start from the same BOLD and compute CSD the same way, OR accept that the CSD estimation step is an additional source of 5-10% error and document it. Alternatively, generate long BOLD time series (500+ scans) where MAR CSD estimation is more accurate.

5. **Appropriate relative error metric for near-zero parameters**
   - What we know: The 10% relative error criterion from the roadmap is problematic for parameters near zero (e.g., absent connections clamped to zero by mask have true value 0, and any non-zero estimate gives infinite relative error).
   - What's unclear: What the right metric is for absent connections.
   - Recommendation: Use a hybrid metric: relative error for |parameter| > 0.01, absolute error (< 0.02) for |parameter| <= 0.01. Document this convention.

## SPM12 DCM Struct Field Reference

### Required Fields for `spm_dcm_estimate` (Task DCM)

| Field | Type | Shape | Description |
|-------|------|-------|-------------|
| `DCM.a` | double | (N, N) | Binary endogenous connectivity mask |
| `DCM.b` | double | (N, N, 0) | Modulatory masks (empty for linear DCM) |
| `DCM.c` | double | (N, M) | Binary driving input mask |
| `DCM.d` | double | (N, N, 0) | Nonlinear masks (empty for bilinear DCM) |
| `DCM.Y.y` | double | (v, N) | BOLD time series |
| `DCM.Y.dt` | double | scalar | TR in seconds |
| `DCM.Y.X0` | double | (v, 1) | Confound regressors (ones = constant) |
| `DCM.Y.Q` | cell | (N,) | Error precision components from `spm_Ce` |
| `DCM.U.u` | double | (T_micro, M) | Inputs at microtime resolution |
| `DCM.U.dt` | double | scalar | Input sampling interval (TR/16 typical) |
| `DCM.U.name` | cell | (M,) | Input names |
| `DCM.n` | double | scalar | Number of regions |
| `DCM.v` | double | scalar | Number of scans |
| `DCM.TE` | double | scalar | Echo time in seconds (0.04 typical) |
| `DCM.delays` | double | (1, N) | Slice timing delays per region |
| `DCM.options.nonlinear` | double | 0 or 1 | Nonlinear DCM flag |
| `DCM.options.two_state` | double | 0 or 1 | Two-state model flag |
| `DCM.options.stochastic` | double | 0 or 1 | Stochastic DCM flag |
| `DCM.options.centre` | double | 0 or 1 | Mean-centre inputs flag |
| `DCM.options.induced` | double | 0 | Must be 0 for task DCM |
| `DCM.options.nograph` | double | 1 | Suppress GUI (required for batch) |
| `DCM.options.maxit` | double | 128 | Max VL iterations |

### Output Fields from `spm_dcm_estimate`

| Field | Type | Description |
|-------|------|-------------|
| `DCM.Ep.A` | double (N, N) | Posterior mean of A (free parameters, NOT parameterized) |
| `DCM.Ep.C` | double (N, M) | Posterior mean of C |
| `DCM.Cp` | sparse | Full posterior covariance over all parameters |
| `DCM.F` | double | Free energy (lower bound on log model evidence) |
| `DCM.y` | double (v, N) | Predicted BOLD |
| `DCM.R` | double (v, N) | Residuals |

### Critical Note on SPM12 A Matrix Parameterization

SPM12's `DCM.Ep.A` stores FREE parameters, not the parameterized A matrix. To compare with our `parameterize_A(A_free)`:
- **Diagonal:** SPM stores free param; actual self-connection = `-exp(Ep.A_ii) / 2`
- **Off-diagonal:** SPM stores the connection strength directly (same as our off-diagonal)

Our comparison should either:
1. Compare free parameters directly: our `A_free` vs SPM's `Ep.A`
2. Compare parameterized: our `parameterize_A(A_free)` vs SPM's parameterized (apply same transform to `Ep.A`)

Option 1 is simpler and avoids the nonlinear transform amplifying small differences.

### Additional Fields for `spm_dcm_fmri_csd` (Spectral DCM)

Same as task DCM but with:
| Field | Value | Description |
|-------|-------|-------------|
| `DCM.options.induced` | 1 | Enables CSD analysis mode |
| `DCM.options.analysis` | 'CSD' | Analysis type string |
| `DCM.options.order` | 8 | MAR model order for CSD estimation |

Output additionally includes:
| Field | Description |
|-------|-------------|
| `DCM.Ep.a` | Neuronal fluctuation parameters (2, 1) |
| `DCM.Ep.b` | Global observation noise (2, 1) |
| `DCM.Ep.c` | Regional observation noise (1, N) |
| `DCM.Hc` | Predicted CSD |
| `DCM.Hz` | Frequency vector |

### tapas rDCM API Reference

```matlab
% Function signature (from GitHub source):
[output, options] = tapas_rdcm_estimate(DCM, type, options, methods)

% type: 's' = simulated, 'r' = real/empirical
% methods: 1 = original (rigid), 2 = sparse
% options: struct with fields:
%   .SNR           - signal-to-noise ratio (for type='s')
%   .y_dt          - sampling interval
%   .p0_all        - sparsity hyperparameter grid (default 0.05:0.05:0.95)
%   .iter          - number of permutations per region
%   .filter_str    - temporal filter strength
%   .restrictInputs - prune driving inputs flag

% Output struct:
%   output.Ep      - posterior expectations (struct with A, C, ...)
%   output.logF    - negative free energy
%   output.Ip      - binary indicators for connections (sparse only)
```

## Expected Discrepancies Between VL and SVI

### Posterior Means
- **Expected range:** 5-15% relative error on off-diagonal A elements
- **Diagonal (self-connections):** May differ more due to nonlinear parameterization (`-exp(x)/2`) amplifying small differences in free parameter space
- **Near-zero parameters:** Absolute differences < 0.02 expected; relative error meaningless
- **Root cause:** Different optimization algorithms (Gauss-Newton vs Adam), different starting points, different regularization from prior handling

### Posterior Uncertainties
- **VL tendency:** Underestimates uncertainty (overconfident) due to Laplace approximation at a single mode
- **SVI tendency:** Mean-field factorization ignores posterior correlations, which can either over- or under-estimate marginal variances
- **Not directly comparable:** VL gives full (non-diagonal) covariance; SVI gives diagonal (independent) scales
- **Recommendation:** Do NOT compare posterior variances as a validation criterion. Focus on means.

### Free Energy / ELBO
- **Absolute values:** Will differ significantly. SPM's F includes log-determinant of full posterior covariance, analytical integration terms, and hyperparameter estimation. Pyro's -ELBO is a Monte Carlo estimate using the mean-field guide.
- **Rankings:** Should agree on clearly differentiated models (correct vs. wrong mask). May disagree on closely matched models (within noise).
- **Recommendation:** Test ranking agreement on 3+ scenarios where the correct model is clearly better (missing a true connection vs. having it).

## Sources

### Primary (HIGH confidence)
- SPM12 source code (local install: `C:/Users/aman0087/Documents/Github/spm12/`):
  - `spm_dcm_estimate.m` -- Full DCM struct specification, estimation pipeline, output fields
  - `spm_dcm_fmri_csd.m` -- Spectral DCM estimation, CSD computation from BOLD
  - `spm_dcm_fmri_priors.m` -- Prior specifications (pA=64, A prior = A/128, pC.A = A/pA + I/pA)
  - `spm_dcm_specify_ui.m` -- DCM struct construction, U/Y field format
  - `spm_dcm_generate.m` -- Synthetic data generation reference
- `scipy.io.savemat` documentation (verified locally) -- Nested struct export works correctly
- Existing pyro_dcm codebase (local) -- All simulators, models, and inference verified

### Secondary (MEDIUM confidence)
- tapas rDCM GitHub (`github.com/translationalneuromodeling/tapas/tree/master/rDCM/code`):
  - `tapas_rdcm_estimate.m` signature: `[output, options] = tapas_rdcm_estimate(DCM, type, options, methods)`
  - `tapas_rdcm_tutorial.m` -- Working example code
  - Repository archived November 2025
- Zeidman et al. (2024), "A primer on Variational Laplace (VL)", NeuroImage (PMC10951963):
  - VL overconfidence in posterior estimates documented
  - ADVI more accurate when Laplace assumption violated

### Tertiary (LOW confidence)
- Julia `RegressionDynamicCausalModeling.jl` -- Referenced in Phase 3 research but not verified on this system (Julia not installed)
- Web search results on VL vs SVI discrepancies -- General statements, not DCM-specific empirical results

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - SPM12 source verified locally, scipy.io tested
- Architecture (export/import pipeline): HIGH - All components verified individually
- SPM DCM struct format: HIGH - Read directly from SPM12 source code
- tapas rDCM API: MEDIUM - Function signature from GitHub, not tested locally
- Julia rDCM: LOW - Not installed, cannot verify
- Expected VL vs SVI discrepancies: MEDIUM - Based on published VL primer and general VI theory
- Pitfalls: HIGH - Derived from reading SPM12 source code (scaling, microtime, struct format)

**Research date:** 2026-03-28
**Valid until:** 2026-04-28 (SPM12 is mature/stable; tapas archived)
