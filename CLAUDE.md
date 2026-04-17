# CLAUDE.md -- Project Instructions for Pyro-DCM

## Project Overview

Research-grade Python implementation of Dynamic Causal Modeling (DCM) for neuroimaging,
using Pyro for probabilistic inference. See `.planning/PROJECT.md` for full vision and
`.planning/ROADMAP.md` for current phase.

## Critical Rules

### 1. No Placeholders -- Ever

Every function must compute real mathematics. The following patterns are FORBIDDEN:

```python
# FORBIDDEN:
def compute_csd(timeseries):
    pass  # TODO: implement

def balloon_ode(state, t, params):
    return torch.zeros_like(state)  # placeholder

def transfer_function(omega, A):
    # simplified version -- replace later
    return torch.eye(A.shape[0])
```

If the math is not fully specified, do not write the function. Consult
`.planning/REFERENCES.md` first.

### 2. Every Equation Must Be Cited

Every function implementing a mathematical equation MUST have a docstring citing
the reference ID and equation number from `.planning/REFERENCES.md`:

```python
def balloon_ode(state, t, params):
    """Balloon-Windkessel hemodynamic model.

    Implements [REF-002] Eq. 2-5 (Stephan et al. 2007).

    Parameters
    ----------
    state : torch.Tensor
        Hemodynamic state vector (s, f, v, q) per region.
    t : float
        Time point.
    params : dict
        Keys: kappa, gamma, tau, alpha, E0.

    Returns
    -------
    torch.Tensor
        Time derivatives of hemodynamic states.
    """
```

### 3. Test Before Integrate

No module is integrated into the pipeline until it passes its own standalone test:
- Simulate known ground truth
- Run the module
- Assert recovery within documented tolerances

### 4. Numerical Stability

- All ODE integrations must be tested for 500s simulation duration without NaN
- Log-transform positive parameters (kappa, gamma, tau, alpha) in the Pyro model
- Use `torch.linalg.solve` not `torch.inverse` for matrix operations
- Clip eigenvalues of A to ensure stability: real(lambda) < 0

## Tech Stack

- **Python 3.11+**
- **PyTorch 2.x** -- tensor computations, autograd
- **Pyro 1.9+** -- probabilistic programming, SVI, ELBO
- **torchdiffeq** -- ODE integration (odeint, odeint_adjoint)
- **Zuko** -- normalizing flows for amortized guides
- **NumPyro** -- NUTS validation only (JAX backend)
- **scipy** -- signal processing (CSD computation), validation
- **matplotlib** -- plotting for diagnostics and benchmarks
- **pytest** -- testing framework
- **ruff** -- linting + formatting (line-length 88, NumPy docstrings)
- **mypy** -- type checking

## Directory Structure (src/ layout)

```
dcm_pytorch/
├── src/
│   └── pyro_dcm/
│       ├── __init__.py
│       ├── forward_models/
│       │   ├── __init__.py
│       │   ├── neural_state.py      # dx/dt = Ax + Cu  [REF-001]
│       │   ├── balloon_model.py     # Balloon-Windkessel ODEs  [REF-002]
│       │   ├── bold_signal.py       # BOLD observation equation  [REF-002]
│       │   ├── spectral_transfer.py # H(w) = (iwI - A)^-1  [REF-010]
│       │   ├── csd_computation.py   # Cross-spectral density  [REF-010]
│       │   └── rdcm_likelihood.py   # Frequency-domain regression  [REF-020]
│       ├── models/
│       │   ├── __init__.py
│       │   ├── task_dcm_model.py       # Pyro model for task-based DCM [v0.3.0: + bilinear B path]
│       │   ├── spectral_dcm_model.py   # Pyro model for spectral DCM
│       │   ├── rdcm_model.py           # Pyro model for regression DCM
│       │   ├── guides.py               # SVI guide factory (AutoNormal/AutoLowRankMVN/AutoIAF/...)
│       │   └── amortized_wrappers.py   # Amortized task/spectral DCM wrappers
│       ├── guides/
│       │   ├── __init__.py
│       │   ├── meanfield.py         # Baseline Gaussian guide
│       │   └── amortized_flow.py    # Normalizing flow amortized guide
│       ├── connectivity/
│       │   ├── __init__.py
│       │   ├── static_a.py          # Fixed A matrix prior
│       │   └── structural_mask.py   # Binary mask for allowed connections
│       ├── simulators/
│       │   ├── __init__.py
│       │   ├── task_simulator.py
│       │   ├── spectral_simulator.py
│       │   └── rdcm_simulator.py
│       ├── inference/
│       │   ├── __init__.py
│       │   ├── svi_runner.py        # SVI training loop with diagnostics
│       │   ├── nuts_validator.py    # NumPyro NUTS for posterior validation
│       │   └── model_comparison.py  # ELBO-based Bayesian model comparison
│       └── utils/
│           ├── __init__.py
│           ├── ode_integrator.py    # Wrapper around torchdiffeq
│           ├── spectral_utils.py    # FFT, CSD, frequency grids
│           └── diagnostics.py       # Convergence checks, posterior plots
├── tests/
│   ├── conftest.py
│   ├── test_balloon.py
│   ├── test_neural_state.py
│   ├── test_spectral.py
│   ├── test_rdcm.py
│   ├── test_pyro_models.py
│   ├── test_task_dcm_recovery.py
│   ├── test_spectral_dcm_recovery.py
│   ├── test_rdcm_recovery.py
│   └── test_model_comparison.py
├── validation/
│   ├── compare_spm.py
│   └── recovery_results/
├── benchmarks/
│   ├── run_all_benchmarks.py
│   └── results/
├── scripts/
│   └── train_amortized_guide.py
├── docs/
│   ├── 00_current_todos/
│   ├── 01_project_protocol/
│   ├── 02_pipeline_guide/
│   ├── 03_methods_reference/
│   └── 04_scientific_reports/
├── figures/
├── .planning/
├── pyproject.toml
├── CLAUDE.md
└── README.md
```

## Coding Conventions

Follows `project_utils/CODING_STANDARDS.md`:

- **Docstrings**: NumPy-style, enforced by ruff (`convention = "numpy"`)
- **Type hints**: Python 3.10+ native syntax (`list[float]`, `str | None`)
- **Imports**: `from __future__ import annotations` in every module
- **Naming**: snake_case functions/vars, PascalCase classes, UPPER_SNAKE constants
- **Math notation**: Three-layer system (class internals: math symbols; API: descriptive; scripts: domain English)
- **Fitted attributes**: Trailing underscore (`K_`, `x_post_`, `P_post_`)
- **Line length**: 88 characters
- **Function size**: Target 20 lines, hard limit 50
- **Tensor shapes**: Documented in docstrings as `# shape: (n_regions, n_timepoints)`
- **No global mutable state**: All config via function arguments or dataclasses

## Tensor Shape Conventions

| Tensor | Shape | Description |
|--------|-------|-------------|
| A | (N, N) | Effective connectivity matrix |
| C | (N, M) | Driving input weights |
| B_j | (N, N) | Modulatory input j |
| u | (T, M) | Experimental inputs over time |
| bold | (T, N) | BOLD time series |
| csd | (F, N, N) | Cross-spectral density (complex) |
| hemo_state | (T, N, 4) | (s, f, v, q) per region |

N = regions, M = inputs, T = time points, F = frequency bins

## When Stuck

1. Check `.planning/REFERENCES.md` for the relevant paper and equation
2. Check `.planning/STATE.md` for prior decisions
3. If a mathematical detail is ambiguous, flag it -- do not guess
4. If SPM source code is needed for clarification:
   - `spm_dcm_fmri.m` (task DCM)
   - `spm_dcm_csd.m` (spectral DCM)
   - `spm_fx_fmri.m` (neural state equation)
   - `spm_gx_fmri.m` (BOLD observation)
   - `spm_csd_mtf.m` (CSD computation)
