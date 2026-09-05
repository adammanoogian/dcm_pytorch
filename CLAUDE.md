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

- **Python 3.10+**
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
├── src/pyro_dcm/
│   ├── forward_models/          # deterministic forward models
│   │   ├── neural_state.py          # dx/dt = Ax + Su.Bx + Cu  [REF-001]
│   │   ├── balloon_model.py         # Balloon-Windkessel ODEs  [REF-002]
│   │   ├── bold_signal.py           # BOLD observation equation  [REF-002]
│   │   ├── coupled_system.py        # coupled neural + hemodynamic system
│   │   ├── spectral_transfer.py     # H(w) = (iwI - A)^-1  [REF-010]
│   │   ├── spectral_noise.py        # innovation / measurement noise spectra
│   │   ├── csd_computation.py       # cross-spectral density  [REF-010]
│   │   ├── mar_csd.py               # MAR-based CSD (SPM parity)
│   │   ├── rdcm_forward.py          # frequency-domain regression  [REF-020]
│   │   ├── rdcm_posterior.py        # rDCM analytic VB posterior  [REF-020]
│   │   ├── latent_observation.py    # latent-circuit observation
│   │   ├── cmc_neural_mass.py       # canonical microcircuit  (v0.8.0)
│   │   ├── cmc_priors.py            # CMC prior specification
│   │   ├── _cmc_network.py          # shared N-source CMC core
│   │   ├── erp_coupled_system.py    # hierarchical extrinsic coupling
│   │   ├── erp_input.py             # evoked input (spm_erp_u port)
│   │   ├── erp_leadfield.py         # single-dipole lead field + scalp proj.
│   │   ├── mmn_reference.py         # 5-source auditory MMN network
│   │   └── collision_reference.py   # 2/3-node collision networks
│   ├── inference/               # inference engines
│   │   ├── variational_laplace.py   # SPM12 spm_nlsi_GN port (the VL engine)
│   │   ├── vl_forward_models.py     # ForwardModel protocol implementors
│   │   ├── csd_precision.py         # CSD precision / hyperpriors
│   │   └── sbi_*.py                 # SBI/NPE (SBC calibration OPEN, 2/9)
│   ├── model_selection/
│   │   └── bmr.py                   # Bayesian Model Reduction  [REF-030]
│   ├── models/                  # Pyro generative models
│   │   ├── task_dcm_model.py        # task DCM (+ bilinear B path)
│   │   ├── spectral_dcm_model.py    # spectral DCM
│   │   ├── rdcm_model.py            # regression DCM
│   │   ├── latent_circuit_dcm_model.py
│   │   ├── erp_dcm_model.py         # ERP-DCM  (v0.8.0)
│   │   ├── hybrid_vae_dcm.py        # amortized encoder + DCM ODE decoder
│   │   ├── guides.py                # SVI guide factory
│   │   └── amortized_wrappers.py
│   ├── guides/                  # amortized / flow guides
│   ├── simulators/              # task, spectral, rdcm, latent, erp, meg
│   ├── io/                      # MNE + BIDS loaders
│   ├── rnn/                     # CT-RNN training, PCA, fixed points
│   ├── neural_data_models/      # LSTM autoencoder, latent CSD
│   ├── foundation/              # TRIBE / LaBraM / BrainOmni extractors
│   └── utils/
│       ├── ode_integrator.py        # torchdiffeq wrapper
│       ├── local_linearization.py   # spm_int_L exp-Euler  (v0.8.0)
│       └── circuit_viz.py
├── tests/                       # unit + recovery + SPM parity ladders
├── validation/
│   ├── matlab_scripts/          # SPM12 .m bridges (getenv SPM12_PATH)
│   ├── data/                    # byte-frozen SPM12 .mat fixtures
│   ├── run_validation.py        # SVI-path SPM12 orchestrator
│   └── run_vl_validation.py     # VL-path SPM12 orchestrator (Phase 32)
├── benchmarks/                  # runners, metrics, recovery matrix
├── cluster/                     # DCCN Slurm jobs (see cluster/README.md)
│   ├── lib/cluster_env.sh       # shared env activation + MATLAB setup
│   ├── sbatch/                  # job scripts
│   └── scripts/                 # Python entrypoints
├── scripts/                     # demos + training entrypoints
├── docs/
├── .planning/                   # GSD roadmap / phases / state
├── config.py                    # ALL path constants (CCDS v2)
├── pyproject.toml
└── CLAUDE.md
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
- **Tensor shapes**: Documented in NumPy-style ``Parameters`` blocks with explicit shape annotations (e.g., ``A : torch.Tensor, shape (N, N)``)
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

## Compute Routing (DCCN)

Compute runs on the **DCCN cluster** (`mentat`), Slurm-scheduled. Migrated from
Monash M3 on 2026-09-05 -- `--partition=comp`, conda envs, and `/home/aman0087/`
paths anywhere in this repo are pre-migration provenance.

- Anything over **~3 minutes of saturating laptop CPU** goes to the cluster:
  `pytest -m slow`, full-suite runs, SVI/NUTS fits, VL sweeps, identifiability
  grids, recovery harnesses. Fast unit tests (<30 s) stay local.
- **This binds subagents too.** No laptop-only carve-out.
- Partition `batch`; **always pass `--mem`** (DCCN defaults to 1 GB).
- No conda on the cluster -- jobs activate a uv venv via
  `cluster/lib/cluster_env.sh`. Never install inside a job.
- **MATLAB/SPM12 run LOCALLY**: the workstation has R2025b (valid licence) and a
  complete SPM12. The cluster has MATLAB modules but no system SPM12.
- The v0.8.0 ERP parity ladders are fixture-keyed and need no MATLAB; only
  regenerating `validation/data/*.mat` does.

See `cluster/README.md` and the `dccn-hpc` skill.

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

## Research track

This repo serves two research tracks:

- **DCM + Machine Learning** (`dcm-ml`) -- `Tracks/Track - DCM + Machine Learning.md`
- **Joint DCM-Behavioural Modelling** (`joint-dcm-behaviour`) -- `Tracks/Track - Joint DCM-Behavioural Modelling.md`

Differentiable DCM, and the inference engine joint fitting needs. `dcm_hgf_mixed_models` depends on the `pyro_dcm` interface.

A *track* is a research programme -- one scientific question pursued across many
repos, over years. The roster, the full repo-to-track map, and the cross-track
couplings live in the **`research-tracks` skill**
(`~/.claude/skills/research-tracks/SKILL.md`); the authoritative notes live in
the Obsidian vault at `C:\Users\adaman\Documents\Obsidian Vault\Tracks\`.

Read the track note before substantive work here -- it carries the current
state, the open questions, and the decision log, none of which are duplicated in
this repo. When work produces something durable (a derivation, a protocol, a
transferable failure), file it to the vault layer that matches its lifetime
rather than leaving it in this repo's docs.
