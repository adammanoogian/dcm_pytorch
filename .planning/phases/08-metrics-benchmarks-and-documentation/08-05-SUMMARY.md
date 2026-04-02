---
phase: 08-metrics-benchmarks-and-documentation
plan: 05
subsystem: documentation
tags: [docstrings, api, exports, numpy-style, ruff, examples]

# Dependency graph
requires:
  - phase: 08-metrics-benchmarks-and-documentation
    plan: 01
    provides: "Benchmark infrastructure and pyproject.toml extras"
provides:
  - "Complete NumPy-style docstrings across all ~18 public source files"
  - "Examples sections on top ~15 most important public functions"
  - "Expanded __init__.py with 23 public API exports organized by category"
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns: ["NumPy-style docstrings with Examples sections"]

key-files:
  created: []
  modified:
    - "src/pyro_dcm/__init__.py"
    - "src/pyro_dcm/forward_models/neural_state.py"
    - "src/pyro_dcm/forward_models/balloon_model.py"
    - "src/pyro_dcm/forward_models/bold_signal.py"
    - "src/pyro_dcm/forward_models/spectral_transfer.py"
    - "src/pyro_dcm/forward_models/csd_computation.py"
    - "src/pyro_dcm/forward_models/rdcm_forward.py"
    - "src/pyro_dcm/forward_models/rdcm_posterior.py"
    - "src/pyro_dcm/forward_models/coupled_system.py"
    - "src/pyro_dcm/models/task_dcm_model.py"
    - "src/pyro_dcm/models/spectral_dcm_model.py"
    - "src/pyro_dcm/models/rdcm_model.py"
    - "src/pyro_dcm/models/guides.py"
    - "src/pyro_dcm/guides/amortized_flow.py"
    - "src/pyro_dcm/guides/summary_networks.py"
    - "src/pyro_dcm/guides/parameter_packing.py"
    - "src/pyro_dcm/simulators/task_simulator.py"
    - "src/pyro_dcm/simulators/spectral_simulator.py"
    - "src/pyro_dcm/simulators/rdcm_simulator.py"
    - "src/pyro_dcm/utils/ode_integrator.py"

key-decisions:
  - "D104 (missing __init__.py docstrings) accepted as ignorable -- package __init__ files are re-export hubs"
---

# Summary: 08-05 API Docstring Audit + Expanded Exports

## What Was Built

1. **Forward models docstring audit** — Added Examples sections to all public functions
   in neural_state.py, balloon_model.py, bold_signal.py, spectral_transfer.py,
   csd_computation.py, rdcm_forward.py, rdcm_posterior.py, and coupled_system.py.
   All math functions retain [REF-XXX] citations.

2. **Models/guides/simulators/utils docstring audit** — Added Examples sections to
   run_svi(), create_guide(), extract_posterior_params(), simulate_task_dcm(),
   simulate_spectral_dcm(), simulate_rdcm(), make_random_stable_A(), make_block_stimulus(),
   task_dcm_model(), spectral_dcm_model(), and AmortizedFlowGuide class.

3. **Expanded top-level exports** — __init__.py now exports 23 public API symbols
   organized into: Generative models, Inference, Amortized inference, Simulators,
   and Utilities sections.

## Verification

- `ruff check --select D src/pyro_dcm/` — 4 D104 violations (package __init__.py files only, accepted)
- `python -c "from pyro_dcm import simulate_task_dcm, extract_posterior_params, parameterize_A"` — OK
- Tests: all passing (docstring-only changes, no logic modifications)

## Commits

| Hash | Message |
|------|---------|
| 3176656 | docs(08-05): add Examples sections to forward_models docstrings |
| 757db31 | docs(08-05): add Examples sections to models, guides, simulators, utils |
| ee941ab | chore: add .gitignore and expand top-level exports |
| d7fa601 | docs(08-05): add Examples sections to remaining forward_models docstrings |

## Issues

None.
