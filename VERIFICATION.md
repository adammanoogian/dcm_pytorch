# Pyro-DCM v0.1.0-foundation -- Integration Verification Report

Generated: 2026-03-31
Branch: gsd/phase-08-metrics-benchmarks-and-documentation
Scope: Cross-phase wiring, E2E flow completeness, API consistency

---

## Integration Check Complete

### Wiring Summary

**Connected:** 18 exports properly used across phases
**Orphaned:** 2 categories of exports not in top-level API
**Missing:** 1 expected connection (scripts package importability)

### API Coverage

**Consumed:** All 7 benchmark runners have callers via RUNNER_REGISTRY
**Orphaned:** 0 runner registrations without callers

### E2E Flows

**Complete:** 6 of 8 flows work end-to-end
**Broken:** 2 flows have confirmed runtime bugs

---

## Connected Exports (Verified Working)

**Phase 1 forward models -> Phase 4 task_dcm_model:**
- `bold_signal` imported in task_dcm_model.py (line 25) and amortized_wrappers.py (line 35)
- `CoupledDCMSystem` imported in task_dcm_model.py (line 26) and amortized_wrappers.py (line 36)
- `parameterize_A` imported in task_dcm_model.py (line 27), spectral_dcm_model.py (line 29), amortized_wrappers.py (line 37), and all SVI benchmark runners
- `integrate_ode` and `make_initial_state` imported in task_dcm_model.py (line 28) and amortized_wrappers.py (lines 44-46)

**Phase 2 spectral forward model -> Phase 4 spectral_dcm_model:**
- `spectral_dcm_forward` imported in spectral_dcm_model.py (line 30) and amortized_wrappers.py (line 38)
- `decompose_csd_for_likelihood` from spectral_dcm_model.py used in amortized_wrappers.py (line 43) and spectral_svi.py benchmark runner (line 33)

**Phase 3 rDCM forward model -> Simulator and Benchmarks:**
- `create_regressors`, `generate_bold`, `get_hrf` imported in rdcm_simulator.py (lines 24-28) and rdcm_vb.py benchmark runner (lines 25-28)
- `rigid_inversion`, `sparse_inversion` imported in rdcm_simulator.py (lines 29-32) and rdcm_vb.py benchmark runner (lines 31-32)

**Phase 4 guides infrastructure -> Phase 7 amortized:**
- `TaskDCMPacker`, `SpectralDCMPacker` imported in amortized_flow.py (line 30) and amortized_wrappers.py (line 39)
- `amortized_task_dcm_model` wired in task_amortized.py (line 40), used at line 156
- `amortized_spectral_dcm_model` wired in spectral_amortized.py (line 41), used at line 185

**Phase 8 benchmark internal wiring:**
- RUNNER_REGISTRY in benchmarks/runners/__init__.py wires all 7 (variant, method) pairs
- run_all_benchmarks.py imports RUNNER_REGISTRY (line 46) and dispatches via runner(config) (line 312)
- benchmarks/plotting.py::generate_all_figures imported inline and called in run_all_benchmarks.py (lines 353-358)
- BenchmarkConfig.quick_config and full_config imported and used in run_all_benchmarks.py (lines 301-302)

---

## Orphaned Exports

**1. rigid_inversion, sparse_inversion, create_regressors, generate_bold, get_hrf -- not in top-level __init__.py**

- Defined in: src/pyro_dcm/forward_models/rdcm_posterior.py and rdcm_forward.py
- Exported from: src/pyro_dcm/forward_models/__init__.py (lines 20-36)
- Missing from: src/pyro_dcm/__init__.py
- Impact: Users cannot access rDCM analytic VB via `import pyro_dcm`. The stated rDCM E2E flow is simulate_rdcm -> rdcm_posterior but neither rigid_inversion nor sparse_inversion are reachable at top level.
- Severity: Medium. Does not break existing code (all consumers import from submodules directly).

**2. decompose_csd_for_likelihood -- in models/__init__.py __all__ but not pyro_dcm/__init__.py**

- Defined in: src/pyro_dcm/models/spectral_dcm_model.py
- Exported from: src/pyro_dcm/models/__init__.py (line 11)
- Missing from: src/pyro_dcm/__init__.py
- Impact: Low. Used only internally; not a primary user-facing function.

---

## Missing Connections

**scripts/ directory has no __init__.py and pytest has no pythonpath setting**

Expected: `from scripts.generate_training_data import invert_A_to_A_free` works in tests.

Files affected:
- tests/test_amortized_benchmark.py line 64
- tests/test_amortized_spectral_dcm.py line 43
- tests/test_amortized_task_dcm.py line 38
- tests/test_parameter_packing.py line 219
- benchmarks/runners/task_amortized.py line 46
- benchmarks/runners/spectral_amortized.py line 47

Mitigation in place: run_all_benchmarks.py adds project root to sys.path (lines 39-41). Python 3 namespace packages allow the import without __init__.py when project root is on sys.path. However pyproject.toml does not include `pythonpath = ["."]" under [tool.pytest.ini_options], making test execution fragile.

Fix: Add `pythonpath = ["."]" to [tool.pytest.ini_options] in pyproject.toml.

---

## Broken Flows

### Flow 1: Amortization Gap Computation (task and spectral runners)

**Status: BROKEN -- logic error produces meaningless gap values**

Broken at:
- benchmarks/runners/task_amortized.py lines 403-411
- benchmarks/runners/spectral_amortized.py lines 385-393

Description: The amortization gap is defined as the ELBO difference between a per-subject SVI guide and the amortized guide. Both runners instead pass `svi_guide` to `Trace_ELBO().loss()` in both calls, computing the SVI ELBO twice rather than once each for SVI and amortized.

Relevant code in task_amortized.py lines 403-408:

    # BUG: svi_guide is used, not the amortized guide
    elbo = Trace_ELBO()
    amort_loss = elbo.loss(
        task_dcm_model, svi_guide,
        *model_args,
    )

Additional complication: amortized_task_dcm_model requires `packer` as an 8th positional argument, but model_args at this point is a 7-element tuple (bold, stimulus, a_mask, c_mask, t_eval, TR, dt_model). A corrected evaluation must use amortized_task_dcm_model, the AmortizedFlowGuide, and model_args extended with packer.

Impact: All amortization_gap_list values in benchmark JSON output are wrong. The computed gap will be near zero (same guide evaluated twice) rather than measuring the true amortization cost.

The same bug is present in spectral_amortized.py lines 385-393.

### Flow 2: default_frequency_grid argument swap in spectral_amortized fallback

**Status: BROKEN -- swapped positional arguments produce wrong frequency grid**

Broken at: benchmarks/runners/spectral_amortized.py line 288

Function signature (src/pyro_dcm/forward_models/spectral_transfer.py):

    def default_frequency_grid(TR: float = 2.0, n_freqs: int = 32) -> torch.Tensor

Actual call at line 288:

    freqs = default_frequency_grid(32, TR=2.0)

This passes 32 as the TR argument. Result: a frequency grid from 1/128 Hz to 1/(2*32) = 0.015625 Hz instead of the correct Nyquist 0.25 Hz. The frequency axis covers only 6% of the expected spectral range.

This path is triggered only when loading a pre-trained guide checkpoint that does not contain a freqs key (line 282: checkpoint.get("freqs", None) returns None). The inline training path is unaffected.

Fix: `freqs = default_frequency_grid(TR=2.0, n_freqs=32)`

---

## E2E Flow Status

| Flow | Status | Notes |
|------|--------|-------|
| Task DCM: simulate -> SVI -> extract | COMPLETE | All data contracts satisfied |
| Spectral DCM: simulate -> SVI -> extract | COMPLETE | decompose_csd wired correctly |
| rDCM analytic VB: simulate_rdcm | COMPLETE | Internal pipeline fully wired |
| Amortized task: training + sample_posterior | COMPLETE | Training and inference correct |
| Amortized spectral: training + sample_posterior | COMPLETE | Training and inference correct |
| Amortization gap metric | BROKEN | svi_guide passed instead of amortized guide in both runners |
| Benchmark CLI: JSON + figures | STRUCTURALLY COMPLETE | Gap values in output will be wrong |
| Model comparison via ELBO | COMPLETE | run_svi result used correctly |
| SPM validation (Python side) | COMPLETE | All imports resolve |
| SPM validation (full) | MATLAB REQUIRED | Hardcoded path; not exercised in CI |
| Fallback freq grid (spectral_amortized) | BROKEN | Args swapped: default_frequency_grid(32, TR=2.0) |

---

## Parameter and API Consistency

**A_free / A naming:** Consistent across all three Pyro models and all benchmark runners. All models use Pyro sample site name A_free; all consumers call parameterize_A on the extracted median.

**Log-space contracts:**
- TaskDCMPacker packs noise_prec in log-space (parameter_packing.py line 119). amortized_task_dcm_model calls .exp() on the unpacked value (amortized_wrappers.py line 215). CONSISTENT.
- SpectralDCMPacker packs csd_noise_scale in log-space (parameter_packing.py line 305). amortized_spectral_dcm_model calls .exp() (amortized_wrappers.py line 259). CONSISTENT.

**Tensor shape contracts:**
- simulate_task_dcm returns bold (T_TR, N); task_dcm_model expects (T, N). MATCH.
- simulate_spectral_dcm returns csd (F, N, N) complex128; spectral_dcm_model expects (F, N, N) complex128. MATCH.
- generate_bold returns {"y", "y_clean", "x", "hrf"}; simulate_rdcm accesses ["hrf"] and ["y"]. MATCH.
- rigid_inversion returns {"A_mu", "C_mu", "F_total", "F_per_region", "mu_per_region", "Sigma_per_region"}; rdcm_vb.py rigid runner accesses all required keys. MATCH.
- sparse_inversion adds "z_per_region" to return dict; rdcm_vb.py sparse runner accesses it. MATCH.

**generate_bold return keys vs simulate_rdcm consumption:** generate_bold (rdcm_forward.py line 374) returns {"y", "y_clean", "x", "hrf"}. simulate_rdcm (rdcm_simulator.py lines 312-313) accesses bold_result["hrf"] and bold_result["y"]. MATCH.

---

## Action Items (Priority Order)

### Critical (break benchmark output correctness)

**1. Fix amortization gap computation**

Files: benchmarks/runners/task_amortized.py (lines 403-411), benchmarks/runners/spectral_amortized.py (lines 385-393).

In task_amortized.py: replace the elbo.loss(task_dcm_model, svi_guide, *model_args) block with a call using amortized_task_dcm_model, the AmortizedFlowGuide instance (variable name `guide`), and model_args extended with packer: (bold, stimulus, a_mask, c_mask, t_eval, TR, dt_model, packer). Note that `guide` is in scope from the _train_task_guide_inline return or the pretrained load branch.

In spectral_amortized.py: same fix using amortized_spectral_dcm_model, the AmortizedFlowGuide (`guide`), and model_args (csd, freqs, a_mask, packer).

**2. Fix default_frequency_grid argument order**

File: benchmarks/runners/spectral_amortized.py line 288.

Change from: `default_frequency_grid(32, TR=2.0)`
Change to:   `default_frequency_grid(TR=2.0, n_freqs=32)`

### Non-Critical (API and test reliability)

**3. Add pythonpath to pytest config**

File: pyproject.toml [tool.pytest.ini_options].
Add: `pythonpath = ["."]" to make `from scripts.generate_training_data import invert_A_to_A_free` reliably importable in tests across all environments.

**4. Export rDCM analytic VB functions from top-level API**

File: src/pyro_dcm/__init__.py.
Add: `rigid_inversion`, `sparse_inversion` (and optionally `create_regressors`, `generate_bold`, `get_hrf`) to complete the stated rDCM user-facing flow.

---

*Report generated by integration checker. All file line references are verified against the current codebase.*
