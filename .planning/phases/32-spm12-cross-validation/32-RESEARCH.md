# Phase 32: SPM12 Cross-Validation (Local / MATLAB) - Research

**Researched:** 2026-06-11
**Domain:** VL inference cross-validation against MATLAB `spm_nlsi_GN` / `spm_dcm_fmri_csd`; reuse of the `validation/` SPM bridge
**Confidence:** HIGH (grounded entirely in this codebase: VL engine source, Phase 28/29 docs, existing `validation/` files, Phase 31 recovery harness, and the verbatim S1-S4 pitfalls)

## Summary

Phase 32 cross-validates the Phase 28 Variational Laplace (VL) engine against MATLAB's actual
`spm_nlsi_GN` (driven via `spm_dcm_fmri_csd`) on a **prior-matched** spectral DCM problem. Almost
everything needed already exists. The VL engine is `run_variational_laplace` /
`run_variational_laplace_generic` in `src/pyro_dcm/inference/variational_laplace.py`; it already
exposes the three SPM-matching knobs (`hyperprior_mean=8.0`, `hyperprior_precision=128.0`,
`prior_mean_a_offset=a_mask/128`) that make S1/S2 avoidable. The result object
(`VariationalLaplaceResult`) carries `theta_post["A_free"]` (the free-parameter posterior mean to
compare against SPM `DCM.Ep.A`) and `free_energy: list[float]` (last element compared to SPM
`DCM.F`). The existing `validation/` bridge (`export_spectral_dcm_for_spm`,
`run_spm_spectral_dcm.m`, `load_spm_results`, `compute_free_param_comparison`,
`compare_model_ranking`) is reused verbatim. Phase 32 adds exactly two things per VLSPM-03: a new
`validation/run_vl_validation.py` orchestrator (a VL-path twin of the existing SVI-path
`run_spectral_dcm_validation`), and a new `compare_free_energies()` in
`validation/compare_results.py`.

The work is small and well-scoped: write the VL orchestrator, write `compare_free_energies()`, and
write an SPM-gated test (`tests/test_vl_spm_cross_validation.py`) marked `@pytest.mark.spm` +
`@pytest.mark.slow` that auto-skips via `check_matlab_available()`. The matched problem MUST use a
**reciprocal-edge N=2** ground-truth `A` (Phase 31 identifiability finding: lone off-diagonal /
feed-forward chains are CSD-indistinguishable from the empty graph). The C-order CSD round-trip test
(`tests/test_csd_corder_roundtrip.py`, 2 tests, `@pytest.mark.vl`) must be green before any
comparison run (VLSPM-03 / S4).

The single real design decision (an open question, NOT a locked decision) is **how the two engines
see the same CSD**: the existing SVI orchestrator exports BOLD and lets SPM compute CSD via its
internal MAR model while the Python side computes CSD via Welch — a documented 5-10% discrepancy.
For a clean prior-matched comparison the planner should make both engines consume the SAME CSD (see
Open Questions Q1). This is the highest-leverage planning decision.

**Primary recommendation:** Add `validation/run_vl_validation.py` (VL twin of
`run_spectral_dcm_validation`) + `compare_free_energies()` in `compare_results.py`; fit a
reciprocal-edge N=2 spectral problem with `run_variational_laplace(..., hyperprior_mean=8.0,
hyperprior_precision=128.0, prior_mean_a_offset=a_mask/128.0)`; compare `theta_post["A_free"]` vs
SPM `Ep_A` in free-parameter space (~10%), and `free_energy[-1]` vs `DCM.F` (~5%) on the SAME
matched problem; gate the test with `@pytest.mark.spm` + `skipif(not check_matlab_available())`.

## Standard Stack

This is an internal-integration phase; no new external libraries. The "stack" is the existing
codebase surface Phase 32 binds together.

### Core (the things Phase 32 wires together)
| Component | Location | Purpose | Phase 32 use |
|-----------|----------|---------|--------------|
| `run_variational_laplace` | `src/pyro_dcm/inference/variational_laplace.py:401` | Spectral-DCM VL entry point (wraps `_run_vl_generic` with `SpectralDCMForward`) | THE engine under test; call with SPM-matched priors |
| `run_variational_laplace_generic` | same module (re-exported `pyro_dcm.inference`) | Model-agnostic VL entry (used by Phase 31 harness) | alt entry; takes a `SpectralDCMForward` + `context={"freqs": freqs}` |
| `VariationalLaplaceResult` | same module:233 | Result dataclass | read `theta_post["A_free"]`, `free_energy`, `sigma_post`, `predicted_csd` |
| `SpectralDCMForward` | `src/pyro_dcm/inference/forward_models.py` | CSD forward model protocol impl | passed to generic entry; `context={"freqs": freqs}` |
| `simulate_spectral_dcm` | `src/pyro_dcm/simulators/spectral_simulator.py:37` | Analytic CSD generator; returns `{"csd": (F,N,N) c128, "freqs": (F,) f64, ...}` | generate the matched-problem ground-truth CSD |
| `make_sparse_ground_truth_A` | `benchmarks/bmr_recovery.py:47` | Builds sparse reciprocal-edge `A` (diag `-0.5`, edges `0.15`), validates stability | build the N=2 reciprocal ground truth |
| `export_spectral_dcm_for_spm` | `validation/export_to_mat.py:160` | Writes SPM12 DCM struct (induced=1, analysis=CSD) from BOLD | reuse (REUSE-mandated by VLSPM-03) |
| `run_spm_spectral_dcm.m` | `validation/matlab_scripts/run_spm_spectral_dcm.m` | MATLAB batch: `spm_dcm_fmri_csd`; saves `results.Ep_A`, `results.Cp`, `results.F` (+ `Ep_transit`,`Ep_decay`,`Hc`,`Hz`) | reuse; invoked via subprocess like existing runner |
| `load_spm_results` | `validation/compare_results.py:23` | Loads `results` struct: `Ep_A`, `F`, optional `Cp`,`Ep_C`,`Ep_transit`,... | reuse to read SPM side |
| `compute_free_param_comparison` | `validation/compare_results.py:249` | Free-parameter-space A comparison (delegates to `compare_posterior_means`, hybrid rel/abs metric) | reuse for `Ep` ~10% comparison (S1/S2) |
| `compare_model_ranking` | `validation/compare_results.py:187` | Pairwise ranking agreement (`spm_F` vs `pyro_elbo`); pass >= 0.80 | reuse for ranking; substitute VL `free_energy` for `pyro_elbo` |
| `check_matlab_available` | `validation/run_validation.py:67` | Subprocess MATLAB probe; returns bool | reuse for pytest skip gate |

### Supporting (already present)
| Component | Location | Use |
|-----------|----------|-----|
| `MATLAB_PATH` / `MATLAB_SCRIPTS_DIR` / `DEFAULT_OUTPUT_DIR` | `validation/run_validation.py:58-64` | constants the new VL orchestrator should import/mirror. `MATLAB_PATH = "C:/Program Files/MATLAB/R2022a/bin/matlab"` |
| `config.py MATLAB_PATH` | repo-root `config.py:52-57` (env-overridable; per 29-VERIFICATION) | env-overridable variant; prefer this over the hardcoded one if the planner wants override support |
| `offdiag_indices` | `benchmarks/bmr_recovery.py:113` | C-order off-diagonal flat indices (for any BMR/ranking over edges) |
| `bmr_tensors_from_vl_result` | `benchmarks/bmr_recovery.py:140` | extracts full `A_free` covariance sub-block + zero-mean prior from a VL result (only if ranking-via-BMR is desired) |
| pytest markers | `pyproject.toml:85-92` | `spm` (L87), `slow` (L86), `vl` (L92) all registered |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `run_variational_laplace` (spectral wrapper) | `run_variational_laplace_generic` + `SpectralDCMForward` | identical engine; the spectral wrapper is simpler (no manual forward-model/context construction) and already threads the three SPM-matching kwargs. Phase 31 used the generic form. Either is fine. |
| Reusing `run_spectral_dcm_validation` SVI path | New `run_vl_validation.py` | VLSPM-03 MANDATES a new `run_vl_validation.py`; the existing function wires the **SVI guide**, not VL. Do NOT retrofit VL into the SVI function. |

**Installation:** none. No new dependencies. (Optionally `git` branch `gsd/phase-32-spm12-cross-validation`.)

## Architecture Patterns

### Recommended file layout (per VLSPM-03 — minimal extension)
```
validation/
├── run_vl_validation.py     # NEW: VL-path orchestrator (twin of run_spectral_dcm_validation)
├── compare_results.py       # EXTEND: add compare_free_energies()
├── run_validation.py        # REUSE: check_matlab_available, MATLAB_* constants
├── export_to_mat.py         # REUSE: export_spectral_dcm_for_spm
└── matlab_scripts/
    └── run_spm_spectral_dcm.m   # REUSE: already saves Ep_A, Cp, F
tests/
└── test_vl_spm_cross_validation.py   # NEW: @pytest.mark.spm + slow, auto-skip
```

### Pattern 1: The VL orchestrator (`run_vl_validation.py`)
**What:** A function (e.g. `run_vl_spectral_dcm_validation(...) -> dict`) structurally mirroring
`run_spectral_dcm_validation` (`validation/run_validation.py:358-540`) but replacing the SVI block
(`create_guide` / `run_svi` / `extract_posterior_params`) with a VL fit.

**Reference (the exact existing SVI block being replaced), `run_validation.py:487-512`:**
```python
freqs = default_frequency_grid(TR, n_freqs=32)
observed_csd = bold_to_csd_torch(bold_ts, fs=1.0 / TR, freqs=freqs)
a_mask_torch = torch.ones(N, N, dtype=torch.float64)
model_args = (observed_csd, freqs, a_mask_torch)
guide = create_guide(spectral_dcm_model)
svi_result = run_svi(spectral_dcm_model, guide, model_args, num_steps=..., lr=0.01, ...)
posterior = extract_posterior_params(guide, model_args)
pyro_A_free = posterior["median"]["A_free"].detach().cpu().numpy()
```

**VL replacement (verified against the VLBMR-01 fit pattern, `tests/test_bmr_vlbmr01_recovery.py:106-124`):**
```python
# matched-problem ground truth: RECIPROCAL edges (Phase 31 identifiability finding)
A_true = make_sparse_ground_truth_A(N, present_edges=[(0, 1), (1, 0)])  # N=2 reciprocal
sim = simulate_spectral_dcm(A_true, TR=2.0, n_freqs=32, seed=seed)
observed_csd = sim["csd"].to(torch.complex128)        # (F, N, N) complex128
freqs = sim["freqs"].double()                          # (F,) float64
a_mask = torch.ones(N, N, dtype=torch.float64)

result = run_variational_laplace(
    observed_csd=observed_csd,
    freqs=freqs,
    a_mask=a_mask,
    N=N,
    max_iter=64,
    hyperprior_mean=8.0,                       # SPM hE (S2)
    hyperprior_precision=128.0,                # SPM ihC (S2)
    prior_mean_a_offset=a_mask / 128.0,        # SPM A prior-mean offset (S2)
)
vl_A_free = result.theta_post["A_free"].detach().cpu().numpy()   # compare vs SPM Ep_A (S1)
vl_free_energy = result.free_energy[-1]                          # compare vs SPM DCM.F (S3-aware)
```

**Result-object contract (`VariationalLaplaceResult`, source:233-264) — exact attribute names:**
- `theta_post: dict[str, torch.Tensor]` with keys `"A_free"`, `"A"`, `"noise_a"`, `"noise_b"`,
  `"noise_c"`, `"P_transit"`, `"P_decay"`, `"P_epsilon"` (set at source:784-793). **`"A_free"` is
  the free-parameter posterior mean** to compare against SPM `Ep.A`. `"A"` is the parameterized
  matrix (diag `-exp(x)/2`) — do NOT compare `"A"` against SPM `Ep.A`.
- `free_energy: list[float]` — per-iteration trace; **use `free_energy[-1]`** as the VL free energy.
- `sigma_post: torch.Tensor` — full posterior covariance in full parameter space (`A_free` packed
  FIRST; leading `(N*N, N*N)` block is the A_free covariance). **Never compare element-wise vs SPM
  `Cp` (S3).**
- `converged: bool`, `n_iterations: int`, `predicted_csd: torch.Tensor (F,N,N)`,
  `n_reduced_params: int`.

### Pattern 2: `compare_free_energies()` (new, in `compare_results.py`)
**What:** A relative-tolerance comparison of a single matched-problem free energy. Mirror the
docstring/return style of the sibling comparators (`compare_results.py:126-275`).
**Signature recommendation:**
```python
def compare_free_energies(
    vl_free_energy: float,
    spm_F: float,
    rel_tolerance: float = 0.05,
) -> dict:
    """Compare VL free energy vs SPM DCM.F on the SAME matched problem.

    VLSPM-02: relative tolerance ~5% on the matched problem ONLY. This function
    must NEVER be used to compare absolute F across DIFFERENT models (pitfall S3);
    cross-model comparison goes through compare_model_ranking (relative ranking).
    """
    rel_err = abs(vl_free_energy - spm_F) / max(abs(spm_F), 1e-12)
    return {
        "vl_free_energy": float(vl_free_energy),
        "spm_F": float(spm_F),
        "relative_error": float(rel_err),
        "within_tolerance": bool(rel_err < rel_tolerance),
    }
```

### Pattern 3: SPM-gated test (auto-skip)
**What:** Module-level `pytestmark` exactly as `tests/test_spm_spectral_dcm_validation.py:29-36`:
```python
from validation.run_validation import check_matlab_available  # reuse
pytestmark = [
    pytest.mark.spm,
    pytest.mark.slow,
    pytest.mark.skipif(
        not check_matlab_available(),
        reason="MATLAB/SPM12 not available",
    ),
]
```
(`tests/test_model_ranking_validation.py:63-66` uses an autouse fixture variant — either works; the
module-level `pytestmark` is the cleaner, more common pattern in this repo.)

### Anti-Patterns to Avoid
- **Comparing `result.theta_post["A"]` vs SPM `Ep.A`:** `"A"` is parameterized (diag `-exp(x)/2`);
  SPM `Ep.A` is free. Compare `theta_post["A_free"]` (S1).
- **Comparing element-wise `sigma_post` vs SPM `Cp`:** forbidden by VLSPM-02/S3. SPM `Cp` is full
  (incl. hyperparameters) and SVD-reduced differently; only ranking + matched-F are valid.
- **Comparing absolute F across models:** forbidden by S3. Use `compare_model_ranking` (relative).
- **Feed-forward / lone off-diagonal ground truth:** unidentifiable in spectral DCM (Phase 31).
  Use reciprocal edges.
- **Retrofitting VL into `run_spectral_dcm_validation`:** that function is the SVI path; VLSPM-03
  mandates a separate `run_vl_validation.py`.
- **Default (non-matched) priors when comparing F:** "Never compare F values between SPM and
  dcm_pytorch runs that used different hyperpriors" (S2).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Free-parameter A comparison | new element-wise diff | `compute_free_param_comparison` (`compare_results.py:249`) | already implements the hybrid rel/abs metric and the same-space contract (S1) |
| Model-ranking agreement | new pairwise loop | `compare_model_ranking` (`compare_results.py:187`) | pairwise agreement-rate already implemented; substitute VL `free_energy` for `pyro_elbo` |
| SPM result loading | new `loadmat` parsing | `load_spm_results` (`compare_results.py:23`) | handles nested struct access + optional `Cp`/`F`/`Ep_A` |
| SPM DCM export | new `.mat` writer | `export_spectral_dcm_for_spm` (`export_to_mat.py:160`) | sets `induced=1`, `analysis='CSD'`, float64 casts, microtime U |
| MATLAB availability gate | new probe | `check_matlab_available` (`run_validation.py:67`) | subprocess probe already used by all SPM tests |
| Reciprocal ground-truth A | new builder | `make_sparse_ground_truth_A` (`bmr_recovery.py:47`) | builds + stability-checks sparse reciprocal A; unambiguous true edges |
| C-order CSD index map | new index math | `tests/test_csd_corder_roundtrip.py` (already green) | locks the S4 contract; just keep it green |

**Key insight:** Per VLSPM-03 the deliverable surface is intentionally tiny — one new orchestrator
file + one new comparator function + one new test. Everything else is reuse. Resist building new
loaders, exporters, or metrics.

## Common Pitfalls

These are the named S1-S4 the success criteria hinge on, quoted verbatim from
`.planning/research/v0.7.0/PITFALLS.md`. The planner MUST surface S1-S4 in the plan's `must_haves`.

### Pitfall S1: SPM12 Parameter Parameterization Mismatch (A Matrix Diagonal)
**Verbatim (PITFALLS.md:83-105):**
> **What goes wrong:** SPM12 stores `DCM.Ep.A` as free parameters (not the parameterised A matrix).
> For the diagonal, SPM's actual self-connection is `-exp(Ep.A_ii) / 2`, matching
> `parameterize_A()` in this codebase. When comparing posterior means, naively subtracting
> `spm_A - our_A_free` compares apples to oranges if one side has applied the transform and the
> other has not. This doubles or halves apparent errors on self-connections.
>
> **Prevention:** Always compare in the SAME space. The cleaner choice is free-parameter space
> (compare `our A_free` vs `SPM Ep.A` directly), because the nonlinear transform amplifies small
> differences in a direction that is hard to normalise. Document which space is used... and assert
> the transform has been applied (or not) consistently before computing any error metric. Add a
> unit test: simulate A_free, apply `parameterize_A`, verify the diagonal formula matches SPM's
> `-exp(x)/2` convention explicitly.

**Phase 32 action:** compare `result.theta_post["A_free"]` (NOT `"A"`) vs SPM `Ep_A` via
`compute_free_param_comparison` at ~10%.

### Pitfall S2: SPM12 Prior Conventions — Scale, Mean, and Hyperprior
**Verbatim (PITFALLS.md:109-140), incl. the table:**
> | Parameter | SPM12 value | dcm_pytorch default |
> |-----------|-------------|---------------------|
> | A prior variance (pC.A) | `a_mask / 64` (= 1/64 where mask=1) | 1/64 (matches) |
> | A prior mean offset | `a_mask / 128` | 0 (does NOT match) |
> | Hyperprior mean (hE) | 8.0 for spectral DCM | computed from data: -log(var(y))+4 |
> | Hyperprior precision (ihC) | 128.0 for spectral DCM | exp(4) ≈ 54.6 |
> | Hemodynamic prior variance | 1/256 | 1/256 (matches) |
>
> **Prevention:** When running SPM12 cross-validation, use `hyperprior_mean=8.0`,
> `hyperprior_precision=128.0`, and `prior_mean_a_offset=a_mask / 128` in `run_variational_laplace`
> / `_run_vl_generic` (all three parameters are already exposed in the API as of commit e1934e1).
> ... Never compare F values between SPM and dcm_pytorch runs that used different hyperpriors.

**Phase 32 action:** pass all three kwargs on the VL fit. (Verified present in the signature,
`variational_laplace.py:413-415`.)

### Pitfall S3: Free Energy F Is Not Directly Comparable Across Implementations
**Verbatim (PITFALLS.md:144-171):**
> **What goes wrong:** SPM's `DCM.F` and dcm_pytorch's `result.free_energy[-1]` are both lower
> bounds on log model evidence, but they are computed with different approximations: SPM includes
> log-det terms from the full (non-diagonal) posterior covariance computed via the full parameter
> vector (including hyperparameters); dcm_pytorch's L_1 + L_2 + L_3 decomposition operates in the
> SVD-reduced subspace... The absolute difference can easily be tens to hundreds of nats even when
> posterior means agree.
>
> **Prevention:** NEVER use absolute F values for the SPM cross-validation criterion. The valid
> comparison is MODEL RANKING: on a set of synthetic conditions where the correct model is clearly
> better (has a true connection vs lacks it), both engines should agree on which model has higher
> evidence. For ranking tests, require a separation gap of > 3 nats (equivalent to Bayes factor
> ~20) before counting a disagreement as meaningful.

**Phase 32 nuance (reconcile with VLSPM-02):** VLSPM-02 says VL `free_energy` ~= SPM `DCM.F` within
~5% **on the matched problem**. S3 forbids absolute-F comparison **across different models**. These
are consistent: the ~5% check is for the SINGLE prior-matched problem (same priors, same data, same
model) where the two F bounds should be close *because the priors are matched*; the never-compare
rule applies to comparing F between DIFFERENT models — there, use `compare_model_ranking`
(relative). The plan must keep these two comparisons distinct: `compare_free_energies()` for the
one matched problem; `compare_model_ranking()` for cross-model ranking. NOTE: S3 itself warns the
matched-problem F gap "can easily be tens to hundreds of nats"; the ~5% matched-F target in
VLSPM-02 is therefore an ASPIRATION that depends on the prior-matching being exact and the SAME-CSD
path (Open Q1). If the 5% target proves unreachable even when priors are matched, the defensible
fallback is to keep the ranking-agreement + ~10% Ep checks as the hard gates and report the
matched-F gap descriptively (see Open Questions Q2). The planner should treat the ~5% as the goal
but make the test's HARD assertion the ranking agreement + Ep tolerance, not necessarily the 5% F.

### Pitfall S4: MATLAB/Python Data Layout — Column-Major vs Row-Major
**Verbatim (PITFALLS.md:175-202):**
> **What goes wrong:** MATLAB stores matrices in column-major (Fortran) order; Python/NumPy/PyTorch
> use row-major (C) order. `scipy.io.loadmat` partially handles this by transposing 2D arrays, but
> complex CSD arrays and 3D tensors require explicit attention. This project already encountered a
> C-order bug in `csd_precision.py` (fixed in commit 64e326f): the index mapping
> `j = idx % N; i = (idx // N) % N; w = idx // (N*N)` must match the PyTorch `.reshape(-1)`
> convention (C-order, fastest index last, which means j varies fastest)...
>
> **Prevention:** When loading any SPM result via `scipy.io.loadmat`, explicitly check the shape
> and apply `.T` where MATLAB column-major convention has transposed a matrix vs the expected
> layout. ... print `A_spm[0,1]` and `A_spm[1,0]` for a known asymmetric ground truth and confirm
> the sign pattern matches expectation before running any comparison. Add a round-trip test: export
> a known asymmetric matrix to .mat, reload with `loadmat`, and assert element `[i,j]` matches the
> expected value.

**Phase 32 action:** the round-trip regression already exists
(`tests/test_csd_corder_roundtrip.py`, 2 `@pytest.mark.vl` tests, both green per 29-VERIFICATION).
VLSPM-03 requires it green BEFORE any comparison run — the plan should add an explicit gate (run
`pytest tests/test_csd_corder_roundtrip.py -m vl` first). For the N=2 reciprocal A (asymmetric only
if edge strengths differ; reciprocal-equal edges are symmetric), add the `A_spm[0,1]` vs `A_spm[1,0]`
sanity print, and consider asymmetric edge strengths (e.g. `(0,1)=0.15`, `(1,0)=0.10`) so the layout
check has teeth.

### (Context) Related pitfalls the planner should be aware of
- **C1 (PITFALLS.md:17-48):** Laplace overconfidence breaks ABSOLUTE BMR pruning — relevant if the
  plan does any BMR-based ranking; use relative ranking only (already handled by `rank_connections`).
- **N4 (PITFALLS.md:492-517):** VL finds the prior-nearest local mode; spectral H(w) is non-convex.
  At N=2 reciprocal this is benign (Phase 30: sign 1.0, A-RMSE ~0.04), but if Ep disagrees, suspect
  a local optimum before suspecting an engine bug — `initial_p` enables multi-restart.
- **N3 (PITFALLS.md:462-488):** float64 everywhere; cast every array to `np.float64` before
  `savemat` (the existing exporters already do this).

## Code Examples

### Generate the matched reciprocal-edge N=2 problem (verified pattern)
```python
# Source: tests/test_bmr_vlbmr01_recovery.py:106-109 + benchmarks/bmr_recovery.py:47
from benchmarks.bmr_recovery import make_sparse_ground_truth_A
from pyro_dcm.simulators.spectral_simulator import simulate_spectral_dcm

N = 2
A_true = make_sparse_ground_truth_A(N, present_edges=[(0, 1), (1, 0)])  # reciprocal
sim = simulate_spectral_dcm(A_true, TR=2.0, n_freqs=32, seed=42)
observed_csd = sim["csd"].to(torch.complex128)   # (F, 2, 2) complex128
freqs = sim["freqs"].double()                    # (F,) float64
```

### Prior-matched VL fit (the S2 fix; verified signature)
```python
# Source: src/pyro_dcm/inference/variational_laplace.py:401-470
from pyro_dcm.inference import run_variational_laplace
a_mask = torch.ones(N, N, dtype=torch.float64)
result = run_variational_laplace(
    observed_csd=observed_csd, freqs=freqs, a_mask=a_mask, N=N, max_iter=64,
    hyperprior_mean=8.0, hyperprior_precision=128.0,
    prior_mean_a_offset=a_mask / 128.0,
)
vl_A_free = result.theta_post["A_free"].detach().cpu().numpy()   # (N, N) free params (S1)
vl_F = result.free_energy[-1]                                    # float
```

### SPM side (reuse) + comparisons
```python
# Source: validation/run_validation.py:457-516 (export+subprocess+load), compare_results.py
from validation.export_to_mat import export_spectral_dcm_for_spm
from validation.compare_results import load_spm_results, compute_free_param_comparison
# ... export BOLD/CSD, run subprocess([MATLAB_PATH, "-batch", ...]) on run_spm_spectral_dcm ...
spm = load_spm_results(results_path)              # spm["Ep_A"], spm["F"], spm.get("Cp")
ep_cmp = compute_free_param_comparison(vl_A_free, spm["Ep_A"], tolerance=0.10)   # S1/S2, ~10%
f_cmp  = compare_free_energies(vl_F, spm["F"], rel_tolerance=0.05)               # NEW, matched-F
```

### Model-ranking agreement across models (S3-safe; reuse)
```python
# Source: validation/compare_results.py:187-246
# Build >=2 model masks (e.g. correct reciprocal vs diagonal-only), fit VL on each,
# collect {"spm_F": spm_F_k, "pyro_elbo": vl_F_k} and require agreement_rate >= 0.80.
from validation.compare_results import compare_model_ranking
ranking = compare_model_ranking([
    {"spm_F": spm_F_correct, "pyro_elbo": vl_F_correct},
    {"spm_F": spm_F_diag,    "pyro_elbo": vl_F_diag},
])
assert ranking["agreement_rate"] >= 0.80
```
Note: `compare_model_ranking`'s key is literally `"pyro_elbo"` (higher = better); VL `free_energy`
(higher = better log-evidence bound) substitutes directly. The planner may rename the key in a new
helper for clarity, but reuse is allowed.

## State of the Art

| Old Approach (existing SVI path) | Phase 32 (VL path) | Why it changes |
|----------------------------------|--------------------|----------------|
| `run_spectral_dcm_validation` uses `create_guide`/`run_svi` (mean-field SVI) | new `run_vl_validation.py` uses `run_variational_laplace` | VL is the Phase 28 engine being validated; SVI was the v0.1-v0.5 inference |
| Default priors; tolerance relaxed to 15% (SVI + MAR-vs-Welch CSD) | SPM-matched priors (hE=8, ihC=128, offset=a_mask/128); ~10% Ep | prior matching removes the S2 bias so a tighter tolerance is justified |
| BOLD exported; SPM computes CSD via MAR; Python via Welch (5-10% gap noted) | SHOULD feed SAME CSD to both (Open Q1) | apples-to-apples matched problem |
| Ranking via `pyro_elbo` (negative SVI loss) | Ranking via VL `free_energy[-1]` | VL F is a proper 3-term free energy, not an ELBO |

**Deprecated/outdated for Phase 32:** the SVI orchestrator's 15% tolerance + "csd_method_note" are
SVI-specific and should NOT be copied as the VL acceptance bar.

## Open Questions

1. **Same-CSD path (HIGHEST LEVERAGE).** The existing SVI orchestrator exports BOLD and lets SPM
   compute CSD internally (MAR), while the Python side computes CSD (Welch) — a documented 5-10%
   discrepancy. For a clean prior-matched comparison, both engines should consume the SAME CSD.
   - What we know: `simulate_spectral_dcm` produces an analytic `(F,N,N)` CSD; the VL engine
     consumes it directly. SPM's `spm_dcm_fmri_csd` normally recomputes CSD from BOLD.
   - What's unclear: whether to (a) export BOLD and accept SPM's MAR-CSD (simplest, reuses
     `export_spectral_dcm_for_spm` unchanged, but adds CSD-estimation discrepancy), or (b) inject
     the Python CSD into the SPM DCM struct so SPM skips its CSD estimation (cleaner apples-to-apples
     but requires a MATLAB-side change to `run_spm_spectral_dcm.m` to set `DCM.Y.csd`/`DCM.Y.Hz` and
     bypass `spm_dcm_fmri_csd_data`). PITFALLS.md Open Q3 (lines 581-585) frames exactly this choice.
   - Recommendation: start with path (a) for VLSPM-01 (>=1 problem, minimal MATLAB change), and if
     the ~5% matched-F or ~10% Ep targets fail, escalate to path (b). The plan should make path-(a)
     the first task and path-(b) a conditional follow-up, not assume (b) up front.

2. **Is the ~5% matched-F a hard gate or an aspiration?** S3 explicitly warns the matched-problem F
   gap "can easily be tens to hundreds of nats." VLSPM-02 asks for ~5%. These can only both hold if
   prior-matching is exact AND the CSD is identical (Open Q1 path b) AND the SVD-reduction
   normalization differences are small.
   - Recommendation: make the HARD test assertions be (i) Ep within ~10% in free-parameter space
     and (ii) model-ranking agreement >= 0.80; treat matched-F within ~5% as a target that, if
     missed, is REPORTED (printed + recorded in `VALIDATION_REPORT.md`) rather than failing the
     suite. Confirm this framing with the user if a strict 5% gate is required by acceptance.

3. **How many models for ranking?** `compare_model_ranking` needs >=2 scenarios. The existing
   ranking test uses correct / missing-connection / diagonal-only masks
   (`test_model_ranking_validation.py:5-8`). For N=2 reciprocal, natural masks are: full reciprocal
   (correct), diagonal-only (wrong). That gives 1 pairwise comparison — enough to show agreement but
   thin. Recommendation: include >=3 masks (e.g. correct reciprocal, single-direction, diagonal-only)
   for >=3 pairwise comparisons; note that single-direction may be near-unidentifiable (Phase 31),
   which is itself a useful ranking signal (it should rank near diagonal-only).

4. **N=2 reciprocal symmetry vs S4 teeth.** Reciprocal-equal edges make A symmetric, so a
   transpose bug wouldn't surface. Recommendation: use asymmetric edge strengths (e.g. `(0,1)=0.15`,
   `(1,0)=0.10`) for the matched problem so the S4 `A_spm[0,1]` vs `A_spm[1,0]` check is meaningful,
   while staying stable (verify `make_sparse_ground_truth_A` stability check passes).

5. **MATLAB R2022a path / SPM12 path are hardcoded.** `run_validation.py:58` hardcodes
   `C:/Program Files/MATLAB/R2022a/bin/matlab`; `run_spm_spectral_dcm.m:18` hardcodes
   `C:/Users/aman0087/Documents/Github/spm12`. `config.py` (repo root) exposes an env-overridable
   `MATLAB_PATH`. Recommendation: the new orchestrator should import `MATLAB_PATH` from `config.py`
   (env-overridable) rather than re-hardcoding, so the suite is portable; keep the SPM12 addpath as
   is (it's the local install) but document it.

## Sources

### Primary (HIGH confidence — all in-repo, read directly)
- `src/pyro_dcm/inference/variational_laplace.py` — `run_variational_laplace` signature (L401-470,
  three SPM-match kwargs L413-415), `VariationalLaplaceResult` (L233-264), `theta_post` keys
  (L784-793), `free_energy` (L243/259).
- `src/pyro_dcm/inference/__init__.py` — public exports (`run_variational_laplace`,
  `run_variational_laplace_generic`, `SpectralDCMForward`, `VariationalLaplaceResult`).
- `validation/run_validation.py` — `check_matlab_available` (L67), `run_spectral_dcm_validation`
  (L358-540, the SVI twin to mirror), `MATLAB_PATH`/`MATLAB_SCRIPTS_DIR`/`DEFAULT_OUTPUT_DIR` (L58-64).
- `validation/compare_results.py` — `load_spm_results` (L23), `compare_posterior_means` (L126),
  `compare_model_ranking` (L187), `compute_free_param_comparison` (L249).
- `validation/export_to_mat.py` — `export_spectral_dcm_for_spm` (L160-240).
- `validation/matlab_scripts/run_spm_spectral_dcm.m` — saves `results.Ep_A`, `results.Cp`,
  `results.F` (+ `Ep_transit`,`Ep_decay`,`Hc`,`Hz`) (L72-92).
- `.planning/research/v0.7.0/PITFALLS.md` — S1 (L83-105), S2 (L109-140), S3 (L144-171), S4
  (L175-202); C1 (L17-48), N3/N4 (L462-517); Open Q3 (L581-585).
- `tests/test_csd_corder_roundtrip.py` — the S4 round-trip regression (2 `@pytest.mark.vl` tests).
- `tests/test_bmr_vlbmr01_recovery.py` — canonical reciprocal-edge spectral VL fit pattern (L106-124).
- `benchmarks/bmr_recovery.py` — `make_sparse_ground_truth_A` (L47), `offdiag_indices` (L113),
  `bmr_tensors_from_vl_result` (L140).
- `tests/test_spm_spectral_dcm_validation.py` — SPM-gated test pattern (`pytestmark` L29-36).
- `.planning/phases/28-variational-laplace-engine/28-CONSOLIDATION-SUMMARY.md` — VL engine as-built.
- `.planning/phases/29-vl-validation-infra-bmr-rank/29-VERIFICATION.md` — BenchmarkConfig VL fields,
  C-order test green, `vl` marker, `config.py MATLAB_PATH` env-overridable.
- `.planning/ROADMAP.md` L587-611 — Phase 32 goal + success criteria; L465 prior-matching values.
- `benchmarks/config.py` — `BenchmarkConfig` VL fields (`max_iter`, `hyperprior_mean`,
  `hyperprior_precision`, `prior_mean_a_offset`).
- `pyproject.toml` L85-92 — `slow`/`spm`/`vl` markers registered.

### Secondary / Tertiary
- None — no WebSearch/Context7 needed; this is an internal-integration phase fully specified by the
  codebase and planning docs.

## Metadata

**Confidence breakdown:**
- Standard stack / API surface: HIGH — exact signatures, attribute names, and file:line read from source.
- Architecture (file layout + 3 patterns): HIGH — mirrors existing SVI orchestrator + VLBMR-01 fit.
- Pitfalls S1-S4: HIGH — quoted verbatim from `.planning/research/v0.7.0/PITFALLS.md`.
- Same-CSD design decision (Open Q1) + 5%-F gate framing (Open Q2): MEDIUM — these are genuine,
  unresolved design choices that the plan must make explicit; flagged honestly rather than guessed.

**Research date:** 2026-06-11
**Valid until:** stable (internal code) — re-verify only if `variational_laplace.py`,
`validation/`, or `benchmarks/bmr_recovery.py` change before planning.
