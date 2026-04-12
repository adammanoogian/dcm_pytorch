# Phase 10 Research: Guide Variants

**Researched:** 2026-04-12
**Pyro version:** 1.9.1 (installed, verified via `pyro.__version__`)
**Overall confidence:** HIGH -- all findings verified against installed Pyro source via `inspect.signature` and runtime tests

---

## 1. Autoguide Constructor Signatures (Verified)

All signatures verified against installed pyro-ppl 1.9.1 using `inspect.signature()`.

### 1.1 AutoDelta

```python
AutoDelta(model, init_loc_fn=init_to_median, *, create_plates=None)
```

- **MRO:** AutoDelta -> AutoGuide -> PyroModule -> Module
- **init_scale:** NOT accepted (raises TypeError)
- **Methods:** `median()` (own), NO `quantiles()`
- **Variational params:** 1 per latent (Delta locations only)
- **Behavior with Predictive:** All N samples are identical (deterministic)
- **Mean-field:** Yes (trivially -- point estimate)

### 1.2 AutoNormal

```python
AutoNormal(model, *, init_loc_fn=init_to_feasible, init_scale=0.1, create_plates=None)
```

- **MRO:** AutoNormal -> AutoGuide -> PyroModule -> Module
- **init_scale:** YES (default 0.1; project uses 0.01)
- **Methods:** `median()` (own), `quantiles()` (own)
- **Variational params:** 2 per latent (loc + scale)
- **Param store keys:** `AutoNormal.locs.<site>`, `AutoNormal.scales.<site>`
- **Mean-field:** YES (independent Normal per site)

### 1.3 AutoLowRankMultivariateNormal

```python
AutoLowRankMultivariateNormal(model, init_loc_fn=init_to_median, init_scale=0.1, rank=None)
```

- **MRO:** AutoLowRankMVN -> AutoContinuous -> AutoGuide -> PyroModule
- **init_scale:** YES (default 0.1)
- **rank:** `None` = auto `ceil(sqrt(latent_dim))`; context decision says use rank=2
- **Methods:** `median()`, `quantiles()` (inherited from AutoContinuous), `get_posterior()` (own)
- **Variational params:** D (loc) + D (cov_diag) + D*rank (cov_factor)
- **Mean-field:** NO (captures correlations via low-rank factor)

### 1.4 AutoMultivariateNormal

```python
AutoMultivariateNormal(model, init_loc_fn=init_to_median, init_scale=0.1)
```

- **MRO:** AutoMVN -> AutoContinuous -> AutoGuide -> PyroModule
- **init_scale:** YES (default 0.1)
- **Methods:** `median()`, `quantiles()` (inherited from AutoContinuous), `get_posterior()` (own)
- **Variational params:** D (loc) + D*(D+1)/2 (scale_tril) -- quadratic scaling
- **Mean-field:** NO (full covariance)

### 1.5 AutoIAFNormal

```python
AutoIAFNormal(model, hidden_dim=None, init_loc_fn=None, num_transforms=1, **init_transform_kwargs)
```

- **MRO:** AutoIAFNormal -> AutoNormalizingFlow -> AutoContinuous -> AutoGuide
- **init_scale:** NOT a named param, but absorbed through `**init_transform_kwargs` (passed to `affine_autoregressive`). Behavior unclear -- do NOT rely on this.
- **hidden_dim:** Singular (not `hidden_dims`). Set to None = latent_dim. Passed as `hidden_dims=hidden_dim` to underlying `affine_autoregressive`.
- **num_transforms:** Number of stacked IAF layers. Context decision: 2.
- **init_loc_fn:** Deprecated (warning emitted). Do not pass.
- **Methods:** `median()`, `quantiles()` (inherited from AutoContinuous)
- **Mean-field:** NO (autoregressive flow captures arbitrary correlations)

**IMPORTANT:** The context decision says `hidden_dims=[20]` but the constructor takes `hidden_dim` (singular) which is an int or None, not a list. The underlying `affine_autoregressive` does accept a list for `hidden_dims`, but `AutoIAFNormal` converts the singular `hidden_dim` int to the `hidden_dims` kwarg. Passing `hidden_dims=[20]` through kwargs IS possible (it gets absorbed by `**init_transform_kwargs`), but this is fragile. Recommend using `hidden_dim=20` (singular int) instead.

### 1.6 AutoLaplaceApproximation

```python
AutoLaplaceApproximation(model, init_loc_fn=init_to_median)
```

- **MRO:** AutoLaplace -> AutoContinuous -> AutoGuide -> PyroModule
- **init_scale:** NOT accepted (raises TypeError)
- **Methods:** `laplace_approximation(*args, **kwargs)` -> returns `AutoMultivariateNormal`, `get_posterior()` -> returns Delta (MAP), `median()`, `quantiles()` from AutoContinuous
- **Variational params:** D (loc only, Delta distribution for MAP)
- **Mean-field:** Not applicable (two-phase: MAP then Hessian)

**Two-phase workflow (verified):**
1. Phase 1: Run SVI with AutoLaplaceApproximation as guide (optimizes MAP location via Delta distribution)
2. Phase 2: Call `guide.laplace_approximation(*model_args)` -- computes Hessian at MAP, returns `AutoMultivariateNormal` with loc=MAP and scale_tril from inverse Hessian
3. Phase 3: Use returned AutoMultivariateNormal for all posterior queries (median, quantiles, Predictive sampling)

**Integration with run_svi:** The MAP optimization IS the SVI training (Delta distribution guide). The `laplace_approximation()` call is a post-processing step that must be called after SVI completes, before posterior extraction. The returned AutoMultivariateNormal is a NEW guide with FROZEN parameters -- it cannot be further optimized.

---

## 2. init_scale Compatibility Matrix

| Guide | Accepts init_scale | Default | How init_scale Works |
|-------|-------------------|---------|---------------------|
| AutoDelta | NO | N/A | Point estimate, no scale concept |
| AutoNormal | YES | 0.1 | Initial std for each independent Normal |
| AutoLowRankMVN | YES | 0.1 | Init scale for cov_diag and cov_factor |
| AutoMultivariateNormal | YES | 0.1 | Init scale for scale_tril diagonal |
| AutoIAFNormal | ABSORBED* | N/A | Passes through `**init_transform_kwargs`; unclear effect |
| AutoLaplaceApprox | NO | N/A | MAP point, no scale init needed |

*AutoIAFNormal absorbs init_scale via kwargs but it's not documented behavior.

**Implication for create_guide factory:** The factory cannot blindly pass `init_scale=0.01` to all guide constructors. It must:
- Pass `init_scale=0.01` to AutoNormal, AutoLowRankMVN, AutoMVN
- Skip init_scale for AutoDelta, AutoLaplaceApproximation
- For AutoIAFNormal: do NOT pass init_scale (unclear behavior)

---

## 3. ELBO Variant Signatures (Verified)

### 3.1 Trace_ELBO

```python
Trace_ELBO(
    num_particles=1,
    max_plate_nesting=inf,
    vectorize_particles=False,
    strict_enumeration_warning=True,
    ...
)
```

- **MRO:** Trace_ELBO -> ELBO -> object
- **Compatible with all guides:** YES
- **Current usage:** Hardcoded in `run_svi` with `num_particles=1`

### 3.2 TraceMeanField_ELBO

```python
TraceMeanField_ELBO(
    num_particles=1,
    max_plate_nesting=inf,
    vectorize_particles=False,
    strict_enumeration_warning=True,
    ...
)
```

- **MRO:** TraceMeanField_ELBO -> Trace_ELBO -> ELBO -> object
- **Same constructor as Trace_ELBO** (inherits parameters)
- **Benefit:** Uses analytic KL divergences when available (tighter gradient estimates)
- **Requirement (per docs):** Guide must have mean-field structure and all latent vars must be reparameterized

**CRITICAL FINDING -- TraceMeanField_ELBO does NOT enforce mean-field requirement at runtime.**

Verified by testing:
- AutoMultivariateNormal + TraceMeanField_ELBO: **runs without error**
- AutoIAFNormal + TraceMeanField_ELBO: **runs without error**
- AutoLaplaceApproximation + TraceMeanField_ELBO: **runs without error**
- AutoDelta + TraceMeanField_ELBO: **runs without error**

The only validation is `_check_mean_field_requirement()` which checks that model and guide **sample sites are ordered identically** -- a sufficient but not necessary condition. For AutoContinuous-based guides (MVN, LowRank, IAF, Laplace), the guide registers a single `_Auto*_latent` site, so the mean-field check is trivially satisfied (one site = trivially factored). For AutoNormal, each site is independent by construction.

**What actually happens with non-mean-field guides:**
- AutoContinuous-based guides register ONE joint sample site
- TraceMeanField_ELBO computes analytic KL for that ONE site (e.g., KL(MVN || MVN))
- This IS mathematically correct -- the "mean-field" label is misleading in this context
- The benefit of TraceMeanField_ELBO over Trace_ELBO is reduced for these guides since there's only one KL to compute anyway

**Recommendation:** Our `run_svi` should still raise ValueError for TraceMeanField_ELBO + non-mean-field guides (as per context decision), because:
1. The user INTENDED to get analytic mean-field KL benefits
2. Using it with AutoMVN gives no benefit over Trace_ELBO (just one KL computation either way)
3. Preventing misleading configurations is good UX
4. The context decision explicitly says "raise ValueError"

**Mean-field compatible guides:** AutoDelta, AutoNormal
**Non-mean-field guides:** AutoLowRankMVN, AutoMVN, AutoIAFNormal, AutoLaplaceApproximation

### 3.3 RenyiELBO

```python
RenyiELBO(
    alpha=0,
    num_particles=2,
    max_plate_nesting=inf,
    vectorize_particles=False,
    strict_enumeration_warning=True,
)
```

- **MRO:** RenyiELBO -> ELBO -> object
- **alpha:** Must not equal 1.0 (raises ValueError). Default 0. Alpha < 1 gives tighter bound than standard ELBO. Alpha = 0 gives importance-weighted autoencoder (IWAE) objective.
- **num_particles:** Default 2 (not 1 like Trace_ELBO). Can be 1 but benefits from more particles.
- **Compatible with all guides:** YES
- **Context decision:** alpha=0.5 (from success criteria). This is a good middle ground -- tighter than ELBO but more stable than IWAE (alpha=0).

**Recommendation for alpha:** Use 0.5. This gives a tighter variational bound than Trace_ELBO while being more stable than the IWAE objective (alpha=0). The success criteria explicitly mention `RenyiELBO(alpha=0.5)`.

**num_particles consideration:** RenyiELBO benefits from multiple particles. Default is 2. For benchmarking, use `num_particles=4` for better gradient estimates (2x default). This increases computation per step but improves convergence quality.

---

## 4. Latent Dimension Analysis (for AutoMVN Blocklist)

Verified by tracing actual model execution with `pyro.poutine.trace`.

### Task DCM (N regions, M=1 input)

| N | D (latent) | AutoNormal params | AutoMVN params | AutoLowRank(r=2) params |
|---|-----------|-------------------|----------------|------------------------|
| 3 | 13 | 26 | 104 | 52 |
| 5 | 31 | 62 | 527 | 124 |
| 10 | 111 | 222 | 6,327 | 444 |
| 15 | 241 | 482 | 29,402 | 964 |
| 20 | 421 | 842 | 89,252 | 1,684 |

D_task = N*N + N*M + 1 (A_free + C + noise_prec)

### Spectral DCM (N regions)

| N | D (latent) | AutoNormal params | AutoMVN params | AutoLowRank(r=2) params |
|---|-----------|-------------------|----------------|------------------------|
| 3 | 24 | 48 | 324 | 96 |
| 5 | 48 | 96 | 1,224 | 192 |
| 10 | 143 | 286 | 10,439 | 572 |
| 15 | 288 | 576 | 41,904 | 1,152 |

D_spectral = N*N + 2*N + 2 + 2*N + 1 (A_free + noise_a + noise_b + noise_c + csd_noise_scale)

### Memory vs. Optimization Difficulty

Memory is NOT the bottleneck (10K float64 params = 0.08 MB). The real concern is:
1. **Optimization difficulty:** O(D^2) parameters to learn with O(D) data constraints
2. **Convergence speed:** Full-rank MVN needs much more data/steps to learn correlations reliably
3. **Numerical stability:** Cholesky factor can become ill-conditioned with large D

**Blocklist recommendation:** Block AutoMVN at N >= 8 (D >= 73 for task, D >= 97 for spectral). At these dimensions, AutoLowRankMVN with rank=2 is a much better trade-off (captures key correlations with 3x fewer params than even AutoNormal's 2D). The block threshold is conservative -- AutoMVN may work at N=8-10, but convergence will be slow and unreliable for benchmarking.

---

## 5. extract_posterior_params Redesign

### Current Implementation (AutoNormal-specific)

```python
def extract_posterior_params(guide, model_args):
    median = guide.median(*model_args)     # AutoNormal-specific: works
    params = {k: v.detach().clone() for k, v in pyro.get_param_store().items()}
    return {"median": median, "params": params}
```

**Problems with current approach:**
1. `guide.median()` is available on AutoNormal and AutoContinuous but NOT consistently across all guides
2. `guide.quantiles()` is on AutoNormal and AutoContinuous but NOT on AutoDelta
3. Param store structure differs between AutoNormal (per-site locs/scales) and AutoContinuous (single joint loc/scale_tril)
4. AutoDelta has no scale/uncertainty information
5. AutoLaplaceApproximation's MAP guide gives misleading "posterior" -- the real posterior is from `laplace_approximation()`

### Recommended: Sample-Based Extraction via Predictive

Verified working with all 6 guide types:

```python
from pyro.infer import Predictive

def extract_posterior_params(guide, model_args, num_samples=1000):
    pred = Predictive(
        model, guide=guide, num_samples=num_samples,
        return_sites=<latent_sites>,
    )
    samples = pred(*model_args)
    result = {}
    for site, tensor in samples.items():
        result[site] = {
            "mean": tensor.mean(dim=0),
            "std": tensor.std(dim=0),
            "samples": tensor,  # (num_samples, *site_shape)
        }
    return result
```

**Advantages:**
- Works identically for all 6 guide types
- For AutoDelta: all samples identical, std=0 (correct)
- For AutoNormal: samples from independent Normals (equivalent to median + quantiles)
- For AutoMVN/LowRankMVN: samples capture posterior correlations
- For AutoIAFNormal: samples from flow-transformed distribution
- For AutoLaplaceApproximation: must call on the returned AutoMVN from `laplace_approximation()`, not the original guide

**Predictive verified with all guide types** (tested at runtime).

### Coverage Computation Change

Current runners use `guide.quantiles([0.025, 0.975], *model_args)` for 95% CI. With sample-based extraction:

```python
# Sample-based quantiles (works for all guides)
samples = result["A_free"]["samples"]  # (num_samples, N, N)
lo = torch.quantile(samples, 0.025, dim=0)
hi = torch.quantile(samples, 0.975, dim=0)
```

This is numerically equivalent for AutoNormal but more accurate for non-Gaussian guides (AutoIAF).

---

## 6. AutoLaplaceApproximation Integration with run_svi

### The Challenge

AutoLaplaceApproximation has a two-phase workflow:
1. MAP optimization (run SVI with Delta guide)
2. Hessian computation (`laplace_approximation()`)
3. Use returned AutoMVN for posterior queries

This does not fit the current `run_svi` -> `extract_posterior_params` pattern because:
- After `run_svi`, the guide is still a Delta (MAP) guide
- `laplace_approximation(*model_args)` must be called separately
- The returned AutoMVN is a DIFFERENT guide object

### Design Options

**Option A: Handle in create_guide (wrap with Laplace logic)**
- `create_guide` returns a LaplaceGuideWrapper that auto-calls `laplace_approximation()` after training
- Transparent to run_svi and extract_posterior_params

**Option B: Handle in run_svi (special-case Laplace)**
- `run_svi` detects AutoLaplaceApproximation, runs MAP SVI, calls `laplace_approximation()`, returns the MVN guide
- Return dict gains `"guide"` key with the post-Laplace guide

**Option C: Handle at the call site (runner level)**
- Runners check if guide is AutoLaplaceApproximation and call the extra step
- Most explicit, least magic

**Recommendation: Option B.** The `run_svi` function already orchestrates the SVI loop. Adding a post-SVI step for Laplace is natural. The returned dict can include `"guide"` pointing to the post-Laplace AutoMVN. The original guide stays unchanged (for inspection). This keeps runners clean.

### MAP Pre-fit Steps

Context decision: 1000 MAP pre-fit steps. But `run_svi` already has `num_steps`. For Laplace:
- The `num_steps` parameter to `run_svi` controls how many MAP steps to run (same as for other guides)
- After SVI completes, `laplace_approximation()` is called automatically
- The 1000 is a recommended default, not a separate parameter

---

## 7. Call Site Migration Analysis

### Sites Using `create_guide(model, init_scale=0.01)`

These must be updated to `create_guide(model, init_scale=0.01)` or `create_guide(model)` when init_scale becomes a kwarg:

| File | Line | Current Call |
|------|------|-------------|
| tests/test_svi_integration.py | 149,161,162,314,345,375 | `create_guide(model, init_scale=0.01)` or `create_guide(model)` |
| tests/test_task_dcm_recovery.py | 221 | `create_guide(task_dcm_model, init_scale=0.01)` |
| tests/test_spectral_dcm_recovery.py | 198 | `create_guide(spectral_dcm_model, init_scale=0.01)` |
| tests/test_amortized_benchmark.py | 223 | `create_guide(spectral_dcm_model, init_scale=0.01)` |
| tests/test_elbo_model_comparison.py | 373,408,433,480,502,561,577,665,696,732,751,807,819 | `create_guide(*, init_scale=0.01)` |
| benchmarks/runners/task_svi.py | 173 | `create_guide(task_dcm_model, init_scale=0.01)` |
| benchmarks/runners/spectral_svi.py | 166 | `create_guide(spectral_dcm_model, init_scale=0.01)` |
| benchmarks/runners/task_amortized.py | 401 | `create_guide(task_dcm_model, init_scale=0.01)` |
| benchmarks/runners/spectral_amortized.py | 395 | `create_guide(spectral_dcm_model, init_scale=0.01)` |

### Sites Using `AutoNormal(model, init_scale=0.01)` directly (not via factory)

| File | Line | Current Call |
|------|------|-------------|
| tests/test_spectral_dcm_model.py | 258,288 | `AutoNormal(spectral_dcm_model, init_scale=0.01)` |
| tests/test_rdcm_model.py | 229,261 | `AutoNormal(rdcm_model, init_scale=0.01)` |
| tests/test_task_dcm_model.py | 281,314,350 | `AutoNormal(task_dcm_model, init_scale=0.01)` |

These bypass the factory and should be migrated to use `create_guide` for consistency.

### Total migration scope

- **~30 call sites** across tests and runners
- **13 unique test files** that import or use create_guide
- **4 runner files** that call create_guide
- **5 test files** that use AutoNormal directly

The migration is mechanical: `create_guide(model, init_scale=0.01)` stays as-is (init_scale moves to kwargs but the call syntax is the same with the new factory signature). The default `guide_type='auto_normal'` ensures backward compatibility.

---

## 8. TraceMeanField_ELBO Compatibility Policy

### Recommendation: Strict enforcement in our code

Even though Pyro allows all guide+ELBO combinations at runtime, we should enforce the mean-field restriction ourselves because:

1. TraceMeanField_ELBO provides no benefit with non-mean-field guides (one joint site => one KL computation either way)
2. Users choosing `tracemeanfield_elbo` expect analytic per-site KL benefits
3. Silent acceptance hides a configuration mistake

### Compatibility Matrix (18 cells)

| Guide \ ELBO | trace_elbo | tracemeanfield_elbo | renyi_elbo |
|--------------|-----------|--------------------|-----------| 
| auto_delta | OK | OK | OK |
| auto_normal | OK | OK (analytic KL) | OK |
| auto_lowrank_mvn | OK | REJECT | OK |
| auto_mvn | OK | REJECT | OK |
| auto_iaf | OK | REJECT | OK |
| auto_laplace | OK | REJECT | OK |

**Valid combinations: 14 out of 18.**
**Rejected: 4** (TraceMeanField_ELBO with non-mean-field guides)

### Where to enforce

Enforce in `run_svi` before creating the ELBO object. The guide type string is known at that point. Error message pattern:

```
ValueError: TraceMeanField_ELBO requires a mean-field guide 
(auto_delta or auto_normal), got 'auto_mvn'. 
Use 'trace_elbo' or 'renyi_elbo' instead.
```

---

## 9. String Key Naming Convention

Recommendation for guide_type string keys:

| Pyro Class | String Key | Rationale |
|-----------|-----------|-----------|
| AutoDelta | `"auto_delta"` | Matches class name, snake_case |
| AutoNormal | `"auto_normal"` | Matches class name, current default |
| AutoLowRankMultivariateNormal | `"auto_lowrank_mvn"` | Abbreviated (full name too long) |
| AutoMultivariateNormal | `"auto_mvn"` | Standard abbreviation |
| AutoIAFNormal | `"auto_iaf"` | IAF is the salient feature |
| AutoLaplaceApproximation | `"auto_laplace"` | Clear, concise |

For ELBO types:

| Pyro Class | String Key |
|-----------|-----------|
| Trace_ELBO | `"trace_elbo"` |
| TraceMeanField_ELBO | `"tracemeanfield_elbo"` |
| RenyiELBO | `"renyi_elbo"` |

---

## 10. RenyiELBO Alpha and num_particles

### Alpha Recommendation: 0.5

- Success criteria specify `RenyiELBO(alpha=0.5)` explicitly
- alpha=0.5 provides a tighter variational bound than standard ELBO (alpha -> 1)
- alpha=0.5 is more numerically stable than IWAE (alpha=0)
- alpha=1.0 is FORBIDDEN by Pyro (raises ValueError)
- The alpha parameter should be configurable via kwargs for future flexibility

### num_particles Recommendation: 4

- RenyiELBO defaults to `num_particles=2`
- More particles = tighter bound but more compute per step
- For benchmarking, `num_particles=4` is a reasonable balance
- This should be passed through `run_svi`'s existing `num_particles` parameter
- When `elbo_type='renyi_elbo'` and `num_particles=1`, use 2 (Renyi's default minimum for meaningful estimation)

---

## 11. NaN ELBO Handling During Sweeps

### Current Behavior

`run_svi` raises `RuntimeError("NaN ELBO at step {step}")` immediately.

### Recommendation for 6x3 Matrix Tests

Keep the RuntimeError raise for individual `run_svi` calls. In the test matrix:
- Each (guide, elbo) combo runs independently
- If one combo NaN-s, catch RuntimeError at the test level and record it as a failure
- Do NOT change `run_svi` to return sentinel -- the NaN indicates a real problem that should be investigated

For the benchmark runners (Phase 11), the existing try/except pattern already handles NaN via `n_failed` counting.

---

## 12. Blocklist Storage Recommendation

### Option: Hardcoded dict in create_guide

```python
_BLOCKLIST: dict[str, set[int]] = {
    "auto_mvn": {8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20},
    "auto_iaf": {15, 16, 17, 18, 19, 20},  # IAF is expensive at high D
}
```

Check `n_regions` against blocklist before allocating the guide. Raise ValueError with suggestion:

```
ValueError: AutoMultivariateNormal is not recommended for N=10 regions 
(D=111 latent parameters, 6327 variational parameters). 
Use 'auto_lowrank_mvn' (rank=2) instead, which captures key posterior 
correlations with only 444 parameters.
```

### Alternative: Max-D threshold instead of region count

Since the latent dimension depends on the model variant (task vs spectral), a D-based threshold is more principled:

```python
_MAX_D_MVN = 80  # ~N=7 for spectral, ~N=8 for task
_MAX_D_IAF = 200  # ~N=13 for spectral
```

But this requires knowing D before guide construction, which means running a model trace first. The region-count approach is simpler and sufficient for Phase 10.

**Recommendation:** Use region count with guide_type key. This is simple, deterministic, and matches the context decision about N=10.

---

## 13. Posterior Sample Count for extract_posterior_params

### Recommendation: 1000 samples (default)

- 1000 samples give reliable mean/std estimates for parameters with dimension up to ~20x20
- For AutoDelta: all 1000 are identical, overhead is trivial
- For AutoNormal: 1000 Normal samples, fast
- For AutoMVN/LowRank: 1000 MVN samples, fast (single Cholesky solve reused)
- For AutoIAF: 1000 flow forward passes, moderate cost but acceptable
- Configurable via `num_samples` parameter

---

## 14. Phase Ordering Implications

### Suggested Implementation Order (3 subtasks)

**Subtask 10-01: create_guide factory + tests**
- Extend `create_guide` to accept `guide_type` string and `**kwargs`
- Map string keys to Pyro classes
- Handle init_scale per guide type (pass where accepted, skip where not)
- Add blocklist checking (N-based)
- Update return type annotation to `AutoGuide` (generic)
- Migrate all v0.1.0 call sites
- Tests: each guide type instantiates and runs 1 SVI step without error

**Subtask 10-02: ELBO plumbing + run_svi extension**
- Add `elbo_type` parameter to `run_svi`
- Map string keys to Pyro ELBO classes
- Enforce TraceMeanField_ELBO + mean-field-only check
- Handle RenyiELBO num_particles (minimum 2)
- Handle AutoLaplaceApproximation post-SVI step
- Add `elbo_type` to BenchmarkConfig
- Tests: all 14 valid (guide, ELBO) pairs converge on spectral DCM N=3

**Subtask 10-03: extract_posterior_params redesign + runner updates**
- Replace current AutoNormal-specific extraction with Predictive-based sampling
- Update all runners to use new extraction API
- Update coverage computation to use sample-based quantiles
- Verify backward compatibility (v0.1.0 behavior preserved with defaults)
- Tests: posterior extraction works with all 6 guide types

### Dependency Chain

```
10-01 (create_guide) -> 10-02 (ELBO plumbing) -> 10-03 (extraction + runners)
```

10-01 is prerequisite for 10-02 because ELBO compatibility checks reference guide_type strings. 10-03 depends on both because extraction tests need the full guide+ELBO infrastructure.

---

## 15. Confidence Assessment

| Finding | Confidence | Source |
|---------|-----------|--------|
| Constructor signatures | HIGH | `inspect.signature()` on installed Pyro 1.9.1 |
| init_scale compatibility | HIGH | Runtime testing with each guide type |
| TraceMeanField_ELBO non-enforcement | HIGH | Runtime testing -- all guide types pass without error |
| Predictive-based extraction | HIGH | Runtime testing with all 6 guide types |
| AutoLaplaceApproximation workflow | HIGH | Runtime testing of full 3-phase flow |
| RenyiELBO alpha constraints | HIGH | Runtime testing + source inspection |
| AutoMVN parameter counts | HIGH | Computed from verified latent dimensions |
| Blocklist thresholds | MEDIUM | Based on optimization difficulty heuristics, not empirical testing |
| AutoIAFNormal hidden_dim behavior | MEDIUM | Source inspection + some runtime testing |

---

## Sources

- Pyro 1.9.1 installed source (via `inspect.getsource()`, `inspect.signature()`)
- [Pyro Autoguide documentation](https://docs.pyro.ai/en/stable/infer.autoguide.html)
- [Pyro SVI documentation](https://docs.pyro.ai/en/stable/inference_algos.html)
- [Pyro AutoIAFNormal source](https://github.com/pyro-ppl/pyro/blob/dev/pyro/infer/autoguide/guides.py)
- [Pyro RenyiELBO source](https://github.com/pyro-ppl/pyro/blob/dev/pyro/infer/renyi_elbo.py)
- [Pyro TraceMeanField_ELBO source](https://github.com/pyro-ppl/pyro/blob/dev/pyro/infer/trace_mean_field_elbo.py)
- [Pyro Predictive class](https://docs.pyro.ai/en/stable/infer.predictive.html)
- [Working with AutoLaplaceApproximation (forum)](https://forum.pyro.ai/t/working-with-autolaplaceapproximation/2846)
- v0.2.0 research files: `.planning/research/v0.2.0/STACK.md` (prior findings on guide variants)
