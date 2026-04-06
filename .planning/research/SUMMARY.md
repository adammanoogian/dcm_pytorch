# Research Summary: Pyro-DCM v0.2.0 Cross-Backend Inference Benchmarking

**Project:** Pyro-DCM
**Domain:** Probabilistic DCM / cross-backend inference benchmarking
**Researched:** 2026-04-06
**Confidence:** HIGH (Pyro/NumPyro side), MEDIUM (NUTS performance on ODE models)

---

## Executive Summary

v0.2.0 answers one scientific question: how much does the variational approximation cost
for fMRI Dynamic Causal Modeling? The benchmark adds 6+ Pyro guide variants, NumPyro
NUTS/ADVI/Laplace backends, and a systematic calibration study. The closest published
analog is Baldy et al. (2025) in J. R. Soc. Interface, which benchmarked DCM inference
in NumPyro/PyMC/Stan — but for ERP/EEG DCM, not fMRI DCM. Their key finding is directly
relevant: mean-field ADVI produced systematically underdispersed posteriors, while
full-rank ADVI and Laplace were "equivalent" to NUTS. Our existing coverage numbers
(0.44–0.78 at 90% nominal) are consistent with the expected mean-field ceiling of
~0.80–0.88 on correlated posteriors. This is a known design limitation, not a bug.

The recommended build strategy is Pyro-first: all new guide variants (AutoLowRank,
AutoMultivariateNormal, AutoIAFNormal) are zero-dependency drop-in changes to the
existing `create_guide` factory. This immediately extends the benchmark to 6+ methods
with no new infrastructure, and answers whether richer guide families fix the coverage
problem before investing in NumPyro integration. NumPyro is the highest-effort phase
because it requires JAX reimplementation of the Balloon-Windkessel and spectral transfer
forward models; this is mechanical but must be validated against SPM12 references.

The two critical risks are: (1) NUTS on ODE-based task DCM will be very slow — expect
30–120 minutes per 3-region dataset, making the compute matrix non-trivial, and
(2) the existing amortization gap metric is scientifically incorrect (uses an RMSE-ratio
proxy instead of actual ELBO evaluation) and must be fixed before any amortization
analysis is reported. Both risks are well-scoped and have clear fixes.

---

## Key Findings

### Recommended Stack

See `.planning/research/v0.2.0/STACK.md` for full detail.

v0.2.0 adds no new core PyTorch/Pyro dependencies. All needed Pyro guide variants
(`AutoLowRankMultivariateNormal`, `AutoStructured`, `AutoGaussian`, `RenyiELBO`,
`TraceMeanField_ELBO`, `Predictive`) ship in pyro-ppl >= 1.9.1, already installed.

**New optional dependencies:**

- **numpyro >= 0.16 + jax[cpu]**: NumPyro backends (NUTS gold standard, ADVI, Laplace).
  Structured as a separate `[numpyro]` optional group because JAX installation is
  platform-specific. CPU-only is sufficient — NUTS is sequential and GPU adds no benefit.
- **diffrax >= 0.6**: JAX-native ODE solver. API maps directly from torchdiffeq
  (`odeint` -> `diffeqsolve`, `Dopri5` solver). Required for NUTS on task DCM.
- **arviz >= 0.20, < 1.0**: MCMC diagnostics (ESS, R-hat, trace plots, from_dict for SVI
  posteriors). Pin below 1.0 — arviz 1.0 requires Python 3.12 and restructured the API.
- **.npz (NumPy)**: Cross-backend data interchange. Already available. Framework-neutral:
  `torch.from_numpy(np.load(...)["A_true"])` and `jnp.array(np.load(...)["A_true"])`.

**Critical non-finding:** ins-amu/DCM_PPLs implements ERP-DCM (Jansen-Rit neural mass,
EEG observation), NOT fMRI-DCM (bilinear neural state, Balloon-Windkessel, BOLD). It
cannot be used as a drop-in NumPyro reference. NumPyro models must be written from
scratch using our validated math, not adapted from DCM_PPLs.

**What NOT to add:** posteriordb-python (DCM not in database), h5py (overkill for small
arrays), blackjax/PyMC/Stan (complexity without clear gain over NumPyro NUTS).

---

### Expected Features

See `.planning/research/v0.2.0/FEATURES.md` for full detail.

**Must have (table stakes) — MVP blockers:**

- **TS-6: Fix amortization gap metric** — Replace RMSE-ratio proxy with real ELBO via
  `Trace_ELBO().differentiable_loss(model, guide, *args)`. Unblocks all honest reporting.
  Low effort, high scientific impact.
- **TS-2: Extended guide variants** — AutoDelta, AutoLowRank, AutoMultivariateNormal,
  AutoIAFNormal. One-line guide changes in `create_guide`. Primary calibration insight.
- **TS-1: NUTS reference** — NumPyro NUTS for spectral DCM at 3–5 regions as gold
  standard. 4 chains, R-hat < 1.01, bulk ESS > 400. Task DCM NUTS is a stretch goal
  (slow ODE integration).
- **TS-3: Coverage calibration plot** — Per-parameter coverage curves (expected vs
  observed at 50%, 75%, 90%, 95%). The key paper figure.
- **TS-7: Cross-backend comparison table** — Unified 9+ methods × 3 variants × 3 sizes
  table (RMSE, coverage@90%, correlation, wall time).
- **TS-8: Reproducible benchmark runner** — `generate_fixtures.py` + extended
  `run_all_benchmarks.py` CLI with `--guide`, `--network-size`, `--parameterization`.
- **TS-4: ELBO variant comparison** — Trace_ELBO vs TraceMeanField_ELBO vs
  RenyiELBO(alpha=0.5). Low effort.
- **TS-5: Network size scaling** — 3, 5 regions primary; 10 regions stretch goal.

**Should have (differentiators):**

- **D-4: Practical recommendation guide** — Decision tree for guide selection by
  compute budget and network size. The single most useful output for users.
- **D-5: Per-parameter posterior comparison plots** — Violin/ridge overlay of all
  methods per A_ij element. The "money figure" for reviewers.
- **D-2: Amortized + refinement pipeline** — Semi-amortized: amortized init + 50–100
  SVI refinement steps. Closes 60–80% of amortization gap.
- **D-6: Wall-clock timing breakdown** — ODE time, guide eval, gradient time.

**Defer to v0.3+:**

- **D-1: Full SBC** — Computationally prohibitive (27K+ inference runs for 500
  datasets × 9 methods × 3 variants). Add as supplementary after primary results.
- **D-3: Regularization study** — NCP, prior scale sweep. Interesting but orthogonal
  to the main paper story. Run after primary method comparison is complete.
- **10-region benchmarks for all methods** — Only feasible for mean-field SVI and rDCM.

**Anti-features (explicitly don't build):**

- Cross-PPL runtime benchmarking (Baldy et al. 2025 already did this; repeating it
  adds no value and invites unfair comparisons).
- Real-data benchmarking (ground truth unknown; simulation is the correct approach).
- Novel guide architectures (scope creep; use Pyro's built-in autoguides).
- Automatic method selection (the benchmark characterizes tradeoffs; users decide).

---

### Architecture Approach

See `.planning/research/v0.2.0/ARCHITECTURE.md` for full detail.

The existing benchmark architecture (RUNNER_REGISTRY, BenchmarkConfig, metrics.py) is
well-suited for extension. The three new architectural concerns are: (1) pre-generated
shared fixtures for cross-backend fairness, (2) a parameterized guide factory replacing
per-guide runner files, and (3) a `numpyro_models/` package with JAX forward models.

**Major new/modified components:**

1. **`benchmarks/generate_fixtures.py`** — Pre-generate all synthetic datasets as `.npz`
   files (3 variants × 3 sizes × 20–50 datasets). Runs once before benchmarks. Eliminates
   per-runner data generation with different PRNGs. All subsequent runners load from
   `benchmarks/fixtures/`.

2. **Extended `create_guide(model, guide_type, init_scale, rank)`** — Factory replacing
   hardcoded `AutoNormal`. `guide_type` string dispatches to AutoDelta, AutoNormal,
   AutoLowRank, AutoMultivariateNormal, AutoIAFNormal, AutoLaplaceApproximation. Existing
   runners gain a `guide_type` parameter; the registry key becomes a 3-tuple
   `(variant, method, guide_type)`.

3. **Extended `BenchmarkConfig`** — Adds `guide_type`, `n_regions_list`, `prior_scale`,
   `parameterization` ("centered" | "non_centered"), `elbo_type`, `fixtures_dir`.
   All new fields have defaults that preserve v0.1.0 behavior.

4. **`src/pyro_dcm/numpyro_models/`** — New package. `task_dcm_numpyro.py`,
   `spectral_dcm_numpyro.py`, and `_forward_jax.py` (JAX reimplementation of Balloon-
   Windkessel and spectral transfer function using diffrax). Wrapped with
   `try: import numpyro` guards; runners skip gracefully when JAX not installed.

5. **`benchmarks/runners/numpyro_nuts.py`, `numpyro_advi.py`, `numpyro_laplace.py`** —
   Separate files for NumPyro backends (JAX arrays, different inference API). Not
   parameterized into existing SVI runners — the frameworks are too different to abstract.

6. **`benchmarks/comparison.py`** — Aggregates nested JSON results into a flat CSV
   comparison table. Provides `rank_methods()` by metric per (variant, size).

**Key architectural rules:**
- Do NOT build a framework abstraction layer wrapping both Pyro and NumPyro. The APIs
  are fundamentally different; abstraction leaks.
- Do NOT run NUTS in CI. NUTS is a manual/scheduled run; CI only validates that the
  NumPyro model executes a few steps without error.
- Metrics stay in torch; NumPyro runners convert JAX → numpy → torch at the runner
  boundary before calling existing `compute_rmse`, `compute_coverage_from_ci`.
- Do NOT duplicate forward model code. JAX version is written from the same reference
  equations and validated against the same `.npz` test fixtures, not from the torch code.

---

### Critical Pitfalls

See `.planning/research/v0.2.0/PITFALLS.md` for full detail.

**P1 — Mean-field coverage ceiling is ~0.80–0.88 by design (not a bug).**
AutoNormal assumes parameter independence. DCM posteriors have strong A_free
correlations. The 0.44–0.78 coverage in v0.1.0 is the expected mean-field behavior,
confirmed by Baldy et al. (2025). Prevention: report per-parameter coverage curves,
not just aggregate coverage. Accept the ceiling; show improvement from AutoLowRank →
AutoMultivariateNormal as the scientific finding.

**P2 — NUTS can fail silently for DCM (multimodality, not divergences).**
Baldy et al. found that NUTS chains with diffuse priors converged to different modes
with R-hat = 1.007–1.009 — formally "converged" but not to the same posterior. For
DCM with N ≥ 5 (100+ parameters), the risk is higher. Prevention: run ≥ 8 chains with
initialization at prior tails (2.5th/97.5th percentile, which Baldy et al. found
achieves 100% convergence). Check cross-chain energy distance, not just R-hat.

**P3 — Cross-framework comparison requires strict numerical controls.**
JAX defaults to float32; our codebase uses float64. If NumPyro runs in float32, CSD
differences dominate the comparison. Different PRNG implementations mean the same seed
produces different data. Prevention: set `jax_enable_x64 = True` for all NumPyro runs;
use shared `.npz` fixtures (not per-backend generation); never compare ELBO values
across backends (only compare downstream metrics: RMSE, coverage, correlation).

**P4 — The existing amortization gap metric is scientifically incorrect.**
`task_amortized.py` line 409 uses `svi_result["final_loss"] * (1.0 + max(0.0,
rmse_amort/rmse_svi - 1.0))` as a proxy — not a valid ELBO measurement. The real
amortization gap = ELBO improvement from fine-tuning the amortized guide on a single
test subject. Prevention: compute actual ELBO via
`Trace_ELBO().differentiable_loss(model, guide, *args)` for both amortized and
per-subject SVI with matching guide families.

**P5 — SBC can pass while detecting nothing** if implemented naively.
Marginal rank statistics are uniform for mean-field VI even when joint calibration is
wrong (Modrak et al. 2023). 20 datasets (current full_config) is too few. Prevention:
add joint test quantities (max eigenvalue of A, Frobenius norm of A); require ≥ 500
datasets for reliable KS tests; treat SBC as a separate expensive computation, not
embedded in the benchmark loop.

**P6 — Full-rank guide memory explosion at N=10.**
AutoMultivariateNormal at N=10 spectral DCM: D=143 latent dims → 10,439 guide params +
full ODE forward graph. May exceed 8GB GPU memory or cause severe CPU slowdown.
Prevention: only run AutoMultivariateNormal at N=3 and N=5. Use AutoLowRank at N=10.

**P9 — Simpson's paradox in aggregated tables.**
rDCM RMSE (0.10–0.15) is 5–10× higher than spectral DCM RMSE (0.01–0.02). Naive
averaging across variants produces rankings driven by rDCM scale. Prevention: never
aggregate across variants; use rank-based aggregation if cross-variant summaries are
needed.

**P11 — Combinatorial explosion without compute tiering.**
Full benchmark matrix = 9 methods × 3 variants × 3 sizes × 5 seeds × 20 datasets =
8,100 runs; at 30–120s each, that is 68–270 hours. Prevention: three-tier approach —
Tier 1 (all Pyro guides × spectral × N=3), Tier 2 (NUTS vs best-Pyro × all variants ×
N=3,5), Tier 3 (full matrix × N=10).

---

## Implications for Roadmap

### Suggested Phase Structure

**Phase 1 — Fixture Generation and Benchmark Config Extension** (LOW risk, HIGH value)

Deliverables: `benchmarks/generate_fixtures.py`, `.npz` fixtures for 3 variants × 3
sizes, `BenchmarkConfig` extension, fixture-loading in existing runners.

Rationale: Foundation for cross-backend fairness. Every subsequent runner depends on
shared fixtures. No new dependencies. Low risk, can be done before any NumPyro work.

Features: TS-8 (benchmark runner), TS-6 (fix amortization gap — done in parallel).
Pitfalls to avoid: P3 (shared fixtures prevent PRNG mismatch), P10 (train amortized
guide on wider distribution than test).

---

**Phase 2 — Pyro Guide Variants** (LOW risk, HIGH impact)

Deliverables: Extended `create_guide` factory (6 guide types), 3-tuple RUNNER_REGISTRY,
benchmark runs for spectral DCM (Tier 1). Coverage calibration multi-level plot.

Rationale: Zero new dependencies. Answers whether richer approximation families fix the
0.44–0.78 coverage problem. The primary calibration finding of the paper. Must precede
NumPyro work to set expectations for what VI can and cannot achieve.

Features: TS-2 (guide variants), TS-3 (calibration analysis), TS-4 (ELBO variants),
D-5 (per-parameter plots).
Pitfalls to avoid: P1 (coverage ceiling; report per-parameter, not aggregate), P6
(full-rank memory; only at N=3,5), P9 (no cross-variant aggregation), P12 (report
per-dataset scatter, not just means).

Research flag: STANDARD PATTERNS. Pyro autoguide API is well-documented; no additional
research needed.

---

**Phase 3 — Regularization Study** (MEDIUM risk)

Deliverables: `poutine.reparam(LocScaleReparam)` integration for non-centered
parameterization, prior scale sensitivity sweep (1/128, 1/64, 1/32, 1/16), ELBO
variant comparison (Trace vs TraceMeanField vs Renyi). Centered vs NCP interaction
analysis.

Rationale: Tests whether the coverage problem is a parameterization/prior issue or a
fundamental mean-field limitation. Determines whether NCP helps Pyro SVI (literature
suggests it primarily helps NUTS, not SVI). Should run AFTER Phase 2 so NCP × guide
type combinations can be compared.

Features: D-3 (regularization study).
Pitfalls to avoid: P7 (NCP may hurt centered SVI; test empirically), P8 (hold prior
fixed across methods in primary comparison; present sensitivity as separate analysis).

Research flag: NEEDS EMPIRICAL VALIDATION. The centered vs NCP tradeoff for Pyro SVI
is under-studied. Phase 3 is itself the research; don't pre-judge the outcome.

---

**Phase 4 — NumPyro Backend Integration** (HIGH effort, HIGH scientific value)

Deliverables: `src/pyro_dcm/numpyro_models/` package, JAX forward models (_forward_jax.py:
Balloon-Windkessel + spectral transfer via diffrax), NumPyro model definitions (task,
spectral), NUTS/ADVI/Laplace runners, optional dependency handling in pyproject.toml,
arviz-based MCMC diagnostics, forward model cross-validation against SPM12 (.npz fixtures).

Rationale: The NUTS reference posterior is the scientific anchor for all VI comparisons.
Cannot report "variational approximation cost" without MCMC gold standard. Most effort
phase due to JAX reimplementation of validated forward models. Placed after Pyro phases
so the Pyro-side analysis is complete when NUTS results arrive.

Sub-ordering: spectral DCM NUTS first (no ODE, fastest), then task DCM NUTS (slow ODE,
possible timeout at N ≥ 5 — treat as stretch goal).

Features: TS-1 (NUTS reference), TS-7 (cross-backend table begins populating).
Pitfalls to avoid: P2 (≥ 8 chains, prior-tail initialization, cross-chain energy
distance), P3 (jax_enable_x64, shared fixtures, no ELBO cross-backend comparison),
P4 (real ELBO measurement, not proxy), P11 (set per-run timeouts; Tier 2 only).

Research flag: NEEDS RESEARCH during implementation — specifically the JAX forward
model validation step. Verify that Pyro and NumPyro models produce identical predicted
BOLD/CSD (atol=1e-6) given identical parameters before running NUTS.

---

**Phase 5 — Cross-Backend Comparison and Analysis** (LOW risk, pure analysis)

Deliverables: `benchmarks/comparison.py` (aggregate JSON → CSV), coverage calibration
figure (the key paper figure), wall-time vs RMSE Pareto frontier, recommendation
decision tree (D-4), updated benchmark narrative.

Rationale: No new infrastructure. Synthesizes all prior results. Outputs the final
deliverables for the paper.

Features: TS-7 (comparison table), D-4 (recommendation guide), D-6 (timing analysis).
Pitfalls to avoid: P9 (present results per-variant, use rank aggregation for summaries),
P12 (report median+IQR, failure rates, paired comparisons), P13 (Pareto frontier not
raw wall time; report amortized training cost separately).

Research flag: STANDARD PATTERNS.

---

**Phase 6 — Amortized + Refinement Study** (MEDIUM effort, adds D-2)

Deliverables: Semi-amortized runner (amortized init → 50–100 SVI refinement steps).
ELBO/coverage/RMSE at steps 0, 10, 50, 100. Practical answer to "how many refinement
steps to close the amortization gap?"

Rationale: Can only run AFTER Phase 1 (correct amortization gap metric) and Phase 2
(guide variants for comparison). Conceptually important but requires careful engineering
of guide initialization transfer.

Features: D-2 (amortized + refinement), TS-6 (already fixed in Phase 1).
Pitfalls to avoid: P4 (correct gap decomposition), P10 (amortized training distribution
must cover test distribution).

Research flag: STANDARD PATTERNS. Amortized refinement is well-described in the SBI
literature (semi-amortized VI, Cremer et al. 2018).

---

### Critical Path

```
Phase 1 (fixtures)
    |
    +-- Phase 2 (Pyro guides) --> Phase 3 (regularization) -->+
    |                                                          |
    +-- Phase 4 (NumPyro) ------------------------------------->
                                                               |
                                                          Phase 5 (analysis)
                                                               |
                                                          Phase 6 (amortized refinement)
```

Phase 1 unblocks everything. Phases 2 and 4 can run in parallel if staffing allows.
Phase 6 is last; depends on both the amortization gap fix (Phase 1) and the guide
variant baseline (Phase 2).

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All Pyro guide variants and ELBO estimators verified against pyro-ppl source. NumPyro/JAX versions verified on PyPI. ArviZ version pinning documented. |
| Features | MEDIUM-HIGH | Table stakes are well-established by Baldy et al. (2025) and standard benchmarking practice. Differentiators are reasoned from the literature, not empirically pre-verified for this specific DCM setting. |
| Architecture | HIGH (Pyro side), MEDIUM (NumPyro side) | Pyro extension is straightforward. JAX forward model equivalence needs empirical validation. DCM_PPLs non-usability is confirmed (HIGH). |
| Pitfalls | HIGH | Five critical pitfalls grounded in Baldy et al. (2025) data and verified bugs in v0.1.0 codebase. NUTS multimodality finding directly replicated from published DCM benchmark. |

**Overall confidence:** HIGH for Pyro-side work (no unknowns). MEDIUM for NumPyro
integration (forward model parity requires empirical validation). LOW for absolute
NUTS runtime estimates on Windows CPU (highly hardware-dependent).

### Gaps to Address

- **NUTS runtime on task DCM at N=5**: Estimates (2–8 hours per dataset) are
  extrapolated from forum reports on non-DCM PK-PD models. Actual runtime may vary by
  2–5× depending on ODE stiffness and step count. Resolution: run a pilot on 1 dataset
  before committing to the full Tier 2 matrix.

- **NCP for Pyro SVI**: The centered vs non-centered tradeoff for Pyro auto-guides is
  not well-studied in the literature. Baldy et al. (2025) did not discuss it. Phase 3
  is exploratory; don't assume NCP helps. Resolution: empirical testing on spectral DCM
  N=3 before generalizing.

- **Full-rank guide at N=5**: Memory profiling needed before the Phase 2 benchmark
  run. Resolution: one SVI step at N=5 with AutoMultivariateNormal + peak memory
  measurement before committing to full runs.

- **JAX forward model numerical parity**: The Balloon-Windkessel ODE reimplemented in
  diffrax will not be bit-identical to the torchdiffeq version. Acceptable tolerance
  (atol=1e-6 for BOLD predictions given identical parameters) needs empirical
  confirmation. Resolution: cross-validation script against shared `.npz` test cases
  at the start of Phase 4.

---

## Sources

### Primary (HIGH confidence)
- Baldy, Woodman, Jirsa & Hashemi (2025). Dynamic causal modelling in probabilistic
  programming languages. J. R. Soc. Interface. — NUTS vs ADVI vs Laplace for DCM,
  multimodality finding, mean-field underdispersion, Laplace equivalence to NUTS
- Pyro autoguide documentation (docs.pyro.ai/en/stable/infer.autoguide.html) —
  AutoLowRank, AutoStructured, AutoGaussian, AutoIAF specifications
- Pyro SVI / ELBO documentation — RenyiELBO, TraceMeanField_ELBO, Predictive API
- NumPyro autoguide/MCMC documentation (num.pyro.ai) — guide variants, NUTS config
- ins-amu/DCM_PPLs GitHub — verified to be ERP-DCM, not fMRI-DCM (HIGH)
- pyro-ppl 1.9.1, numpyro 0.20.1, diffrax 0.7.2, arviz 0.22.0 on PyPI — version pinning
- v0.1.0 benchmark results and 08-VERIFICATION.md — current coverage numbers, known bugs

### Secondary (MEDIUM confidence)
- Vehtari et al. (2021). Rank-normalization, folding, and localization: An improved
  R-hat. Bayesian Analysis. — R-hat < 1.01 threshold, bulk/tail ESS
- Modrak et al. (2023). Simulation-Based Calibration Checking for Bayesian Computation:
  The Choice of Test Quantities. Bayesian Analysis. — SBC blind spots (P5)
- Cremer et al. (2018). Inference Suboptimality in Variational Autoencoders. — Correct
  amortization gap decomposition (P4 fix)
- Pyro forum: NUTS with Diffrax ODE models — runtime estimates for ODE-based NUTS
- posteriordb (AISTATS 2025) — benchmark methodology reference, NOT used directly

### Tertiary (LOW confidence)
- TREND (Transformer DCM, 2024) — scaling to large networks; full text not accessed
- Amortized VI Systematic Review (JAIR 2023) — general amortization principles;
  PDF extraction failed, claims from abstract only

---

*Research completed: 2026-04-06*
*Ready for roadmap: yes*
