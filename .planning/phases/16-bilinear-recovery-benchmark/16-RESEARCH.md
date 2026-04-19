# Phase 16: 3-Region Bilinear Recovery Benchmark — Research

**Researched:** 2026-04-18
**Domain:** Benchmark runner infrastructure integration; bilinear DCM SVI recovery diagnostics
**Confidence:** HIGH (all load-bearing interfaces verified by direct source read; empirical timing extrapolated from Phase 15 `test_bilinear_svi_smoke_3region_converges` which ran 40 steps in 29.8s at N=3, J=1)

---

## Summary

1. **The v0.2.0 benchmark infrastructure is production-ready and Phase 16 should plug into it, not reinvent.** The `benchmarks/` layout has `runners/` (registry-dispatched; `(variant, method) -> callable(BenchmarkConfig) -> dict`), `fixtures.py` + `generate_fixtures.py` (`.npz` per-dataset with `manifest.json`), `config.py` (`BenchmarkConfig` dataclass), `metrics.py` (RMSE / multi-level coverage / summary stats), `plotting.py` (scatter + strip figures; **no forest-plot helper** — Phase 16 must add one), and `run_all_benchmarks.py` (CLI glue with `VALID_COMBOS`). Phase 16 = 1 new runner + 1 new fixture generator + 1 forest-plot helper + 1 registry-tuple entry (`("task_bilinear", "svi")`).

2. **`run_svi` cannot forward keyword-only `b_masks` / `stim_mod`.** It takes `model_args: tuple` positionally (`guides.py:195`) and calls `svi.step(*model_args)` (line 332). This is a **known gap** documented in `test_posterior_extraction.py:281-284` ("we use a bare SVI loop because run_svi takes a positional model_args tuple"). Phase 16 **must** use a bare SVI loop OR add a `model_kwargs` parameter to `run_svi`. Recommendation below: add `model_kwargs: dict | None = None` to `run_svi` (5-line additive change) so the benchmark runner stays thin and the factory serves a cleaner API; this is a Phase 16 src-side change outside `benchmarks/`.

3. **Recommended ground truth (3-region, 1 driving, 1 modulator, B[1,0]=B[2,1]≠0):** A = `make_random_stable_A(N=3, density=0.5, seed=seed_i)` (matches `task_svi.py` + fixture family for RECOV-03 comparator); C = `[[0.5], [0], [0]]` (midpoint of `task_svi.py`'s `1.0` and `test_posterior_extraction.py`'s `0.25`); `B[1,0] = +0.4`, `B[2,1] = +0.3` (both positive, both > 0.1 to satisfy RECOV-04/05 masks, both ≤ 0.5·σ_prior to avoid Pitfall B1 eigenvalue blowup and stay within the safe-ROI per Phase 13 BILIN-06 3σ test); single modulator epoch schedule: **4 epochs of 12s ON at [20, 65, 110, 155]s over 200s total** (yields `n_eff ≈ 48/2 = 24` effective events per free B element, comfortably above Rowe's 20-threshold from Pitfall B2).

4. **Linear baseline for RECOV-03 = run inline within Phase 16's own runner** (same seeds, bilinear fixture family with `b_masks=None`). This is the cleanest approach because (a) the v0.2.0 committed `benchmark_results.json` only has 5-seed spectral/rdcm results (no 10-seed linear task baseline at fixed topology to compare against), (b) `b_masks=None` activates the linear short-circuit (bit-exact; MODEL-04 L3), and (c) the 1.25× threshold is inherently comparative so inline is most defensible. Runtime cost: doubles wall-time budget, but per-seed linear baseline is ~6-7s at 500 steps (verified: `benchmark_results.json:167` shows spectral SVI `mean_time = 6.78s` at similar step count).

5. **Plan-sizing recommendation: 3 plans, 2 waves.** Wave 1 = plan 16-01 (fixture generator + runner skeleton + ground truth). Wave 2 parallel = 16-02 (metric helpers + forest plot + acceptance table) and 16-03 (factory-callable hook + linear-baseline inline + `run_svi` kwarg extension + end-to-end integration test). Critical path: 16-01 (~400 LoC); 16-02 and 16-03 (~300 LoC each).

**Primary recommendation:** Add `model_kwargs: dict | None = None` to `run_svi` in plan 16-03 so the bilinear runner can call the shared SVI helper cleanly; then replicate the `task_svi.py` pattern with bilinear kwargs + new metrics module (`benchmarks/bilinear_metrics.py`) + forest-plot helper (`benchmarks/plotting.py::plot_bilinear_b_forest`).

---

## 1. v0.2.0 Benchmark Infrastructure

### 1.1 Directory Map (verified by `ls` and direct read)

```
benchmarks/
├── __init__.py
├── calibration_analysis.py          (19 KB; coverage-by-guide calibration)
├── calibration_sweep.py             (19 KB; guide-type sweep harness)
├── config.py                        (5 KB; BenchmarkConfig dataclass)
├── fixtures.py                      (4 KB; load_fixture / get_fixture_count)
├── generate_fixtures.py             (12 KB; .npz generator for task/spectral/rdcm)
├── metrics.py                       (10 KB; RMSE, coverage, summary stats)
├── plotting.py                      (51 KB; scatter, strips, amortization gap)
├── run_all_benchmarks.py            (12 KB; CLI + registry dispatch)
├── timing_profiler.py               (10 KB; per-step SVI timing helpers)
├── figures/                         (gitignored; contains output PNG/PDFs)
├── results/                         (committed; has benchmark_results.json)
└── runners/
    ├── __init__.py                  (RUNNER_REGISTRY mapping)
    ├── rdcm_vb.py                   (22 KB)
    ├── spectral_amortized.py        (16 KB)
    ├── spectral_svi.py              (11 KB)
    ├── spm_reference.py             (5 KB)
    ├── task_amortized.py            (16 KB)
    └── task_svi.py                  (11 KB; closest Phase 16 template)
```

**Phase 16 additions (minimum):**

- `benchmarks/runners/task_bilinear.py` — new runner (~400 LoC, mirroring `task_svi.py`).
- `benchmarks/bilinear_metrics.py` — `compute_b_metrics(...)` producing sign/coverage/shrinkage (~150 LoC).
- `benchmarks/plotting.py` — add `plot_bilinear_b_forest(...)` (~150 LoC; section 5.3 below).
- `benchmarks/generate_fixtures.py` — add `generate_task_bilinear_fixtures(...)` subcommand (~80 LoC).
- `benchmarks/runners/__init__.py` — add `("task_bilinear", "svi"): run_task_bilinear_svi` to `RUNNER_REGISTRY`.
- `benchmarks/run_all_benchmarks.py` — extend `VALID_COMBOS` + `VARIANT_EXPANSION` (~5 LoC).

**Phase 16 does NOT need:**

- New `BenchmarkConfig` fields (injected at runtime; see Section 6).
- New `.npz` field decoding logic (`fixtures.py` loops over `data.files` generically; new field names auto-load).
- Changes to `calibration_*.py` (orthogonal to recovery benchmark).

### 1.2 `BenchmarkConfig` (file: `benchmarks/config.py`)

**Existing fields (`config.py:56-69`):**

```python
variant: str
method: str
n_datasets: int = 20
n_regions: int = 3
n_svi_steps: int = 3000
seed: int = 42
quick: bool = False
output_dir: str = "benchmarks/results"
save_figures: bool = True
figure_dir: str = "figures"
guide_type: str = "auto_normal"
n_regions_list: list[int] = field(default_factory=lambda: [3])
elbo_type: str = "trace_elbo"
fixtures_dir: str | None = None
```

**Defaults for Phase 16 via `BenchmarkConfig.quick_config("task_bilinear", "svi")`:** must be added to `quick_config.defaults` dict (`config.py:92-98`) and `full_config.defaults` (`config.py:130-136`). Recommended:

```python
defaults["task_bilinear"] = {"n_datasets": 3, "n_svi_steps": 500}    # quick
defaults["task_bilinear"] = {"n_datasets": 10, "n_svi_steps": 1500}  # full
```

**Key insight — `BenchmarkConfig` supports new variants without schema changes.** `quick_config` and `full_config` use `defaults.get(variant, fallback)` (`config.py:99`, `:137`), so adding a new variant key is a 2-line additive edit. No dataclass field additions required.

**CONTEXT.md confirms:** factory for `stimulus_mod` is **NOT stored in `BenchmarkConfig`** (kept for `.npz` reproducibility); it is injected at runtime at the runner's call site. See Section 6.

### 1.3 `.npz` Fixture Infrastructure (`fixtures.py` + `generate_fixtures.py`)

**Layout convention (verified, `fixtures.py:6-14`):**

```
benchmarks/fixtures/
    task_bilinear_3region/
        dataset_000.npz
        dataset_001.npz
        ...
        manifest.json          # {"variant": "task_bilinear", "n_regions": 3, ...}
```

**`load_fixture` behavior (`fixtures.py:35-90`):**

- Generic: loops over `data.files` and converts every key to a `torch.Tensor` (no schema validation).
- **Special-case complex reconstruction** at `fixtures.py:27-32` for `("csd_real", "csd_imag", "csd")` pairs; Phase 16 needs no complex fields so `_COMPLEX_PAIRS` is not touched.
- Returns `dict[str, torch.Tensor]`.

**Fixture fields Phase 16 must save per `.npz` (additive to `task_svi.py`'s fields):**

```python
save_dict = {
    # Carried from task_svi.py fixture shape (`generate_fixtures.py:112-123`):
    "A_true": A_true.numpy(),                      # (N, N)
    "C": C.numpy(),                                # (N, M) driving
    "bold": sim["bold"].detach().numpy(),          # (T, N)
    "bold_clean": sim["bold_clean"].detach().numpy(),
    "stimulus_times": stim["times"].numpy(),       # driving breakpoints
    "stimulus_values": stim["values"].numpy(),
    "TR": np.array(2.0),
    "SNR": np.array(3.0),                          # RECOV says SNR=3 (not 5)
    "duration": np.array(200.0),
    "seed": np.array(seed_i),
    # NEW bilinear fields (Phase 16):
    "B_true": B_true.numpy(),                      # (J, N, N) stacked
    "b_mask_0": b_masks[0].numpy(),                # (N, N) — one per modulator
    "stim_mod_times": stim_mod_dict["times"].numpy(),
    "stim_mod_values": stim_mod_dict["values"].numpy(),
    "J": np.array(1),                              # number of modulators
}
```

Loader is generic — **no `fixtures.py` changes needed** to read these. The runner handles reconstructing `b_masks: list[Tensor]` from `b_mask_0`, `b_mask_1`, ...

**Fixture generator pattern** (mirrors `generate_task_fixtures` at `generate_fixtures.py:60-133`):

```python
def generate_task_bilinear_fixtures(
    n_regions: int,          # locked = 3 for Phase 16
    n_datasets: int,
    seed: int,
    output_dir: str,
    *,
    stim_mod_factory: Callable | None = None,   # HGF hook (Section 6)
) -> None:
    subdir = Path(output_dir) / f"task_bilinear_{n_regions}region"
    os.makedirs(subdir, exist_ok=True)
    for i in range(n_datasets):
        seed_i = seed + i
        torch.manual_seed(seed_i)
        A_true = make_random_stable_A(n_regions, density=0.5, seed=seed_i)
        C = torch.zeros(n_regions, 1, dtype=torch.float64); C[0, 0] = 0.5
        b_mask = torch.zeros(n_regions, n_regions, dtype=torch.float64)
        b_mask[1, 0] = 1.0; b_mask[2, 1] = 1.0
        B_true = torch.zeros(1, n_regions, n_regions, dtype=torch.float64)
        B_true[0, 1, 0] = 0.4; B_true[0, 2, 1] = 0.3
        stim = make_block_stimulus(n_blocks=5, block_duration=20.0,
                                   rest_duration=20.0, n_inputs=1)
        if stim_mod_factory is None:
            stim_mod_dict = make_epoch_stimulus(
                event_times=[20.0, 65.0, 110.0, 155.0],
                event_durations=[12.0] * 4,
                event_amplitudes=[1.0] * 4,
                duration=200.0, dt=0.01, n_inputs=1,
            )
        else:
            stim_mod_dict = stim_mod_factory(seed=seed_i)
        stim_mod = PiecewiseConstantInput(
            stim_mod_dict["times"], stim_mod_dict["values"],
        )
        sim = simulate_task_dcm(
            A_true, C, stim, duration=200.0, dt=0.01, TR=2.0, SNR=3.0,
            seed=seed_i, solver="rk4",
            B_list=[B_true[0]], stimulus_mod=stim_mod,
        )
        # ... save ...
```

### 1.4 Figure Pipeline (`plotting.py`)

**Public entry points (existing):**

- `plot_true_vs_inferred(results, output_dir, formats=("png",))` — scatter of A elements.
- `plot_metric_strips(results, output_dir, formats=("png",))` — 2x2 RMSE / Coverage / Correlation / Wall-time strips.
- `plot_amortization_gap(results, output_dir, formats=("png",))` — bar chart (not used by Phase 16).
- `generate_all_figures(results, output_dir, formats=("png",))` — aggregator called by `run_all_benchmarks.py:403`.

**What Phase 16 needs that doesn't exist:**

- **Forest plot for per-element B** (headline figure per CONTEXT.md):
  - For each of the 9 B elements (3×3), plot posterior median + 95% CI + `B_true` reference dot.
  - Inline shrinkage annotation per element (`std_post/std_prior` text or color tint).
- **Pass/fail acceptance table** as a figure or text artifact (CONTEXT.md "matches `/gsd:verify-work` format").

**Recommended signature (add to `plotting.py`):**

```python
def plot_bilinear_b_forest(
    results: dict,                        # full results JSON
    output_dir: str,
    *,
    formats: tuple[str, ...] = ("png", "pdf"),
    ci_level: float = 0.95,
) -> None:
    """Per-element forest plot of B recovery across seeds.

    For the task_bilinear entry in results, reads per-seed posterior
    samples of B, computes per-element median + CI_level CI across seeds,
    and plots horizontal error bars with B_true reference dots. Annotates
    each row with std_post / std_prior shrinkage (RECOV-07 inline).
    """
```

**Figure naming convention** (existing):
- PNG: `{output_dir}/{figure_name}.png` at `dpi=150` (`plotting.py:126`).
- PDF: save with `formats=("png", "pdf")` (`plotting.py:111-127`).

**Recommended Phase 16 figure filenames:**
- `b_forest_recovery.png` (headline).
- `acceptance_gates.png` (pass/fail table rendered as text or matplotlib table).
- `a_recovery_scatter.png` (reuse `plot_true_vs_inferred` — filter to only `task_bilinear` entries).

### 1.5 `run_all_benchmarks.py` Integration

**Required edits (3 lines total):**

1. `VALID_COMBOS` at line 58 — add `("task_bilinear", "svi"),`.
2. `VARIANT_EXPANSION` at line 69 — add `"task_bilinear": ["task_bilinear"],` (and optionally `"all"` expansion).
3. `--variant` choices at line 246 — add `"task_bilinear"`.

---

## 2. Bilinear SVI Configuration

### 2.1 Empirical timings (verified, Phase 15)

**From Phase 15 `test_bilinear_svi_smoke_3region_converges` (summary `15-01-SUMMARY.md:157`):**

- **40 SVI steps, 3 regions, J=1, 30s BOLD at TR=2.0, dt=0.5 → 29.8s wall time.**
- Linear `test_svi_loss_decreases` at 50 steps on the same machine ran in ~60s baseline.
- **Observed bilinear/linear per-step ratio at N=3, J=1: ~2x** (comfortably below the Pitfall B10 3-6× budget).

**Extrapolation to Phase 16 full-seed run** (conservative):

- 200s BOLD (~6.7× the smoke test duration) at 1500 SVI steps (~37.5× the smoke budget) with J=1:
  - Per-step ODE cost scales linearly with duration: 6.7× per-step.
  - Per-seed wall = 29.8s × 6.7 × (1500/40) = **~7500s ≈ 125 min per seed**.
- This is too slow. **Three mitigations available:**
  - (a) Reduce `dt` from 0.01 (simulator default) to 0.5 in the model's `t_eval` — already standard (`task_svi.py:173`).
  - (b) Reduce `num_steps` from 1500 to 500 — linear baseline at 500 converges acceptably (`task_svi.py:191-197`).
  - (c) Cap `n_datasets = 10` for full run (RECOV floor is ≥ 10 seeds).
- **Revised estimate: ~500 SVI steps, 10 seeds, 200s BOLD. Per seed wall ≈ 235s (v0.1.0 linear baseline)  × 2 ≈ 470s. Total run ≈ 10 × 470 ≈ 4700s ≈ 80 min.** Within 2-hour acceptance budget.

### 2.2 Locked config for Phase 16 SVI

| Parameter | Value | Source |
|-----------|-------|--------|
| `num_steps` | 500 | task_svi quick-config baseline + Phase 15 convergence |
| `lr` | 0.005 | `task_svi.py:193` |
| `clip_norm` | 10.0 | `task_svi.py:194` (= `run_svi` default) |
| `lr_decay_factor` | 0.01 | `task_svi.py:194` |
| `init_scale` | **0.005** | L2 locked (Phase 15 `15-01-SUMMARY.md:157`) — bilinear half-default |
| `optimizer` | `ClippedAdam` | `guides.py:307` (not overrideable; acceptable) |
| `elbo_type` | `trace_elbo` | `config.py:68` default |
| `num_particles` | 1 | `guides.py:200` default |
| Model `dt` | 0.5 | `task_svi.py:173` (Pitfall B12 compliance; epoch widths 12s ≫ 0.5s) |
| Model `TR` | 2.0 | `task_svi.py:177` |
| Simulator `dt` | 0.01 | `task_svi.py:165` |

### 2.3 `run_svi` Fitness for Bilinear — CRITICAL GAP

**Verified from source (`guides.py:192-203`):**

```python
def run_svi(
    model, guide, model_args: tuple[Any, ...],
    num_steps=2000, lr=0.01, clip_norm=10.0, lr_decay_factor=0.01,
    num_particles=1, elbo_type="trace_elbo", guide_type=None,
) -> dict[str, Any]:
    ...
    svi = SVI(model, guide, optimizer, loss=elbo)
    for step in range(num_steps):
        loss = svi.step(*model_args)       # <-- POSITIONAL forward only
```

**The problem:** `task_dcm_model` has keyword-only `b_masks` and `stim_mod` (`task_dcm_model.py:127-129`). `svi.step(*model_args)` cannot reach them. Confirmed by Phase 15 precedent in `test_posterior_extraction.py:281-284`:

```python
# IMPORTANT: do NOT import run_svi -- we use a bare SVI loop because
# run_svi takes a positional model_args tuple, which cannot forward
# task_dcm_model's keyword-only b_masks / stim_mod kwargs.
```

**Two mitigation paths, ranked:**

**Path A (RECOMMENDED) — extend `run_svi` with `model_kwargs` parameter.**

5-line additive change (`guides.py`):

```python
def run_svi(
    model, guide, model_args,
    num_steps=2000, lr=0.01, clip_norm=10.0, lr_decay_factor=0.01,
    num_particles=1, elbo_type="trace_elbo", guide_type=None,
    model_kwargs: dict[str, Any] | None = None,     # NEW
) -> dict[str, Any]:
    ...
    kw = model_kwargs or {}
    for step in range(num_steps):
        loss = svi.step(*model_args, **kw)          # CHANGED
```

Benefits:
- Phase 16 runner stays thin (no bare SVI loop duplicated).
- All existing `run_svi` callers unchanged (default `None → {}`).
- Post-Laplace `laplace_approximation(*model_args)` still works (no `**kwargs` needed; bilinear is deferred from AutoLaplace anyway per Phase 15).

Risk: none — purely additive; any tests asserting `run_svi` signature shape would need updating, but `grep -n "def run_svi"` in the codebase shows no such tests.

**Path B — inline bare SVI loop inside `task_bilinear.py` runner.**

Replicates the bare loop from `test_posterior_extraction.py:322-340`. Same semantics, more code in the runner. Rejected because:
- Duplicates ELBO/optimizer setup, LR decay, and NaN ELBO guard already in `run_svi`.
- Drift risk: future `run_svi` improvements won't propagate.

**Recommendation:** Path A, scoped to plan 16-03.

### 2.4 NaN Handling

**Two layers already exist; both are needed for bilinear:**

- **Model-level (post-Phase 15, `task_dcm_model.py:379-381`):** `torch.isnan(predicted_bold).any() or torch.isinf(...).any() → zero_like(...).detach()`. This prevents NaN likelihood. Ported from `amortized_wrappers.py:143-145`.
- **`run_svi`-level (`guides.py:335-337`):** `if math.isnan(loss): raise RuntimeError(f"NaN ELBO at step {step}")`. This is the last-line guard; with the model-level NaN zero-fill, it should NEVER fire in practice. If it does, inspect the specific draw.

**Phase 16 runner must:**

- Wrap each seed's SVI in `try: ... except RuntimeError as e: n_failed += 1` (mirror `task_svi.py:270-272`).
- Silence `pyro_dcm.stability` logger for the duration of SVI (same autouse-fixture pattern as Phase 15 `test_task_dcm_model.py:735-739`). Without this silencing, test/CI logs fill with D4 stability warnings — non-fatal but noisy.

### 2.5 Guide Selection

**CONTEXT.md Claude's Discretion:** pick one primary guide for acceptance; optional 2-3 guide sidebar on subset of seeds.

**Options and tradeoffs** (verified from `guides.py:44-51` + `15-02-SUMMARY.md:129`):

| Guide | Pros | Cons | Bilinear cost (N=3, J=1) |
|-------|------|------|--------------------------|
| `auto_normal` | Fastest; mean-field; robust; matches `task_svi.py` default; supports `init_scale` | Mean-field CI underestimates correlations (Pitfall v0.2.0 P1 — WORSE under bilinear) | 1× |
| `auto_lowrank_mvn` | Captures A↔B correlations; supports `init_scale`; rank=2 default | Slightly slower; full-rank memory irrelevant at N=3 | ~1.3× |
| `auto_iaf` | Captures non-Gaussian posterior; auto-discovers bilinear sites | No `init_scale`; slower; hidden_dim tuning needed | ~2× |

**Recommendation for Phase 16 acceptance:** **`auto_normal` with `init_scale=0.005`** (L2 locked). Primary because:
- Matches v0.2.0 task_svi precedent (single-config comparator for RECOV-03).
- Fastest (1× baseline) — important for ≥10 seed run within budget.
- Known to converge on bilinear 3-region J=1 per Phase 15 smoke.
- Mean-field coverage on B is a risk for RECOV-06 (P1 worsens under bilinear) — **primary risk for Phase 16**; if coverage fails, sidebar comparison with `auto_lowrank_mvn` at 3 seeds informs the narrative.

**Configurability:** Benchmark runner reads `config.guide_type` (`config.py:66`). Default via `BenchmarkConfig.quick_config("task_bilinear", "svi")` should propagate as "auto_normal" with `init_scale=0.005` passed separately through a new `init_scale` field on `BenchmarkConfig` OR hardcoded in the runner (cleaner: **hardcode in runner** — L2 says "callers pass explicitly").

---

## 3. Ground-Truth Fixture Design

### 3.1 A matrix (3 regions)

**Use `make_random_stable_A(n_regions=3, density=0.5, seed=seed_i)`** from `task_simulator.py:946`.

**Why this specific helper:**

- **Matches v0.2.0 task_svi fixture family** (`generate_fixtures.py:93-95`). This is REQUIRED for RECOV-03 ("A RMSE ≤ 1.25× linear-baseline RMSE") to be an apples-to-apples comparison.
- Diagonal = `-0.5` (self-inhibition at SPM default, matching `A_free=0 → -exp(0)/2 = -0.5`, `task_simulator.py:971-972`).
- Off-diagonal: sparse (density=0.5), strength range `(0.0, 0.3)` (`task_simulator.py:950`). Spectral radius < 0.5 confirmed by `torch.linalg.eigvals(A).real.max() < 0` test (`task_simulator.py:998`).
- Seed varies per dataset; 10 seeds gives 10 distinct A topologies — this is intended for RECOV's "≥10 seeds" requirement (different A's per seed exercises recovery across connectivity patterns).

### 3.2 B matrices (non-zero: B[1,0], B[2,1])

**Recommended magnitudes:** `B[1,0] = 0.4`, `B[2,1] = 0.3` (both positive; asymmetric hierarchy V1→V5→SPL as per CONTEXT.md).

**Rationale:**

- **Stability (Pitfall B1 + D4):** Gershgorin bound says `max Re(eig(A_eff))` at u_mod=1.0 with diag(A)=-0.5 plus B row-sum `{0.4 + 0.3} = 0.7` off-diagonal → worst-case row-sum in A_eff = `-0.5 + 0.7 = +0.2`. Borderline-positive, but sustained only for 12s epochs → `exp(0.2 × 12) ≈ 11` amplification (not the `exp(1.5 × 10) ≈ 3e6` catastrophic case from Phase 15 `15-RESEARCH.md` Section 9). Safe margin.
- **Phase 13 BILIN-06 3σ safety anchor:** at B ≈ 3σ = 3×1.0 = 3.0 the bilinear forward is numerically stable per `13-03-SUMMARY.md:150`. 0.4 and 0.3 are well within that envelope.
- **RECOV-04 signal-above-threshold:** `|B_true| > 0.1` mask includes both elements (0.4 > 0.1, 0.3 > 0.1). Threshold RMSE ≤ 0.20 is meaningful (non-trivial fraction of signal).
- **RECOV-05 sign recovery:** both positive; asymmetric to avoid B[1,0] = B[2,1] = same-magnitude symmetry degeneracy.
- **RECOV-06 coverage_of_zero:** the 7 null B elements (`|B_true| = 0`) are all within `0.5 × σ_prior = 0.5` threshold by construction.

**Alternative magnitudes considered:**

| B[1,0], B[2,1] | Rationale | Rejected because |
|----------------|-----------|------------------|
| (0.5, 0.5) | Simpler; exactly at 0.5·σ_prior | Symmetric; harder to detect asymmetric recovery quality; Gershgorin bound hits +0.5 row sum |
| (0.3, 0.3) | Conservative; well within RECOV-04 margin | Symmetric; low SNR vs BOLD noise at SNR=3 |
| (-0.4, -0.3) | Same magnitude, negative signs | Symmetry broken but negative B means inhibitory modulator — less biologically natural for V1→V5 hierarchy |
| (0.4, 0.3) | **RECOMMENDED** | Asymmetric; stable; above RECOV-04 signal floor; positive (biologically natural) |

### 3.3 C matrix (driving inputs)

**Recommended:** `C = [[0.5], [0.0], [0.0]]` — driving input on region 0 only.

**Why 0.5:**

- `task_svi.py:152-153` uses `C[0, 0] = 1.0`.
- `test_posterior_extraction.py:297` uses `C = [[0.25], [0.0], [0.0]]`.
- 0.5 is the midpoint — sufficient signal to drive the network without maxing out BOLD amplitude.
- Matches C prior `N(0, 1)` (`task_dcm_model.py:285-289`) at roughly 0.5σ, comfortably recoverable.

**c_mask:** `c_mask = [[1.0], [0.0], [0.0]]` — only `C[0,0]` is a free parameter.

### 3.4 Driving stimulus

**Recommended:** `make_block_stimulus(n_blocks=5, block_duration=20.0, rest_duration=20.0, n_inputs=1)`.

Total `5 × 40 = 200s` covers the full simulation duration. Matches `task_svi.py`'s `n_blocks=5, block_duration=15.0, rest_duration=15.0` shape but stretched slightly longer per block to ensure modulator epochs land predominantly during ON windows (the bilinear term `u_mod(t) · B` is most informative when x(t) is non-zero, which requires driving ON).

### 3.5 Modulator schedule (epochs)

**Recommended:** `make_epoch_stimulus(event_times=[20, 65, 110, 155], event_durations=[12]×4, event_amplitudes=[1]×4, duration=200.0, dt=0.01, n_inputs=1)`.

**Rationale:**

- **Pitfall B2 (Rowe 2015) identifiability floor:** `n_eff = Σ u_mod² / max(u_mod)² / J_free_per_element = (4 epochs × 12s) / 1 = 48` effective event-seconds, distributed across 2 free B elements → **24 per element**. Rowe's threshold is ≥ 20 per element. **Satisfied with margin.**
- **Pitfall B12 (rk4 mid-step blur):** `make_epoch_stimulus` is boxcar, NOT sticks — epoch width 12s ≫ model `dt=0.5s`, so all rk4 stages within the epoch see the same amplitude.
- **Overlap with driving blocks:** driving is ON during [0,20), [40,60), [80,100), [120,140), [160,180). Modulator epochs at [20,32), [65,77), [110,122), [155,167) → modulator fires during driving-OFF AND driving-ON windows (informative for B estimation because it decouples u_mod from u_drive variance).
- **Amplitude 1.0 (locked per CONTEXT.md).**

**Duration 200s (not 90s from `task_svi.py`):**

- 90s = 5 epoch-blocks × 18s = 5 × 18s (cut 3 blocks' worth of idle time).
- More simulation time means more identifiability → mitigation for Pitfall B2.
- Trade-off: 200s at TR=2.0 → T=100 BOLD samples; at dt=0.5 → T_fine=400 fine steps. Phase 15 smoke ran 60 fine steps in 29.8s → 400 fine steps ≈ 198s/40 steps ≈ 5s/step. 500 steps ≈ 2500s per seed. **Runtime estimate matches Section 2.1.**

### 3.6 SNR

**Recommended:** `SNR=3.0`. Per RECOV definition ("on ≥10 seeds at SNR=3"). `task_svi.py` uses SNR=5, `generate_fixtures.py` task uses SNR=5, `generate_fixtures.py` rDCM uses SNR=3. Phase 16 bilinear MUST use SNR=3 per requirements.

### 3.7 b_mask contract

```python
b_mask = torch.zeros(3, 3, dtype=torch.float64)
b_mask[1, 0] = 1.0     # B[1,0] free
b_mask[2, 1] = 1.0     # B[2,1] free
b_masks = [b_mask]     # J=1 list
```

All 7 other elements (including diagonal) are masked = 0 — so parameterize_B zeroes them out post-sampling. No `DeprecationWarning` fired (zero-diagonal mask, Phase 13 MODEL-03).

---

## 4. Metric Computation (RECOV-03..08)

### 4.1 RECOV-03: A RMSE ≤ 1.25 × linear-baseline RMSE

**Formula (per-seed A-RMSE):**

```python
A_inferred = parameterize_A(posterior["A_free"]["mean"])   # (N, N)
a_rmse = compute_rmse(A_true, A_inferred)                  # scalar
```

**Existing helper:** `benchmarks.metrics.compute_rmse` (`metrics.py:18-36`) — RMSE over the full `(N, N)` matrix. No mask applied — the full matrix is the convention (matches `task_svi.py:222`).

**Aggregate across seeds:** `mean(a_rmse_list)` for pass/fail; `compute_summary_stats(a_rmse_list)` for median/IQR.

**Linear baseline source** (Section 7): Phase 16 **runs the linear short-circuit inline** on the same seeds, reusing the bilinear fixture with `b_masks=None`. Computes `a_rmse_linear_list` → `mean(a_rmse_linear_list)`. Pass/fail:

```python
pass_rmse = mean(a_rmse_bilinear_list) <= 1.25 * mean(a_rmse_linear_list)
```

### 4.2 RECOV-04: B RMSE ≤ 0.20 on |B_true| > 0.1 elements

**Formula (per-seed):**

```python
B_posterior = posterior["B"]["mean"]               # (J, N, N) masked
mask_nonzero = torch.abs(B_true) > 0.1             # (J, N, N) bool
b_rmse_nonzero = torch.sqrt(
    ((B_true - B_posterior)[mask_nonzero] ** 2).mean()
).item()
```

**NEW helper to add in `benchmarks/bilinear_metrics.py`:**

```python
def compute_b_rmse_magnitude(
    B_true: torch.Tensor,                      # (J, N, N)
    B_inferred: torch.Tensor,                  # (J, N, N)
    *,
    magnitude_threshold: float = 0.1,
) -> float:
    """B RMSE restricted to |B_true| > threshold elements (RECOV-04)."""
    mask = torch.abs(B_true) > magnitude_threshold
    if not mask.any():
        return 0.0                             # vacuous; no non-null elements
    return torch.sqrt(((B_true - B_inferred)[mask] ** 2).mean()).item()
```

**Aggregate across seeds:** `mean(b_rmse_nonzero_list)`.

### 4.3 RECOV-05: sign_recovery_nonzero ≥ 80% on |B_true| > 0.1

**Formula (per-seed):**

```python
mask_nonzero = torch.abs(B_true) > 0.1
sign_match = torch.sign(B_posterior) == torch.sign(B_true)
sign_recovery_seed = sign_match[mask_nonzero].float().mean().item()
```

**Aggregate:** For 2 non-zero elements × 10 seeds = 20 (seed, element) pairs. Two interpretations:

- **(a) Pooled:** compute `sign_recovery_total = sum(matches) / sum(elements_nonzero)` across all seeds. Simpler; matches natural reading of "80% recovery."
- **(b) Per-seed mean, then average:** `mean(sign_recovery_seed_list)`. More robust to seed-to-seed variance but harder to interpret at 2 elements per seed.

**Recommendation: (a) pooled** — for 2 elements per seed, per-seed mean is always in `{0, 0.5, 1.0}` which produces a pathological distribution. Pooled gives a clean fraction over all 20 non-zero instances.

### 4.4 RECOV-06: coverage_of_zero ≥ 85% on |B_true| < 0.5·σ_prior

**Formula:**

- σ_prior = √`B_PRIOR_VARIANCE` = √1.0 = 1.0. Threshold = 0.5.
- All 7 zero-truth B elements (of the 9 per modulator, 2 are non-zero) have `|B_true| = 0 < 0.5` — included in the mask.
- Coverage: does the 95% CI of the B posterior contain zero?

**Implementation (95% CI from posterior samples):**

```python
B_samples = posterior["B"]["samples"]          # (S, J, N, N)
ci_lo = torch.quantile(B_samples.float(), 0.025, dim=0)
ci_hi = torch.quantile(B_samples.float(), 0.975, dim=0)
mask_zero = torch.abs(B_true) < 0.5
contains_zero = (ci_lo <= 0) & (0 <= ci_hi)
coverage_of_zero_seed = contains_zero[mask_zero].float().mean().item()
```

**Aggregate:** `mean(coverage_of_zero_list)` or pool as (c) above — recommend pool because per-seed denominator is 7 (more stable than 2).

**New helper in `bilinear_metrics.py`:**

```python
def compute_coverage_of_zero(
    B_true: torch.Tensor,
    B_samples: torch.Tensor,                   # (S, J, N, N)
    *,
    null_threshold: float = 0.5,
    ci_level: float = 0.95,
) -> float:
    """Fraction of |B_true| < null_threshold elements whose CI contains zero."""
    alpha = (1.0 - ci_level) / 2.0
    lo = torch.quantile(B_samples.float(), alpha, dim=0)
    hi = torch.quantile(B_samples.float(), 1.0 - alpha, dim=0)
    mask = torch.abs(B_true) < null_threshold
    if not mask.any():
        return 0.0
    contains = (lo <= 0) & (0 <= hi)
    return contains[mask].float().mean().item()
```

### 4.5 RECOV-07: Identifiability shrinkage `std_post / std_prior ≤ 0.7` (soft)

**σ_prior = 1.0 (constant from `B_PRIOR_VARIANCE`).**

**std_post extraction depends on guide family:**

| Guide | How to compute `std_post` for site `B_free_j` |
|-------|-----------------------------------------------|
| `auto_normal` | `posterior["B_free_j"]["std"]` — already computed by `extract_posterior_params` via `samples.float().std(dim=0)` (`guides.py:472-473`). Shape `(N, N)` — element-wise. |
| `auto_lowrank_mvn` | Same — `extract_posterior_params` uses `Predictive` which samples in original shape, then computes std on samples. Includes cross-site correlation automatically. |
| `auto_iaf` | Same — same Predictive path; samples reflect the flow-pushed posterior shape. |

**Key insight: `extract_posterior_params` is GUIDE-AGNOSTIC** (`guides.py:359-483`). It runs `Predictive(model, guide=guide, num_samples=1000)`, draws samples, computes `mean`/`std`/`samples` per site. The `std` field is **the posterior standard deviation estimated from samples** — works uniformly for all six guide families. No per-guide-family branching needed.

**Phase 16 formula (per B element):**

```python
std_post_B = posterior["B_free_0"]["std"]       # (N, N); per-element posterior std
sigma_prior = 1.0                               # √B_PRIOR_VARIANCE
shrinkage = std_post_B / sigma_prior            # (N, N); lower is better
```

**Soft target:** `shrinkage <= 0.7` for the 2 free (non-null) B elements per D3. Reported per element in the forest plot; DOES NOT block acceptance (CONTEXT.md + RECOV-07 explicit "does NOT block acceptance").

**Aggregate across seeds:** per free element, compute `mean(shrinkage_seed_list)`. Report in forest plot subtitle or color-tint.

### 4.6 RECOV-08: Wall-time vs linear baseline

**Measurement:** `t0 = time.time(); ...; elapsed = time.time() - t0` around the SVI loop (mirror `task_svi.py:190-197`). `config.n_svi_steps` from config.

**Per-seed comparison table:**

```python
time_bilinear = mean(time_bilinear_list)
time_linear   = mean(time_linear_list)    # from inline baseline runs
ratio = time_bilinear / time_linear
```

**Flag condition:** `ratio > 10` → warning emitted; does not fail acceptance.

**CONTEXT.md baseline reference:** "linear baseline: 235s" — this is the `v0.1.0 N=3 linear 500-step SVI` number from `PITFALLS.md:463` (`CLAUDE.md` citation). It is **NOT in the current `benchmark_results.json`** (the committed file's `task` entry doesn't exist; only `spectral`, `rdcm_rigid`, `rdcm_sparse`). The 235s figure comes from historical v0.1.0 development, not from the current codebase.

**Implication for Phase 16:** the 235s baseline is a **historical reference only**. Phase 16 **must** compute its own linear-baseline wall-time inline using the same fixture + same machine (to eliminate machine-dependence variance in the ratio).

---

## 5. HGF Forward-Compat Hook

### 5.1 Signature

**Per CONTEXT.md:** `stimulus_mod_factory: Callable[[seed], PiecewiseConstantInput]`. Injected at runtime at the runner's call site; NOT stored in `BenchmarkConfig`.

**Recommended concrete signature:**

```python
from typing import Callable, TypedDict
from pyro_dcm.utils.ode_integrator import PiecewiseConstantInput

class StimulusModBreakpoints(TypedDict):
    times: torch.Tensor       # (K,) float64
    values: torch.Tensor      # (K, J) float64

# Factory contract:
StimulusModFactory = Callable[[int], StimulusModBreakpoints]

def run_task_bilinear_svi(
    config: BenchmarkConfig,
    *,
    stimulus_mod_factory: StimulusModFactory | None = None,
) -> dict[str, Any]:
    """If stimulus_mod_factory is None, use make_epoch_stimulus with the
    default [20, 65, 110, 155] s × 12 s epoch schedule. v0.3.1 HGF plugin
    will pass a factory returning a LinearInterpolatedInput (SIM-06)
    wrapped in the same breakpoint dict shape."""
```

**Why factory returns a `dict[str, torch.Tensor]` (breakpoint dict), NOT a `PiecewiseConstantInput`:**

- Matches the existing `make_block_stimulus` / `make_epoch_stimulus` output shape (`task_simulator.py:487`, `:943`).
- The runner handles `.npz` saving of `stim_mod_times` + `stim_mod_values` columns — which requires raw tensors, not a class instance.
- The runner wraps in `PiecewiseConstantInput(dict["times"], dict["values"])` at the call site before handing to `task_dcm_model` (same pattern as `task_svi.py:140-143`).

**Factory input parameter:** just `seed: int`. Duration/dt/n_inputs come from the factory's closure (set at construction time by the caller). Rationale: factories have closure-captured context for their specific signal type; duration coupling is a caller concern.

**v0.3.1 HGF factory shape (for planner's reference):**

```python
def make_hgf_factory(
    duration: float, dt: float,
    belief_trajectory_generator: Callable[[int], torch.Tensor],
) -> StimulusModFactory:
    def factory(seed: int) -> StimulusModBreakpoints:
        trajectory = belief_trajectory_generator(seed)
        times = torch.arange(0, duration, dt, dtype=torch.float64)
        values = trajectory.reshape(-1, 1)   # (K, 1) one-modulator
        return {"times": times, "values": values}
    return factory
```

### 5.2 Placeholder mock factory for Phase 16 tests

**Purpose:** exercise the factory hook end-to-end in Phase 16 tests (CONTEXT.md: "exercised by a placeholder mock factory in Phase 16's test suite so the indirection is proven wired — not a theoretical API").

**Concrete sketch:**

```python
def mock_sinusoid_factory(
    duration: float = 200.0, dt: float = 0.01, amplitude: float = 0.5,
) -> StimulusModFactory:
    """Mock factory producing a 0.05 Hz sinusoid modulator.

    NOT physiologically meaningful; used only to verify the factory hook
    plumbing in Phase 16 tests. v0.3.1 SIM-06 replaces with HGF trajectory.
    """
    def factory(seed: int) -> StimulusModBreakpoints:
        torch.manual_seed(seed)
        times = torch.arange(0, duration, dt, dtype=torch.float64)
        values = (amplitude * torch.sin(0.05 * 2 * torch.pi * times)).unsqueeze(-1)
        return {"times": times, "values": values}
    return factory
```

**Test assertion (plan 16-03):**

```python
def test_factory_hook_wiring():
    """Default factory + mock factory produce different stim_mod values."""
    config = BenchmarkConfig.quick_config("task_bilinear", "svi")
    config.n_datasets = 1
    default_result = run_task_bilinear_svi(config)
    mock_result = run_task_bilinear_svi(
        config,
        stimulus_mod_factory=mock_sinusoid_factory(duration=200.0),
    )
    # Different stim_mod → different SVI trajectory → different RMSE.
    assert default_result["rmse_list"][0] != mock_result["rmse_list"][0]
```

### 5.3 Factory is NOT stored in `BenchmarkConfig`

Confirmed by CONTEXT.md ("Config enum/alternative was rejected; factory keeps the v0.2.0 `.npz` reproducibility path clean").

**Implications:**

- `run_all_benchmarks.py` CLI always uses the default factory (cannot pass a callable via CLI).
- Custom factories (HGF, mock, sweep experiments) are invoked by calling `run_task_bilinear_svi(config, stimulus_mod_factory=custom)` directly in Python scripts/tests, bypassing `run_all_benchmarks.py`.
- `.npz` fixtures pre-generate with the default factory; if a custom factory is used, the runner generates fixtures on-the-fly (like the `fixtures_dir=None` path in `task_svi.py:146-167`).

---

## 6. Linear Baseline Strategy (RECOV-03)

**Decision: run linear baseline inline within Phase 16's runner, same seeds, same fixtures.**

**Rationale:**

1. **No committed task-linear 3-region baseline at SNR=3, duration=200s.** The v0.1.0 figure "235s" is a historical CLAUDE.md reference (per `PITFALLS.md:463`). Current `benchmark_results.json` has no task entry (only spectral/rdcm). The `task_svi.py` default SNR=5 + duration=90s does not match Phase 16's SNR=3 + duration=200s design. Cached values from v0.2.0 cannot serve as a direct comparator.

2. **Bit-exact linear short-circuit** (Phase 15 MODEL-04 L3): `task_dcm_model(..., b_masks=None)` is bit-exact equivalent to pre-Phase-15 linear task-DCM. The runner can fit the linear model on the same fixtures by passing `b_masks=None, stim_mod=None`.

3. **Single-seed sharing eliminates variance.** Using identical seeds for linear and bilinear means the A_true / noise pattern is identical, so `a_rmse_bilinear - a_rmse_linear` isolates the B-pricing inflation (Pitfall B13) as intended.

4. **Runtime cost is acceptable.** Linear 500-step SVI at N=3 in the current era is ~6-7s per seed (per `benchmark_results.json:167` spectral baseline which has similar compute). 10 seeds = ~70s total linear baseline. Added to bilinear total ~80 min → 81 min — negligible.

**Implementation within runner:**

```python
def run_task_bilinear_svi(config, *, stimulus_mod_factory=None):
    for i in range(config.n_datasets):
        seed_i = config.seed + i
        data = _generate_or_load_bilinear_fixture(i, config, stimulus_mod_factory)

        # --- LINEAR BASELINE: same fixture, b_masks=None ---
        linear_posterior, linear_time = _fit(
            task_dcm_model, model_args_linear=(
                data["bold"], data["stimulus"], data["a_mask"],
                data["c_mask"], data["t_eval"], 2.0, 0.5,
            ),
            model_kwargs={},   # empty → linear short-circuit
        )
        a_rmse_linear = compute_rmse(
            data["A_true"],
            parameterize_A(linear_posterior["A_free"]["mean"]),
        )

        # --- BILINEAR: same fixture, b_masks non-empty ---
        bilinear_posterior, bilinear_time = _fit(
            task_dcm_model, model_args=(...),
            model_kwargs={"b_masks": data["b_masks"], "stim_mod": data["stim_mod"]},
        )
        a_rmse_bilinear = compute_rmse(
            data["A_true"],
            parameterize_A(bilinear_posterior["A_free"]["mean"]),
        )

        # Append both to tracking lists.
```

**Pass/fail (RECOV-03):**

```python
pass_recov03 = mean(a_rmse_bilinear_list) <= 1.25 * mean(a_rmse_linear_list)
```

**Reporting:** a 2-row pass/fail table entry:

```
| Criterion | Observed | Threshold | Pass? |
| RECOV-03  | 0.X (bilinear) vs 0.Y (linear) | 1.25× linear | True/False |
```

---

## 7. Pitfalls & New Risks

### 7.1 Pitfalls inherited from v0.3.0 PITFALLS.md

| # | Title | Phase 16 status | Mitigation |
|---|-------|-----------------|------------|
| **B1** | Sustained large u·B causes positive eigenvalues | MITIGATED | Phase 15 NaN-safe BOLD guard + L2 init_scale=0.005 + B_true ≤ 0.4 (Gershgorin bound max Re = +0.2, sustained only for 12s epoch → exp(+2.4) ≈ 11× amplification, safe) |
| **B2** | B non-identifiable under sparse/short events | MITIGATED | Epoch schedule 4×12s → n_eff = 24 per free element > Rowe threshold 20; RECOV-07 shrinkage metric surfaces any residual identifiability issue |
| **B10** | Per-step ODE cost 3-6× linear | MONITORED | RECOV-08 tracks; 2× observed at Phase 15 smoke; runtime budget ~80 min for 10 seeds |
| **B11** | SVI convergence slower for bilinear | LOW RISK | 500 SVI steps proven adequate at Phase 15 smoke (40 steps already decreases ELBO); Phase 16 uses 500 steps with `lr_decay_factor=0.01` |
| **B12** | rk4 mid-step blur on sticks | NOT APPLICABLE | Phase 16 uses boxcar epochs (`make_epoch_stimulus`, duration 12s ≫ model dt 0.5s) |
| **B13** | A-RMSE inflation under bilinear pricing | MITIGATED | RECOV-03 uses relative threshold 1.25× |

### 7.2 Phase 16-specific NEW risks

**N1: Mean-field guide understates B posterior correlation with A — RECOV-06 coverage risk.**

Pitfall v0.2.0 P1 (mean-field coverage ceiling) **WORSENS** under bilinear (PITFALLS.md:648): "Mean-field coverage on B will be strictly worse than on A. Expected ceiling drops ~0.80 → ~0.65 for B elements." 

RECOV-06 target is 85% coverage_of_zero. If A↔B posterior correlations dominate, `auto_normal` may underperform by 20+ percentage points.

**Mitigation:**
- Primary run uses `auto_normal` but if coverage_of_zero fails, sidebar run with `auto_lowrank_mvn` on a 3-seed subset to validate. Document the gap in the report.
- If `auto_lowrank_mvn` also fails, this is a milestone-level insight — not a Phase 16 bug, but a guide-capacity finding worth documenting in `docs/03_methods_reference/`.

**Severity:** MEDIUM. Direct threat to RECOV-06 acceptance.

**N2: A-matrix RMSE variance across seeds.**

At N=3, 10 seeds gives 10 distinct A_true's. If one seed draws an A with near-singular structure, its per-seed RMSE could be 3-5× the mean — inflating the aggregate and failing RECOV-03 without being a systemic issue.

**Mitigation:**
- Report both `mean(rmse)` and `median(rmse)` + IQR (already done by `compute_summary_stats`).
- Acceptance uses mean (matches threshold language "A RMSE ≤ 1.25 × linear-baseline RMSE"), but the forest plot and pass/fail table include median for robustness context.
- If a single seed drives failure, re-seeded re-run (11th seed) is acceptable — document the excluded seed transparently.

**Severity:** LOW-MEDIUM. Easy to diagnose from `rmse_stats` distribution.

**N3: `run_svi` kwarg extension regression risk.**

Adding `model_kwargs: dict | None = None` to `run_svi` is additive but `task_svi.py` / `spectral_svi.py` / `rdcm_vb.py` callers all pass positional `model_args` only. Default `None → {}` means existing call sites get an empty dict — zero behavioral change.

**Mitigation:** grep the codebase for `run_svi(` calls before submitting the change; all ~10 call sites should work unchanged. Add a unit test `test_run_svi_with_model_kwargs_forwards_bilinear` in `test_svi_runner.py` to guard against regression.

**Severity:** LOW.

**N4: Forest plot across 10 seeds / 9 elements is cluttered.**

9 elements × 10 seeds = 90 dots on one figure. Options:

- (a) Pool across seeds: 1 row per element, error bar across seeds × samples.
- (b) Stack by seed: 9 rows per seed, 10 sub-plots. Too big.
- (c) Summary: per-element median of per-seed medians + IQR of per-seed 95% CIs.

**Recommendation: (c).** Use per-seed posterior median as the "point estimate" for each element, then aggregate across seeds via median + IQR. Keeps the figure readable at ~9 rows.

**Severity:** LOW (presentation concern only).

**N5: Factory hook + fixture reproducibility tension.**

`.npz` fixtures save `stim_mod_times` / `stim_mod_values` — so if a user later re-runs with a DIFFERENT factory, the fixtures are stale. Either:

- (a) Re-generate fixtures on every call (slow).
- (b) Store a factory-hash in `manifest.json` and raise on mismatch (complex).
- (c) Document that `--fixtures-dir` uses frozen default-factory data; custom factories skip fixture cache.

**Recommendation: (c).** Matches CONTEXT.md "`.npz` reproducibility path clean"; custom factories are test/sweep artifacts, not reproducible-run artifacts.

**Severity:** LOW.

---

## 8. Plan-Sizing Recommendation

**3 plans, 2 waves, ~1000 LoC total (src + test + benchmark).**

```
Wave 1 (foundation, blocks all):
┌────────────────────────────────────────────────────┐
│ 16-01: task_bilinear runner + fixture generator   │
│  - benchmarks/generate_fixtures.py additions       │
│  - benchmarks/runners/task_bilinear.py             │
│  - benchmarks/runners/__init__.py registry entry  │
│  - benchmarks/run_all_benchmarks.py CLI updates   │
│  - No metrics extensions yet; uses existing        │
│    compute_rmse and compute_coverage_from_samples  │
│  - End-to-end SVI path works on 3 seeds (quick)   │
│  - ~400 LoC src + ~150 LoC test                   │
└────────────────────────────────────────────────────┘
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
Wave 2a (parallel):    Wave 2b (parallel):
┌─────────────────────┐  ┌────────────────────────────┐
│ 16-02: B metrics +  │  │ 16-03: HGF hook + run_svi  │
│  forest plot +      │  │  kwargs + linear baseline  │
│  acceptance table   │  │  inline + placeholder mock │
│  - bilinear_metrics │  │  - guides.py:run_svi +=     │
│    module           │  │    model_kwargs             │
│  - plotting.py +=   │  │  - task_bilinear.py         │
│    plot_bilinear_   │  │    linear-baseline loop     │
│    b_forest         │  │  - factory injection test   │
│  - acceptance       │  │  - mock_sinusoid_factory    │
│    pass/fail table  │  │                             │
│  - ~200 LoC src +   │  │  - ~250 LoC src + ~150 LoC  │
│    ~100 LoC test    │  │    test                     │
└─────────────────────┘  └────────────────────────────┘
```

### Plan 16-01 Scope

**Title:** Bilinear task-DCM runner skeleton + fixture generator

**Deliverables:**

1. `benchmarks/generate_fixtures.py`: add `generate_task_bilinear_fixtures(n_regions=3, n_datasets, seed, output_dir)` subcommand. Exercises `simulate_task_dcm(..., B_list=[b], stimulus_mod=stim_mod)` with ground-truth magnitudes from Section 3.
2. `benchmarks/runners/task_bilinear.py`: `run_task_bilinear_svi(config) -> dict`. Mirrors `task_svi.py` structure but passes bilinear kwargs. Uses bare SVI loop (16-03 will refactor to `run_svi` with kwargs).
3. `benchmarks/runners/__init__.py`: register `("task_bilinear", "svi"): run_task_bilinear_svi`.
4. `benchmarks/run_all_benchmarks.py`: extend `VALID_COMBOS`, `VARIANT_EXPANSION`, CLI `--variant` choices.
5. `benchmarks/config.py`: extend `quick_config` / `full_config` defaults dicts.
6. Test: `tests/test_task_bilinear_benchmark.py::test_smoke_runs_3_seeds_quick` — runs `BenchmarkConfig.quick_config("task_bilinear", "svi")` with `n_datasets=3`, asserts no exceptions and `rmse_list` length = 3.

**Closes requirements:** RECOV-01, RECOV-02 (structural).

**Runtime target:** 3-seed quick test completes in < 15 min (3 × ~5 min at reduced steps).

### Plan 16-02 Scope

**Title:** Bilinear metrics module + acceptance gates

**Deliverables:**

1. `benchmarks/bilinear_metrics.py` (new): `compute_b_rmse_magnitude`, `compute_sign_recovery`, `compute_coverage_of_zero`, `compute_shrinkage`. Pure functions; ~150 LoC.
2. `benchmarks/plotting.py`: add `plot_bilinear_b_forest(results, output_dir)` — per-element forest plot with shrinkage annotation. ~150 LoC additive.
3. `benchmarks/plotting.py`: add `plot_acceptance_gates_table(results, output_dir)` — text or matplotlib table. ~80 LoC.
4. `benchmarks/plotting.py::generate_all_figures`: dispatch to new functions when `"task_bilinear"` entry present.
5. Tests: `tests/test_bilinear_metrics.py` with 5 unit tests covering each metric's edge cases (empty mask, 100% recovery, 0% recovery, etc.).

**Closes requirements:** RECOV-04, RECOV-05, RECOV-06, RECOV-07 (metric side).

### Plan 16-03 Scope

**Title:** `run_svi` kwarg extension + factory hook + linear baseline + integration test

**Deliverables:**

1. `src/pyro_dcm/models/guides.py`: add `model_kwargs: dict | None = None` to `run_svi`. 5-line change; docstring update; existing callers unchanged.
2. Refactor `task_bilinear.py` runner to use `run_svi(..., model_kwargs={"b_masks": ..., "stim_mod": ...})` instead of bare loop.
3. `task_bilinear.py`: add inline linear-baseline loop (Section 7); append to `a_rmse_linear_list`, `time_linear_list`; include in output dict.
4. `task_bilinear.py`: add `stimulus_mod_factory: Callable | None = None` parameter. If None, use default epoch schedule; else call `factory(seed=seed_i)` and pack into `PiecewiseConstantInput`.
5. `benchmarks/runners/task_bilinear.py`: define `mock_sinusoid_factory` at module level (for test).
6. Tests:
   - `tests/test_svi_runner.py::test_run_svi_with_model_kwargs` — verify `run_svi(..., model_kwargs={"b_masks": ...})` forwards to `svi.step(**kwargs)`.
   - `tests/test_task_bilinear_benchmark.py::test_factory_hook_wiring` — mock factory produces different results than default factory.
   - `tests/test_task_bilinear_benchmark.py::test_linear_baseline_matches_task_svi` — linear-baseline RMSE within ±20% of a standalone `run_task_svi` call on the same seeds.
7. End-to-end gate: `tests/test_task_bilinear_benchmark.py::test_10_seed_acceptance_gates` — runs `full_config` (10 seeds, 500 steps) and asserts all 4 RECOV acceptance criteria pass. **This test is the Phase 16 acceptance gate.** Gate runtime target ~80 min.

**Closes requirements:** RECOV-03, RECOV-08 (and closes the full RECOV-01..08 set via the end-to-end gate test).

### Rejected Alternative Decompositions

- **1 plan:** too large (~1000 LoC), hard to review, cannot parallelize.
- **4 plans:** fragmentation — no clean split for plan 4.
- **Merge 16-02 and 16-03:** rejected — metrics module and factory hook touch different files with no dependency; parallel execution faster.

---

## 9. Open Questions for Planner

1. **Guide variant lock:** primary = `auto_normal` (speed + Phase 15 precedent). Sidebar 3-seed comparison with `auto_lowrank_mvn` — **should be a plan 16-03 stretch goal or deferred to post-acceptance narrative?** Research recommends IN-SCOPE sidebar (3 seeds × 1 guide = marginal runtime cost ~5-10 min).

2. **Number of SVI steps:** 500 recommended (Section 2.2). **Planner may tighten to 300 if runtime budget pressure mounts** — Phase 15 smoke decreased ELBO within 40 steps, so 300 is viable but margin against N2 (seed variance) is thinner.

3. **n_datasets = 10 vs n_datasets = 15:** RECOV floor is ≥10. Research recommends 10 (minimum), **but planner may choose 15 for N2 variance robustness.** Cost: 1.5× runtime (80 min → 120 min).

4. **`model_kwargs` extension scope:** Section 2.3 Path A recommended. **Planner must decide: add to `run_svi` (shared infra) vs keep bare loop (localized).** Research recommends Path A for cleanliness.

5. **Forest plot aggregation:** Section 7.2 N4 recommended (c) — per-seed median → aggregate across seeds. **Planner should confirm or pick an alternative (a) or (b).**

6. **Acceptance gate test runtime:** Plan 16-03 end-to-end gate runs ~80 min. **Planner should decide if this test gets `@pytest.mark.slow` (skipped by default in CI) or runs nightly.** Research recommends `@pytest.mark.slow` + separate make-target `make acceptance-gate`.

7. **Forest plot CI level:** 95% (RECOV-06 implicit) or 90% (matches `task_svi.py:111`). Research recommends 95% to align with RECOV-06's implicit 95% CI for `coverage_of_zero`.

8. **Sidebar guide sweep: part of Phase 16 or deferred?** CONTEXT.md "Design note sidebar comparing 2-3 variants on a subset of seeds if runtime allows" — **planner decides based on Q1 decision.**

9. **Fixture regeneration policy:** `--fixtures-dir` uses default factory only (Section 7.2 N5 option (c)). **Planner confirms this matches the reproducibility-clean intent.**

10. **`run_all_benchmarks.py --variant all` expansion:** should `"all"` include `"task_bilinear"` by default, or keep Phase 16 behind explicit `--variant task_bilinear`? Research recommends EXPLICIT-ONLY for now (runtime gate; prevents accidental 80-min runs on default CI).

---

## 10. Sources

### Primary (HIGH confidence — direct code/doc read 2026-04-18)

- `benchmarks/config.py` — full read (146 lines).
- `benchmarks/fixtures.py` — full read (139 lines).
- `benchmarks/metrics.py` — full read (326 lines).
- `benchmarks/generate_fixtures.py` — full read (439 lines).
- `benchmarks/runners/task_svi.py` — full read (336 lines). Phase 16 template.
- `benchmarks/runners/__init__.py` — full read (31 lines).
- `benchmarks/run_all_benchmarks.py` — full read (414 lines).
- `benchmarks/plotting.py` — partial read (lines 1-450 of 51 KB; covers `plot_true_vs_inferred`, `plot_metric_strips`, helpers).
- `benchmarks/results/benchmark_results.json` — sampled first 100 lines + key timing fields. Confirmed no `task` entry; spectral `mean_time = 6.78s` at 500 steps.
- `src/pyro_dcm/models/task_dcm_model.py` — full read (400 lines). Post-Phase-15 bilinear signature confirmed.
- `src/pyro_dcm/models/guides.py` — full read (484 lines). `run_svi(model_args)` positional confirmed; `extract_posterior_params` guide-agnostic confirmed.
- `src/pyro_dcm/simulators/task_simulator.py` — sampled `make_block_stimulus` (425-488), `make_event_stimulus` (490-705), `make_epoch_stimulus` (707-943), `make_random_stable_A` (946-1000).
- `tests/test_task_dcm_model.py` — sampled lines 700-800 (bilinear SVI smoke + L2 init_scale=0.005 usage).
- `tests/test_posterior_extraction.py` — sampled lines 280-360 (bare SVI loop precedent + bilinear Predictive pattern).
- `.planning/phases/16-bilinear-recovery-benchmark/16-CONTEXT.md` — full read (152 lines).
- `.planning/phases/15-pyro-bilinear-model/15-01-SUMMARY.md` — sampled Phase 16 readiness section (lines 140-175).
- `.planning/phases/15-pyro-bilinear-model/15-RESEARCH.md` — full read (833 lines; structural template).
- `.planning/research/v0.3.0/PITFALLS.md` — relevant sections B1, B2, B10, B11, B12, B13 (lines 105-600).
- `.planning/REQUIREMENTS.md` — RECOV-01..08 (lines 53-60).

### Secondary (MEDIUM confidence — inherited references)

- SPM12 `spm_dcm_fmri_priors.m` — prior variance convention (cited via PITFALLS.md B8, Phase 15 research).
- Rowe et al. (2015), PMC4335185 — identifiability floor, cited by PITFALLS.md B2.
- Zeidman et al. (2019), PMC6711459 — self-connection identifiability, cited by PITFALLS.md B2.
- Baldy et al. (2025) — ADVI under-dispersion at 10-params, cited by PITFALLS.md B11.

### Tertiary (LOW confidence — not load-bearing)

- 235s v0.1.0 linear baseline (CLAUDE.md citation via PITFALLS.md:463) — historical reference only; not a Phase 16 comparator.

---

## Metadata

**Confidence breakdown:**

- v0.2.0 infrastructure (Section 1): HIGH — all files read; all integration points verified.
- Bilinear SVI config (Section 2): HIGH — `run_svi` signature confirmed by direct read; Phase 15 timings empirical.
- Ground truth (Section 3): HIGH — magnitudes chosen from Gershgorin bound + RECOV thresholds + Phase 13 BILIN-06 3σ anchor.
- Metric computation (Section 4): HIGH on formulas; MEDIUM on aggregate-across-seeds convention choice (two reasonable interpretations documented).
- HGF hook (Section 5): HIGH — signature follows CONTEXT.md exactly.
- Linear baseline (Section 6): HIGH — inline-run is the only path that matches SNR=3/duration=200s topology.
- Pitfalls (Section 7): HIGH on inherited; MEDIUM on N1 coverage concern (mean-field under bilinear is known risk but magnitude is empirical).
- Plan sizing (Section 8): HIGH — dependency-forced 3-plan structure; LoC estimates based on `task_svi.py`'s 336-line template.

**Research date:** 2026-04-18
**Valid until:** 2026-05-18 (30 days; stable upstream: Pyro 1.9+, benchmark infra unchanged since v0.2.0, Phase 15 task_dcm_model landed and verified)

## RESEARCH COMPLETE
