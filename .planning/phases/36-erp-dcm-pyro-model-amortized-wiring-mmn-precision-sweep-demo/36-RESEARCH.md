# Phase 36: ERP-DCM Pyro Model, Amortized Wiring & MMN Precision-Sweep Demo — Research

**Researched:** 2026-06-26
**Domain:** Pyro generative ERP-DCM model + amortized flow wiring + 5-source auditory-MMN precision-sweep demo (the actinf_physics hand-off artifact), v0.8.0 CAPSTONE
**Confidence:** HIGH (all forward/lead-field/integrator pieces are shipped + SPM12-parity-verified through Phase 35; this phase is wiring + a demo, not new physics)

## Summary

Phase 36 is the v0.8.0 capstone. **Nothing in the forward stack is new** — Phases 33/34/35 shipped and parity-verified the entire CMC→evoked→lead-field→scalp pipeline (`cmc_network_f`, `apply_condition_modulation` with the `diag(B)→G[:,6]` precision path, `integrate_local_linearization`, `build_lead_field`/`project_to_scalp`, `simulate_erp_dcm`, and `ERPDCMForward`). Phase 36 only adds: (1) a Pyro generative wrapper (`erp_dcm_model.py`, ERPDCM-01), (2) the amortized packer + wrapper (ERPDCM-02), (3) a public 5-source MMN network builder (the topology is already locked in `validation/export_to_mat.py` as the private `_MS_*` constants, ERPDCM-03), (4) the precision-sweep demo script + transfer curve (ERPDCM-04), (5) a thin consumer adapter (ERPDCM-05), and (6) a gate that refuses to emit the figure until the **already-green** Phase-35 LFP forward-parity check passes (ERPDCM-06).

**Two decisions resolve the open questions in the objective:**

- **ERPDCM-06 reuses the existing `validation/data/erp_leadfield_fixtures.mat` (LFP).** No new ECD fixture is needed. The demo's reference parameter point is **byte-identical** to that fixture (both are the locked `_MS_*` 5-source topology), and the Phase-35 ladder (`tests/test_spm_erp_leadfield_validation.py`) already gates the production forward + difference wave at ≤1e-7 against frozen `spm_gen_erp`+`spm_lx_erp` arrays.
- **The difference-wave SIGN (negative-going) and frontal-dominance are established in LFP source-space, NOT via ECD.** In LFP-identity mode each scalp channel *is* a source's superficial-pyramidal voltage (`P.J = e_2`), so the rIFG channel difference is directly readable; Phase 35 already recorded the peak-channel sign as `-1` (negative) as a non-gating diagnostic. **ECD orientation + MNI coords are NOT required for the headline demo** and stay deferred — they would only be needed for true sensor-space topography (out of v0.8.0 scope).

**Primary recommendation:** Build `erp_dcm_model` and the amortized wrapper as **thin wrappers that reuse `simulate_erp_dcm` / `ERPDCMForward.predict` as the deterministic forward** — never re-assemble the forward, so the parity-gated pipeline is the *only* forward in the codebase. Sample the named priors, delegate the physics, condition a Gaussian likelihood on the flattened scalp residual. Establish the MMN sign/frontal-dominance in LFP source-space against the reused Phase-35 fixture.

## Standard Stack

Zero new dependencies (locked decision E5). Everything is in-tree + already-installed.

### Core (all shipped, reused verbatim)
| Symbol | Location | Role in Phase 36 |
|--------|----------|------------------|
| `simulate_erp_dcm(p, x_design, n, ns, dt, ons_ms, dur_ms, sus, l_full)` | `simulators/erp_simulator.py` | The deterministic forward for `erp_dcm_model` AND the demo. Returns `{"states","pst","inputs","difference_wave","scalp":(Cnd,ns,Nc),"difference_wave_scalp":(ns,Nc)}` when `l_full` supplied. Threads a **sampled** `p["B"]` straight through `apply_condition_modulation`. |
| `ERPDCMForward(...)` | `inference/forward_models.py:693` | The VL/amortized forward; `pack_params`/`unpack_params`/`param_count` define the FROZEN parameter ordering the packer must mirror. |
| `apply_condition_modulation(p, x_design_row)` | `forward_models/erp_coupled_system.py:111` | `spm_gen_Q` port: folds `B` into all 4 `A` blocks AND `diag(B)→Q.G[:,0]→G[:,6]` (the precision path). The sweep + demo route B through here. |
| `parameterize_cmc_network` / `cmc_network_f` | `forward_models/erp_coupled_system.py` | extrinsic routing + EOM (`spm_fx_cmc`). |
| `integrate_local_linearization` | `utils/local_linearization.py:89` | `spm_int_L` exp-Euler; `jacrev` is differentiable → SVI/VL-safe. |
| `build_lead_field` / `project_to_scalp` / `cmc_default_pj` / `lfp_spatial` | `forward_models/erp_leadfield.py` | LFP lead field (`P.J=e_2`, `P.L=ones`→identity). |
| `cmc_prior_moments(a_mask, c_mask, n)` | `forward_models/cmc_priors.py:30` | the prior means/variances (`spm_cmc_priors.m`) for the Pyro priors and `build_prior_cov`. |
| `J_PERM = (6,1,2,3,0,4,5,7,8,9)` | `forward_models/cmc_neural_mass.py:55` | the permutation guard: `P.G[:,0]→G[:,6]` (sp self-inhibition = precision). |

### Pyro/amortized idioms to mirror (shipped)
| Pattern | Reference file | What to copy |
|---------|----------------|--------------|
| Pyro generative model | `models/spectral_dcm_model.py`, `models/task_dcm_model.py` | `pyro.sample(site, dist.Normal(zeros, std).to_event(k))`; `pyro.deterministic`; per-effect `B_free_{j}` loop (NO `pyro.plate`, `task_dcm_model.py:303`); HalfCauchy noise scale; Gaussian likelihood with `obs=`. |
| Amortized wrapper | `models/amortized_wrappers.py:51,155` | `_sample_latent_and_unpack(packer)` samples a single `_latent ~ N(0,I)`, `unstandardize`, `unpack`; `_run_*_forward_model(...)` runs the SAME forward + likelihood; NaN-guard (`zeros_like(...).detach()`). |
| Packer | `guides/parameter_packing.py` (`TaskDCMPacker`, `SpectralDCMPacker`) | `pack`/`unpack`/`fit_standardization`/`standardize`/`unstandardize`; `n_features`; **log-space contract** (but see note — CMC free params are already unconstrained). |
| Flow guide | `guides/amortized_flow.py` | `AmortizedFlowGuide(summary_net, n_features, packer=packer)`; single `_latent` site via `ZukoToPyro`. |
| Summary net | `guides/summary_networks.py` (`BoldSummaryNet`/`CsdSummaryNet`) | needs a small **`ErpSummaryNet`** addition (ERP obs is `(Cnd, ns, Nc)`, neither BOLD nor CSD shape). |
| Demo script | `scripts/demo_bilinear_consumer.py`, `scripts/demo_task_dcm.py` | self-contained `main()`, ground-truth build, forward, metrics print; `matplotlib` for figures. |
| Adapter precedent | `scripts/demo_bilinear_consumer.py` (`simulate_bilinear_bold` thin wrapper) | the "thin consumer map" idiom. |

## Architecture Patterns

### Recommended file footprint (additive-only)
```
NEW:
  src/pyro_dcm/models/erp_dcm_model.py            # ERPDCM-01 Pyro generative model
  src/pyro_dcm/forward_models/mmn_reference.py    # ERPDCM-03/05 public 5-source net + adapter
  scripts/demo_mmn_precision_sweep.py             # ERPDCM-04/06 demo + gate + transfer curve
  tests/test_erp_dcm_model.py                     # structural + SVI-smoke (laptop unit / M3 fit)
  tests/test_mmn_reference.py                     # net topology == _MS_* + permutation/precision wiring
  tests/test_amortized_erp.py                     # packer round-trip + flow-trains-without-error
MODIFIED (append-only symbols; existing bodies untouched):
  src/pyro_dcm/guides/parameter_packing.py        # + class ERPDCMPacker
  src/pyro_dcm/guides/summary_networks.py         # + class ErpSummaryNet
  src/pyro_dcm/models/amortized_wrappers.py       # + amortized_erp_dcm_model + _run_erp_forward_model
  src/pyro_dcm/models/__init__.py                 # + erp_dcm_model export
  src/pyro_dcm/guides/__init__.py                 # + ERPDCMPacker, ErpSummaryNet exports
  src/pyro_dcm/forward_models/__init__.py         # + build_mmn_5source_network, mmn_cmc_params exports
  cluster/sbatch/erp_pytest.sbatch                # + Phase-36 test files in default TEST_TARGET
```

### Plan decomposition (3 waves)

**Wave 1 — `erp_dcm_model.py` + amortized wiring (ERPDCM-01, ERPDCM-02).**
- `erp_dcm_model.py`: samples A/B/C/T/G/S/R, reuses `simulate_erp_dcm` as the forward, Gaussian likelihood on flat scalp residual. Works with the existing `create_guide(...)` factory (AutoNormal/AutoLowRankMVN/AutoIAF) — no factory edit (auto-discovers sites, MODEL-06 precedent).
- `ERPDCMPacker` + `amortized_erp_dcm_model` + `_run_erp_forward_model` + `ErpSummaryNet`. **Amortized holds B FIXED** (mirrors `ERPDCMForward` + the v0.3.0-D5 "amortized defers B" precedent); the full SVI model samples B.
- Tests: packer round-trip (`unpack(pack(x))==x`), a 1-source/2-source structural SVI-smoke (a few steps, laptop), flow-trains-without-error on a handful of `erp_simulator` draws.
- **Compute:** packer/structural tests laptop; any real SVI/amortized fit (jacrev × 128 steps × 2 conditions is slow) → M3.

**Wave 2 — 5-source MMN network builder + consumer adapter + parity-reuse decision (ERPDCM-03, ERPDCM-05).**
- `mmn_reference.py`: `build_mmn_5source_network()` returning the canonical masks/B/C/x_design (promotes the locked `_MS_*` topology to a public, importable API); `mmn_cmc_params(sp_inhibition_gain, a1_b_gain, rifg_b_gain, fwd_bwd_flag)` (the actinf_physics adapter).
- Tests assert the public builder reproduces the `_MS_*` edge lists element-wise and the permutation/precision-node wiring (perturb `P.G[:,0]` at rIFG → `G[:,6]` changes).
- **Settle ERPDCM-06 here:** reuse `erp_leadfield_fixtures.mat`; the demo's reference point == fixture's `_reference_p()`.
- **Compute:** all forward-only; laptop.

**Wave 3 — demo + gated parity + transfer curve (ERPDCM-04, ERPDCM-06).**
- `demo_mmn_precision_sweep.py`: (a) **gate first** — assert the Phase-35 LFP forward parity is green (`scalp`/`diff_wave` reproduce the frozen fixture ≤1e-7) and **refuse to emit any figure otherwise**; (b) sweep `P.G[:,0]` at {rIFG, A1L, A1R}; (c) emit `gain→|MMN|` transfer curve + low/baseline/high difference-wave overlay; (d) assert monotone attenuation + negative-going + frontal-dominant.
- **Compute:** a coarse sweep (≤~12 gain points) is laptop-feasible; a fine sweep >3 min → M3. Figures via `matplotlib` (already a dep).

### Anti-patterns to avoid
- **Re-assembling the forward inside `erp_dcm_model`.** Reuse `simulate_erp_dcm`; a second forward implementation can silently diverge from the parity-gated one (pitfall V1). The model's only job is priors + likelihood.
- **Editing `ERPDCMForward.predict`/`variational_laplace.py`** to support a sampled B in the amortized path. Hold B fixed in amortized (matches VL); sample B only in the SVI Pyro model.
- **Sweeping `G[:,0]` directly** (the permutation trap C3a/S1). Always perturb the FREE `P.G[:,0]`; assert it moves `G[:,6]`.
- **Asserting the MMN sign via ECD/coords.** Establish it in LFP source-space; ECD stays deferred.
- **Hard-coding MNI coords from memory.** They are MUST-VERIFY (see Open Questions) and not load-bearing for the LFP demo.

## Per-File Spec

### 1. `models/erp_dcm_model.py` (ERPDCM-01)

Mirror `spectral_dcm_model.py`. Signature:
```python
def erp_dcm_model(
    observed_scalp: torch.Tensor,        # (Cnd, ns, Nc) float64
    a_masks: list[torch.Tensor],         # 4 x (N,N) extrinsic routing masks
    b_masks: list[torch.Tensor],         # list of (N,N) between-trial B masks (live positions)
    c_mask: torch.Tensor,                # (N, M)
    x_design: torch.Tensor,              # (Cnd, n_effects)
    l_full: torch.Tensor,                # (Nc, 8N) precomputed LFP lead field
    N: int | None = None,
    *, dt: float = 0.004, ns: int = 128,
    ons_ms: float = 60.0, dur_ms: float = 16.0, sus: float = 0.0,
) -> None
```
Sample sites + prior scales (free/log space — `dist.Normal(mean, sqrt(var))`, transcribe from `cmc_prior_moments` / `spm_cmc_priors.m`):

| Site | shape | mean | sqrt(var) | SPM ref | mask-dead value |
|------|-------|------|-----------|---------|-----------------|
| `A_free` | `(4,N,N)` `.to_event(3)` | 0 | `(1/16)**0.5` | `spm_cmc_priors.m:80-81` | **−32** (CMC dead = `exp(−32)·E0≈0`; NOT 0) |
| `B_free_{j}` (per-effect loop, no plate) | `(N,N)` `.to_event(2)` | 0 | **VERIFY** (provisional `(1/8)**0.5`) | `spm_dcm_erp` B prior — confirm exact `pC.B` | **0** (B is an additive log-offset; 0 = no modulation) |
| `C_free` | `(N,M)` `.to_event(2)` | 0 | `(1/32)**0.5` | `:114-116` | **−32** |
| `T` | `(N,4)` `.to_event(2)` | 0 | `(1/32)**0.5` | `:121` | — |
| `G` | `(N,4)` `.to_event(2)` | 0 | `(1/32)**0.5` | `:122` | — |
| `S` | `(N,1)` `.to_event(2)` | 0 | `(1/64)**0.5` | `:124` | — |
| `R` | `(M,2)` `.to_event(2)` | 0 | `(1/16)**0.5` | `:133` | — |

Body (after sampling + masking):
```python
p = {"T": T, "G": G, "C": c_masked, "S": S, "R": R,
     "A": [a_free_list[i] for i in range(4)], "B": B_free_list}
sim = simulate_erp_dcm(p, x_design, N, ns=ns, dt=dt, ons_ms=ons_ms,
                       dur_ms=dur_ms, sus=sus, l_full=l_full)
pred = sim["scalp"]                       # (Cnd, ns, Nc)
pred = torch.nan_to_num(pred)             # NaN guard (amortized_wrappers idiom)
pyro.deterministic("predicted_scalp", pred)
noise_scale = pyro.sample("scalp_noise_scale", dist.HalfCauchy(tensor(1.0, float64)))
pyro.sample("obs_erp",
            dist.Normal(pred.reshape(-1), noise_scale).to_event(1),
            obs=observed_scalp.reshape(-1))
```
- **Mask asymmetry (load-bearing):** A/C dead → `−32`; B dead → `0`. (A/C are `exp(P)·E0` strengths; B is an additive-in-log offset.) Mirror `ERPDCMForward._masked_free` for A/C.
- **B IS sampled here** (objective requirement), unlike `ERPDCMForward` (B fixed). Reuse the `task_dcm_model.py:296-319` per-effect `B_free_{j}` loop verbatim (no `pyro.plate`; enables AutoGuide auto-discovery — MODEL-06).
- Flatten with `.reshape(-1)` (C-order) to match the locked `(Cnd,ns,Nc)` layout used everywhere downstream.

### 2. `ERPDCMPacker` (ERPDCM-02, in `guides/parameter_packing.py`)

Mirror `ERPDCMForward.pack_params`/`unpack_params` EXACTLY (so a packed latent ↔ VL theta are interchangeable). B is **fixed** in the amortized path (excluded from the packer, matching `ERPDCMForward`).
```
n_features = 4*N*N + N*M + 4*N + 4*N + N + 2*M
order      = A_free(4NN) | C_free(NM) | T(4N) | G(4N) | S(N) | R(2M)
unpack shapes: A_free (4,N,N) | C_free (N,M) | T (N,4) | G (N,4) | S (N,1) | R (M,2)
```
- **Log-space contract note:** unlike `TaskDCMPacker` (which `log()`s `noise_prec`), **every CMC free param is already unconstrained** (`P.*` free log-params), so pack/unpack are identity reshapes — no `.exp()` at unpack, no positive-constrained scalar. State this explicitly in the docstring.
- Provide `fit_standardization(dataset)`, `standardize`, `unstandardize` identical to the sibling packers (NSF spline needs ~zero-mean/unit-var input).

### 3. `amortized_erp_dcm_model` + `_run_erp_forward_model` (ERPDCM-02, `models/amortized_wrappers.py`)

Mirror `amortized_task_dcm_model`:
```python
def amortized_erp_dcm_model(observed_scalp, a_masks, b_masks, c_mask, x_design,
                            l_full, forward: ERPDCMForward, packer: ERPDCMPacker,
                            *, dt=0.004, ns=128, ...):
    params = _sample_latent_and_unpack(packer)   # A_free,C_free,T,G,S,R (unconstrained)
    theta = forward.pack_params(**params)        # reuse the frozen pack order
    _run_erp_forward_model(forward, theta, observed_scalp)
```
`_run_erp_forward_model` calls `forward.predict(theta, observed_scalp, N)` (the parity-gated forward, B fixed inside `forward`), NaN-guards, then `pyro.sample("obs_erp", dist.Normal(pred_flat, noise_std).to_event(1), obs=observed_scalp.reshape(-1))`. Hold `b_masks` fixed by constructing `forward = ERPDCMForward(l_full, x_design, a_masks, b_masks, c_mask, ...)` once outside the model.

### 4. `ErpSummaryNet` (ERPDCM-02, `guides/summary_networks.py`)

Small MLP/CNN over the flattened scalp `(Cnd·ns·Nc,) → embed_dim` (default 128, float64), matching the `BoldSummaryNet` surface (`embed_dim` attr, `forward(x)->(...,embed_dim)`). The flow guide `AmortizedFlowGuide(ErpSummaryNet(...), packer.n_features, packer=packer)` then needs **zero** changes.

### 5. `forward_models/mmn_reference.py` (ERPDCM-03 + ERPDCM-05)

`build_mmn_5source_network()` — promote the locked `_MS_*` constants (`validation/export_to_mat.py:660-701`) to a public, importable builder so the demo, the adapter, AND `actinf_physics` share one source of truth. The topology (0-indexed A1L=0,A1R=1,STGL=2,STGR=3,rIFG=4):
- **Forward** (`A{1}` sp→ss, `A{2}` sp→dp): A1L→STGL, A1R→STGR, STGL→rIFG, STGR→rIFG
- **Lateral reciprocal** STGL↔STGR (added to forward blocks; triggers the `(1+4L)` reduction)
- **Backward** (`A{3}` dp→sp, `A{4}` dp→ii): rIFG→STGL, rIFG→STGR, STGL→A1L, STGR→A1R
- **Input C**: A1L, A1R only
- **B precision nodes** (`diag(B)→G[:,6]`): rIFG, A1L, A1R; `B_EDGE=0.3` on every extrinsic edge, `B_DIAG=0.5` on the precision diag
- Returns `{"a_masks":[4×(5,5)], "b_masks":[(5,5)], "c_mask":(5,1), "x_design":(2,n_eff), "source_names":[...], "precision_nodes":(4,0,1)}`. **MNI coords are NOT included** (LFP scope; coords are MUST-VERIFY, Phase-36+ ECD only).

`mmn_cmc_params(sp_inhibition_gain, a1_b_gain, rifg_b_gain, fwd_bwd_flag="both")` (ERPDCM-05):
- `sp_inhibition_gain` → `P.G[node,0] = sp_inhibition_gain` for `node ∈ {rIFG, A1L, A1R}` (the swept precision knob; flows to `G[:,6]` via `J_PERM[0]=6`).
- `a1_b_gain` → `B_DIAG` at A1L, A1R; `rifg_b_gain` → `B_DIAG` at rIFG.
- `fwd_bwd_flag ∈ {"forward","backward","both"}` → which extrinsic blocks carry the `B_EDGE` modulation (Garrido/Ranlund model-space toggle).
- Returns a ready-to-simulate bundle `{"p":p_struct, "a_masks","b_masks","c_mask","x_design","l_full"}`. **Forward-only, no fitting** — the contract for the actinf_physics Phase-133 adapter.

### 6. `scripts/demo_mmn_precision_sweep.py` (ERPDCM-04 + ERPDCM-06)

```
1. net = build_mmn_5source_network(); l_full = build_lead_field(cmc_default_pj(), lfp_spatial(ones(5),5))
2. GATE (ERPDCM-06): load erp_leadfield_fixtures.mat; run the production forward at the
   reference params (== _reference_p()); assert scalp & diff_wave reproduce the frozen
   arrays <= 1e-7. If the fixture is absent OR the check fails -> raise/exit BEFORE any
   figure is written (reuse tests/test_spm_erp_leadfield_validation.py::_production_scalp logic).
3. SWEEP: for gain in linspace(...): p = mmn_cmc_params(sp_inhibition_gain=gain, a1_b_gain=0.5,
   rifg_b_gain=0.5); sim = simulate_erp_dcm(p, x_design, 5, l_full=l_full);
   mmn = sim["difference_wave_scalp"]; record |mmn| peak at rIFG channel.
4. ASSERT monotone attenuation (|MMN| decreasing in gain), negative-going peak at rIFG,
   and frontal-dominance (|mmn[:,rIFG]| max > |mmn[:,A1]| max).
5. FIGURE: gain->|MMN| transfer curve (n_gain,) + low/baseline/high difference-wave overlay
   -> figures/mmn_precision_sweep.{png,pdf}.
```

## The ERPDCM-06 Fixed-Reference Parity-Gate Spec (DECIDED)

**Reuse `validation/data/erp_leadfield_fixtures.mat` (LFP). No new ECD fixture.**

| Question | Answer | Why |
|----------|--------|-----|
| Reuse Phase-35 LFP fixture or new ECD fixture? | **Reuse the LFP fixture.** | The demo's reference params are byte-identical to the fixture (`_reference_p()` = the `_MS_*` topology). The Phase-35 ladder already gates production `scalp`/`diff_wave` ≤1e-7 vs frozen `spm_gen_erp`+`spm_lx_erp`. A new ECD fixture would require a sensor montage + verified MNI coords + `spm_cond_units` export — all deferred. |
| How is the gate enforced "BEFORE any sweep output"? | The demo runs the production forward at the reference point and asserts ≤1e-7 vs the frozen `y_scalp`/`diff_wave`; on failure or missing fixture it **raises before writing the figure**. | S3: the headline artifact is only credible once the single reference point is SPM-validated. |
| Does the sign assertion need ECD/coords? | **No.** LFP-identity scalp channel = source sp-voltage; the rIFG channel difference gives sign + frontal-dominance directly. Phase-35 already recorded peak sign `−1`. | ECD orientation/coords only matter for sensor-space topography (out of scope). |
| Gate compute | Laptop (fixture committed; torch-vs-frozen-array, no MATLAB). M3 only if a NEW fixture is ever regenerated. | `validation/data/` is committed; `test_spm_erp_leadfield_validation.py` runs on laptop. |

## The MMN-Demo Acceptance Spec

| Criterion | How established | Source |
|-----------|-----------------|--------|
| **Monotone gain→\|MMN\| attenuation** | Sweep FREE `P.G[:,0]` at {rIFG,A1L,A1R}; ↑ → larger `G[:,6]` (sp self-inhibition, `uu_sp = g[:,7]·s[:,0] − g[:,6]·s[:,2]`) → lower sp gain → smaller PE → smaller `|deviant−standard|`. Assert `|MMN|` non-increasing across the sweep. | FEATURES §4; `cmc_neural_mass.py:235`, `J_PERM` |
| **Sweep hits the right knob** | Reuse the Phase-33 permutation guard: assert perturbing `P.G[:,0]` changes `G[:,6]`, not `G[:,0]` (`tests/test_cmc_forward.py`). | C3a / S1 |
| **Negative-going difference wave** | In LFP source-space the rIFG channel = its sp-voltage; assert `diff_wave_scalp[peak, rIFG] < 0` (deviant−standard). Phase-35 diagnostic already `−1`. | LEAD-03 diagnostic, 35-03-SUMMARY |
| **Frontal-dominant** | Assert `max\|diff_wave_scalp[:,rIFG]\| > max\|diff_wave_scalp[:,A1L/A1R]\|` (Ranlund winning model: precision modulation at rIFG + bilateral A1). | FEATURES §5 |
| **Transfer curve emitted** | `gain→|MMN|` array (n_gain,) + overlay figure → consumable as a *function* by actinf_physics (D3). | ERPDCM-04 |
| **Gated** | Criteria 1–4 only run AFTER the ERPDCM-06 reference-parity check is green. | S3 |

**Watch item (must measure during planning/exec):** confirm frontal-dominance actually emerges at the reference params — the Phase-35 fixture peak channel index was recorded but not named. If A1 happens to dominate rIFG at baseline, tune `rifg_b_gain` vs `a1_b_gain` in the demo (forward-only, allowed) and document, OR document the channel ranking honestly. Do not fake it (V-discipline).

## Don't Hand-Roll

| Problem | Don't build | Use instead |
|---------|-------------|-------------|
| The ERP forward inside the Pyro model | a second integrate→project loop | `simulate_erp_dcm(..., l_full=...)["scalp"]` |
| The amortized forward | re-derive predict | `ERPDCMForward.predict` (B fixed) |
| The 5-source topology | re-type edge lists | promote `_MS_*` from `export_to_mat.py` into `mmn_reference.build_mmn_5source_network()` |
| The reference-parity check | a fresh MATLAB run | the committed `erp_leadfield_fixtures.mat` + `_production_scalp` logic |
| The LFP lead field | a head model | `build_lead_field(cmc_default_pj(), lfp_spatial(ones(N),N))` |
| The precision permutation | index `G[:,0]` | `P.G[:,0]` → `J_PERM[0]=6` → `G[:,6]` |

## Common Pitfalls (Phase-36-specific; the C1–C5/V1–V5 forward traps are already gated upstream)

- **S1 — sweeping the wrong knob.** Perturb FREE `P.G[:,0]`, not `G[:,0]`; reuse the permutation guard. A wrong-knob sweep still produces *some* monotone curve (looks right, isn't the mechanism).
- **S2 — sign not pinned.** Pin subtraction order `deviant−standard`, `P.J=e_2` (sp voltage, idx 2 NOT 6), LFP source-space readout. Cross-check standard & deviant each match SPM before differencing (the Phase-35 fixture already does).
- **S3 — demo not gated.** No figure before the reference-parity check is green.
- **B-mask dead value asymmetry.** A/C dead=`−32`, B dead=`0`. Getting B dead wrong (e.g. −32) silently kills the modulation (no MMN) or injects a spurious one.
- **Mutagen `models/` ignore footgun.** `erp_dcm_model.py` lives in `src/pyro_dcm/models/`. The Mutagen sync ignore pattern `models/` (intended for top-level output dirs) **silently excludes `src/pyro_dcm/models/` from M3 sync** (known footgun, `reference_mutagen_models_ignore_footgun`). Before any M3 run importing the new model: **anchor the ignore** (`/models/` not `models/`) OR `scp` the file as a stopgap. Verify the file lands on M3 (`ssh m3 'ls .../src/pyro_dcm/models/erp_dcm_model.py'`) before sbatch.
- **Amortized B-packing scope creep.** Keep B fixed in the amortized/packer path (matches VL + v0.3.0-D5). Only the SVI Pyro model samples B.
- **B prior variance unverified.** `cmc_prior_moments` does not include B (B is Phase-34 condition modulation, fixed in the parity gate). The `pC.B` value for the SVI recovery path is MUST-VERIFY from `spm_cmc_priors.m`/`spm_dcm_erp` — provisional `1/8`. Low stakes for the headline demo (fixed B); matters only for D1/D2 recovery.

## Compute Routing & Footguns

| Task | Where | Note |
|------|-------|------|
| Packer round-trip, structural model trace, topology-equality tests | Laptop (<30s) | unit tests |
| SVI fit / amortized flow training (jacrev × 128 × 2 cond) | **M3** (>3 min) | `cluster/sbatch/erp_pytest.sbatch` with `TEST_TARGET=...`; add Phase-36 test files to the default `TEST_TARGET` line |
| Fine precision sweep (>~12 gain pts) | **M3** if >3 min; coarse sweep laptop-OK | demo |
| ERPDCM-06 reference-parity gate | Laptop | fixture committed; torch-vs-frozen-array (no MATLAB) |
| New MATLAB fixture (only if ever needed) | M3 (R2022a + Carrick spm12) | not needed this phase |

**Mutagen sync:** new `src/pyro_dcm/models/erp_dcm_model.py` is at risk from the `models/` ignore footgun (above) — anchor or scp before M3. New `forward_models/`, `guides/`, `scripts/` files sync normally. Per project rule, use **Mutagen** (not git push/pull) for M3 deploy; **never `pip install` in array/sbatch jobs**.

## Citations (do NOT fabricate `[REF-xxx]`)

Inline citation policy this milestone: **SPM source file + line + author/year only**. The `REF-ERP-*`/`REF-MMN-*` Zotero keys are **not yet confirmed** — do not write `[REF-xxx]` in docstrings until the paper is in Zotero (CLAUDE.md `.bib` rule). Papers the user must add to Zotero before any keyed citation:

| Paper | Role | Zotero? |
|-------|------|---------|
| David O, Friston KJ (2003), NeuroImage 20:1743 | neural-mass + lead-field canonical ref (head of every `spm_fx_*`/`spm_lx_erp`) | VERIFY |
| Bastos AM et al. (2012), Neuron 76:695 | CMC for predictive coding (sp=ascending PE) | VERIFY |
| Kiebel SJ, David O, Friston KJ (2006), NeuroImage 30:1273 | `spm_lx_erp`/`spm_erp_L` lead-field parameterization | VERIFY |
| Garrido MI et al. (2009), Clin. Neurophysiol. 120:453 | MMN predictive-coding review; fwd/bwd modulation | VERIFY |
| Ranlund S et al. (2016), Hum. Brain Mapp. 37:351 | 5-source MMN net; winning model = intrinsic gain at rIFG + bilateral A1; **MNI coords source** | VERIFY |
| Adams RA et al. (2013), Front. Psychiatry 4:47 | aberrant-precision / sp self-inhibition account | VERIFY |

## State of the Art

| Old (within this milestone) | Now (shipped through Phase 35) | Impact for Phase 36 |
|------------------------------|--------------------------------|---------------------|
| Forward pieces planned | `simulate_erp_dcm`, `ERPDCMForward`, LFP parity gate all shipped + green | Phase 36 is wiring + demo; **no new physics, no new fixture** |
| Sign/coords "TBD, maybe ECD" | LFP source-space gives sign + frontal-dominance | ECD/coords stay deferred |

## Open Questions

1. **MNI source coordinates (MUST-VERIFY, LOW confidence).** Commonly-cited Garrido/Ranlund values: A1L (−42,−22,7), A1R (46,−14,8), STGL (−61,−32,8), STGR (59,−25,8), rIFG (46,20,8). **Not load-bearing for the LFP demo** (sign/frontal come from source-space). Verify against Ranlund 2016 / Garrido before EVER hard-coding for an ECD path. Recommendation: do NOT hard-code coords in Phase 36; keep them out of `build_mmn_5source_network()`.
2. **B prior variance for the SVI recovery path (MEDIUM).** `pC.B` not in `cmc_prior_moments`. Confirm from `spm_cmc_priors.m`/`spm_dcm_erp`; provisional `1/8`. Only affects D1/D2 recovery, not the fixed-B headline demo.
3. **Frontal-dominance at the reference params (verify empirically).** Confirm `max|diff_wave_scalp[:,rIFG]| > max|diff_wave_scalp[:,A1]|` holds at baseline; if not, tune `rifg_b_gain` (forward-only) and document honestly.
4. **Amortized summary-net architecture (LOW stakes).** `ErpSummaryNet` over flattened `(Cnd·ns·Nc,)` — a small MLP suffices for "trains without error" (ERPDCM-02); no need for a CNN unless training is unstable.

## Sources

### Primary (HIGH — codebase read directly)
- `inference/forward_models.py:693-1008` (`ERPDCMForward` — frozen pack order, `_masked_free`, predict)
- `simulators/erp_simulator.py` (`simulate_erp_dcm` — the reusable forward + scalp/diff-wave)
- `forward_models/erp_coupled_system.py` (`apply_condition_modulation` `diag(B)→G[:,0]`; `cmc_network_f`)
- `forward_models/cmc_neural_mass.py:55,101-211` (`J_PERM`, `parameterize_cmc`, sigmoid, EOM `uu_sp`)
- `forward_models/cmc_priors.py:30-77` (`cmc_prior_moments` — prior means/variances)
- `forward_models/erp_leadfield.py` (LFP lead field, `cmc_default_pj`, `project_to_scalp`)
- `utils/local_linearization.py` (`integrate_local_linearization`, jacrev differentiable)
- `models/spectral_dcm_model.py`, `models/task_dcm_model.py` (Pyro idiom; `B_free_{j}` loop)
- `models/amortized_wrappers.py`, `guides/parameter_packing.py`, `guides/amortized_flow.py` (amortized idiom)
- `validation/export_to_mat.py:660-1184` (`_MS_*` 5-source MMN topology constants)
- `validation/matlab_scripts/run_spm_erp_dcm_leadfield.m` + `tests/test_spm_erp_leadfield_validation.py` (the green LFP parity ladder; `_production_scalp`, `_reference_p`)
- `.planning/phases/35-*/35-03-SUMMARY.md` (LEAD-05 gated ≤1e-7; sign/ECD deferred to Phase 36; peak sign `−1`)
- `cluster/sbatch/erp_pytest.sbatch` (M3 pytest runner, `TEST_TARGET`)

### Primary (HIGH — milestone research, read)
- `.planning/research/v0.8.0/{SUMMARY,FEATURES,ARCHITECTURE,PITFALLS}.md` (precision mechanism, 5-source net, S1–S3 demo pitfalls, C1–C5/V1–V5)
- `.planning/REQUIREMENTS.md` (ERPDCM-01..06, milestone decisions E1–E5)

### Secondary (MEDIUM/LOW — flagged for verification)
- MNI coords (LOW — verify vs Ranlund 2016/Garrido before any ECD hard-coding)
- `pC.B` variance (MEDIUM — verify vs `spm_cmc_priors.m`)

## Metadata

**Confidence breakdown:**
- Standard stack / reuse seam: HIGH — all pieces shipped + parity-verified; read in-tree.
- Architecture (model/packer/amortized wiring): HIGH — direct mirror of 3 shipped precedents.
- ERPDCM-06 reuse decision + LFP-sign decision: HIGH — the fixture + ladder are green; sign diagnostic already recorded.
- MMN demo acceptance: MEDIUM-HIGH — mechanism + sign are pinned; frontal-dominance is an empirical check to confirm at the reference params.
- Coords / B prior variance: LOW/MEDIUM — flagged MUST-VERIFY; not load-bearing for the headline.

**Research date:** 2026-06-26
**Valid until:** ~30 days (stable; gated on shipped, parity-verified code)

## RESEARCH COMPLETE
