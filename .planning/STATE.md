# State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-10)

**Core value:** A matrix (effective connectivity) remains explicit and interpretable with full posterior uncertainty
**Current focus:** Planning v0.7.0 — VL validation matrix + deferred real-data application (real Cam-CAN M/EEG, real foundation-model runs, SBI calibration). Seed: `.planning/v0.7.0-VL-RECONCILIATION-DRAFT.md`.

## Current Position

**Milestone:** v0.7.0 Variational Laplace Validation — **ROADMAPPED 2026-06-10 (VL-validation-led)**.
**Phase: 29 — VL Validation Infrastructure & BMR Rank Functions. Status: In progress (1/5 plans — 29-03 numerical-robustness guards done 2026-06-10).**
Roadmap drafted: 4 phases (29-32), 19/19 requirements mapped. Critical path 29 -> 30 -> 31; Phase
32 (SPM12, local/MATLAB) runs in parallel with Phase 30 (recovery sweep, M3 cluster). Confirmed
scope: synthetic recovery matrix (N×SNR), SPM12 cross-validation (user has MATLAB), VL+BMR
comparison + overconfidence fix (relative ranking only), numerical robustness. **No real data;
SBI deferred to v0.8.0+.** Phase numbering continues from 28 → v0.7.0 = Phases 29-32.
**Next:** `/gsd:plan-phase 29`. **PREREQUISITE for Phase 30 latent-circuit M3 runs:** fix the
Mutagen `models/` ignore (recreate session with anchored ignores).

<details><summary>v0.6.0 — SHIPPED 2026-06-10 (scope-cut), archived + tagged</summary>
**Phase:** All 34 plans executed (Phases 20-27 + retroactive Phase 28 VL engine).
**Status:** Goal-backward audit (`.planning/v0.6.0-AUDIT.md`) found every plan executed but the
**real-data scientific claims undelivered** (pivoted to synthetic or built-but-not-run). v0.6.0
**scope-cut** to its delivered core; real-data application **deferred to v0.7.0** (recorded as
deferred, NOT failed). User-approved both decisions 2026-06-10.

  - ✅ **Delivered:** Phase 20 synthetic recovery (via VL: A-RMSE 0.026, B-RMSE 0.0048, pooled-R²
    0.961, BMR 3/3) · Phase 21 CT-RNN · Phase 23 BMR (~93× faster) · Phase 27 pub artifacts ·
    **Phase 28 SPM12-grade VL inference engine** (the path that actually delivered v0.6.0 inference).
  - ⚠️ **Synthetic/infra-only → v0.7.0:** Phase 22 (pivoted from real Cam-CAN to synthetic OU;
    gates 1-2 unmet) · Phase 24 (real extractors + real parcellation built, never run on real
    weights; M/EEG pipeline scripts deleted in merge).
  - ✅ **Phase 25 HVAE-02 CONFIRMED 2026-06-10** — eval-only re-run (job 56331599) on the trained
    checkpoint reproduced RMSE 0.0761 + unmasked 0.4425 exactly, and **masked sign recovery 0.7745
    > 0.6 → PASS**. Phase 25 now 4/4. The 0.4425 was purely the `sign(0)` artifact.
  - ❌ **→ v0.7.0:** Phase 26 SBI SBC failed 2/9 (structural); real-M/EEG demo unplanned.
**Last activity:** 2026-06-10 -- Milestone audit + scope-cut: ROADMAP reconciled (stale
"0/N Planned" → audited status), Phase 28 VL consolidation written, 4 gaps triaged.

> **✅ Planning-doc drift RESOLVED 2026-06-10.** The post-consolidation Variational Laplace work
> (ForwardModel protocol, SVI→VL default, ReML M-step, `spm_dcm_csd_Q` precision, SVD reduction,
> SPM12 hyperpriors, analytical hemodynamic Jacobian, `LatentCircuitForward`) is now documented as
> **retroactive Phase 28** (`.planning/phases/28-variational-laplace-engine/28-CONSOLIDATION-SUMMARY.md`)
> inside v0.6.0 — it's what delivered v0.6.0's inference. The forward-looking VL *validation matrix*
> + *real-data application* is reserved for v0.7.0 (`.planning/v0.7.0-VL-RECONCILIATION-DRAFT.md`,
> Phases B–E) via `/gsd:new-milestone`.

</details>

**Milestone status:**
- v0.1.0 ✅ | v0.2.0 ✅ | v0.4.0 ✅ (Phase 17) | v0.5.0 ✅ (Phases 18-19) | **v0.6.0 ✅ shipped 2026-06-10 (scope-cut)**
- v0.3.0 ⏸ still in progress — Phase 16.1 (RECOV-04 B-RMSE diagnostic) never executed; the only genuinely-open prior milestone.
- v0.7.0 📋 next — VL validation + deferred real-data (`/gsd:new-milestone`).

## Decisions

- **[20-01-D1] hemodynamic=False as keyword-only after stability_check_every.** No positional break for existing callers; bit-exact backward compat preserved.
- **[20-01-D2] simulate_latent_circuit reuses _normalize_B_list/_normalize_stimulus_to_input_fn from task_simulator.** DRY: bilinear path is identical; private helpers imported directly.
- **[20-01-D3] Initial state torch.zeros(N) not make_initial_state (5N).** make_initial_state returns wrong shape for hemodynamic=False mode.
- **[20-03-D1] pyro.deterministic() appears as type='sample' in this Pyro version.** Tests must check by site name, not by type. Pattern documented in 20-03-SUMMARY.md key-decisions.
- **[20-03-D2] AutoIAFNormal hidden_dim must exceed latent_dim.** For N=4, M=1 model: latent_dim=21 (N^2 + N*M + 1). Use hidden_dim=[32] not [20] to avoid AutoRegressiveNN ValueError.
- **[20-03-D3] LC_A_PRIOR_VARIANCE=1/16 confirmed as separate constant from BOLD A prior (1/64).** Addresses pitfall LC4. Stored in latent_circuit_dcm_model as module-level constant, re-exported from models/__init__.py.
- **[20-04-D1] importlib.import_module required to access LC_*_PRIOR_VARIANCE for monkey-patching.** pyro_dcm.models.__init__ re-exports latent_circuit_dcm_model function under the submodule name; import-as resolves to function not module. Fixed with importlib.import_module("pyro_dcm.models.latent_circuit_dcm_model").
- **[20-04-D2] 100s / dt=0.01 ODE = ~16s per SVI step on laptop.** Full acceptance runs (1000+ steps, 10+ restarts, 10 seeds) must go to M3 cluster. Smoke test uses _duration_override=2.0 for API verification only.
- **[20-04-D3] All LC acceptance thresholds provisional.** A-RMSE 0.15, B-RMSE 0.20, sign recovery 0.80, CI coverage 0.85, trajectory R2 0.95. Plan 20-05 recalibration pending.
- **[21-01-D1] alpha = dt/tau is a plain float attribute, not nn.Parameter.** Fixed for v0.6.0; avoids accidental gradient computation through it; learnable timescales deferred.
- **[21-01-D2] Euler integration chosen over torchdiffeq for CT-RNN training.** Matches Langdon & Engel (2025) trainRNNbrain exactly; faster and deterministic for fixed-dt neurogym observations.
- **[21-01-D3] Langdon & Engel (2025) formal REF-ID deferred to Phase 25 (PUB-03).** Cited by author/year in docstring as interim placeholder.
- **[21-02-D1] neurogym labels shape is (T, B) not (T*B,).** ngym.Dataset() v2.3.1 returns labels as (seq_len, batch_size); must .reshape(-1) before CrossEntropyLoss and accuracy computation.
- **[21-02-D2] neurogym imported inside train_rnn/eval_rnn_performance only.** Optional dependency guard: try/except ImportError with install hint. Callers without neurogym can still import ContinuousTimeRNN.
- **[21-02-D3] Early stopping checks every log_every steps; 3 consecutive checks >= criterion_acc trigger return.** Count resets if accuracy drops below threshold at any log checkpoint.
- **[21-03-D1] Module docstring must precede `from __future__ import annotations`.** ruff E402 treats any non-import statement (including docstrings) before imports as breaking import-block contiguity; PEP 257 module docstring is first statement.
- **[21-03-D2] output_r_squared_gate fail test uses orthogonal equal-variance embedding.** Correlated factor mixing allows 1 PC to capture >90% variance; disjoint H/3-block basis guarantees PC1 ~33% and gate fails with N=1.
- **[21-03-D3] classify_stability parameter named jacobian_matrix to avoid shadowing module import.** Module imports `jacobian` from `torch.autograd.functional`; parameter named `jacobian` would shadow it inside the function.
- **[21-03-D4] extract_trajectories metadata stored as Python dict under `__meta__` key.** np.ndarray cannot hold heterogeneous scalar types (dt_seconds, tau, alpha); dict is correct container.
- **[20-02] n_restarts=1 path is bit-exact with pre-Phase-20 single-run path.** Return dict has exactly {losses, final_loss, num_steps}; no extended keys. Backward compat verified by existing test suite.
- **[20-02] guide_factory required when n_restarts>1 (ValueError on None).** Prevents silent reuse of a pre-trained guide across restarts.
- **[20-02] Param store restored via get_state/set_state after all restarts.** Avoids performance cost of re-running the best restart.
- **[20-01-D4] make_stable_latent_circuit_A uses self_inhibition=1.0 Hz (vs 0.5 Hz SPM12 default).** RNN latent states evolve faster than BOLD; stronger self-inhibition appropriate.
- **[19-02] t_eval for task_dcm_model constructed from DURATION+DT_MODEL, not from simulate_task_dcm output times.** Model contract requires t_eval spacing == dt; using simulation times_fine (DT_SIM=0.01) would violate this and make SVI prohibitively slow.
- **[19-02] Single-edge B mask (0->1 only) used in task DCM demo for clarity.** Simpler than demo_bilinear_consumer's two-edge mask; makes recovery metric output less cluttered for demo purposes.
- **[19-01] Demo scripts use simulate_* CSD for SVI fitting, not MNE noise epochs.** epochs_to_csd is demonstrated for IO bridge visibility (shapes printed); recovery metrics require ground-truth CSD from the generative model. Comments in script explain the distinction.
- **v0.6.0 phase structure = 6 phases (20-25).** Derived from 10 requirement categories clustered into 6 delivery boundaries. Phase 20 (14 reqs) is the scientific core; Phases 20 and 21 can run in parallel.
- **C_obs fixed at identity for v0.6.0.** Addresses pitfall LC5 (rotation ambiguity). Learned C_obs deferred to v0.7.0+.
- **Multi-start SVI (>=10 restarts) non-optional.** Addresses pitfall LC11; L&E uses 100.
- **Prior recalibration mandatory.** LC_A_PRIOR_VARIANCE separate from BOLD priors. Addresses pitfall LC4.
- **[22-01-D1] eig_clamp=None disables clamping entirely; -1.0 recommended for MEG.** Default -1/32 preserves fMRI behavior. None relies on parameterize_A upstream.
- **[22-01-D2] prior_a_var uses variance (not std) to match SPM12 convention.** Standard deviation computed as prior_a_var**0.5.
- **[22-03-D1] OU process uses Euler-Maruyama with dt=1/sfreq, not adaptive ODE solver.** Simple, fast, matches plan specification; spectral DCM linear model doesn't need adaptive stepping.
- **[22-03-D2] CSD consistency test uses 100 samples x 20s at 50 Hz with correlation > 0.5 threshold.** Finite-sample OU noise requires many realizations and lenient threshold; 50 Hz keeps test fast (~4s).
- **[23-01-D1] BMR delta_F uses Laplace approximation, not VFE difference.** delta_F = log p(mu_f|m_r) - log p(mu_f|m_f) + 0.5*[log|Sigma_r| - log|Sigma_f|]. No trace term. Validated against exact conjugate Gaussian Bayes factor.
- **[23-01-D2] BMR antisymmetry holds only for equal-covariance prior pairs.** When full and reduced priors differ only in mean (same cov), delta_F(A->B) = -delta_F(B->A) exactly. Different covariances break antisymmetry due to distinct reduced posterior precisions.
- **[24-02-D1] TRIBE v2 import guarded with try/except ImportError, not added to pyproject.toml.** Optional GPU dependency; requires A100; install via git URL.
- **[24-02-D2] Pipeline scripts use lazy imports after argparse.** Heavy torch/pyro imports slow; --help should be fast.
- **[24-02-D3] compute_empirical_csd with fs=1.0 Hz for TRIBE v2.** TRIBE v2 outputs at 1 Hz (fMRI TR); Nyquist at 0.5 Hz.
- **[25-02-D1] SVI smoke test uses windowed average (first 5 vs last 5 finite losses).** Early SVI steps produce NaN losses from ODE divergence; NaN guard prevents gradient corruption but losses are NaN. Windowed comparison is more robust.
- **[25-02-D2] packer.total_dim used (not n_features) for LatentCircuitDCMPacker.** Sparse packing attribute name differs from TaskDCMPacker's n_features.
- **[25-04-D1] KL annealing uses poutine.scale with mutable beta container.** SVI created once with scaled_model closure; beta_container[0] updated per epoch. Avoids SVI recreation overhead.
- **[25-04-D2] Beta floor is 1e-3 (not 0.0) at epoch 0.** When scale=0.0, poutine.scale zeros all log-probs, causing degenerate ELBO (all NaN). 1e-3 floor ensures valid gradients from first epoch.
- **[25-04-D3] KL estimated analytically from encoder z_loc/z_scale vs N(0,I).** Avoids Trace_ELBO decomposition; 0.5*(scale^2 + loc^2 - 1 - 2*log(scale)).sum() is exact for diagonal Gaussian vs standard normal.
- **[27-02-D1] Generated figures are gitignored; script is source of truth.** figures/*.png and figures/*.pdf excluded by .gitignore. Regenerate via `python scripts/generate_publication_figures.py`.
- **[29-03-D1] _TASK_PRECISION_MAX_DIM=5000 caps the dense (T*N,T*N) task-DCM precision.** TaskDCMForward.build_precision fails loud (ValueError with expected-vs-actual size) above the cap; enforces the dt>=0.1 floor (VLROBUST-02, pitfall N1). Tractable path unchanged.
- **[29-03-D2] C-order CSD index contract (j fastest, i, w) locked by regression test.** tests/test_csd_corder_roundtrip.py guards the commit-64e326f fix against silent column-major/transpose regression (VLREC-05, pitfall S4). Registered the `vl` pytest marker (was unregistered).
- Prior v0.3.0/v0.4.0/v0.5.0 decisions: see earlier STATE.md history in git log.

## Blockers

**AUDIT COMPLETE 2026-06-10 (`.planning/v0.6.0-AUDIT.md`).** The 4 gaps below are triaged.
**Nothing hard-blocks v0.6.0 completion under the scope-cut** — real-data gaps (Phase 22/24/26)
are formally **deferred to v0.7.0**; the only in-scope open item is the **HVAE-02 masked-metric
re-eval** (a <5-min M3 job, or accept-with-caveat). Recommended next action: optionally close
HVAE-02, then `/gsd:complete-milestone`. Detail retained below.

**Triage summary:**
1. **[20-05]** ✅ closed (VL). 2. **[25/HVAE-02]** ✅ CONFIRMED 2026-06-10 (masked 0.7745, job
56331599) — Phase 25 now 4/4. 3. **[26/SBI-03]** → v0.7.0 Phase D (structural, not a v0.6.0
deliverable). 4. **[24-01 parcellation]** ✅ No-Placeholders violation resolved; runtime
validation → v0.7.0. Plus **[vl-overconfidence-for-bmr]** → v0.7.0 Phase C.

**No remaining in-scope blockers — v0.6.0 is clean for `/gsd:complete-milestone`.**

<details><summary>Original 4-gap analysis (pre-audit, 2026-06-09 — retained)</summary>

1. **[Phase 20-05] ✅ FULLY CLOSED via Variational Laplace 2026-06-09 — SYNTH-01/02/03 all pass.**
   - **SYNTH-01/02** (job 56268248, 10 seeds): A-RMSE 0.026, **B-RMSE 0.0048** (vs SVI ~0.31),
     sign 1.00, CI cov 1.00, **pooled trajectory-R² 0.961**. R² "failure" was a metric bug
     (recovered R² == oracle R² → 0.95 unachievable); fixed via variance-pooled R², gate 0.95→0.90.
   - **SYNTH-03** (job 56270544, 3 seeds): BMR evidence ranking recovers the true chain {4,9,14}
     3/3 (sep 14×/13×/1.8×). Caveat: VL Laplace overconfidence suppresses *absolute* BMR pruning;
     *relative* ranking is the robust signal (→ todo on tempering VL posterior for BMR).

   <details><summary>Original SVI failure analysis (resolved; retained for the record)</summary>

   A-RMSE passes; B-RMSE, trajectory R², and ELBO model-selection fail (SVI). Full analysis in
   `20-05-SUMMARY.md`. Three distinct causes: (a) **B under-identified by experiment design** —
   the 50s CPU-feasibility rework + 80/20 split leaves only ONE 8s modulator window in
   training, so B collapses to ~0.31 RMSE (same pathology as unresolved Phase 16.1 ~0.34);
   (b) **R² fail is downstream** — held-out window contains a modulator epoch a collapsed-B
   model can't reproduce; (c) **ELBO model selection is methodologically invalid** — candidates
   N∈{2..6} fit datasets of different observed dimensionality so −ELBO scales with N and
   min-loss always picks N=2 (BMR/Phase 23 is the correct tool). **Tier-A methodology fixes
   APPLIED 2026-06-09:** modulator epochs retimed to fractions of duration (all in training
   split) + `compute_elbo_model_selection` gained a fail-loud cross-dimensional guard; covered
   by `tests/test_latent_circuit_metrics.py` (8 tests pass). **Tier-B decision:** use
   Variational Laplace (already full-covariance → no structured SVI guide needed).
   **`LatentCircuitForward` adapter BUILT 2026-06-09** (`pyro_dcm.inference.forward_models`):
   direct-obs + bilinear B + time-domain residual for `_run_vl_generic`, validated by
   `tests/test_latent_circuit_vl.py` (VL recovers A/B signs, full covariance, R²>0.7).
   **DONE:** `cluster/scripts/lc_vl_acceptance_run.py` + `lc_vl_acceptance.sbatch` ran as job
   56268248 (10 seeds, all gates pass); `lc_vl_bmr_selection.py` (job 56270544) closed SYNTH-03.

   </details>
2. **[Phase 25 / HVAE-02] ~RESOLVED 2026-06-09 — metric artifact (like 20-05 R²).** Sign
   recovery was computed over ALL 16 A_free entries; with ~6 structural zeros per matrix and
   `sign(0)=0` never matching a non-zero prediction, each zero is a guaranteed miss. 0.4425 =
   7.08/16 → **~0.71 masked** (passes >0.6). Fixed: added `masked_sign_recovery` (|A_true|>0.1,
   unit-tested) used by the train script. Remaining: add an eval-only path to recompute the
   EXACT masked number on the existing checkpoint (job, no retraining) — see todo.
3. **[Phase 26 / SBI-03] SBC calibration fails — DIAGNOSED 2026-06-09, structural.** Job 55772094:
   2/9 pass. Failure mode = **parameter-specific bias** (not under-training: 50k sims; not
   overconfidence). Fixed a real plumbing bug (`--num-transforms/--hidden-features/--max-epochs`
   never reached `train_npe`). Retrain with a larger flow (job 56274446) **still 2/9 — the bias
   just redistributed**, so capacity is NOT the cause: the miscalibration is **structural**
   (likely `eig_clamp` non-injectivity near the stability boundary). Next: restrict the prior to
   the stable region / reparameterize, or accept that **VL/SVI is the calibrated path** and SBI
   is an optional speed-up, not a v0.6.0 blocker. See `2026-06-09-sbi-sbc-calibration-gap.md`.
4. **[Phase 24-01] Parcellation placeholder — violates "No Placeholders" critical rule.**
   `src/pyro_dcm/foundation/parcellation.py:146` assigns vertices to ROIs by naive equal-size
   contiguous blocks instead of the real Schaefer atlas vertex-to-parcel mapping. Fetches real
   atlas labels but averages the wrong vertices → scientifically invalid ROI timeseries for any
   real Phase 24 foundation-model analysis. Needs the nilearn surface-projection pipeline.
   *(2026-06-10: RESOLVED — rewrite confirmed real by audit; runtime validation → v0.7.0.)*

</details>

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 001 | Cluster sbatch infrastructure for Phase 16 | 2026-04-19 | 6bade20 | [001](./quick/001-cluster-sbatch-phase-16-acceptance/) |
| 002 | Structure-audit migration waves 1-5 | 2026-04-29 | 33e03e1 | [002](./quick/002-structure-audit-migration-waves-1-5/) |

### Pending Todos

6 pending — see `.planning/todos/pending/`. (HVAE-02 sign-recovery todo → `done/` 2026-06-10.)
- **Mutagen `models/` ignore (HIGH, INFRA)** — unanchored ignore excludes `src/pyro_dcm/models/`
  from M3 sync; recreate session with anchored ignores before v0.7.0 M3 runs
- Neural ODE DCM (Approach 2) — separate milestone v0.7.0+ after v0.6.0 informs whether bilinear suffices
- ROI projection for latent circuit DCM — map PCA circuit nodes back to brain ROIs; blocked on behavioral + neuroimaging dataset
- **Parcellation runtime validation → v0.7.0** — placeholder REMOVED; only nilearn runtime check remains
- **Phase 26 SBI-03 SBC calibration → v0.7.0 Phase D** — structural; VL is the calibrated path
- **VL overconfidence for BMR → v0.7.0 Phase C** — absolute prune threshold suppressed; relative ranking works

## Key Risks

- **[INFRA, surfaced 2026-06-10] Mutagen silently excludes `src/pyro_dcm/models/` from M3
  sync.** The `dcm-pytorch` session's unanchored `models/` ignore matches the source package;
  M3 was frozen at May 29. Invisible (reports "Watching", no conflict). Past impact nil (only
  the June-9 metric helper was stale; VL engine is in the synced `inference/`). Future hazard:
  edits under `models/` won't deploy. Stopgap: `scp` (path is ignored → safe). Fix: recreate
  session with anchored ignores — todo `mutagen-models-ignore`, memory
  `reference-mutagen-models-ignore-footgun`.


- **Bilinear misspecification (LC1).** Bilinear DCM is a first-order approximation of nonlinear RNN dynamics. Mitigated by linearization quality diagnostic and L&E nonlinear comparison.
- **Prior scale mismatch (LC4).** BOLD-calibrated priors wrong for RNN hidden states. Mitigated by mandatory recalibration on 5+ synthetic RNNs in Phase 20.
- **Rotational degeneracy (LC2).** PCA basis is arbitrary. Mitigated by Procrustes alignment and perturbation validation.
- **PCA discards task-relevant dynamics (LC3).** Mitigated by output-R-squared gate (>= 0.90) in Phase 21.
- **Multi-start convergence (LC11).** ELBO landscape has local optima. Mitigated by >=10 random restarts.

## Session Continuity

Last session: 2026-06-10 (executed Phase 29 Plan 03 — VL numerical-robustness guards)
Stopped at: Completed 29-03-PLAN.md. Added C-order CSD round-trip regression test (VLREC-05,
  guards commit 64e326f) + TaskDCMForward.build_precision intractability guard (VLROBUST-02,
  expected-vs-actual ValueError, dt>=0.1 floor) + 2 regression tests; registered `vl` marker.
  Commits bc30477, d12eb84, 09c9375. Phase 29 now 1/5 plans complete.
Next: execute remaining Phase 29 plans (29-01, 29-02, 29-04, 29-05) via `/gsd:execute-phase 29`.
  Note: pre-existing failures in tests/test_vl_forward_model_protocol.py task-DCM cases
  (make_block_stimulus/simulate_task_dcm signature drift) are unrelated to 29-03 — worth a
  cleanup pass. INFRA reminder: fix the Mutagen `models/` ignore before any v0.7.0 M3 run.
Resume file: None
