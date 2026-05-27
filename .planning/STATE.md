# State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-24)

**Core value:** A matrix (effective connectivity) remains explicit and interpretable with full posterior uncertainty
**Current focus:** v0.6.0 DCM Interpretability for Neural Data Models -- DCM as interpretability tool for deep learning on simulated/real M/EEG data

## Current Position

**Milestone:** v0.6.0 (restructured 2026-05-27; simulation-first pivot)
**Phase:** Phase 23 Plan 01 COMPLETE (BMR core)
**Plan:** Phase 23-01: BMR core function implemented and tested
**Status:** Phase 23-01 COMPLETE. bayesian_model_reduction() + make_reduced_prior_zero_connection() in pyro_dcm.model_selection, 8/8 tests pass, validated against exact conjugate Gaussian Bayes factor.
**Last activity:** 2026-05-27 -- Phase 23-01 complete (commit 97ef99a). BMR Laplace approximation formula derived and validated.

**Prior milestones in flight:**
- v0.5.0: Phase 18 COMPLETE; Phase 19 COMPLETE (both plans done)
- v0.3.0: Phase 16.1 pending (RECOV-04 B-RMSE diagnostic)
- v0.6.0: Phase 20 partial (A-RMSE passes, B/R2/ELBO fail) | Phase 21 dropped | Phase 22 COMPLETE (raw=1/9, latent=2/9) | Phase 23-01 COMPLETE (BMR core)

Progress: v0.1.0 [==========] 100% | v0.2.0 [==========] 100% | v0.3.0 [=========-] 16.1 pending | v0.4.0 [==========] Phase 17 complete | v0.5.0 [==========] Phases 18+19 complete | v0.6.0 [============------] Ph20 partial; Ph21 dropped; Ph22 DONE; Ph23 executing; Ph24-27 planned

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
- Prior v0.3.0/v0.4.0/v0.5.0 decisions: see earlier STATE.md history in git log.

## Blockers

None currently.

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 001 | Cluster sbatch infrastructure for Phase 16 | 2026-04-19 | 6bade20 | [001](./quick/001-cluster-sbatch-phase-16-acceptance/) |
| 002 | Structure-audit migration waves 1-5 | 2026-04-29 | 33e03e1 | [002](./quick/002-structure-audit-migration-waves-1-5/) |

### Pending Todos

1 pending — see `.planning/todos/pending/`.
- Neural ODE DCM (Approach 2) — separate milestone v0.7.0+ after v0.6.0 informs whether bilinear suffices

## Key Risks

- **Bilinear misspecification (LC1).** Bilinear DCM is a first-order approximation of nonlinear RNN dynamics. Mitigated by linearization quality diagnostic and L&E nonlinear comparison.
- **Prior scale mismatch (LC4).** BOLD-calibrated priors wrong for RNN hidden states. Mitigated by mandatory recalibration on 5+ synthetic RNNs in Phase 20.
- **Rotational degeneracy (LC2).** PCA basis is arbitrary. Mitigated by Procrustes alignment and perturbation validation.
- **PCA discards task-relevant dynamics (LC3).** Mitigated by output-R-squared gate (>= 0.90) in Phase 21.
- **Multi-start convergence (LC11).** ELBO landscape has local optima. Mitigated by >=10 random restarts.

## Session Continuity

Last session: 2026-05-27 (Phase 23-01 complete)
Stopped at: Phase 23-01 BMR core implementation complete. 8/8 tests pass, commit 97ef99a.
Next: Phase 23 remaining plans (if any). Phase 20-05 acceptance still needs rework.
Resume file: None
