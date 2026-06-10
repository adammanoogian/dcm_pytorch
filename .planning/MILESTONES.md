# Project Milestones: Pyro-DCM

## v0.6.0 Latent Circuit DCM — DCM Interpretability for Neural Data Models (Shipped: 2026-06-10, scope-cut)

**Delivered:** A synthetic-validated DCM-recovery methodology plus an SPM12-grade Variational Laplace inference engine (the structured-posterior path that closed synthetic recovery), with ready — but un-run — real-data infrastructure (neural-data-model pipeline, foundation-model extractors, hybrid VAE-DCM, SBI). Real-data scientific application was audited as undelivered and **deferred to v0.7.0** (deferred, not failed).

**Phases completed:** 20-27 + retroactive Phase 28 (VL engine) — 34 plans total

**Key accomplishments:**

- **Variational Laplace inference engine** matching SPM12 `spm_nlsi_GN` (Gauss-Newton E-step, ReML M-step, SVD parameter reduction, 3-term free energy, full posterior covariance) generalized via a `ForwardModel` protocol (spectral / task / latent-circuit); now the DCM default inference.
- **Synthetic recovery validated** (Phase 20 via VL): A-RMSE 0.026, B-RMSE 0.0048, sign 1.00, CI coverage 1.00, pooled-R² 0.961 — the B-collapse mean-field SVI couldn't fix.
- **Bayesian Model Reduction** (Friston & Penny 2011), validated vs brute-force ELBO (~93× faster); recovers true circuit structure 3/3 via relative evidence ranking.
- **Hybrid VAE-DCM** delivered 4/4: amortized encoder + DCM ODE decoder, A-RMSE 0.076, masked sign recovery 0.77, no posterior collapse (KL 18.8), 0.76 ms inference.
- **Ready real-data infrastructure**: neural-data-model interpretability pipeline (synthetic-validated), real foundation-model extractors (TRIBE v2 / LaBraM / BrainOmni) + real Schaefer parcellation, SBI/NPE for spectral DCM.
- **Goal-backward milestone audit** that honestly separated delivered (synthetic + engine) from deferred (real-data), driving the scope-cut.

**Stats:**

- 174 commits over 18 days (2026-05-24 → 2026-06-10)
- 8 phases (20-27) + 1 retroactive (28), 34 plans
- Library ~18,800 LOC Python (cumulative)

**Git range:** `8ed4c2d` → `1bba88c`

**What's next:** v0.7.0 — VL validation matrix + the deferred real-data application (real Cam-CAN M/EEG interpretability, real foundation-model runs + cross-modal comparison, SBI SBC calibration). Seed: `.planning/v0.7.0-VL-RECONCILIATION-DRAFT.md`.

---

## v0.2.0 Cross-Backend Inference Benchmarking (Shipped: 2026-04-13)

**Delivered:** Systematic calibration study across 6 Pyro guide families, 3 ELBO objectives, and 3 DCM variants, with tiered benchmark sweep infrastructure, 9 publication-quality figure types, and a practical recommendation guide with Mermaid decision tree.

**Phases completed:** 9-12 (11 plans total)

**Key accomplishments:**

- Shared fixture infrastructure for reproducible benchmarks (3 variants x 3 sizes x N seeds)
- 6 SVI guide types in `create_guide` factory with blocklist guards (AutoMVN blocked at N>=8)
- 3 ELBO objectives (Trace_ELBO, TraceMeanField_ELBO, RenyiELBO) with compatibility enforcement
- Multi-level coverage calibration (4 CI levels) with per-parameter breakdown (diagonal/off-diagonal A)
- Tiered calibration sweep orchestrator (42 configs at tier=all) with resume support
- 9 publication figure types: calibration curves, comparison tables, scaling study, violin plots, Pareto frontier, timing breakdown
- Mermaid decision tree guide for guide selection by variant/size/budget with 5 dedicated warnings
- Benchmark narrative rewrite with zero TBD entries (14 v0.1.0 placeholders replaced)

**Stats:**

- 47 commits over 6 days
- ~26,300 lines of Python across library, benchmarks, and tests
- 4 phases, 11 plans
- 12 requirements: 12 shipped, 0 dropped

**Git range:** `75cb91a` -> `9eaac48`

**What's next:** v0.3+ candidates include NumPyro backends (NUTS/ADVI), regularization study, amortized refinement pipeline, and SPM12 cross-validation.

---

## v0.1.0 Foundation (Shipped: 2026-04-03)

**Delivered:** Complete DCM framework with three variants (task, spectral, regression), Pyro probabilistic inference, amortized normalizing flow guides, SPM12 cross-validation, and benchmark suite.

**Phases completed:** 1-8 (26 plans total)

**Key accomplishments:**

- Three DCM forward models with full mathematical fidelity (every equation cites paper reference)
- Pyro generative models with mean-field SVI achieving RMSE < 0.02 on spectral DCM
- Amortized Neural Spline Flow guides for instant (<1s) posterior inference
- SPM12 cross-validation infrastructure with .mat export and MATLAB batch scripts
- rDCM analytic VB matching Julia reference implementation (model ranking 100% agreement)
- Benchmark CLI with 7 runners, publication-quality figures, and methods section (MD + LaTeX)

**Stats:**

- 127 commits over 9 days
- ~6,200 lines of library code across 21 source files
- 57+ tests (unit, integration, recovery, validation)
- 8 phases, 26 plans

**Git range:** `d4b3e7f` → `33fc134`

---
