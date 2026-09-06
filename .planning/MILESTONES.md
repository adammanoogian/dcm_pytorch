# Project Milestones: Pyro-DCM

## v0.8.0 DCM for Evoked Responses (EEG/MEG ERP) (Complete: 2026-06-26)

**Delivered:** A complete time-domain ERP forward stack — canonical microcircuit neural mass → evoked response → single-dipole lead field → scalp ERP — verified against SPM12 at *every* stage on byte-frozen MATLAB fixtures, plus the ERP-DCM Pyro model, amortized wiring, and a 5-source auditory-MMN precision-sweep demo.

**Phases completed:** 33-36 (4 phases, 12 plans), each gsd-verifier-passed

**Key accomplishments:**

- **SPM12 forward parity at every stage** — the strongest validation in the repo. `cmc_f` is bit-exact against `spm_fx_cmc`; the exponential-Euler integrator is bit-exact against `spm_int_L` (measured `matrix_exp` floor 8.6e-11); the multi-source network matches `spm_gen_Q` + `spm_gen_erp` at N=5; scalp projection matches `spm_lx_erp`; full-pipeline gate 6.4e-11.
- **`utils/local_linearization.py`** — the SPM12 frozen-Jacobian exp-Euler integrator (Ozaki 1992), CMC-agnostic, fixture-verified before anything was built on it.
- **`ERPDCMForward`** — a 4th `ForwardModel` protocol implementor, so the Phase 28 VL engine was reused unchanged.
- **MMN mechanism shipped**: monotone precision → |MMN| attenuation via the diag→G path, delivered as a gated demo script.
- **Parity ladders are fixture-keyed and MATLAB-independent** — they assert pure-torch output against committed `.mat` files, which is why the whole regression suite survived the M3 → DCCN move intact.

**Honest limits:**

- **Forward/synthetic only** — no empirical ERP data was fitted.
- **Frontal scalp topography was NOT recovered.** ERPDCM-04 turned out to be an ECD phenomenon, unrecoverable with the single-dipole LFP lead field. Reclassified into a defined follow-up (Phase 37) rather than dropped or quietly downgraded.
- Delay-free (`D = I`) only; the `P.M` term is unimplemented.

**Stats:**

- 60 commits over 2 days (2026-06-25 → 2026-06-26), `4122624` → `cb0167c`
- 79 ERP tests green (M3 job 57904695), 11 ERP/CMC test files

**What's next:** Phase 37 (ECD dipole lead field + frontal-dominant scalp MMN), **blocked on verified 5-source MNI coordinates from the user**; may become v0.8.1. The milestone has not been formally closed with `/gsd:complete-milestone`.

---

## v0.7.0 Variational Laplace Validation (Complete: 2026-06-12)

**Delivered:** A validation-breadth milestone proving the Variational Laplace engine is trustworthy across a Cartesian product of network sizes, SNR levels and forward models — and establishing precisely where it is not.

**Phases completed:** 29-32 (4 phases, 14 plans), each gsd-verifier-passed

**Key accomplishments:**

- **Full recovery matrix**: N{2,4} × SNR{1,3} × {spectral, task, latent-circuit}, 10 seeds per cell. **10/10 cells classified, 0 errored — 6 PASS, 4 documented identifiability limits with evidence.** No silent failures.
- **BMR relative-evidence ranking** recovers true sparse structure 5/5 seeds at both N=2 and N=4 with a positive separation gap, and agrees with brute-force VL refits.
- ~~**Two hard identifiability findings**: spectral DCM cannot identify a lone off-diagonal A entry, and a feed-forward chain A produces a CSD *bit-identical* to the empty graph.~~ **⚠️ RETRACTED 2026-09-06** — this was a defect in `compute_transfer_function_hemodynamic` (`pinv` of a defective eigenvector basis), not a property of spectral DCM. SPM12 separates chain from empty graph by 1.427; the port collapsed it to 9e-16. Fixed in `007b686`. The reciprocal-edge *results* below stand; the reason for choosing reciprocal edges does not. What spectral DCM can actually identify is now an open question.
- **VL determinism** across all three forward models (fixed-seed, within-machine, atol 1e-8), with the cross-machine caveat documented.
- Task DCM recovers cleanly at N=2 (sign 1.0, A-RMSE ~0.04); **task N=4 is a genuine identifiability limit** (sign 0.57, coverage 0.0), reported as a finding rather than patched away.

**Honest limits:**

- **The strict 5% VL-vs-SPM12 matched-free-energy gate was MISSED.** `relative_error = 0.8776`, traced to a **constant 269.895-nat offset** (M3 job 56407192). Cross-model ranking agreement was 1.0 and free-space `Ep` off-diagonals 17%/47% against a ~10% target. The miss is recorded in JSON, not hidden — but VL absolute free energy is *not* SPM-comparable, and this is the largest open scientific question in the repo.
- **Absolute-ΔF BMR pruning remains structurally broken** (posterior std ~0.001-0.01× prior at high SNR). Only relative ranking is defensible.
- **Posterior tempering is exploratory, not a headline claim.** T=2.0 restores coverage on task-N4 without changing the ranking, but the same T breaks positive-definiteness on task-N2 — a recorded cross-condition hazard.
- No real data; SBI SBC calibration (2/9 parameters) deferred.

**Stats:**

- 76 commits over 2 days (2026-06-10 → 2026-06-12), `b833ea5` → `4122624`
- 19/19 requirements mapped and delivered (VLINFRA, VLREC, VLBMR, VLSPM, VLROBUST)

**What's next:** v0.8.0 DCM for Evoked Responses.

---

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

## v0.5.0 MNE-Python Integration (Shipped: 2026-05-24)

**Delivered:** M/EEG data ingestion via MNE-Python and BIDS, plus end-to-end pipeline demonstrations.

**Phases completed:** 18-19 (4 plans)

**Key accomplishments:**

- `src/pyro_dcm/io/` — `mne_loader.py` and `bids_loader.py`, with a 17/17-must-have IO test suite
- End-to-end pipeline demos (10/10 must-haves verified)
- `mne` optional dependency group and pytest marker

**What's next:** v0.6.0 Latent Circuit DCM.

---

## v0.4.0 Circuit Explorer (Shipped: 2026-05-21)

**Delivered:** Interactive serialization and rendering for DCM model configs and fitted posteriors.

**Phases completed:** 17 (1 plan, verified 15/15 must-haves)

**Key accomplishments:**

- `utils/circuit_viz.py` — CircuitViz JSON serializer plus a static circuit-explorer HTML template
- Structural acceptance (JSON schema validity, round-trip equality, planned↔fitted toggle) rather than RMSE/coverage gates

**What's next:** closed jointly with v0.3.0 in commit `76c3ced`.

---

## v0.3.0 Bilinear DCM Extension (Shipped: 2026-05-21)

**Delivered:** The full Friston 2003 bilinear neural state equation `dx/dt = Ax + Σ_j u_j B_j x + Cu` propagated end-to-end through forward model, simulator, Pyro model and priors.

**Phases completed:** 13-16 (12 plans). Phase 16.1 was inserted then **superseded**.

**Key accomplishments:**

- Bilinear neural state + stability monitor, stimulus utilities, bilinear simulator, B priors and masks
- **The defining result is a negative one turned positive:** the Phase 16 recovery benchmark FAILED its RECOV-04 B-RMSE acceptance gate under SVI (B-RMSE 0.3467). Rather than tune the guide, the failure was diagnosed to SVI's first-order mean-field approximation and resolved by writing the **Variational Laplace engine**, which recovered B at RMSE 0.0170 on the same problem — proving the forward model correct.
- That engine (`inference/variational_laplace.py`) became the backbone of v0.6.0, v0.7.0 and v0.8.0, and was retroactively formalized as Phase 28.
- Phase 16.1 (the planned B-RMSE shrinkage diagnostic) was therefore never executed.

**Stats:** 27/27 v0.3.0 requirements complete. Closed with v0.4.0 in commit `76c3ced`.

**What's next:** v0.5.0 MNE-Python Integration.

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
