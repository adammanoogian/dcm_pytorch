# Project Milestones: Pyro-DCM

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
