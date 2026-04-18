# Phase 16: 3-Region Bilinear Recovery Benchmark - Context

**Gathered:** 2026-04-18
**Status:** Ready for planning

<domain>
## Phase Boundary

Deliver a runnable bilinear DCM recovery benchmark on simulated data for a
3-region network (1 driving input, 1 modulatory input, 2 non-zero B elements)
that gates the v0.3.0 milestone via four RECOV acceptance criteria on >=10
seeds at SNR=3. Integrates with the v0.2.0 shared `.npz` fixture
infrastructure, `BenchmarkConfig`, and the existing figure pipeline.

Produces: `benchmarks/runners/task_bilinear.py` runner, ground-truth fixture
generators, acceptance-gated final report (pass/fail table + per-element B
recovery figure), and a bilinear-vs-linear wall-time comparison.

Out of scope: HGF trajectory signals themselves (v0.3.1 SIM-06), LinearInterpolatedInput (v0.3.1), amortized bilinear inference (v0.3.1 per D5), spectral/rDCM bilinear backends (v0.4+). Phase 16 guarantees the **hook** for HGF is in place but does not ship HGF itself.

</domain>

<decisions>
## Implementation Decisions

### Ground-truth network topology

- 3-region network, B[1,0] and B[2,1] non-zero (**asymmetric hierarchy**);
  remaining 7 B elements = 0. Tests downstream propagation of modulatory
  effect along a V1 -> V5 -> SPL-style hierarchy (SPM-canonical pattern).
- Null elements (|B_true| = 0) are used for RECOV-06 coverage_of_zero >= 85%;
  non-zero elements (B[1,0], B[2,1]) for RECOV-04 B RMSE <= 0.20 and RECOV-05
  sign_recovery_nonzero >= 80%.
- Specific B magnitudes for the two non-zero elements are **Claude's
  Discretion** during planning (guided by Pitfall B1 stability: |B_true| <=
  sigma_prior, and RECOV-04's 0.20 absolute-RMSE threshold). Likely in the
  range |B| ~ 0.3-0.5.
- A matrix and driving-input (C) topology: Claude's Discretion during
  planning; must match the v0.2.0 linear 3-region fixture family so RECOV-03
  (A RMSE <= 1.25 * linear-baseline RMSE) has a meaningful comparator.

### Modulatory-input signal family (ground truth for Phase 16)

- **Boxcar epochs via `make_epoch_stimulus` (SIM-02)** for the ground-truth
  modulator. SPM-standard; blur-robust under rk4 (mitigates Pitfall B12).
- Fixed amplitude per epoch (amplitude = 1.0); a handful of epochs across the
  total simulation duration. Specific epoch count, duration, and inter-epoch
  spacing: Claude's Discretion during planning.
- Explicitly NOT variable-amplitude in Phase 16 (that bridges to HGF but is
  not required for the v0.3.0 acceptance gate).
- Driving input (C): block design via `make_block_stimulus` on region 0.

### HGF trajectory forward-compatibility hook

- **Runner-level factory callable.** The bilinear benchmark runner signature
  accepts a `stimulus_mod_factory: Callable[[seed], PiecewiseConstantInput]`
  parameter (exact type: Claude's Discretion). Default factory uses
  `make_epoch_stimulus`; v0.3.1 will add an HGF factory returning a
  `LinearInterpolatedInput` (SIM-06) without requiring runner changes.
- The hook is **exercised by a placeholder mock factory in Phase 16's test
  suite** so the indirection is proven wired (not a theoretical API).
- The config/enum alternative was explicitly rejected: factory keeps the
  v0.2.0 `.npz` reproducibility path clean (config still stores the seed;
  factory is injected at runtime by the caller).
- **HGF ground truth re-uses Phase 16's (A, B, topology)** when SIM-06 lands
  in v0.3.1. Apples-to-apples: event/epoch vs HGF trajectory on identical
  neural dynamics. This is a **lock-in decision**, not an open question.

### Figures and final report format

- **Headline B figure: per-element forest plot.** For each of the 9 B
  elements, posterior median + 95% CI across seeds, with B_true as a
  reference dot. Readable at a glance for both non-zero and null elements.
- **Identifiability shrinkage (RECOV-07)** is **annotated inline** in the
  same B figure (e.g. color tint on the CI band, or subtitle per element
  listing `std_post/std_prior`). Single figure shows recovery +
  identifiability together. No dedicated shrinkage figure.
- **Acceptance gates: pass/fail table.** Row per criterion (RECOV-03, -04,
  -05, -06), columns = observed value | threshold | pass/fail. One-glance
  milestone gate. Matches `/gsd:verify-work` format.
- **Wall-time (RECOV-08): inline metric in summary table.** Single number
  comparison: `bilinear: X s | linear baseline: 235s | ratio: Yx`. Flag with
  warning note if > 10x per Pitfall B10. No dedicated runtime figure.
- **A-matrix recovery** (RECOV-03): posterior-median-vs-truth scatter or
  summary RMSE table: Claude's Discretion (standard format from v0.2.0 is
  acceptable).
- Per-seed raw posterior artifacts saved under `benchmarks/results/` per
  v0.2.0 convention; aggregate figures saved under `benchmarks/figures/`.

### Claude's Discretion

- **Guide variant for the acceptance benchmark.** Phase 15 verified bilinear
  works across AutoNormal, AutoLowRankMVN, AutoIAFNormal. Researcher/planner
  picks which variant(s) run the acceptance gate (single guide is lighter
  per Pitfall B10; multi-guide comparison is more defensible for milestone).
  Default expectation: **single guide for acceptance, with a design-note
  sidebar comparing 2-3 variants on a subset of seeds if runtime allows**.
- **Seed count, CI subset, SVI step count, wall-time budget per invocation.**
  RECOV floor is >=10 seeds; exact N, step count, early-stopping, CI-vs-local
  split are planner calls guided by the expected 3-6x slowdown (Pitfall B10).
- Specific A / C values, B magnitudes, epoch count/duration/spacing for the
  ground-truth fixtures (within the topology and signal-family locked
  above).
- Choice of per-seed artifact format and v0.2.0 figure-pipeline integration
  specifics.
- A-matrix figure format.

</decisions>

<specifics>
## Specific Ideas

- "Same topology when HGF lands" — apples-to-apples comparison in v0.3.1.
  The benchmark should ship with fixtures that will survive the HGF signal
  swap without ground-truth re-design.
- Compact report over comprehensive: pass/fail table + one B figure is the
  target, not a multi-page diagnostic dashboard. Sidebars and per-seed
  artifacts exist but don't dominate the acceptance view.
- RECOV-08's wall-time is a **milestone risk flag** (not an acceptance
  criterion); keep it visible but terse.

</specifics>

<deferred>
## Deferred Ideas

- **SIM-06: `LinearInterpolatedInput` for smooth-ramp modulatory inputs
  (HGF belief-update trajectories).** Belongs in v0.3.1. Phase 16 ships the
  factory-callable hook so this lands without runner churn.
- **v0.3.1 HGF recovery benchmark variant.** When SIM-06 lands, add a
  `benchmarks/runners/task_bilinear_hgf.py` sibling that re-uses Phase 16's
  `(A, B, topology)` ground truth but injects an HGF trajectory through the
  same factory hook. Compares recovery quality between epoch and HGF
  modulator shapes on identical neural dynamics.
- **Multi-guide acceptance sweep** (AutoNormal vs AutoLowRankMVN vs AutoIAF
  across full seed grid). Not required for v0.3.0 acceptance; could inform
  guide-selection guidance in future docs.
- **Variable-amplitude epoch modulator.** A partial bridge between the
  fixed-amplitude Phase 16 boxcars and v0.3.1 HGF trajectories. Could be a
  middle-state v0.3.1 benchmark if HGF proves complex.
- **Per-criterion distribution plots** (seed-wise histograms with threshold
  overlays for each RECOV gate). Not shipped in the Phase 16 report; can
  live as a diagnostic sidebar if needed during the run.
- **Bar-chart wall-time figure.** Pass/fail table treatment is sufficient
  for v0.3.0.

</deferred>

---

*Phase: 16-bilinear-recovery-benchmark*
*Context gathered: 2026-04-18*
