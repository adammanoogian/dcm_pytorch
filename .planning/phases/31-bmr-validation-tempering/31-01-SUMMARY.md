---
phase: 31-bmr-validation-tempering
plan: 01
subsystem: testing
tags: [bmr, variational-laplace, spectral-dcm, model-selection, rank_connections, identifiability]

# Dependency graph
requires:
  - phase: 29-vl-validation-infra-bmr-rank
    provides: "rank_connections (relative single-prune BMR ranking + separation gap)"
  - phase: 28-variational-laplace-engine
    provides: "run_variational_laplace_generic + SpectralDCMForward + result.theta_post/sigma_post packing"
provides:
  - "benchmarks/bmr_recovery.py: sparse ground-truth A builder + VL-result -> rank_connections tensor extractor (full A_free covariance slice) + C-order off-diagonal index set"
  - "tests/test_bmr_vlbmr01_recovery.py: VLBMR-01 end-to-end relative-ranking structure-recovery test (spectral N=2/N=4, 5 seeds, 5/5)"
  - "Empirical finding: a feed-forward chain A is unidentifiable from spectral CSD (CSD bit-identical to the empty graph); reciprocal edges are recoverable"
affects: [31-03, bmr-tempering, model-comparison, spectral-dcm-identifiability]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Sparse constructed ground truth (explicit present_edges) over reading structure off a dense A (pitfall R2)"
    - "Full A_free posterior covariance via result.sigma_post[:N*N,:N*N] (A_free packed first), NOT the diagonal-only marginal std"
    - "C-order flat index i*N+j with round-trip divmod guard (pitfall S4)"

key-files:
  created:
    - benchmarks/bmr_recovery.py
    - tests/test_bmr_vlbmr01_recovery.py
  modified: []

key-decisions:
  - "[31-01-D1] A feed-forward chain is unidentifiable by spectral DCM; VLBMR-01 ground truth uses RECIPROCAL edges."
  - "[31-01-D2] N=2 saturated-reciprocal case has no absent prunable edge, so the separation_after_rank==K cut is degenerate and is asserted conditionally."
  - "[31-01-D3] # type: ignore[import-untyped] on pyro_dcm imports (no py.typed), per the established [30-01-D4] convention; pyproject untouched."

patterns-established:
  - "VLBMR-01 gate: top-K essential off-diagonal edges == true present edges (set equality) + positive separation_gap + cut at K when an absent edge exists; NEVER an absolute-delta-F threshold (pitfall C1)."

# Metrics
duration: ~25min
completed: 2026-06-11
---

# Phase 31 Plan 01: VLBMR-01 Recovery Harness Summary

**BMR relative-evidence ranking on a real spectral-DCM VL posterior recovers the true sparse circuit structure 5/5 across seeds for N=2 and N=4 — top-K essential off-diagonal edges == true present edges, positive separation gap, cut at K — using RECIPROCAL ground-truth edges because a feed-forward chain is unidentifiable from spectral CSD.**

## Performance

- **Duration:** ~25 min (including diagnostic investigation of the chain-unidentifiability finding)
- **Tasks:** 2
- **Files created:** 2
- **Test runtime:** 112.71s laptop CPU (2 parametrizations x 5 seeds, well under the 3-min budget)

## Accomplishments

- `benchmarks/bmr_recovery.py` — three reusable helpers (also feed Plan 31-03):
  - `make_sparse_ground_truth_A` (unambiguous sparse true edges + diagonal-edge guard + stability guard)
  - `offdiag_indices` (C-order off-diagonal flat indices, pitfall S4)
  - `bmr_tensors_from_vl_result` (full `A_free` covariance slice `sigma_post[:N*N,:N*N]` + zero-mean prior, ready for `rank_connections`)
- `tests/test_bmr_vlbmr01_recovery.py` — the `@pytest.mark.vl` VLBMR-01 recovery test: **5/5 seeds recover true structure for both N=2 and N=4.**
- **Identified and documented a genuine spectral-DCM identifiability property** (the primary scientific finding of this plan): a strictly lower-triangular (feed-forward chain) `A` produces a stationary CSD **bit-identical** to the empty-graph CSD (relative difference 0.0), so VL recovers the zero off-diagonal and BMR has no signal to rank. Reciprocal (bidirectional) edges create the cross-spectral coupling spectral DCM can see.

## Task Commits

1. **Task 1: sparse ground-truth + VL-result BMR tensor helpers** — `c081c5c` (feat)
2. **Task 2: VLBMR-01 relative-ranking structure-recovery test** — `766dd24` (feat)

## Test Result

```
N=2: recovered 5/5 seeds (true present edges [1, 2], K=2)  separation_gap=34307.15  cut=rank 1 (degenerate, saturated)
N=4: recovered 5/5 seeds (true present edges [1, 4, 6, 9, 11, 14], K=6)  separation_gap=1110631.39  cut=rank 6 == K
2 passed in 112.71s (0:01:52)
```

ruff + mypy clean on both new files.

## Files Created/Modified

- `benchmarks/bmr_recovery.py` — sparse ground-truth A builder + VL-result -> rank_connections tensor extractor + off-diagonal index helper (pure plumbing; no `rank_connections` call).
- `tests/test_bmr_vlbmr01_recovery.py` — `@pytest.mark.vl` VLBMR-01 end-to-end recovery test.

## Decisions Made

- **[31-01-D1] A feed-forward chain is unidentifiable by spectral DCM; VLBMR-01 ground truth uses RECIPROCAL edges.** The plan specified a feed-forward chain (N=2 `[(1,0)]`, N=4 `[(1,0),(2,1),(3,2)]`). Empirically, the simulated CSD of a strictly lower-triangular `A` is **bit-identical** to the empty-graph CSD (`||csd_chain - csd_zero|| / ||csd_zero|| = 0.0`), so VL collapses `A_free` to exactly zero and every single-prune delta-F is 0.0 (the resulting rank order is float sign-noise `[3,7,11]` = the *transpose* of the true edges). This is a real spectral-DCM identifiability limit, not a code or index bug. Switched the present edges to reciprocal pairs (N=2 `[(0,1),(1,0)]` K=2; N=4 reciprocal chain `[(1,0),(0,1),(2,1),(1,2),(3,2),(2,3)]` K=6), which `make_stable_A_spectral`-style dense recovery and the diagnostic confirmed are cleanly recoverable (5/5). The builder, plumbing, gate semantics, and the NEVER-absolute-delta-F contract are unchanged — only the choice of identifiable present edges changed.
- **[31-01-D2] N=2 saturated-reciprocal case has no absent prunable edge → conditional cut gate.** For N=2 the only identifiable sparse structure is the fully reciprocal pair (both off-diagonals present); a single directed N=2 edge is feed-forward and unidentifiable (D1). With `K == len(prunable)` there is no essential/non-essential boundary, so `separation_after_rank == K` is unsatisfiable (the cut lands at rank 1). Gated that assertion on `has_absent_edges = K < N*(N-1)`; recovery + positive `separation_gap` are still asserted unconditionally. N=4 (K=6 < 12 prunable) exercises the full `cut == K` gate.
- **[31-01-D3] `# type: ignore[import-untyped]` on the pyro_dcm imports.** `pyro_dcm` ships no `py.typed` (same condition as every pyro_dcm-importing benchmark/test; [30-01-D4]). Scoped to the test file's import block; pyproject mypy config untouched.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug in plan premise] Feed-forward chain ground truth is unrecoverable by spectral DCM → reciprocal edges**
- **Found during:** Task 2 (test first run: 0/5 recovery for N=4, top-K = `[3,7,11]` = transpose of true `[4,9,14]`; N=2 A_free all-zero with both prune costs 0.0).
- **Issue:** The plan's feed-forward chain produces a CSD bit-identical to the empty graph (verified `relative diff = 0.0`); VL recovers zero off-diagonal and BMR ranks float sign-noise. This is a genuine spectral-DCM identifiability property.
- **Fix:** Changed `present_edges` to reciprocal pairs (N=2 K=2, N=4 K=6) — same builder, same plumbing, same gate. Documented the finding in the module docstring and [31-01-D1].
- **Files modified:** tests/test_bmr_vlbmr01_recovery.py
- **Verification:** 5/5 recovery both cases; diagnostic showed reciprocal-CSD relative difference 30.5x vs 0.0 for the chain.
- **Committed in:** 766dd24 (Task 2 commit)

**2. [Rule 1 - Bug] `separation_after_rank == K` unsatisfiable when all prunable edges present (N=2)**
- **Found during:** Task 2 (N=4 passed; N=2 failed `separation_after_rank == 2`, got 1).
- **Issue:** With every off-diagonal present (saturated N=2 reciprocal), there is no non-essential edge after rank K, so the cut cannot land at K — the gap is between the two essential edges (rank 1).
- **Fix:** Gated the cut-at-K assertion on `has_absent_edges = K < N*(N-1)`; recovery + positive gap still always asserted.
- **Files modified:** tests/test_bmr_vlbmr01_recovery.py
- **Verification:** 2 passed; N=4 still exercises the full cut==K gate.
- **Committed in:** 766dd24 (Task 2 commit)

**3. [Rule 3 - Blocking] mypy import-untyped on pyro_dcm imports**
- **Found during:** Task 2 (mypy 3 import-untyped errors).
- **Fix:** `# type: ignore[import-untyped]` on the three pyro_dcm import blocks (established [30-01-D4] convention).
- **Files modified:** tests/test_bmr_vlbmr01_recovery.py
- **Committed in:** 766dd24 (Task 2 commit)

---

**Total deviations:** 3 auto-fixed (2 bug/premise, 1 blocking). No architectural changes; no scope creep.
**Impact on plan:** The VLBMR-01 objective, gate semantics, and the relative-only (never absolute-delta-F) contract are delivered exactly as intended. The single substantive change — reciprocal vs feed-forward ground-truth edges — surfaced a real spectral-DCM identifiability property worth carrying forward to 31-03.

## Issues Encountered

- The transpose-looking failure (`topk=[3,7,11]` vs `true=[4,9,14]`) initially read like a C-order index bug, but the round-trip S4 guard and direct inspection proved `A_free` was identically zero — the real cause was CSD-level unidentifiability of the feed-forward chain, not indexing. Resolved by the diagnostic CSD comparison (chain vs empty vs reciprocal).

## Next Phase Readiness

- `benchmarks/bmr_recovery.py` helpers are ready for Plan 31-03 to reuse (the full A_free covariance slice + off-diagonal index set + reciprocal sparse ground truth).
- **Carry-forward for 31-03 / tempering:** spectral-DCM ground truth for BMR validation MUST use reciprocal (bidirectional) edges — feed-forward chains are CSD-invisible. The huge separation_gap magnitudes (1e4–1e6 nats) reflect VL Laplace overconfidence (pitfall C1); they are RELATIVE-ranking signal only and are correctly NOT gated as absolute thresholds — this is exactly the overconfidence regime tempering (31-03) targets.

---
*Phase: 31-bmr-validation-tempering*
*Completed: 2026-06-11*
