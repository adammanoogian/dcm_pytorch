---
created: 2026-04-19T22:30
closed: 2026-04-24
title: Phase 16 step-0 NaN root cause + seed-pool fix
area: benchmarks
files:
  - benchmarks/runners/task_bilinear.py
  - src/pyro_dcm/simulators/task_simulator.py
  - tests/test_task_bilinear_benchmark.py
  - tests/test_bilinear_simulator.py
superseded_initial_fix_commits:
  - 9c50011 (fix(benchmarks): retry bilinear SVI on step-0 NaN at halved init_scale)
---

## Original Problem (as captured 2026-04-19)

Phase 16 v0.3.0 acceptance gate failed on cluster job 54901072 (156 min)
because 3 of 10 seeds (44, 49, 50) NaN'd at SVI step 0 before any gradient
update. Test aborted at the FIX-1 precondition (`n_success >= 10`) — no
RECOV gate was even evaluated.

## Initial Diagnosis (wrong) + First Fix (ineffective)

Commit `9c50011` hypothesized init-scale sensitivity (Pitfall B1/B6) and
added `_fit_bilinear_with_retry` — on step-0 NaN, halve init_scale
(0.005 → 0.001) and retry once. Test was relaxed to `n_success >= 8` with
a UserWarning at 8–9.

Cluster job `54902455` (~171 min) re-ran with the retry landed and
**also failed** on the same 3 seeds — retry at 0.001 also NaN'd at step 0,
n_success = 7 < 8.

## Actual Root Cause (2026-04-24)

Local reproducer on seed 44 with `scripts/debug_phase16_nan_seeds.py`
revealed that `_make_bilinear_ground_truth` produces a BOLD fixture
containing 100 NaN values before SVI ever starts. The step-0 "NaN ELBO"
is then a downstream consequence: the Gaussian likelihood computes
`Normal(predicted_bold=0, noise_std).log_prob(observed_bold=NaN)` which
is NaN regardless of guide init_scale. The `predicted_bold` NaN-safe
guard in `task_dcm_model.py:379` zeros the predicted side but cannot
rescue a NaN observation.

Enumeration across seeds 42..51 (`scripts/debug_phase16_fixture_check.py`)
confirmed the pattern:

| seed | eig(A) max Re | eig(A+B) max Re | bold_nan | status |
|------|--------------:|----------------:|---------:|--------|
| 42   | −0.3407       | −0.5000         |   0      | CLEAN  |
| 43   | −0.5000       | −0.2130         |   0      | CLEAN  |
| **44** | −0.3760     | −0.1561         | **100**  | CORRUPT|
| 45   | −0.3655       | −0.5000         |   0      | CLEAN  |
| 46   | −0.5000       | −0.0839         |   0      | CLEAN  |
| 47   | −0.3147       | −0.2483         |   0      | CLEAN  |
| 48   | −0.5000       | −0.5000         |   0      | CLEAN  |
| **49** | −0.2577     | −0.5000         | **100**  | CORRUPT|
| **50** | −0.5000     | −0.3644         | **100**  | CORRUPT|
| 51   | −0.4769       | −0.3104         |   0      | CLEAN  |

Exactly matches cluster observation (44, 49, 50 corrupt). Critically,
seeds 49 and 50 have stable `A + B` eigenvalues (max Re = −0.50, −0.36)
yet still produce NaN BOLD — so the instability is **not** a simple
eigenvalue condition on `A + sum(B_j)`. Likely drivers are non-normal
matrix transient growth and/or the Balloon-hemodynamic nonlinearity; the
empirical post-hoc NaN check is the right filter regardless of the exact
mechanism.

## Real Fix

1. **`simulate_task_dcm`**: add `simulation_diverged: bool` to the
   return dict, True when bold_clean contains NaN/Inf. Non-intrusive
   diagnostic flag — existing callers unaffected.
2. **`run_task_bilinear_svi`**: replace the fixed-seed loop with a seed
   pool of size `n_datasets * _MAX_POOL_MULTIPLIER` (default 3). For
   each candidate seed, generate the fixture; if BOLD contains NaN/Inf,
   skip the seed and draw the next one. Collect `seeds_used` and
   `seeds_skipped_corrupt` in the return dict and metadata. Fail with
   `status='insufficient_data'` + `pool_exhausted=True` if the pool
   exhausts before `n_datasets` clean fixtures are collected.
3. **`test_acceptance_gates_pass_at_10_seeds`**: tighten back to
   `n_success >= 10` (the pool filters corrupt seeds upstream, so any
   seed that makes it into `seeds_used` should complete SVI cleanly).
   Error message now references `_MAX_POOL_MULTIPLIER` for the raise-
   the-cap escape hatch.
4. **`fixtures_dir` cache**: raise `NotImplementedError` — the
   `.npz` cache is keyed by slot index, which breaks when the pool
   skips seeds. Inline generation only for v0.3.0; cache-by-seed is
   deferred to v0.3.1.

## Files Touched

- `benchmarks/runners/task_bilinear.py` — seed pool + fixture NaN
  filter; `_MAX_POOL_MULTIPLIER = 3` constant; `seeds_used` /
  `seeds_skipped_corrupt` surfaced.
- `src/pyro_dcm/simulators/task_simulator.py` — `simulation_diverged`
  flag added to return dict.
- `tests/test_bilinear_simulator.py` — 2 new unit tests on
  `simulation_diverged`.
- `tests/test_task_bilinear_benchmark.py` — `TestSeedPoolCorruptSkip`
  class (2 tests) + `_FakePiecewise` helper; acceptance test reverted
  to `n_success >= 10`.

## Kept For Defensive Depth

- `_fit_bilinear_with_retry` (init_scale halving) is kept. It is no
  longer the core fix but handles any unrelated step-0 NaN that the
  filter doesn't catch (e.g. pathological guide-param initialization).
  The module-level comment now documents this explicitly.

## Validation

- Reproducer confirmed seed 44 step-0 NaN originates in observed_bold,
  not guide init.
- Fixture enumeration confirmed 44/49/50 corruption matches cluster
  data.
- `scripts/debug_phase16_pool_smoke.py` validates the pool
  end-to-end with SVI stubbed (production SVI run deferred to cluster).
- Local fast test suite (`tests/test_bilinear_simulator.py` +
  `tests/test_task_bilinear_benchmark.py::TestSeedPoolCorruptSkip`)
  must pass before the cluster re-submit.

## Next Step (human)

After local fast tests pass, re-submit cluster job:
`bash cluster/submit_phase16.sh`. Expect 10/10 success on the pool
(likely seeds 42, 43, 45, 46, 47, 48, 51, 52, 53, 54 based on the
42..51 corruption pattern) and full RECOV gate evaluation.
