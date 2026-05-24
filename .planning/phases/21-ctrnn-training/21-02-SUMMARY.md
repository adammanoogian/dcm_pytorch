---
phase: 21-ctrnn-training
plan: "02"
subsystem: rnn
tags: [pytorch, rnn, neurogym, cddm, bptt, cross-entropy, adam, training]

requires:
  - phase: 21-01
    provides: ContinuousTimeRNN(nn.Module) with n_input, n_output, forward() returning (z, h_traj)

provides:
  - train_rnn(): Adam BPTT training on neurogym tasks with grad clip and early stopping
  - eval_rnn_performance(): held-out accuracy evaluation in eval mode
  - 4-test integration suite covering smoke, dimension mismatch, eval, and convergence
  - neurogym inside-function import pattern (optional dependency guard)

affects:
  - 21-04 (cluster training): train_rnn is the entry-point for M3 sbatch array job
  - 21-05 (latent_extraction.py): train_rnn output feeds into extract_trajectories
  - 22-latent-dcm-fit: trained RNN weights and trajectories from train_rnn pipeline

tech-stack:
  added: []
  patterns:
    - "neurogym inside-function import: try/except ImportError with install hint"
    - "ngym.Dataset labels shape is (T, B) not (T*B,) -- reshape(-1) before CrossEntropyLoss"
    - "Early stopping: 3 consecutive log checkpoints >= criterion_acc triggers return"
    - "eval_rnn_performance: rnn.eval() + torch.no_grad() for noise-free inference"

key-files:
  created:
    - src/pyro_dcm/rnn/rnn_trainer.py
    - tests/test_rnn_trainer.py
  modified:
    - src/pyro_dcm/rnn/__init__.py

key-decisions:
  - "neurogym labels are (T, B) not (T*B,) — reshape(-1) required before CrossEntropyLoss and accuracy computation"
  - "neurogym imported inside train_rnn and eval_rnn_performance only — optional dependency pattern"
  - "Early stopping checks every log_every steps; 3 consecutive checks >= criterion_acc triggers exit"

patterns-established:
  - "neurogym optional import: try/except block with ImportError re-raise including install hint"
  - "ob_size and act_size always queried from dataset.env at runtime, never hardcoded"
  - "Labels from ngym.Dataset have shape (T, B); flatten with .reshape(-1) for loss/accuracy"

duration: 15min
completed: "2026-05-25"
---

# Phase 21 Plan 02: RNN Trainer Summary

**train_rnn() and eval_rnn_performance() for CT-RNN BPTT training on neurogym CDDM, with Adam optimizer, gradient clipping, accuracy-based early stopping, and 4-test integration suite.**

## Performance

- **Duration:** 15 min
- **Started:** 2026-05-24T22:37:27Z
- **Completed:** 2026-05-24T22:53:00Z
- **Tasks:** 2/2
- **Files modified:** 3

## Accomplishments

- `src/pyro_dcm/rnn/rnn_trainer.py` created with `train_rnn()` and `eval_rnn_performance()`
- neurogym imported inside functions only — users without neurogym can still import other rnn submodules
- Discovered and handled neurogym API: labels are `(T, B)` not `(T*B,)` — reshape before loss and accuracy
- Early stopping logic: 3 consecutive `log_every` checkpoints >= `criterion_acc` triggers early return
- `eval_rnn_performance()` runs in `rnn.eval()` + `torch.no_grad()` for noise-free inference
- `src/pyro_dcm/rnn/__init__.py` exports `train_rnn` and `eval_rnn_performance`
- 4-test integration suite: smoke (50 steps), dimension mismatch, eval, convergence (slow/latent marked)
- All 3 non-slow smoke tests pass in 6.1s; ruff clean

## Task Commits

Each task was committed atomically:

1. **Task 1: Create rnn_trainer.py with train_rnn and eval_rnn_performance** - `b5e1a9c` (feat)
2. **Task 2: Create RNN trainer integration test** - `3f0d1ea` (test)

**Plan metadata:** (docs commit to follow)

## Files Created/Modified

- `src/pyro_dcm/rnn/rnn_trainer.py` — `train_rnn()` and `eval_rnn_performance()` functions; 248 lines; NumPy docstrings
- `src/pyro_dcm/rnn/__init__.py` — Added `train_rnn` and `eval_rnn_performance` exports
- `tests/test_rnn_trainer.py` — 4 integration tests; 231 lines; all `@pytest.mark.latent`, slow test `@pytest.mark.slow`

## Decisions Made

- **neurogym labels are (T, B) not (T*B,).** The plan spec said labels were `(T*B,)` flattened but
  the actual neurogym 2.3.1 API returns `(T, B)`. Added `.reshape(-1)` before CrossEntropyLoss and
  accuracy computation. Both approaches compute the same loss; this was a corrected expectation.
- **neurogym imported inside functions only.** `import neurogym as ngym` appears inside `train_rnn()`
  and `eval_rnn_performance()` body with `try/except ImportError`. Users without neurogym can still
  do `from pyro_dcm.rnn import ContinuousTimeRNN` without error.
- **Early stopping: 3 consecutive log checks.** Checked every `log_every` steps; 3 consecutive
  checks at or above `criterion_acc` trigger early return. Resets count if accuracy drops below
  threshold at any check.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] neurogym labels shape is (T, B) not (T*B,)**

- **Found during:** Task 1 (rnn_trainer.py implementation)
- **Issue:** Plan spec states `labels: (seq_len * batch_size,)` flattened, but neurogym 2.3.1
  `Dataset()` returns labels with shape `(seq_len, batch_size)`.
- **Fix:** Added `.reshape(-1)` on the labels tensor before CrossEntropyLoss and accuracy
  computation. The z tensor is also reshaped from `(T, B, act_size)` to `(-1, act_size)`. Both
  reshapes are consistent and produce correct per-timestep cross-entropy loss.
- **Files modified:** `src/pyro_dcm/rnn/rnn_trainer.py`
- **Verification:** Verified with `python -c` that loss and accuracy compute without error using
  actual neurogym output shapes.
- **Committed in:** `b5e1a9c` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - Bug: API shape correction)
**Impact on plan:** Required for correctness; no scope creep. The plan's neurogym label shape
assumption was incorrect; the fix is minimal (two `.reshape(-1)` calls).

## Issues Encountered

- `pytest --timeout=120` flag raised `unrecognized arguments` error (pytest-timeout not installed).
  Removed the flag; tests ran without timeout enforcement. Smoke tests complete in 6.1s, well within
  any reasonable timeout.

## User Setup Required

None — neurogym was installed via `pip install pyro-dcm[latent]` (already done for verification).
No external service configuration required.

## Next Phase Readiness

- `train_rnn()` is the entry-point for M3 cluster training in Plan 21-04 (sbatch array job, 20 seeds)
- `eval_rnn_performance()` can gate trajectory extraction (Plan 21-05) on >= 85% accuracy
- `ContinuousTimeRNN` + `train_rnn` are fully importable and tested; Plans 21-03 and 21-04 are unblocked
- The convergence test (`@pytest.mark.slow`) validates full end-to-end training for >= 70% accuracy
  at 2000 steps; actual M3 training will run 3000-5000 steps with H=256 for >= 85% target

---
*Phase: 21-ctrnn-training*
*Completed: 2026-05-25*
