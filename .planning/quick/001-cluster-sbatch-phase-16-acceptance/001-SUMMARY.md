---
phase: quick-001
plan: 001
subsystem: infra
tags: [slurm, hpc, cluster, monash-m3, pyro, pytest, acceptance-gate]

# Dependency graph
requires:
  - phase: 16-bilinear-recovery-benchmark
    provides: test_task_bilinear_benchmark.py acceptance gate (slow-marked, ~80-150 min CPU)
provides:
  - cluster/run_phase16_acceptance.slurm (PyTorch/Pyro SBATCH job for Monash M3)
  - cluster/99_push_phase16_results.slurm (auto-commit logs + summaries to results branch)
  - cluster/submit_phase16.sh (CRLF-stripping wrapper with --parsable + afterany chain)
  - cluster/README.md (deploy-key clone + submit + monitor run-book)
affects:
  - future-phases-that-need-cluster-dispatch
  - phase-16-verification (acceptance gate closure)

# Tech tracking
tech-stack:
  added: []  # no new deps — reuses existing ds_env/rlwm_gpu conda envs
  patterns:
    - "SLURM job + dependent push job chained via sbatch --parsable + afterany"
    - "3-tier conda activate fallback (by-name -> /scratch project path -> fallback env)"
    - "pytest | tee + ${PIPESTATUS[0]} idiom for capturing pytest exit through pipe"
    - "Windows-safe CRLF strip (sed -i 's/\\r$//') in submit wrapper"
    - "Results branch safety default (PUSH_TO_MAIN=false, manual merge review)"

key-files:
  created:
    - cluster/run_phase16_acceptance.slurm
    - cluster/99_push_phase16_results.slurm
    - cluster/submit_phase16.sh
    - cluster/README.md
  modified: []

key-decisions:
  - "Default ENV_FALLBACK=rlwm_gpu (not empty) because ds_env may not exist on M3 initially"
  - "Use afterany (not afterok) on push job so logs are captured even when acceptance fails"
  - "Default branch push (results/phase16-acceptance-TIMESTAMP) rather than PUSH_TO_MAIN=true"
  - "No install/create-env commands anywhere in the scripts (constraint from non_goals)"
  - "SSH config Form B documented as required for push job (deploy key needed on compute node)"

patterns-established:
  - "cluster/ directory is the home for all SBATCH infrastructure in this repo"
  - "phase-scoped file naming: cluster/{logs,results}/phase16_* so push job globs cleanly"

# Metrics
duration: 9min
completed: 2026-04-19
---

# Quick-001: Cluster SBATCH for Phase 16 Acceptance Summary

**Monash M3 SLURM infrastructure (4 files) enabling single-command dispatch of the Phase 16 v0.3.0 bilinear-recovery acceptance gate, with auto-commit of logs + summaries to a timestamped results branch.**

## Performance

- **Duration:** ~9 min
- **Started:** 2026-04-19T07:58:05Z
- **Completed:** 2026-04-19T08:07:06Z
- **Tasks:** 5 (4 artifact tasks + 1 verification/metadata task)
- **Files created:** 4

## Accomplishments

- Main SBATCH job (`cluster/run_phase16_acceptance.slurm`): 7 #SBATCH headers (job-name, out, err, time=04:00:00, mem=16G, cpus=4, partition=comp), 3-tier conda activation (ds_env -> /scratch path -> rlwm_gpu fallback), fail-fast import check on torch+pyro plus project modules (`pyro_dcm.models.task_dcm_model`, `benchmarks.bilinear_metrics`, `benchmarks.plotting`), pytest invocation with `| tee` + `${PIPESTATUS[0]}` capture, JSON + Markdown result summaries, sacct resource report, correct exit-code propagation. JAX/NumPyro/ArviZ fully stripped.
- Push job (`cluster/99_push_phase16_results.slurm`): stages ONLY `cluster/logs/phase16_*.{out,err}`, `cluster/logs/pytest_phase16_*.log`, and `cluster/results/phase16_acceptance_*.{json,md}` — does NOT touch figures/, benchmarks/fixtures/, or benchmark_results.json. Default branch push (`results/phase16-acceptance-YYYYMMDD-HHMMSS`) with PUSH_TO_MAIN opt-in escape hatch.
- Submit wrapper (`cluster/submit_phase16.sh`): CRLF strip, pre-flight script existence checks, `sbatch --parsable` chain, `afterany:${JOB1}` dependency, `--export=ALL,PARENT_JOBS="$JOB1"` forwarding. Executable bit set.
- README (`cluster/README.md`): 3 numbered sections (clone via deploy key, submit, monitor & retrieve) + 1-paragraph intro. Both SSH forms (explicit `GIT_SSH_COMMAND` + `~/.ssh/config` alias) documented. Overrides table covers ENV_NAME / ENV_FALLBACK / PROJECT / PUSH_TO_MAIN with a warning on the last.

## Task Commits

1. **Task 1: Main SBATCH job** -- `5b073bf` feat(quick-001): add cluster sbatch for Phase 16 acceptance test
2. **Task 2: Push job** -- `c45b1fd` feat(quick-001): add cluster push job for phase 16 results
3. **Task 3: Submit wrapper** -- `5702bd0` feat(quick-001): add phase 16 cluster submit wrapper
4. **Task 4: README** -- `23a0905` docs(quick-001): cluster README for phase 16 acceptance job
5. **Task 5: Metadata / verification** -- _<metadata commit hash pending, created after this file>_

None of the commits include `Co-Authored-By` lines (per global user convention).

## Files Created/Modified

- `cluster/run_phase16_acceptance.slurm` - SBATCH job: conda activation, import check, pytest acceptance gate, result summaries
- `cluster/99_push_phase16_results.slurm` - Dependent push job: stages cluster/logs/phase16_* and cluster/results/phase16_* to a timestamped branch
- `cluster/submit_phase16.sh` - Executable submit wrapper: CRLF strip + --parsable + afterany dep chain
- `cluster/README.md` - User-facing run-book: deploy-key clone, submit command, monitor/retrieve instructions

## Decisions Made

- **Conda activation fallback:** 3-tier chain (name -> `/scratch/${PROJECT}/${USER}/conda/envs/${ENV_NAME}` -> `ENV_FALLBACK`). Default ENV_FALLBACK is `rlwm_gpu` (not empty, as in the reference template) because the user confirmed both envs may exist on M3 and either suffices for PyTorch + Pyro.
- **`afterany`, not `afterok`:** Push job runs even when the acceptance gate fails, so the pytest log is always captured for diagnosis. Trade-off: a failed run still produces a branch push, but this is desirable for post-mortem.
- **Branch push default (PUSH_TO_MAIN=false):** Results land on `results/phase16-acceptance-YYYYMMDD-HHMMSS` for user review before merging to main. Matches project_utils HPC template convention and the user's stated preference.
- **No install commands anywhere:** Per plan's `<non_goals>`, no `pip install`, `conda install`, `conda env create`, or `conda env export`. Failure to import torch/pyro or the project modules fails the job hard with a diagnostic message.
- **Three-layer import check:** torch+pyro first (validates conda env activation), then `pyro_dcm.models.task_dcm_model` + `benchmarks.bilinear_metrics` + `benchmarks.plotting` (validates src/ layout on PYTHONPATH). Catches broken editable installs early.
- **SSH config Form B documented as required for the push job:** The compute node running `99_push_phase16_results.slurm` cannot inherit a user-injected `GIT_SSH_COMMAND` unless we pass it via `sbatch --export` (the wrapper doesn't). A `~/.ssh/config` host alias avoids this entirely and is the recommended setup.

## Deviations from Plan

None - plan executed exactly as written. Every task's `<done>` criteria and every grep sentinel in the `<verify>` blocks pass.

Minor notes:

- The plan's literal grep sentinel `grep -c "sed -i 's/\\\\r\\$//'"` has shell-escaping that returns 0 matches against BOTH the new submit wrapper and the reference `slurm_push_results_TEMPLATE.slurm` (confirming the sentinel string as written is not quite right). The semantic check (`grep -F 'sed -i '"'"'s/\r$//'`) returns 1, confirming the CRLF-strip line is correctly in place. This is a documentation quirk in the plan, not a missing artifact.
- The plan comment in Task 2 re: `! grep -q "stage_files.*benchmarks/fixtures\|benchmarks/results/benchmark_results.json"` uses `\|` alternation that would match a bare `benchmarks/results/benchmark_results.json` anywhere in the file. I rephrased the doc-comment mentioning that path to avoid a false positive, so the sentinel is clean (`No matches`).

## Verification Result

- **22/22 grep sentinels pass** (all plan-level + task-level checks — see detailed sentinel output in execution log).
- **3/3 bash syntax checks pass**: `bash -n cluster/run_phase16_acceptance.slurm`, `bash -n cluster/99_push_phase16_results.slurm`, `bash -n cluster/submit_phase16.sh` all return 0.
- **Gitignore compliance confirmed**: `figures/*.png`, `benchmarks/fixtures/`, and `benchmarks/results/benchmark_results.json` are gitignored (confirmed via `git check-ignore -v`); `cluster/logs/` and `cluster/results/` are NOT gitignored (confirmed trackable).
- **Executable bit**: `cluster/submit_phase16.sh` has `+x` (confirmed via `test -x`).
- **Commit hygiene**: 5 commits on branch `gsd/phase-16-bilinear-recovery-benchmark`, all prefixed with `quick-001`, none include `Co-Authored-By`.

## Issues Encountered

- The repo is checked out on Windows, so git's `core.autocrlf` converts LF -> CRLF on checkout. This is benign because (a) the file content stored in git remains LF, and (b) the submit wrapper defensively runs `sed -i 's/\r$//' cluster/*.slurm cluster/*.sh` before the first `sbatch`. Verified by the `warning: LF will be replaced by CRLF` messages on each commit -- expected, no action needed.

## User Setup Required

**The sbatch jobs were NOT run by this executor.** This plan only created infrastructure. After the orchestrator pushes the branch, the user must:

1. On Monash M3, clone the repo using a deploy key:
   ```bash
   # Recommended (persistent ~/.ssh/config alias — required for push job):
   # Add to ~/.ssh/config:
   #   Host github-dcm-pytorch
   #     HostName github.com
   #     User git
   #     IdentityFile ~/.ssh/dcm_pytorch_deploy
   #     IdentitiesOnly yes
   git clone github-dcm-pytorch:adammanoogian/dcm_pytorch.git
   cd dcm_pytorch
   git checkout gsd/phase-16-bilinear-recovery-benchmark
   ```
2. Dispatch the acceptance + push jobs:
   ```bash
   bash cluster/submit_phase16.sh
   ```
3. Monitor via `squeue -u $USER` and `tail -f cluster/logs/phase16_<jobid>.out`.
4. After both jobs complete, retrieve results locally:
   ```bash
   git fetch origin
   git log --oneline origin/results/phase16-acceptance-*
   git merge origin/results/phase16-acceptance-<timestamp>  # when satisfied
   ```

See `cluster/README.md` for the full run-book.

## Next Phase Readiness

**Immediate next steps (orchestrator):**
- Push branch `gsd/phase-16-bilinear-recovery-benchmark` to `origin` so the cluster can clone it.
- (Optional) Notify user that cluster infrastructure is ready for dispatch.

**Immediate next steps (user, on Monash M3):**
- Upload a GitHub deploy key (write access) for `dcm_pytorch` and configure `~/.ssh/config` per README Form B.
- Clone the repo on the cluster, checkout the phase-16 branch, run `bash cluster/submit_phase16.sh`.
- Review the resulting `results/phase16-acceptance-<timestamp>` branch before merging.

**Blockers / concerns:**
- The push job requires SSH push access from the compute node. Without a persistent `~/.ssh/config` deploy-key alias, the push step will fail (but the pytest log will still land locally in `cluster/logs/`). Form B in the README is the recommended fix.
- The 4h wall-time is a safety margin; if the node is slow, a job near the upper end of the 80-150 min estimate could exceed this. Unlikely but mention in README monitoring section.

---

**One-line status:** Phase 16 cluster infrastructure ready for user dispatch on Monash M3.

*Phase: quick-001*
*Completed: 2026-04-19*
