---
phase: quick-001
plan: 001
type: execute
wave: 1
depends_on: []
files_modified:
  - cluster/run_phase16_acceptance.slurm
  - cluster/99_push_phase16_results.slurm
  - cluster/submit_phase16.sh
  - cluster/README.md
autonomous: true

must_haves:
  truths:
    - "User can clone the repo on Monash M3 with an SSH deploy key via a documented command."
    - "User can submit the Phase 16 acceptance job on the cluster with a single `bash cluster/submit_phase16.sh`."
    - "The sbatch job activates an existing conda env (`ds_env` then `rlwm_gpu` fallback) without creating a new env."
    - "The sbatch job fails fast if torch or pyro are not importable in the activated env."
    - "The sbatch job runs `pytest tests/test_task_bilinear_benchmark.py -m slow -k acceptance` and captures the exit code."
    - "After the acceptance job finishes (success OR failure), a push job commits logs and result summaries to a timestamped results branch — never to main by default."
    - "No new environments, pyproject changes, benchmark edits, or figure/fixture commits are produced."
  artifacts:
    - path: "cluster/run_phase16_acceptance.slurm"
      provides: "Main SBATCH job: env activate fallback + import check + pytest acceptance invocation + result summary"
      contains: "#SBATCH --partition=comp, conda activate ds_env, ENV_FALLBACK=rlwm_gpu, pytest tests/test_task_bilinear_benchmark.py"
    - path: "cluster/99_push_phase16_results.slurm"
      provides: "Push job: commits cluster/logs + cluster/results to results/phase16-acceptance-<timestamp> branch"
      contains: "BRANCH_NAME=results/phase16-acceptance, stage cluster/logs/phase16_*, stage cluster/results/phase16_*"
    - path: "cluster/submit_phase16.sh"
      provides: "Submit wrapper: CRLF strip + --parsable chain with afterany dependency"
      contains: "sed -i 's/\\r$//', sbatch --parsable, --dependency=afterany, --export=ALL,PARENT_JOBS"
    - path: "cluster/README.md"
      provides: "User-facing usage doc: deploy-key clone, submit, monitor + retrieve"
      contains: "GIT_SSH_COMMAND, adammanoogian/dcm_pytorch, PUSH_TO_MAIN, ENV_NAME override"
  key_links:
    - from: "cluster/submit_phase16.sh"
      to: "cluster/run_phase16_acceptance.slurm"
      via: "sbatch --parsable (captures JOB1 id)"
      pattern: "JOB1=.*sbatch --parsable"
    - from: "cluster/submit_phase16.sh"
      to: "cluster/99_push_phase16_results.slurm"
      via: "--dependency=afterany:${JOB1} --export=ALL,PARENT_JOBS=$JOB1"
      pattern: "afterany:\\$\\{JOB1\\}"
    - from: "cluster/run_phase16_acceptance.slurm"
      to: "tests/test_task_bilinear_benchmark.py"
      via: "pytest -m slow -k acceptance with tee to cluster/logs/pytest_phase16_${SLURM_JOB_ID}.log"
      pattern: "pytest tests/test_task_bilinear_benchmark.py"
    - from: "cluster/99_push_phase16_results.slurm"
      to: "cluster/logs/ + cluster/results/"
      via: "git add + git push to results/phase16-acceptance-<timestamp>"
      pattern: "stage_files .*cluster/(logs|results)/phase16_"
---

<objective>
Create a Monash-M3 SLURM setup that lets the user run the Phase 16 v0.3.0
acceptance-gate test (`test_acceptance_gates_pass_at_10_seeds`, ~80–150 min
CPU) on the cluster using a pre-existing conda env, and auto-commits logs +
summaries to a results branch the user can review before merging.

Purpose: Plan 16-02 closed all code gaps; the verifier flagged `human_needed`
because the slow `@pytest.mark.slow` acceptance gate cannot be run on the
user's Windows laptop in reasonable time. This plan produces ONLY the cluster
infrastructure (scripts + docs). It does not run the job, does not create a
new env, and does not modify any benchmark / test code.

Output:
- `cluster/run_phase16_acceptance.slurm` (main job, PyTorch/Pyro — NOT JAX)
- `cluster/99_push_phase16_results.slurm` (push job, branch-default)
- `cluster/submit_phase16.sh` (submit wrapper with CRLF strip + dep chain)
- `cluster/README.md` (deploy-key clone + submit + monitor instructions)
</objective>

<execution_context>
@C:\Users\aman0087\.claude/get-shit-done/workflows/execute-plan.md
@C:\Users\aman0087\.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md

# HPC templates (source of truth for conventions — do NOT edit these)
@../project_utils/templates/hpc/README.md
@../project_utils/templates/hpc/slurm_bayesian_cpu_TEMPLATE.slurm
@../project_utils/templates/hpc/slurm_push_results_TEMPLATE.slurm

# Test being run on the cluster (read-only — do NOT modify)
@tests/test_task_bilinear_benchmark.py

# Project gitignore — dictates what the push job may and may not stage
@.gitignore
</context>

<tasks>

<task type="auto">
  <name>Task 1: Create main SBATCH job (cluster/run_phase16_acceptance.slurm)</name>
  <files>cluster/run_phase16_acceptance.slurm</files>
  <action>
Create the main SBATCH job by adapting `../project_utils/templates/hpc/slurm_bayesian_cpu_TEMPLATE.slurm`. Strip ALL JAX/NumPyro/ArviZ content; this project uses PyTorch + Pyro.

Required structure (in this order):

1. Shebang: `#!/bin/bash`

2. Header comment block (≈25 lines): Explain this runs the Phase 16 v0.3.0 acceptance gate (`test_acceptance_gates_pass_at_10_seeds`), ~80–150 min expected, 4h wall-time safety margin. Document usage:
   - Default: `sbatch cluster/run_phase16_acceptance.slurm`
   - Override env: `sbatch --export=ENV_NAME=rlwm_gpu cluster/run_phase16_acceptance.slurm`
   - Override project: `sbatch --export=PROJECT=ft29 cluster/run_phase16_acceptance.slurm`

3. `#SBATCH` headers (EXACTLY these 7 — grep verifier requires >= 6):
   ```
   #SBATCH --job-name=phase16_acceptance
   #SBATCH --output=cluster/logs/phase16_%j.out
   #SBATCH --error=cluster/logs/phase16_%j.err
   #SBATCH --time=04:00:00
   #SBATCH --mem=16G
   #SBATCH --cpus-per-task=4
   #SBATCH --partition=comp
   ```

4. Environment setup:
   - `module load miniforge3`
   - `cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"`
   - `PROJECT_ROOT="$(pwd)"`
   - `mkdir -p cluster/logs cluster/results` (BEFORE any logging — in case logs want to tee here)

5. Stage banner (echo block): Job ID, node, project root, branch (`git rev-parse --abbrev-ref HEAD`), commit (`git rev-parse --short HEAD`), start time. This banner is critical for result-summary reproducibility.

6. Config knobs (before conda activate):
   ```
   _PROJECT="${PROJECT:-fc37}"
   ENV_NAME="${ENV_NAME:-ds_env}"
   ENV_FALLBACK="${ENV_FALLBACK:-rlwm_gpu}"
   ```

7. Conda activation with 3-tier fallback (exactly matching bayesian_cpu template pattern — the grep verifier requires >= 2 `conda activate` calls):
   ```
   if conda activate "$ENV_NAME" 2>/dev/null; then
       echo "Activated $ENV_NAME by name"
   elif conda activate "/scratch/${_PROJECT}/${USER}/conda/envs/${ENV_NAME}" 2>/dev/null; then
       echo "Activated $ENV_NAME from /scratch/${_PROJECT}/${USER}/"
   elif [[ -n "$ENV_FALLBACK" ]] && conda activate "$ENV_FALLBACK" 2>/dev/null; then
       echo "Activated $ENV_FALLBACK (fallback)"
   else
       echo "ERROR: Failed to activate conda environment"
       echo "Tried: $ENV_NAME (by name), /scratch/${_PROJECT}/${USER}/conda/envs/${ENV_NAME}, $ENV_FALLBACK"
       echo "Debug: run 'conda env list' on a login node."
       exit 1
   fi
   ```

8. Import check (PyTorch + Pyro — NOT JAX; grep verifier requires `torch\|pyro` >= 2):
   ```
   python -c "
   import torch, pyro
   print(f'torch {torch.__version__}, pyro {pyro.__version__}')
   print(f'CUDA available: {torch.cuda.is_available()}')
   " || {
       echo "ERROR: torch/pyro import check failed in activated env"
       exit 1
   }
   ```

   Also import the project modules the test actually uses, to fail fast on broken src/ layout:
   ```
   python -c "
   from pyro_dcm.models.task_dcm_model import task_dcm_model
   from benchmarks.bilinear_metrics import compute_acceptance_gates
   from benchmarks.plotting import plot_acceptance_gates_table
   print('project imports OK')
   " || {
       echo "ERROR: project import check failed"
       exit 1
   }
   ```

9. NO JAX setup. NO `JAX_PLATFORMS`, NO `JAX_COMPILATION_CACHE_DIR`. Delete entirely.

10. Pytest invocation (grep verifier requires `pytest tests/test_task_bilinear_benchmark.py` >= 1):
    ```
    echo "[$(date)] Running Phase 16 acceptance gate (10 seeds × 500 SVI steps)"
    START_TIME=$(date +%s)

    pytest tests/test_task_bilinear_benchmark.py \
        -m slow -k acceptance \
        -s -v --tb=short \
        2>&1 | tee "cluster/logs/pytest_phase16_${SLURM_JOB_ID}.log"

    EXIT_CODE=${PIPESTATUS[0]}
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    ```

    Use `${PIPESTATUS[0]}` (NOT `$?`) because `$?` captures `tee`'s exit, not pytest's. This is the correct capture-through-pipe idiom.

11. Write result summary JSON (`cluster/results/phase16_acceptance_${SLURM_JOB_ID}.json`) using a here-doc, containing at minimum:
    ```
    {
      "slurm_job_id": "${SLURM_JOB_ID}",
      "node": "${SLURMD_NODENAME:-unknown}",
      "branch": "<from git rev-parse>",
      "commit_sha": "<from git rev-parse --short HEAD>",
      "env_name": "<the env that activated>",
      "exit_code": <EXIT_CODE>,
      "elapsed_seconds": <ELAPSED>,
      "start_time": "<ISO>",
      "end_time": "<ISO>",
      "pytest_log": "cluster/logs/pytest_phase16_${SLURM_JOB_ID}.log"
    }
    ```

    Track which env actually activated (set an `ACTIVATED_ENV` var in each branch of the fallback chain).

12. Write a short human-readable markdown summary `cluster/results/phase16_acceptance_${SLURM_JOB_ID}.md` with PASS/FAIL, elapsed wall time, node, commit SHA, path to pytest log.

13. sacct resource report (copy verbatim from bayesian_cpu template, lines ≈187-196).

14. Final `exit $EXIT_CODE` so the sbatch job reports the pytest exit code to SLURM.

CRITICAL constraints:
- Line endings MUST be LF (the executor should write with an editor/tool that emits LF; the submit wrapper will strip CRLF defensively anyway).
- Do NOT use `set -e` at the top — we WANT the script to continue past a pytest failure so the result summary and sacct report still get written. Use explicit exit-code captures instead.
- Do NOT run `pip install`, `conda install`, `conda env create`, or `conda env export` anywhere.
  </action>
  <verify>
bash -n cluster/run_phase16_acceptance.slurm  # syntax parse
grep -c "^#SBATCH" cluster/run_phase16_acceptance.slurm    # must be >= 6
grep -c "conda activate" cluster/run_phase16_acceptance.slurm  # must be >= 2
grep -c "ENV_NAME=.*ds_env" cluster/run_phase16_acceptance.slurm  # must be >= 1
grep -c "ENV_FALLBACK=.*rlwm_gpu" cluster/run_phase16_acceptance.slurm  # must be >= 1
grep -c "pytest tests/test_task_bilinear_benchmark.py" cluster/run_phase16_acceptance.slurm  # >= 1
grep -Ec "torch|pyro" cluster/run_phase16_acceptance.slurm  # >= 2
! grep -q "JAX\|jax\|numpyro" cluster/run_phase16_acceptance.slurm  # JAX must be fully stripped
! grep -q "conda env create\|pip install\|conda install" cluster/run_phase16_acceptance.slurm  # no install commands
  </verify>
  <done>
File exists, parses with `bash -n`, all 9 grep sentinels pass, JAX fully stripped, no install commands.
  </done>
</task>

<task type="auto">
  <name>Task 2: Create push job (cluster/99_push_phase16_results.slurm)</name>
  <files>cluster/99_push_phase16_results.slurm</files>
  <action>
Adapt `../project_utils/templates/hpc/slurm_push_results_TEMPLATE.slurm`. Strip ALL MLE/model_comparison patterns; keep the branch-push safety architecture intact.

Required changes from the template:

1. `#SBATCH` headers: keep these identical to template (job-name=push_phase16_results, output/error point to `cluster/logs/push_phase16_%j.*`, time=00:15:00, mem=4G, cpus=1, partition=comp).

2. Header docstring: Retain the template's Windows CRLF warning (gotcha #1) and the `--parsable` explanation. Update examples to reference `cluster/submit_phase16.sh` and `cluster/99_push_phase16_results.slurm`.

3. Defaults:
   ```
   PUSH_TO_MAIN="${PUSH_TO_MAIN:-false}"
   BRANCH_NAME="${BRANCH_NAME:-results/phase16-acceptance-$(date +%Y%m%d-%H%M%S)}"
   PARENT_JOBS="${PARENT_JOBS:-${SLURM_JOB_DEPENDENCY:-unknown}}"
   GIT_REMOTE="${GIT_REMOTE:-origin}"
   ```

4. Retain verbatim: git setup verification block (checks `git`, `rev-parse`, remote exists, remote reachable). These are non-negotiable safety checks.

5. REPLACE the `stage_files` calls (template lines ≈161-171, MLE + model_comparison + figures) with EXACTLY these patterns:
   ```
   stage_files "cluster/logs/phase16_*.out"    "Phase 16 SLURM stdout logs"
   stage_files "cluster/logs/phase16_*.err"    "Phase 16 SLURM stderr logs"
   stage_files "cluster/logs/pytest_phase16_*.log"  "Phase 16 pytest tee logs"
   stage_files "cluster/results/phase16_acceptance_*.json"  "Phase 16 acceptance result summaries"
   stage_files "cluster/results/phase16_acceptance_*.md"    "Phase 16 acceptance markdown reports"
   ```

   Do NOT stage `figures/*` (blocked by `.gitignore`: `figures/*.png`, `figures/*.pdf`).
   Do NOT stage `benchmarks/fixtures/` or `benchmarks/results/benchmark_results.json` (both in .gitignore).
   Do NOT stage `output/` paths — this project has no `output/` directory.

6. Retain the commit message and dual-strategy (PUSH_TO_MAIN=true vs branch push) blocks verbatim from the template, updating only the commit message subject:
   - Branch: `results(phase16): acceptance gate outputs (SLURM jobs: ${PARENT_JOBS})`
   - Main: `results(phase16): acceptance gate outputs (SLURM jobs: ${PARENT_JOBS})`

   Keep the multi-line body including `Node:`, `Date:`, and `Merge with:` instructions.

7. Retain the final summary echo block verbatim.

CRITICAL constraints:
- Do NOT stage anything in `figures/`, `benchmarks/fixtures/`, or `benchmarks/results/benchmark_results.json`.
- Do NOT default `PUSH_TO_MAIN` to true — the user wants to review results before merging.
- Keep `GIT_REMOTE=origin` default; user can override if they set up a `deploy` remote on the cluster.
  </action>
  <verify>
bash -n cluster/99_push_phase16_results.slurm  # syntax parse
grep -c "^#SBATCH" cluster/99_push_phase16_results.slurm    # >= 5
grep -c "BRANCH_NAME=.*results/phase16-acceptance" cluster/99_push_phase16_results.slurm  # >= 1
grep -c "stage_files.*cluster/logs/phase16" cluster/99_push_phase16_results.slurm  # >= 2 (out, err, pytest log)
grep -c "stage_files.*cluster/results/phase16" cluster/99_push_phase16_results.slurm  # >= 2 (json, md)
! grep -q "stage_files.*figures/" cluster/99_push_phase16_results.slurm  # must NOT stage figures
! grep -q "stage_files.*output/mle" cluster/99_push_phase16_results.slurm  # template MLE pattern removed
! grep -q "stage_files.*benchmarks/fixtures\|benchmarks/results/benchmark_results.json" cluster/99_push_phase16_results.slurm
  </verify>
  <done>
File exists, parses, stages only cluster/logs/phase16_* and cluster/results/phase16_*, does NOT stage gitignored paths, defaults to branch push (PUSH_TO_MAIN=false).
  </done>
</task>

<task type="auto">
  <name>Task 3: Create submit wrapper (cluster/submit_phase16.sh)</name>
  <files>cluster/submit_phase16.sh</files>
  <action>
Create a Bash wrapper implementing the README.md "Dependency chaining" snippet, adapted for Phase 16.

Required content (full file):

```bash
#!/bin/bash
# =============================================================================
# Phase 16 Acceptance — Submit Wrapper
# =============================================================================
# Submits the acceptance sbatch job + a dependent push job. Strips CRLF first
# (gotcha #1 for Windows-cloned repos — and THIS repo is cloned from a Windows
# checkout).
#
# Usage:
#   bash cluster/submit_phase16.sh
#
# Overrides (pass through sbatch --export):
#   ENV_NAME=rlwm_gpu bash cluster/submit_phase16.sh
#   PROJECT=ft29 bash cluster/submit_phase16.sh
#   PUSH_TO_MAIN=true bash cluster/submit_phase16.sh   # NOT recommended
# =============================================================================

set -u  # undefined vars are errors; no -e because we want to continue past
        # non-fatal warnings (e.g. sed returning non-zero on a clean file).

# --- 1. Strip CRLF (HPC README gotcha #1) ---------------------------------
# Idempotent; safe on already-clean files.
sed -i 's/\r$//' cluster/*.slurm cluster/*.sh 2>/dev/null || true

# --- 2. Pre-flight: ensure the scripts exist ------------------------------
for f in cluster/run_phase16_acceptance.slurm cluster/99_push_phase16_results.slurm; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: missing required script: $f" >&2
        exit 1
    fi
done

# --- 3. Submit main acceptance job ---------------------------------------
echo "Submitting Phase 16 acceptance job..."
JOB1=$(sbatch --parsable cluster/run_phase16_acceptance.slurm)
if [[ -z "$JOB1" ]]; then
    echo "ERROR: sbatch did not return a job ID for the acceptance job" >&2
    exit 1
fi
echo "  Acceptance job ID: $JOB1"

# --- 4. Submit push job depending on acceptance ---------------------------
echo "Submitting Phase 16 push job (depends on afterany:${JOB1})..."
PUSH_JOB=$(sbatch --dependency=afterany:${JOB1} \
    --export=ALL,PARENT_JOBS="$JOB1" \
    --parsable cluster/99_push_phase16_results.slurm)
if [[ -z "$PUSH_JOB" ]]; then
    echo "ERROR: sbatch did not return a job ID for the push job" >&2
    exit 1
fi
echo "  Push job ID: $PUSH_JOB"

# --- 5. Report ------------------------------------------------------------
echo ""
echo "============================================================"
echo "Phase 16 acceptance dispatched"
echo "============================================================"
echo "Acceptance: $JOB1"
echo "Push:       $PUSH_JOB (runs afterany regardless of acceptance pass/fail)"
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  tail -f cluster/logs/phase16_${JOB1}.out"
echo "  tail -f cluster/logs/phase16_${JOB1}.err"
echo ""
echo "Retrieve results after both jobs complete:"
echo "  git fetch origin"
echo "  git log origin/results/phase16-acceptance-* --oneline"
echo "============================================================"
```

After writing the file, make it executable:
```
chmod +x cluster/submit_phase16.sh
```

CRITICAL:
- The `sed -i 's/\r$//'` line must use EXACTLY that syntax (grep verifier matches on escaped `\r$`).
- No `cd` to an absolute path — this must run from the project root (where the user invokes it).
- `afterany` (not `afterok`) so the push job runs even if pytest fails — we want the logs pushed regardless.
  </action>
  <verify>
bash -n cluster/submit_phase16.sh  # syntax parse (required by the plan's verification section)
test -x cluster/submit_phase16.sh  # executable bit set
grep -c "sed -i 's/\\\\r\\$//'" cluster/submit_phase16.sh  # >= 1 — CRLF strip
grep -c "sbatch --parsable" cluster/submit_phase16.sh  # >= 1
grep -c "afterany" cluster/submit_phase16.sh  # >= 1
grep -c "export=ALL,PARENT_JOBS" cluster/submit_phase16.sh  # >= 1
  </verify>
  <done>
File exists, parses with `bash -n` (verification requirement), has +x bit, all 4 grep sentinels pass.
  </done>
</task>

<task type="auto">
  <name>Task 4: Create user-facing README (cluster/README.md)</name>
  <files>cluster/README.md</files>
  <action>
Create EXACTLY 3 top-level sections (plus a 1-paragraph intro). Do not expand to a methodology doc — this is a run-book.

Required structure:

```markdown
# Cluster — Phase 16 Acceptance Test

Run the v0.3.0 bilinear-recovery acceptance gate (`test_acceptance_gates_pass_at_10_seeds`, ~80–150 min CPU) on Monash M3. Uses an existing conda env (`ds_env` by default, `rlwm_gpu` as fallback) — **do not create a new env**.

## 1. Clone with an SSH deploy key

On the cluster, clone the repo using a deploy key you've uploaded to GitHub (Repo → Settings → Deploy keys, **write access = yes** so the push job can publish results).

**Form A — explicit `GIT_SSH_COMMAND` (no config changes):**

```bash
GIT_SSH_COMMAND='ssh -i ~/.ssh/dcm_pytorch_deploy -o IdentitiesOnly=yes' \
    git clone git@github.com:adammanoogian/dcm_pytorch.git
cd dcm_pytorch
git checkout gsd/phase-16-bilinear-recovery-benchmark
```

For subsequent `git fetch` / `git push` from the cluster you'll need to re-pass `GIT_SSH_COMMAND` OR configure a remote URL alias (next form).

**Form B — `~/.ssh/config` host alias (persistent, recommended for long use):**

```
# ~/.ssh/config
Host github-dcm-pytorch
    HostName github.com
    User git
    IdentityFile ~/.ssh/dcm_pytorch_deploy
    IdentitiesOnly yes
```

Then clone:

```bash
git clone github-dcm-pytorch:adammanoogian/dcm_pytorch.git
cd dcm_pytorch
git checkout gsd/phase-16-bilinear-recovery-benchmark
```

With this form the push job's `git push origin ...` uses the deploy key automatically — no environment variables required.

## 2. Submit

```bash
cd dcm_pytorch
git checkout gsd/phase-16-bilinear-recovery-benchmark
bash cluster/submit_phase16.sh
```

This submits 2 jobs:

1. **Acceptance** (`phase16_acceptance`, ~4h wall limit, 16G, 4 CPUs, partition `comp`) — runs `pytest tests/test_task_bilinear_benchmark.py -m slow -k acceptance`.
2. **Push** (`push_phase16_results`, 15min) — dependent on the acceptance job via `afterany` (runs even if acceptance fails, so logs get captured). Commits `cluster/logs/phase16_*` and `cluster/results/phase16_*` to a timestamped branch `results/phase16-acceptance-YYYYMMDD-HHMMSS`.

### Overrides

| Variable | Default | Purpose |
|---|---|---|
| `ENV_NAME` | `ds_env` | Primary conda env name |
| `ENV_FALLBACK` | `rlwm_gpu` | Second try if `ENV_NAME` is missing |
| `PROJECT` | `fc37` | Project code for `/scratch/${PROJECT}/${USER}/...` |
| `PUSH_TO_MAIN` | `false` | **NOT recommended.** Push job creates a results branch by default so you can review before merging. Setting `PUSH_TO_MAIN=true` pushes directly to `main` via `git pull --rebase` — fails silently on merge conflicts. |

Example overrides:

```bash
# Use rlwm_gpu as primary env instead of ds_env
sbatch --export=ENV_NAME=rlwm_gpu cluster/run_phase16_acceptance.slurm

# Or pass through the submit wrapper (env vars are NOT auto-forwarded — use sbatch directly if you want per-env overrides)
ENV_NAME=rlwm_gpu sbatch --export=ALL,ENV_NAME=rlwm_gpu --parsable cluster/run_phase16_acceptance.slurm
```

## 3. Monitor + retrieve

Watch the queue:

```bash
squeue -u $USER
```

Tail the live logs (replace `<JOB_ID>` with the acceptance job ID from `submit_phase16.sh`):

```bash
tail -f cluster/logs/phase16_<JOB_ID>.out
tail -f cluster/logs/phase16_<JOB_ID>.err
tail -f cluster/logs/pytest_phase16_<JOB_ID>.log
```

After the push job completes, retrieve results locally:

```bash
git fetch origin
git log origin/results/phase16-acceptance-* --oneline
# then on your laptop:
git checkout -b review/phase16-acceptance origin/results/phase16-acceptance-<timestamp>
cat cluster/results/phase16_acceptance_*.md
```

Merge into main when satisfied:

```bash
git checkout main
git merge origin/results/phase16-acceptance-<timestamp>
git push origin main
```

If the acceptance test fails, the push job still runs (due to `afterany`) — you'll get the pytest log on the results branch and can diagnose before re-running.
```

CRITICAL:
- The README must contain EXACTLY the strings the grep verifier checks: `GIT_SSH_COMMAND`, `adammanoogian/dcm_pytorch`, `PUSH_TO_MAIN`.
- Do NOT add a "Setup" / "Install" / "Create env" section — user is explicit that the env already exists.
- Do NOT add troubleshooting-everything. Keep the doc lean (the 3 numbered sections + 1 intro).
  </action>
  <verify>
test -f cluster/README.md
grep -c "GIT_SSH_COMMAND" cluster/README.md    # >= 1
grep -c "adammanoogian/dcm_pytorch" cluster/README.md  # >= 1
grep -c "PUSH_TO_MAIN" cluster/README.md    # >= 1
grep -c "ENV_NAME" cluster/README.md    # >= 1
grep -c "squeue -u \$USER" cluster/README.md    # >= 1
grep -c "gsd/phase-16-bilinear-recovery-benchmark" cluster/README.md  # >= 1 — correct branch name
  </verify>
  <done>
File exists, contains exactly 3 numbered sections + intro, all 6 grep sentinels pass.
  </done>
</task>

<task type="auto">
  <name>Task 5: Verify end-to-end + metadata commit</name>
  <files>cluster/run_phase16_acceptance.slurm, cluster/99_push_phase16_results.slurm, cluster/submit_phase16.sh, cluster/README.md</files>
  <action>
Run the full verification gauntlet from the plan's `<verification>` block, then produce the final metadata commit.

1. Run ALL plan-level grep sentinels and confirm each returns >= the required count:
   ```
   grep -c "^#SBATCH" cluster/run_phase16_acceptance.slurm         # >= 6
   grep -c "conda activate" cluster/run_phase16_acceptance.slurm   # >= 2
   grep -c "ENV_NAME=.*ds_env" cluster/run_phase16_acceptance.slurm  # >= 1
   grep -c "ENV_FALLBACK=.*rlwm_gpu" cluster/run_phase16_acceptance.slurm  # >= 1
   grep -c "pytest tests/test_task_bilinear_benchmark.py" cluster/run_phase16_acceptance.slurm  # >= 1
   grep -Ec "torch|pyro" cluster/run_phase16_acceptance.slurm      # >= 2
   grep -c "sed -i 's/\\\\r\\$//'" cluster/submit_phase16.sh       # >= 1
   grep -c "sbatch --parsable" cluster/submit_phase16.sh           # >= 1
   grep -c "afterany" cluster/submit_phase16.sh                    # >= 1
   grep -c "GIT_SSH_COMMAND" cluster/README.md                     # >= 1
   grep -c "adammanoogian/dcm_pytorch" cluster/README.md           # >= 1
   grep -c "PUSH_TO_MAIN" cluster/README.md                        # >= 1
   ```

   If ANY sentinel fails, fix the corresponding file before committing.

2. Run syntax parse checks:
   ```
   bash -n cluster/submit_phase16.sh
   bash -n cluster/run_phase16_acceptance.slurm
   bash -n cluster/99_push_phase16_results.slurm
   ```

3. Confirm gitignore compliance — NONE of these paths should be visible to git (they are gitignored and MUST NOT be staged):
   ```
   git check-ignore -v figures/phase16_dummy.png    # should match .gitignore
   git check-ignore -v benchmarks/fixtures/x       # should match .gitignore
   git check-ignore -v benchmarks/results/benchmark_results.json  # should match
   ```

   Conversely, `cluster/logs/` and `cluster/results/` must NOT be gitignored:
   ```
   git check-ignore -v cluster/logs/foo.out && echo "BAD: cluster/logs is ignored" || echo "OK"
   git check-ignore -v cluster/results/foo.json && echo "BAD: cluster/results is ignored" || echo "OK"
   ```

4. Do NOT create `.gitkeep` placeholders in `cluster/logs/` or `cluster/results/`. Those directories will be created at runtime by the SBATCH job (`mkdir -p`) and populated with real artifacts the push job will commit. Empty directories don't need to be tracked.

5. Stage and commit each deliverable as a separate commit (per the `<deliverables>` spec), so the git log cleanly maps to the 5 tasks. The executor should produce these 5 commits in order:
   - `feat(quick-001): add cluster sbatch for Phase 16 acceptance test` — stages `cluster/run_phase16_acceptance.slurm`
   - `feat(quick-001): add cluster push job for phase 16 results` — stages `cluster/99_push_phase16_results.slurm`
   - `feat(quick-001): add phase 16 cluster submit wrapper` — stages `cluster/submit_phase16.sh`
   - `docs(quick-001): cluster README for phase 16 acceptance job` — stages `cluster/README.md`
   - `docs(quick-001): complete cluster sbatch setup plan` — stages any updated planning docs (SUMMARY, ROADMAP if touched)

   Do NOT include `Co-Authored-By` lines (per global user convention).
   Do NOT run `git push` — pushing the branch is an orchestrator step, explicitly excluded in `<non_goals>`.

6. Produce the plan SUMMARY at `.planning/quick/001-cluster-sbatch-phase-16-acceptance/001-SUMMARY.md` summarising all 4 artifacts + the 5 commits, and listing the two commands the user needs (deploy-key clone + submit).
  </action>
  <verify>
# All plan-level grep sentinels pass (see list above)
# bash -n on all 3 scripts returns 0
# git log --oneline -n 6 shows the 5 commits in order with correct prefixes
git log --oneline -n 6 | grep -c "quick-001"  # >= 5
# No Co-Authored-By lines
! git log -n 5 --format=%B | grep -q "Co-Authored-By"
# Summary file exists
test -f .planning/quick/001-cluster-sbatch-phase-16-acceptance/001-SUMMARY.md
  </verify>
  <done>
All 12 grep sentinels pass, 3 bash syntax checks pass, gitignore compliance confirmed, 5 commits exist with correct prefixes and no Co-Authored-By, SUMMARY.md exists.
  </done>
</task>

</tasks>

<verification>
Plan-level acceptance (copy of `<verification>` from the planning brief):

```bash
grep -c "^#SBATCH" cluster/run_phase16_acceptance.slurm         # >= 6
grep -c "conda activate" cluster/run_phase16_acceptance.slurm   # >= 2
grep -c "ENV_NAME=.*ds_env" cluster/run_phase16_acceptance.slurm  # >= 1
grep -c "ENV_FALLBACK=.*rlwm_gpu" cluster/run_phase16_acceptance.slurm  # >= 1
grep -c "pytest tests/test_task_bilinear_benchmark.py" cluster/run_phase16_acceptance.slurm  # >= 1
grep -Ec "torch|pyro" cluster/run_phase16_acceptance.slurm      # >= 2
grep -c "sed -i 's/\\\\r\\$//'" cluster/submit_phase16.sh       # >= 1
grep -c "sbatch --parsable" cluster/submit_phase16.sh           # >= 1
grep -c "afterany" cluster/submit_phase16.sh                    # >= 1
grep -c "GIT_SSH_COMMAND" cluster/README.md                     # >= 1
grep -c "adammanoogian/dcm_pytorch" cluster/README.md           # >= 1
grep -c "PUSH_TO_MAIN" cluster/README.md                        # >= 1

bash -n cluster/submit_phase16.sh  # syntax check — required, no execution
```

No pytest run required — this plan only creates infrastructure.
</verification>

<success_criteria>
- [ ] `cluster/run_phase16_acceptance.slurm` exists, 7 #SBATCH headers, JAX fully stripped, conda fallback chain intact (ds_env → /scratch/... → rlwm_gpu), torch + pyro import check, pytest invocation captures `${PIPESTATUS[0]}`, writes JSON + MD result summary.
- [ ] `cluster/99_push_phase16_results.slurm` exists, adapted from `slurm_push_results_TEMPLATE.slurm`, stages ONLY `cluster/logs/phase16_*` and `cluster/results/phase16_*`, defaults to `results/phase16-acceptance-<timestamp>` branch.
- [ ] `cluster/submit_phase16.sh` exists, executable, CRLF strip + `--parsable` chain + `afterany` dependency.
- [ ] `cluster/README.md` exists, exactly 3 numbered sections, both SSH deploy-key forms documented, references `gsd/phase-16-bilinear-recovery-benchmark`.
- [ ] All 12 grep sentinels pass.
- [ ] `bash -n` passes on all 3 shell scripts.
- [ ] 5 commits in git log, each with a `quick-001` scope, none with `Co-Authored-By`.
- [ ] No new env created, no pyproject.toml changes, no benchmark/test edits, no figure/fixture commits.
- [ ] `.planning/quick/001-cluster-sbatch-phase-16-acceptance/001-SUMMARY.md` produced.
</success_criteria>

<output>
After completion, create `.planning/quick/001-cluster-sbatch-phase-16-acceptance/001-SUMMARY.md` summarising:
1. The 4 artifacts created (with absolute paths).
2. The 5 commits produced (hashes + subjects).
3. The two commands the user needs to run on the cluster (post-clone):
   - SSH deploy-key clone (Form A + Form B)
   - `bash cluster/submit_phase16.sh`
4. A one-line status: `Phase 16 cluster infrastructure ready for user dispatch on Monash M3.`

Do NOT push the branch — orchestrator handles that.
</output>
