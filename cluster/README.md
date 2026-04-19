# Cluster -- Phase 16 Acceptance Test

Run the v0.3.0 bilinear-recovery acceptance gate (`test_acceptance_gates_pass_at_10_seeds`, ~80-150 min CPU) on Monash M3. Uses an existing conda env (`actinf-py-scripts` by default) -- **do not create a new env**.

## Dependencies

`pyproject.toml` is the single source of truth. The slurm script runs `pip install -e .[benchmark,dev]` against the activated env on every invocation, so missing project packages (`torch`, `torchdiffeq`, `pyro-ppl`, `zuko`, `scipy`, `numpy`, `matplotlib`, `pytest`) are auto-synced. Idempotent when everything is already present.

**Prior failure note:** The first submission (job 54900993) picked up `ds_env`, which had no `torch`. The failure mode was the fail-fast import check, as intended. The script now defaults to `actinf-py-scripts` and auto-installs any missing deps.

If the cluster env is missing `torch` entirely, the first pip install will pull ~2GB. All subsequent runs are no-op verification (~5s).

## 1. Clone via deploy key

On the cluster, clone the repo using a deploy key you've uploaded to GitHub (Repo -> Settings -> Deploy keys, **write access = yes** so the push job can publish results).

**Form A -- explicit `GIT_SSH_COMMAND` (no config changes):**

```bash
GIT_SSH_COMMAND='ssh -i ~/.ssh/dcm_pytorch_deploy -o IdentitiesOnly=yes' \
    git clone git@github.com:adammanoogian/dcm_pytorch.git
cd dcm_pytorch
git checkout gsd/phase-16-bilinear-recovery-benchmark
```

For subsequent `git fetch` / `git push` from the cluster you'll need to re-pass `GIT_SSH_COMMAND` OR (strongly preferred for the push job) configure a persistent SSH host alias via Form B below -- otherwise `cluster/99_push_phase16_results.slurm` cannot authenticate on the compute node.

**Form B -- `~/.ssh/config` host alias (persistent, recommended for long use):**

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

With this form the push job's `git push origin ...` uses the deploy key automatically -- no environment variables required. This is the **required setup** for the push job to succeed without manually injecting `GIT_SSH_COMMAND` via `sbatch --export`.

## 2. Submit

```bash
cd dcm_pytorch
git checkout gsd/phase-16-bilinear-recovery-benchmark
bash cluster/submit_phase16.sh
```

This submits 2 jobs:

1. **Acceptance** (`phase16_acceptance`, ~4h wall limit, 16G, 4 CPUs, partition `comp`) -- runs `pytest tests/test_task_bilinear_benchmark.py -m slow -k acceptance`.
2. **Push** (`push_phase16_results`, 15min) -- dependent on the acceptance job via `afterany` (runs even if acceptance fails, so logs get captured). Commits `cluster/logs/phase16_*` and `cluster/results/phase16_*` to a timestamped branch `results/phase16-acceptance-YYYYMMDD-HHMMSS`.

### Overrides

| Variable       | Default              | Purpose                                                |
|----------------|----------------------|--------------------------------------------------------|
| `ENV_NAME`     | `actinf-py-scripts`  | Primary conda env name                                 |
| `ENV_FALLBACK` | *(unset)*            | Second try if `ENV_NAME` can't satisfy project deps    |
| `PROJECT`      | `fc37`               | Project code for `/scratch/${PROJECT}/${USER}/...`     |
| `PUSH_TO_MAIN` | `false`              | See warning below                                      |

Example overrides:

```bash
# Use a different env
ENV_NAME=my_env bash cluster/submit_phase16.sh

# Override scratch project code
PROJECT=ft29 bash cluster/submit_phase16.sh
```

The submit wrapper forwards `ENV_NAME`, `ENV_FALLBACK`, and `PROJECT` through `sbatch --export=ALL,...` automatically. You can also sbatch the slurm script directly:

```bash
sbatch --export=ALL,ENV_NAME=my_env cluster/run_phase16_acceptance.slurm
```

**Env fallback logic:** the script tries `ENV_NAME` first, and falls back to `ENV_FALLBACK` if either (a) conda activation fails, (b) `pip install -e .[benchmark,dev]` fails inside it, or (c) the import check fails after install. Prior versions only fell back on activation failure, letting a half-provisioned env pass activate and crash later — fixed.

> **WARNING: `PUSH_TO_MAIN=true` is NOT recommended.** The push job defaults to creating a results branch so you can review before merging. Setting `PUSH_TO_MAIN=true` pushes directly to `main` via `git pull --rebase`, which **fails silently on merge conflicts** and leaves results un-pushed on the compute node -- see HPC template gotcha #4. Only use it if you're certain nobody else has pushed to `main` while the job was running.

## 3. Monitor & retrieve results

Watch the queue:

```bash
squeue -u $USER
```

Tail the live logs (replace `<JOB_ID>` with the acceptance job ID printed by `submit_phase16.sh`):

```bash
tail -f cluster/logs/phase16_<JOB_ID>.out
tail -f cluster/logs/phase16_<JOB_ID>.err
tail -f cluster/logs/pytest_phase16_<JOB_ID>.log
```

Inspect a specific job:

```bash
scontrol show job <JOB_ID>
sacct -j <JOB_ID> --format=JobID,Elapsed,MaxRSS,State
```

After the push job completes, retrieve results locally:

```bash
git fetch origin
git log --oneline origin/results/phase16-acceptance-*
# inspect a specific branch:
git checkout origin/results/phase16-acceptance-<timestamp>
cat cluster/results/phase16_acceptance_*.md
```

Merge into main when satisfied:

```bash
git checkout main
git merge origin/results/phase16-acceptance-<timestamp>
git push origin main
```

If the acceptance test fails, the push job still runs (due to `afterany`) -- the pytest log will be on the results branch and you can diagnose before re-running.
