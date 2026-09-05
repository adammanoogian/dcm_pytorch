# Cluster -- DCCN (Donders / Radboud)

Compute for Pyro-DCM runs on the **DCCN cluster** (`mentat001-007.dccn.nl`),
Slurm-scheduled. Migrated from Monash M3 on **2026-09-05**; anything in this
repo describing `--partition=comp`, conda envs, or `/home/aman0087/...` predates
that move and is provenance only.

## Routing rule

Anything projected to take **more than ~3 minutes of saturating laptop CPU**
goes to the cluster. That includes `pytest -m slow`, full-suite runs, SVI/NUTS
fits, VL sweeps, identifiability grids, recovery harnesses, and model-comparison
sweeps. Single fast unit tests (<30 s) stay local. The rule binds subagents too.

The workstation is for orchestration: editing, linting, quick tests, reading
logs, and -- newly -- the **SPM12/MATLAB bridge** (see below).

## What changed from M3

| | M3 (retired) | DCCN (current) |
|---|---|---|
| Partition | `comp` | `batch` (default), `interactive`, `gpu`, `gpu40g` |
| Env manager | conda (`actinf-py-scripts`) | **uv venv** -- no conda on the cluster |
| Default memory | sensible per node | **1 GB** -- always pass `--mem` |
| Max walltime | varied | 72 h on every partition |
| CPUs per node | -- | `batch` caps at 45 |
| MATLAB | `/usr/local/matlab/r2022a` | `module load matlab/R2024b` |
| SPM12 | `~/fc37/Carrick/spm12` | **not installed** -- run locally instead |
| Code sync | Mutagen | Mutagen (unchanged) |

## Environment (provision once, from the login node)

DCCN has no conda. Jobs activate a uv-managed venv via
`cluster/lib/cluster_env.sh`:

```bash
cd "$DCM_CLUSTER_ROOT"
~/.local/bin/uv venv --python 3.10 .venv
~/.local/bin/uv pip install --python .venv/bin/python -e '.[benchmark,dev]'
```

**Never install inside a job.** Concurrent resolvers in an array job race on the
same venv and corrupt it (`.pth` damage, OOM). Provision once; jobs only
activate. Override the location with `DCM_VENV=/path/to/venv`.

## Writing a job

Every job sources the shared library, which handles activation, thread pinning,
the stack import check, and the header block:

```bash
#SBATCH --job-name=my_job
#SBATCH --output=cluster/logs/my_job_%j.out
#SBATCH --error=cluster/logs/my_job_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G           # REQUIRED -- the default is 1 GB
#SBATCH --cpus-per-task=4
#SBATCH --partition=batch

source cluster/lib/cluster_env.sh
crlf_guard
setup_torch_threads 4
activate_env
verify_torch
print_job_header "My Job"

python cluster/scripts/my_script.py
```

`setup_matlab R2024b` additionally loads MATLAB and hard-checks `SPM12_PATH`.

## MATLAB and SPM12

**Run the SPM bridge on the workstation, not the cluster.** As of 2026-09-05 the
workstation has MATLAB **R2025b with a valid licence** and a complete SPM12 at
`C:/Users/adaman/Documents/external/spm12`. The FlexLM -15 licence failure that
originally forced Phases 32/34/35 onto M3 no longer applies.

The cluster has MATLAB modules but **no system SPM12**, so a cluster SPM run
must point `SPM12_PATH` at a personal checkout; `setup_matlab` fails loudly if
it is unset or wrong.

Paths resolve from `config.py` (`MATLAB_PATH`, `SPM12_PATH`, `TAPAS_RDCM_PATH`),
all environment-overridable. `validation/run_validation.py` and
`validation/run_vl_validation.py` export `SPM12_PATH` into the MATLAB child, and
every `.m` in `validation/matlab_scripts/` reads it via `getenv('SPM12_PATH')`.

Note the v0.8.0 ERP parity ladders are **fixture-keyed**: they assert pure-torch
output against frozen `.mat` files in `validation/data/` and need no MATLAB at
all. Only *regenerating* those fixtures requires SPM12.

## Code sync

Local is the source of truth; Mutagen propagates edits. **Never edit files
directly on the cluster** -- it corrupts the sync direction. See the `dccn-hpc`
skill for `dccn-sync-init`, the ACL model, and the SSH kill switch
(`dccn-unlock` / `dccn-lock`).

The repo requires `.gitattributes` with `* text=auto eol=lf` and local
`core.autocrlf=input`, or every text file becomes a cross-OS conflict.

## Submitting and monitoring

```bash
ssh mentat "cd \$DCM_CLUSTER_ROOT && sbatch cluster/sbatch/recovery_matrix_sweep.sbatch"
ssh mentat "squeue -u \$USER"
ssh mentat "sacct -j <JOBID> --format=JobID,JobName%20,State,Elapsed,ExitCode -X"
ssh mentat "tail -n 200 \$DCM_CLUSTER_ROOT/cluster/logs/<name>_<jobid>.out"
```

Slurm jobs are not harness-tracked. After every `sbatch`, check back twice: once
at ~2 minutes (did it actually leave PENDING, or die instantly on a bad
directive?) and once near the expected finish. **Read the log, not just the
state** -- a job can report `COMPLETED` rc=0 with every task inside silently
errored.

## Layout

| Path | Purpose |
|---|---|
| `cluster/lib/cluster_env.sh` | Shared activation / MATLAB / verification helpers |
| `cluster/sbatch/` | Slurm job scripts (one per experiment) |
| `cluster/scripts/` | Python entrypoints the sbatch files call |
| `cluster/logs/` | Job stdout/stderr (gitignored) |
| `cluster/results/` | Per-job JSON results (gitignored) |

## Retired

`run_phase16_acceptance.slurm`, `submit_phase16.sh`, and
`99_push_phase16_results.slurm` are v0.3.0 Phase-16 M3 machinery carrying conda
activation and in-job `pip install`. Neither works on DCCN. They are kept for
provenance and flagged for removal by the 2026-07-15 code-organization audit
(theme 5.2).
