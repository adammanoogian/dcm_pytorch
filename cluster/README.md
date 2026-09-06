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

## Live setup (provisioned 2026-09-05, verified end to end)

| | |
|---|---|
| Remote root | `/home/affneu/adaman/dcm_pytorch` (network home; **no project allocation yet**) |
| Mutagen session | `dcm-pytorch`, `two-way-safe` |
| Cluster Python | CPython 3.10.21, uv-managed |
| Cluster venv | `$DCM_CLUSTER_ROOT/.venv`, 1.2 GB |
| torch | `2.14.0+cpu` -- **matches the workstation exactly** |
| Home usage | 3.6 GB of 50 GB |

Home is the remote root only because no `/project` allocation has been assigned
yet. **Move the root into `/project/<projid>` once one exists** -- 50 GB is not
much, and a network home is the wrong place for job outputs at scale.

### Why the CPU torch wheel

The GPU nodes are A100s, but the driver is **535.113.01 (CUDA 12.2 max)** while
`pip install torch` resolves to a `+cu130` build needing CUDA 13 -- it cannot
drive these GPUs, and costs ~4.4 GB of CUDA libraries that would never be used.
All 24 of the 25 job scripts run on `batch` (CPU) anyway. Pinning
`torch==2.14.0+cpu` from the PyTorch CPU index saves the space, removes a whole
class of driver-mismatch confusion, and makes cluster and workstation
byte-identical -- which matters, because the VL determinism contract explicitly
carries a cross-machine caveat.

If GPU work is ever needed, install a CUDA **12.x** wheel (not 13.x) into a
separate venv and point `DCM_VENV` at it.

## Environment (provision once, from the login node)

DCCN has no conda. Jobs activate a uv-managed venv via
`cluster/lib/cluster_env.sh`:

```bash
cd "$DCM_CLUSTER_ROOT"
~/.local/bin/uv python install 3.10          # cluster python3 is 3.6; modules stop at 3.4
~/.local/bin/uv venv --python 3.10 .venv
~/.local/bin/uv pip install --python .venv/bin/python -e '.[benchmark,dev]'
~/.local/bin/uv pip install --python .venv/bin/python \
    --index-url https://download.pytorch.org/whl/cpu --reinstall-package torch 'torch==2.14.0'
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

### MATLAB on DCCN (verified 2026-09-06)

MATLAB **is** available and **is licensed on compute nodes** -- confirmed by an
`srun` on `dccn-c040`/`dccn-c087`:

```bash
module load matlab/R2024b     # R2023b .. R2026a available; R2024b is default
matlab -batch my_script       # licenses fine under Slurm (license('inuse') -> matlab)
```

`module avail 2>&1 | grep matlab/` lists the versions; `/opt` is mounted on every
node, so `matlabroot` is `/opt/matlab/R2024b`. See the DCCN HPC wiki:
[software via modules](https://hpc.dccn.nl/docs/cluster_howto/software-modules.html)
and [distributed analysis with MATLAB](https://hpc.dccn.nl/docs/cluster_howto/exercise_matlab/exercise.html).

**SPM12 is the only gap** -- `exist('spm','file')` returns 0 on the nodes. A
cluster SPM run must point `SPM12_PATH` at a personal checkout; `setup_matlab`
fails loudly if it is unset or wrong. Running the SPM bridge on the workstation
remains preferable, since it has both MATLAB R2025b and a complete SPM12.

### Windows quoting trap in MATLAB `-batch`

MATLAB string literals passed to `-batch` **must be single-quoted**. On Windows
the `matlab.exe` launcher re-parses its command line and strips embedded double
quotes, so `-batch 'disp("MATLAB OK")'` arrives as `disp(MATLAB` and fails with
"This statement is incomplete" (rc=1). This silently disabled the entire
`@pytest.mark.spm` suite on this workstation -- `check_matlab_available()`
returned False and all 12 MATLAB-dependent tests SKIPPED. It went unnoticed
because validation previously ran on Linux (M3), where the quotes survive.

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
skill for the ACL model and the SSH kill switch (`dccn-unlock` / `dccn-lock`).

### Do NOT create the session with stock `dccn-sync-init`

Its default ignore list breaks this repo in two ways, both silent:

- **`--ignore='models/'` is unanchored.** Mutagen matches at any depth, so it
  also excludes `src/pyro_dcm/models/` -- the whole DCM model package. This
  actually happened on M3 (2026-06-10): the remote copy froze for twelve days
  while `mutagen sync list` cheerfully reported "Watching for changes",
  surfacing only as a baffling `ImportError` inside a cluster job.
- **`--ignore='*.mat'`** excludes `validation/data/*.mat`, the four byte-frozen
  SPM12 fixtures the entire v0.8.0 parity ladder asserts against. Every parity
  test would fail for a reason that looks like a code bug.

The live session uses a deliberately **minimal** ignore list -- caches, `*.pyc`,
`.venv/`, `.mutagen/`, nothing else. The repo is ~28 MB, so broad exclusions buy
nothing and cost silent breakage. Ignoring `.venv/` is mandatory: a Windows venv
(`Scripts/`) synced over a Linux venv (`bin/`) destroys both.

After creating any session, verify the two traps explicitly:

```bash
ssh mentat "ls ~/dcm_pytorch/src/pyro_dcm/models/*.py | wc -l"   # expect 9
ssh mentat "ls ~/dcm_pytorch/validation/data/*.mat"              # expect 4
```

### Line endings are a sync problem, not just a git problem

Mutagen copies **working-tree bytes**, so git's index normalisation does not
protect you: a CRLF working tree reaches the cluster as CRLF and `sbatch`
rejects it outright (`Batch script contains DOS line breaks`). `.gitattributes`
(`* text=auto eol=lf`) plus `core.autocrlf=input` keeps the working tree LF.

Two traps when checking this from Git Bash -- both hit during the migration:

- **MSYS `grep -P` strips CR in text mode** and reports a CRLF file as clean.
  Check bytes instead: `python -c "print(b'\r\n' in open(F,'rb').read())"`.
- **Python's `pathlib.write_text()` writes CRLF on Windows** (text mode
  translates `\n`). Any script that rewrites repo files must use `write_bytes()`
  or `open(..., newline="")`.

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
