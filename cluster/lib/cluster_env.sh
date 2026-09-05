#!/usr/bin/env bash
# =============================================================================
# cluster_env.sh -- Shared environment setup for SLURM jobs (dcm_pytorch)
# =============================================================================
# Target cluster: DCCN (mentat001-007.dccn.nl), Radboud/Donders.
# Migrated from Monash M3 on 2026-09-05. The differences that matter:
#
#   | thing        | M3 (retired)              | DCCN (current)                |
#   |--------------|---------------------------|-------------------------------|
#   | partition    | comp                      | batch (default), gpu, gpu40g  |
#   | env manager  | conda (actinf-py-scripts) | uv venv -- NO conda installed  |
#   | default mem  | per-node sensible         | 1 GB (!) -- always pass --mem  |
#   | max walltime | varied                    | 72h on every partition        |
#   | CPUs/node    | --                        | batch caps at 45              |
#   | MATLAB       | /usr/local/matlab/r2022a  | `module load matlab/R2024b`   |
#   | SPM12        | ~/fc37/Carrick/spm12      | NOT INSTALLED -- run locally   |
#
# Usage (at top of your .sbatch file, after #SBATCH directives):
#   source cluster/lib/cluster_env.sh
#   crlf_guard
#   setup_torch_threads 4
#   activate_env                      # uv venv activation with fallback ladder
#   verify_torch                      # PyTorch + Pyro import check
#   verify_gpu                        # GPU visibility check (GPU jobs only)
#   print_job_header "My Job"         # standardized job info block
# =============================================================================

# Repo root on the cluster. Every job is submitted from the repo root, so the
# default is the submit directory; override with DCM_CLUSTER_ROOT if a job is
# ever launched from elsewhere.
DCM_CLUSTER_ROOT="${DCM_CLUSTER_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

# The virtualenv used by every job. Kept inside the repo checkout so it lives
# beside the code on project storage, NOT on the 50 GB network home.
DCM_VENV="${DCM_VENV:-${DCM_CLUSTER_ROOT}/.venv}"

activate_env() {
    # DCCN has no conda. The environment is a uv-managed venv; `uv` itself is a
    # single static binary the user keeps in ~/.local/bin.
    local venv="${1:-$DCM_VENV}"

    if [[ -f "${venv}/bin/activate" ]]; then
        # shellcheck disable=SC1091
        source "${venv}/bin/activate"
        echo "Activated venv: ${venv}"
        echo "  python: $(command -v python3)"
        return 0
    fi

    echo "ERROR: no virtualenv at ${venv}"
    echo "  Create it once on the cluster with:"
    echo "    cd ${DCM_CLUSTER_ROOT}"
    echo "    ~/.local/bin/uv venv --python 3.10 .venv"
    echo "    ~/.local/bin/uv pip install --python .venv/bin/python -e '.[dev]'"
    echo "  Override the location with DCM_VENV=/path/to/venv."
    echo "  NEVER run 'uv pip install' inside a SLURM array job -- concurrent"
    echo "  resolvers corrupt the environment. Provision once, from the login node."
    exit 1
}

setup_matlab() {
    # MATLAB is available on DCCN via Environment Modules, but SPM12 is NOT
    # installed cluster-wide. Any SPM-dependent job must supply SPM12_PATH
    # itself (e.g. a personal checkout on project storage).
    #
    # NOTE: as of 2026-09-05 the DCCN *workstation* has MATLAB R2025b with a
    # valid licence and a complete SPM12, so the SPM12 bridge is best run
    # LOCALLY. These cluster hooks exist for jobs too long for the workstation.
    local version="${1:-R2024b}"

    if ! module load "matlab/${version}" 2>/dev/null; then
        echo "ERROR: could not 'module load matlab/${version}'"
        echo "  Available: module avail 2>&1 | grep matlab/"
        exit 1
    fi
    export MATLAB_PATH="${MATLAB_PATH:-$(command -v matlab)}"
    echo "MATLAB: ${MATLAB_PATH} (module matlab/${version})"

    if [[ -z "${SPM12_PATH:-}" ]]; then
        echo "ERROR: SPM12_PATH is unset and DCCN has no system SPM12."
        echo "  Point it at your own checkout, e.g.:"
        echo "    export SPM12_PATH=\$DCM_CLUSTER_ROOT/external/spm12"
        echo "  Or run the SPM bridge on the workstation instead (preferred)."
        exit 1
    fi
    if [[ ! -f "${SPM12_PATH}/spm.m" ]]; then
        echo "ERROR: SPM12_PATH=${SPM12_PATH} does not contain spm.m"
        exit 1
    fi
    echo "SPM12:  ${SPM12_PATH}"
}

setup_torch_threads() {
    local threads="${1:-4}"
    export OMP_NUM_THREADS="$threads"
    export MKL_NUM_THREADS="$threads"
    export OPENBLAS_NUM_THREADS="$threads"
    export TORCH_NUM_THREADS="$threads"
    echo "Thread config: OMP=$threads MKL=$threads TORCH=$threads"
}

verify_torch() {
    echo ""
    echo "PyTorch stack verification:"
    python3 -c "
import torch, pyro, torchdiffeq, zuko, scipy
print(f'  torch {torch.__version__} | pyro {pyro.__version__} | torchdiffeq {torchdiffeq.__version__}')
print(f'  zuko {zuko.__version__} | scipy {scipy.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
" || {
        echo "ERROR: PyTorch stack import check failed."
        exit 1
    }
}

verify_gpu() {
    echo ""
    echo "GPU verification:"
    if [[ "${SLURM_GPUS_ON_NODE:-0}" -eq 0 ]] && [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        echo "  (no GPU requested -- skipping)"
        return 0
    fi
    python3 -c "
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_mem / 1e9
        print(f'  cuda:{i} -- {name} ({mem:.1f} GB)')
else:
    print('  WARNING: No GPU detected by PyTorch.')
    import sys; sys.exit(1)
"
}

print_job_header() {
    local title="${1:-SLURM Job}"
    echo "════════════════════════════════════════════════════════"
    echo "  $title"
    echo "════════════════════════════════════════════════════════"
    echo "  Job ID:     ${SLURM_JOB_ID:-local}"
    echo "  Job name:   ${SLURM_JOB_NAME:-unknown}"
    echo "  Node:       ${SLURMD_NODENAME:-$(hostname)}"
    echo "  Partition:  ${SLURM_JOB_PARTITION:-unknown}"
    echo "  CPUs:       ${SLURM_CPUS_ON_NODE:-unknown}"
    echo "  Mem/node:   ${SLURM_MEM_PER_NODE:-unset} MB"
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        echo "  GPU(s):     $CUDA_VISIBLE_DEVICES"
    fi
    echo "  Repo root:  ${DCM_CLUSTER_ROOT}"
    echo "  Commit:     $(git rev-parse --short HEAD 2>/dev/null || echo 'n/a')"
    echo "  Start:      $(date)"
    echo "════════════════════════════════════════════════════════"
    echo ""
}

crlf_guard() {
    local self="${BASH_SOURCE[1]:-$0}"
    if grep -Pq '\r$' "$self" 2>/dev/null; then
        echo "WARNING: CRLF detected in $self -- fixing in-place"
        sed -i 's/\r$//' "$self"
    fi
}
