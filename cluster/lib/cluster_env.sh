#!/usr/bin/env bash
# =============================================================================
# cluster_env.sh -- Shared environment setup for SLURM jobs (dcm_pytorch)
# =============================================================================
# Source this at the start of any SLURM script for standardized environment
# activation, PyTorch configuration, and thread control.
#
# Usage (at top of your .slurm file, after #SBATCH directives):
#   source cluster/lib/cluster_env.sh
#   activate_env "actinf-py-scripts"  # conda activation with fallback ladder
#   verify_torch                      # PyTorch + Pyro import check
#   verify_gpu                        # GPU visibility check (GPU jobs only)
#   print_job_header                  # standardized job info block
# =============================================================================

_PROJECT="${PROJECT:-fc37}"

activate_env() {
    local env_name="$1"
    if [[ -z "$env_name" ]]; then
        echo "ERROR: activate_env requires an environment name"
        exit 1
    fi

    if module load miniforge3 2>/dev/null; then
        :
    elif module load anaconda 2>/dev/null; then
        eval "$(conda shell.bash hook)" 2>/dev/null || true
    fi

    if conda activate "$env_name" 2>/dev/null; then
        echo "Activated: $env_name (by name)"
    elif conda activate "/scratch/${_PROJECT}/${USER}/conda/envs/${env_name}" 2>/dev/null; then
        echo "Activated: $env_name (from /scratch/${_PROJECT}/${USER}/)"
    else
        echo "ERROR: Failed to activate conda environment: $env_name"
        echo "  Tried:"
        echo "    conda activate $env_name"
        echo "    conda activate /scratch/${_PROJECT}/${USER}/conda/envs/${env_name}"
        exit 1
    fi
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
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        echo "  GPU(s):     $CUDA_VISIBLE_DEVICES"
    fi
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
