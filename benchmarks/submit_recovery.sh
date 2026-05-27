#!/bin/bash
#SBATCH --job-name=dcm_recovery
#SBATCH --output=benchmarks/results/recovery_%j.out
#SBATCH --error=benchmarks/results/recovery_%j.err
#SBATCH --partition=comp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00

# Full parameter recovery validation: N=3,5,10, SNR=5,10,20, 4 patterns, 5 seeds
# Estimated: ~4h (60 conditions x 3-60s each)

set -euo pipefail

module load miniforge3
eval "$(conda shell.bash hook)"
conda activate dec-psilocybin-stats

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

DCM_PYTORCH_DIR="${HOME}/fc37/adam/repos/dcm_pytorch"
cd "$DCM_PYTORCH_DIR"

pip install -e . --no-deps --quiet 2>/dev/null

echo "=== Recovery Validation ==="
echo "Job: ${SLURM_JOB_ID}"
echo "Start: $(date)"

python benchmarks/recovery_validation.py

echo "=== Done: $(date) ==="
