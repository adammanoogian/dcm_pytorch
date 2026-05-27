#!/bin/bash
#SBATCH --job-name=dcm_scaling
#SBATCH --output=benchmarks/results/scaling_%j.out
#SBATCH --error=benchmarks/results/scaling_%j.err
#SBATCH --partition=comp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

# Scaling benchmark: VL vs SVI across N=3..20 (3 seeds each)
# Estimated: ~2h for full sweep

set -euo pipefail

module load miniforge3
eval "$(conda shell.bash hook)"
conda activate dec-psilocybin-stats

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

DCM_PYTORCH_DIR="${HOME}/fc37/adam/repos/dcm_pytorch"
cd "$DCM_PYTORCH_DIR"

pip install -e . --no-deps --quiet 2>/dev/null

echo "=== Scaling Benchmark ==="
echo "Job: ${SLURM_JOB_ID}"
echo "Start: $(date)"
echo "Max N: ${1:-20}, Seeds: ${2:-3}"

python benchmarks/scaling_benchmark.py \
    --max-n "${1:-20}" \
    --seeds "${2:-3}" \
    --vl-max-iter 64 \
    --svi-steps 500

echo "=== Done: $(date) ==="
