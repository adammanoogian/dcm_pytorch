#!/bin/bash
#SBATCH --job-name=val_22_full
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=cluster/logs/val_22_full_%j.out
#SBATCH --error=cluster/logs/val_22_full_%j.err
#SBATCH --partition=comp

source cluster/lib/cluster_env.sh
crlf_guard

ENV_NAME="${ENV_NAME:-actinf-py-scripts}"

setup_torch_threads 4
activate_env "$ENV_NAME"
verify_torch
print_job_header "Phase 22 Full Validation (raw vs latent)"

mkdir -p cluster/logs "cluster/results/phase22_validation"

python scripts/22_run_full_validation.py \
    --output-dir "cluster/results/phase22_validation" \
    --n-train 200 \
    --n-eval 50 \
    --max-iter 128 \
    --tolerance 0.01 \
    --ae-epochs 100 \
    --seed 42

EXIT_CODE=$?
echo ""
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "FULL VALIDATION COMPLETE (exit 0)"
    echo "Results: cluster/results/phase22_validation/raw_vs_latent_comparison.npz"
else
    echo "FULL VALIDATION FAILED (exit ${EXIT_CODE})"
fi
exit $EXIT_CODE
