#!/bin/bash
#SBATCH --job-name=bmr_vs_elbo
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/bmr_vs_elbo_%j.out
#SBATCH --error=cluster/logs/bmr_vs_elbo_%j.err
#SBATCH --partition=comp

source cluster/lib/cluster_env.sh
crlf_guard

ENV_NAME="${ENV_NAME:-actinf-py-scripts}"

setup_torch_threads 4
activate_env "$ENV_NAME"
verify_torch
print_job_header "Phase 23 BMR vs ELBO Validation"

mkdir -p cluster/logs

python -m pytest tests/test_bmr_vs_elbo.py -v -m slow -s 2>&1

EXIT_CODE=$?
echo ""
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "BMR VS ELBO TEST PASSED (exit 0)"
else
    echo "BMR VS ELBO TEST FAILED (exit ${EXIT_CODE})"
fi
exit $EXIT_CODE
