#!/bin/bash
#SBATCH --job-name=roi_perturb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/roi_perturbation_%j.out
#SBATCH --error=cluster/logs/roi_perturbation_%j.err
#SBATCH --partition=batch

source cluster/lib/cluster_env.sh
crlf_guard

setup_torch_threads 4
activate_env
verify_torch
print_job_header "ROI Perturbation Recovery (Approach B)"

mkdir -p cluster/logs

python -m pytest tests/test_roi_perturbation_recovery.py -v -m slow -s 2>&1

EXIT_CODE=$?
echo ""
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "ROI PERTURBATION TESTS PASSED (exit 0)"
else
    echo "ROI PERTURBATION TESTS FAILED (exit ${EXIT_CODE})"
fi
exit $EXIT_CODE
