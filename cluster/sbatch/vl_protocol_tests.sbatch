#!/bin/bash
#SBATCH --job-name=vl_protocol
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=cluster/logs/vl_protocol_%j.out
#SBATCH --error=cluster/logs/vl_protocol_%j.err
#SBATCH --partition=batch

source cluster/lib/cluster_env.sh
crlf_guard

setup_torch_threads 4
activate_env
verify_torch
print_job_header "VL ForwardModel Protocol Tests"

mkdir -p cluster/logs

pip install --no-deps -e . 2>&1 | tail -1

python -m pytest tests/test_vl_forward_model_protocol.py -v -s 2>&1

EXIT_CODE=$?
echo ""
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "VL PROTOCOL TESTS PASSED (exit 0)"
else
    echo "VL PROTOCOL TESTS FAILED (exit ${EXIT_CODE})"
fi
exit $EXIT_CODE
