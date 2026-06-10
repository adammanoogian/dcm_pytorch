#!/bin/bash
# =============================================================================
# Recovery-Matrix Sweep (Phase 30-02, VLREC-01 / VLREC-03)
# =============================================================================
# Each array task runs ONE (variant, N, SNR) cell of the small validation grid
# over GRID_SEEDS=10 seeds and writes one per-cell JSON. The array range MUST
# equal len(enumerate_cells()) -- currently 10 cells:
#   spectral {N2,N4} x {SNR1,SNR3} = 4
#   task     {N2,N4} x {SNR1,SNR3} = 4
#   latent_circuit (fixed N4) x {SNR1,SNR3} = 2
# = 10 cells x 10 seeds = 120 fits. If the grid grows, update --array to match
# (verify with: python -c "from benchmarks.recovery_matrix_grid import
# enumerate_cells; print(len(enumerate_cells()))").
#
# dt>=0.1 is enforced INSIDE the driver (task/latent VL runners assert it; the
# TaskDCMForward precision guard caps T*N). The near-boundary-A exclusion
# (VLREC-03) is applied per seed inside run_one_cell.
#
# Latent-circuit cells import pyro_dcm.models.*; they REQUIRE the Mutagen
# `models/` anchored-ignore fix to be in place so src/pyro_dcm/models/ syncs to
# M3 (per STATE.md the fix landed). Spectral/task cells are unaffected.
#
# Usage:
#   sbatch cluster/sbatch/recovery_matrix_sweep.sbatch
#
# Outputs:
#   cluster/results/recovery_matrix_<jobid>_<taskid>.json   Per-cell result
#   cluster/logs/recov_matrix_<jobid>_<taskid>.{out,err}    SLURM logs
# =============================================================================

#SBATCH --job-name=recov_matrix
#SBATCH --array=0-9
#SBATCH --output=cluster/logs/recov_matrix_%A_%a.out
#SBATCH --error=cluster/logs/recov_matrix_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=comp

source cluster/lib/cluster_env.sh
crlf_guard

ENV_NAME="${ENV_NAME:-actinf-py-scripts}"

setup_torch_threads 4
activate_env "$ENV_NAME"

# NEVER pip install inside this array job. N concurrent SLURM tasks running a
# resolver against the shared conda env race -> OOM + .pth corruption (project
# hard rule). The package is found via sys.path in the Python script; no install
# step here. See m3-hpc.skill convention and cluster/sbatch/lc_vl_acceptance.sbatch.

verify_torch
print_job_header "Recovery-Matrix Sweep (cell ${SLURM_ARRAY_TASK_ID})"

mkdir -p cluster/logs cluster/results

python cluster/scripts/recovery_matrix_cell.py
EXIT_CODE=$?

echo ""
echo "════════════════════════════════════════════════════════"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "  CELL ${SLURM_ARRAY_TASK_ID} COMPLETE (exit 0)"
else
    echo "  CELL ${SLURM_ARRAY_TASK_ID} FAILED (exit ${EXIT_CODE})"
fi
echo "════════════════════════════════════════════════════════"

exit $EXIT_CODE
