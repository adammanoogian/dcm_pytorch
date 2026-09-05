#!/bin/bash
#SBATCH --job-name=tribe_extract
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --output=cluster/logs/tribe_extract_%j.out
#SBATCH --error=cluster/logs/tribe_extract_%j.err

source cluster/lib/cluster_env.sh
crlf_guard

setup_torch_threads 4
activate_env
verify_torch
verify_gpu
print_job_header "Phase 24: TRIBE v2 Extraction + Spectral DCM"

mkdir -p cluster/logs results/phase24_tribe

# --- Pitfall 2: Pin NumPy before tribev2 install ---
pip install 'numpy>=1.26.4,<2.1.0' --quiet

# --- Install tribev2 (no-deps avoids resolver OOM in SLURM) ---
pip install 'tribev2[plotting] @ git+https://github.com/facebookresearch/tribev2.git' \
    --no-deps --quiet

# --- Install nilearn for parcellation ---
pip install 'nilearn>=0.10.3' --quiet

# --- Pitfall 1: Increase HuggingFace download timeout ---
export HF_HUB_DOWNLOAD_TIMEOUT=300

# --- Stimulus path (set via environment or override) ---
STIMULUS_PATH="${STIMULUS_PATH:-stimulus.mp4}"
N_ROIS="${N_ROIS:-100}"
N_REGIONS="${N_REGIONS:-6}"
ROI_INDICES="${ROI_INDICES:-}"
NUM_STEPS="${NUM_STEPS:-2000}"
N_RESTARTS="${N_RESTARTS:-10}"

echo ""
echo "Configuration:"
echo "  STIMULUS_PATH: ${STIMULUS_PATH}"
echo "  N_ROIS:        ${N_ROIS}"
echo "  N_REGIONS:     ${N_REGIONS}"
echo "  ROI_INDICES:   ${ROI_INDICES:-auto (first N)}"
echo "  NUM_STEPS:     ${NUM_STEPS}"
echo "  N_RESTARTS:    ${N_RESTARTS}"
echo ""

# --- Step 1: Extract ROI timeseries from TRIBE v2 ---
echo "=== Step 1: TRIBE v2 Extraction ==="
python scripts/24_extract_tribe_latents.py \
    --video-path "${STIMULUS_PATH}" \
    --output-dir results/phase24_tribe/ \
    --n-rois "${N_ROIS}" \
    --cache-folder ./tribe_cache

EXTRACT_EXIT=$?
if [[ $EXTRACT_EXIT -ne 0 ]]; then
    echo "EXTRACTION FAILED (exit ${EXTRACT_EXIT})"
    exit $EXTRACT_EXIT
fi

# --- Step 2: Fit spectral DCM to ROI timeseries ---
echo ""
echo "=== Step 2: Spectral DCM Fitting ==="

# Build ROI indices argument
ROI_ARGS=""
if [[ -n "$ROI_INDICES" ]]; then
    ROI_ARGS="--roi-indices ${ROI_INDICES}"
fi

python scripts/24_fit_dcm_tribe.py \
    --input-npz results/phase24_tribe/tribe_roi_timeseries.npz \
    --output-dir results/phase24_tribe/ \
    --n-regions "${N_REGIONS}" \
    ${ROI_ARGS} \
    --num-steps "${NUM_STEPS}" \
    --n-restarts "${N_RESTARTS}"

FIT_EXIT=$?

echo ""
if [[ $FIT_EXIT -eq 0 ]]; then
    echo "PIPELINE COMPLETE (exit 0)"
    echo "Results:"
    echo "  ROI timeseries: results/phase24_tribe/tribe_roi_timeseries.npz"
    echo "  DCM results:    results/phase24_tribe/tribe_dcm_results.npz"
else
    echo "DCM FITTING FAILED (exit ${FIT_EXIT})"
fi
exit $FIT_EXIT
