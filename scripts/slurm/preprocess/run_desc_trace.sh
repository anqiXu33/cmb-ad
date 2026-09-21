#!/bin/bash
#SBATCH --job-name=desc_trace
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/desc_trace_%j.log

# CPU-only CMB replay across all 8 films. No GPU, no model loading.
# Total runtime expected ~5 min (~30s per film).
#
# Outputs (one per film):
#   results/desc_trace/desc_trace_<movie>.csv         long form (one row per char per AD)
#   results/desc_trace/review_<movie>.csv             merged with stage2.csv (if exists)
#
# The review CSV is only produced for films whose CMB-AD stage2.csv (or .partial)
# already exists — that way you can re-run this script as more films complete
# without it failing on the missing ones.

set -eo pipefail
mkdir -p logs/ results/desc_trace/

module load miniforge3/25.3.0-3
eval "$(conda shell.bash hook)"
conda activate AD-generation

cd /home/anxu/shot-by-shot

DATA_ROOT="/shares/iict-sp4.ebling.cl.uzh/data/anxu_MA_data"
ANNO_DIR="${DATA_ROOT}/annotations"
IMDB_CHARS="${ANNO_DIR}/imdb_chars_enriched_manual.json"
REP_CROPS="results/preprocess/rep_crops"

MOVIES=(
    8_Tage_im_August
    Baghdad_in_my_Shadow
    La_Tentazione_Di_Esistere
    Moskau_einfach
    Quanto_basta
    Riders_of_justice
    Stuerm
    Trommelwirbel
)

for movie in "${MOVIES[@]}"; do
    echo ""
    echo "=== ${movie} ==="

    anno_csv="${ANNO_DIR}/${movie}_context_faces_threads_shot_scales.csv"
    if [ ! -f "${anno_csv}" ]; then
        echo "[SKIP] missing anno: ${anno_csv}"
        continue
    fi

    pred_csv="results/stage1/${movie}/swissAD_ads/stage1_qwen2vl.csv"
    pred_arg=""
    if [ -f "${pred_csv}" ]; then
        pred_arg="--pred_path ${pred_csv}"
    else
        echo "[INFO] no Stage-1 CSV for ${movie}; falling back to anno GT as text_pred proxy"
    fi

    # Prefer final stage2.csv, fall back to .partial during in-flight runs.
    stage2_csv=""
    for candidate in \
        "results/stage2/cmb_ad/${movie}/stage2.csv" \
        "results/stage2/cmb_ad/${movie}/stage2_qwen3_single.csv" \
        "results/stage2/cmb_ad/${movie}/stage2_qwen3_single.csv.partial"; do
        if [ -f "${candidate}" ]; then
            stage2_csv="${candidate}"
            break
        fi
    done

    review_args=""
    if [ -n "${stage2_csv}" ]; then
        review_args="--stage2_csv ${stage2_csv} \
            --review_output results/desc_trace/review_${movie}.csv \
            --rep_crops_dir ${REP_CROPS}"
        echo "[INFO] joining with: ${stage2_csv}"
    else
        echo "[INFO] no stage2.csv yet; trace-only run"
    fi

    python3 scripts/tools/trace_desc_injection.py \
        ${pred_arg} \
        --anno_path "${anno_csv}" \
        --imdb_chars_path "${IMDB_CHARS}" \
        --output "results/desc_trace/desc_trace_${movie}.csv" \
        ${review_args}
done

echo ""
echo "Done. Outputs in results/desc_trace/"
ls -la results/desc_trace/
