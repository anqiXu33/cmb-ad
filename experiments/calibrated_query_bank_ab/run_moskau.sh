#!/bin/bash
# Side experiment — see experiments/calibrated_query_bank_ab/README.md
#
#SBATCH --job-name=ab_compare_moskau
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="A100|H100|H200"
#SBATCH --time=4:00:00
#SBATCH --output=logs/experiments/calibrated_query_bank_ab/moskau_%j.log

set -eo pipefail

EXP_DIR="experiments/calibrated_query_bank_ab"
OUT_DIR="${EXP_DIR}/results/Moskau_einfach"
mkdir -p logs/experiments/calibrated_query_bank_ab/
mkdir -p ${OUT_DIR}

module load miniforge3/25.3.0-3
module load cuda/12.6.3
eval "$(conda shell.bash hook)"
conda activate AD-generation
export HF_HOME="/home/anxu/data/hf_cache"

cd /home/anxu/shot-by-shot

DATA_ROOT="/shares/iict-sp4.ebling.cl.uzh/data/anxu_MA_data"
MOVIE="Moskau_einfach"
IMDB_ID="tt28132461"
SHARD="results/preprocess/query_bank/query_bank_0.pkl"  # contains tt10949014, tt28132461

python ${EXP_DIR}/run_ab.py \
    --anno_path       ${DATA_ROOT}/annotations/${MOVIE}_context_faces_threads_shot_scales.csv \
    --charbank_path   ${DATA_ROOT}/annotations/imdb_chars_ext.json \
    --query_bank_path ${SHARD} \
    --video_dir       ${DATA_ROOT}/video/ \
    --movie_imdb_id   ${IMDB_ID} \
    --output_path     ${OUT_DIR}/pred_ids_ab.csv \
    --score_thresh    0.2 \
    --det_thresh      0.4 \
    --num_frames      16 \
    --cache_dir       ~/.insightface

echo ""
echo "=== Diff summary ==="
python ${EXP_DIR}/diff.py \
    --csv    ${OUT_DIR}/pred_ids_ab.csv \
    --report ${OUT_DIR}/diff_report.txt

echo ""
echo "Done: ${MOVIE}"
echo "  CSV:    ${OUT_DIR}/pred_ids_ab.csv"
echo "  Report: ${OUT_DIR}/diff_report.txt"
