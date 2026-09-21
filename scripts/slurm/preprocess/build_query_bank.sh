#!/bin/bash
#SBATCH --job-name=build_qb
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="A100|H100|H200"
#SBATCH --time=12:00:00
#SBATCH --output=logs/build_qb_%A_%a.log
#SBATCH --array=0-3

set -eo pipefail
mkdir -p logs/
mkdir -p results/preprocess/rep_crops/ results/preprocess/query_bank/

module load miniforge3/25.3.0-3
module load cuda/12.6.3
eval "$(conda shell.bash hook)"
conda activate AD-generation
export HF_HOME="/home/anxu/data/hf_cache"

cd /home/anxu/shot-by-shot

DATA_ROOT="/shares/iict-sp4.ebling.cl.uzh/data/anxu_MA_data"
ANNO_DIR="${DATA_ROOT}/annotations"

# ---- Discover IMDB IDs from annotation CSVs ----
get_imdbid() {
    python3 -c "import pandas as pd; print(pd.read_csv('$1', nrows=1)['imdbid'].iloc[0])"
}

ALL_IDS=()
for movie in 8_Tage_im_August Moskau_einfach Baghdad_in_my_Shadow Trommelwirbel \
             La_Tentazione_Di_Esistere Quanto_basta Riders_of_justice Stuerm; do
    csv="${ANNO_DIR}/${movie}_context_faces_threads_shot_scales.csv"
    if [ -f "$csv" ]; then
        ALL_IDS+=( "$(get_imdbid "$csv")" )
    else
        echo "[WARN] Missing annotation: $csv"
        ALL_IDS+=( "MISSING" )
    fi
done

# ---- Split into 4 groups of 2 movies each ----
IDX1=$(( SLURM_ARRAY_TASK_ID * 2 ))
IDX2=$(( SLURM_ARRAY_TASK_ID * 2 + 1 ))

MOVIE_IDS="${ALL_IDS[$IDX1]} ${ALL_IDS[$IDX2]}"
echo "Array task ${SLURM_ARRAY_TASK_ID}: processing movies ${MOVIE_IDS}"

python char_recog/build_query_bank.py \
    --imdb_chars_path ${DATA_ROOT}/annotations/imdb_chars_ext.json \
    --actor_images_dir ${DATA_ROOT}/actor_images/ \
    --video_dir ${DATA_ROOT}/video/ \
    --anno_dir ${DATA_ROOT}/annotations/ \
    --output_path results/preprocess/query_bank/query_bank_${SLURM_ARRAY_TASK_ID}.pkl \
    --rep_crops_dir results/preprocess/rep_crops/ \
    --num_rep_crops 3 \
    --score_thresh 0.25 \
    --num_frames 16 \
    --cache_dir ~/.insightface \
    --movie_ids ${MOVIE_IDS}

echo "Done: array task ${SLURM_ARRAY_TASK_ID}"
