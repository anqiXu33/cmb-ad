#!/bin/bash
#SBATCH --job-name=gen_char_desc
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="A100|H100|H200"
#SBATCH --time=4:00:00
#SBATCH --output=logs/gen_char_desc_%j.log

set -eo pipefail
mkdir -p logs/

module load miniforge3/25.3.0-3
module load cuda/12.6.3
eval "$(conda shell.bash hook)"
conda activate AD-generation
export HF_HOME="/home/anxu/data/hf_cache"

cd /home/anxu/shot-by-shot

DATA_ROOT="/shares/iict-sp4.ebling.cl.uzh/data/anxu_MA_data"
IMDB_CHARS="${DATA_ROOT}/annotations/imdb_chars_ext.json"
REP_CROPS="results/preprocess/rep_crops/"

# ---- Movie IMDB IDs per language ----
# Extract from annotation CSVs: head -1 gives header, head -2 | tail -1 gives first data row
get_imdbid() {
    python3 -c "import pandas as pd; print(pd.read_csv('$1', nrows=1)['imdbid'].iloc[0])"
}

ANNO_DIR="${DATA_ROOT}/annotations"

# German movies
DE_IDS=""
for movie in 8_Tage_im_August Moskau_einfach Baghdad_in_my_Shadow Trommelwirbel; do
    csv="${ANNO_DIR}/${movie}_context_faces_threads_shot_scales.csv"
    if [ -f "$csv" ]; then
        mid=$(get_imdbid "$csv")
        DE_IDS="${DE_IDS} ${mid}"
    else
        echo "[WARN] Missing annotation: $csv"
    fi
done

# Italian movies
IT_IDS=""
for movie in La_Tentazione_Di_Esistere Quanto_basta Riders_of_justice Stuerm; do
    csv="${ANNO_DIR}/${movie}_context_faces_threads_shot_scales.csv"
    if [ -f "$csv" ]; then
        mid=$(get_imdbid "$csv")
        IT_IDS="${IT_IDS} ${mid}"
    else
        echo "[WARN] Missing annotation: $csv"
    fi
done

echo "German movie IDs: ${DE_IDS}"
echo "Italian movie IDs: ${IT_IDS}"

# ---- Generate descriptions per language ----

echo ""
echo "=== Generating German descriptions ==="
python char_recog/generate_char_descriptions.py \
    --imdb_chars_path ${IMDB_CHARS} \
    --rep_crops_dir ${REP_CROPS} \
    --output_path ${IMDB_CHARS} \
    --language German \
    --movie_ids ${DE_IDS}

echo ""
echo "=== Generating Italian descriptions ==="
python char_recog/generate_char_descriptions.py \
    --imdb_chars_path ${IMDB_CHARS} \
    --rep_crops_dir ${REP_CROPS} \
    --output_path ${IMDB_CHARS} \
    --language Italian \
    --movie_ids ${IT_IDS}

echo "Done: character descriptions generated for all 8 movies"
