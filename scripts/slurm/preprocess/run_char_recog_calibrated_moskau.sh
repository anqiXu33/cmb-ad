#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# One-movie sanity check: re-run char_recog/main.py on Moskau_einfach with the
# calibrated query bank (FocusedAD-style centroids) instead of raw headshots.
#
# Why: the current $DATA_ROOT/annotations/<movie>_*.csv files were generated in
# Oct 2025 with raw-headshot detection (build_query_bank.py was committed
# 2026-03-31). The query_bank_*.pkl shards now exist but have NEVER been
# applied to regenerate the annotation CSVs. This script does that for
# Moskau only, as a quality gate before running on all 8 films.
#
# Output (does NOT overwrite the original):
#   $DATA_ROOT/annotations_calibrated/Moskau_einfach_context_faces_threads_shot_scales_calibrated.csv
#
# Moskau (tt10949014) lives in query_bank_0.pkl (verified 2026-04-28).
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=charrec_cal_moskau
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="A100|H100|H200"
#SBATCH --time=04:00:00
#SBATCH --output=logs/charrec_cal_moskau_%j.log

set -eo pipefail
mkdir -p logs/

module load miniforge3/25.3.0-3
module load cuda/12.6.3
eval "$(conda shell.bash hook)"
conda activate AD-generation
export HF_HOME="/home/anxu/data/hf_cache"

cd /home/anxu/shot-by-shot

DATA_ROOT="/shares/iict-sp4.ebling.cl.uzh/data/anxu_MA_data"
ANNO_DIR="${DATA_ROOT}/annotations"
OUT_DIR="${DATA_ROOT}/annotations_calibrated"
MOVIE="Moskau_einfach"
INPUT_CSV="${ANNO_DIR}/${MOVIE}_context_faces_threads_shot_scales.csv"
QUERY_BANK="results/preprocess/query_bank/query_bank_0.pkl"

# ── Sanity checks ────────────────────────────────────────────────────────────
if [ ! -f "${INPUT_CSV}" ]; then
    echo "[FATAL] Input anno CSV not found: ${INPUT_CSV}"
    exit 1
fi
if [ ! -f "${QUERY_BANK}" ]; then
    echo "[FATAL] Query bank not found: ${QUERY_BANK}"
    echo "        (Moskau, tt10949014, is in query_bank_0.pkl)"
    exit 1
fi

# Confirm Moskau (tt10949014) is in this query bank shard
python3 -c "
import pickle
with open('${QUERY_BANK}', 'rb') as f:
    qb = pickle.load(f)
assert 'tt10949014' in qb, f'tt10949014 not in query_bank_0.pkl. Keys: {list(qb.keys())}'
print(f'[OK] Moskau (tt10949014) found in query bank with {len(qb[\"tt10949014\"])} centroids.')
"

mkdir -p "${OUT_DIR}"

# ── Run char_recog with --query_bank_path ────────────────────────────────────
# --num_frames 32 matches the existing anno CSV (each row's bboxes is len-32).
# Default --score_thresh 0.2, --det_thresh 0.4 → output filename will be
#   ${OUT_DIR}/swissAD_anno_with_face_reproduced_0.2_0.4.csv
# We rename it to a movie-specific path immediately afterwards.
SCORE_THRESH=0.2
DET_THRESH=0.4

python char_recog/main.py \
    --dataset swissAD \
    --anno_path "${INPUT_CSV}" \
    --charbank_path "${ANNO_DIR}/imdb_chars_ext.json" \
    --video_dir "${DATA_ROOT}/video/" \
    --output_dir "${OUT_DIR}" \
    --score_thresh "${SCORE_THRESH}" \
    --det_thresh   "${DET_THRESH}" \
    --num_frames 32 \
    --cache_dir ~/.insightface \
    --query_bank_path "${QUERY_BANK}"

# ── Rename to movie-specific filename so the file is unambiguous ─────────────
SRC="${OUT_DIR}/swissAD_anno_with_face_reproduced_${SCORE_THRESH}_${DET_THRESH}.csv"
DST="${OUT_DIR}/${MOVIE}_context_faces_threads_shot_scales_calibrated.csv"

if [ ! -f "${SRC}" ]; then
    echo "[FATAL] char_recog produced no output: ${SRC}"
    exit 2
fi

mv "${SRC}" "${DST}"
echo "[OK] Calibrated anno CSV written:"
echo "       ${DST}"
echo
echo "Verify it does NOT clobber the original:"
echo "  Original (raw headshot): ${INPUT_CSV}"
echo "  Calibrated (this run)  : ${DST}"
ls -la "${INPUT_CSV}" "${DST}"

echo "Done."
