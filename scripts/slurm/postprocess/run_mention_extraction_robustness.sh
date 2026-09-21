#!/bin/bash -l
#SBATCH --job-name=mention_extract_robust
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100
#SBATCH --array=0-23%8
#SBATCH --time=03:00:00
#SBATCH --output=/shares/iict-sp4.ebling.cl.uzh/data/anxu_MA_data/logs/mention_robust_%A_%a.log

###############################################################################
# Mention-level evaluation: extract NAME / DESC / PRN mentions from all
# system Stage-2 outputs, for the thesis's primary §6 mention-level metric.
# ─────────────────────────────────────────────────────────────────────────────
# 24 tasks total — 8 films × 3 conditions:
#
#   German (4 films):  8_Tage_im_August, Moskau_einfach,
#                       Baghdad_in_my_Shadow, Trommelwirbel
#   Italian (4 films): La_Tentazione_Di_Esistere, Quanto_basta,
#                       Riders_of_justice, Stuerm
#   Conditions:         cmb_ad, baseline1_no_cmb, baseline2_always_id
#
# Per-task input: ${REPO_ROOT}/results/stage2/<condition>/<movie>/stage2.csv
#                 (matches the save_dir used by the Stage-2 generation
#                  SLURM scripts, which write to a relative path inside
#                  the git working tree).
# Per-task output (under ${WORK_DIR}/mention_robustness/<movie>__<condition>/):
#   text_gen.json     ← system AD text (extractor input)
#   text_gt.json      ← ground-truth AD text (extractor input)
#   text_gen.mentions ← extracted mentions for system output (used by RDA + form-distribution)
#   text_gt.mentions  ← extracted mentions for GT (canonical for 8_Tage_im_August
#                        and Trommelwirbel; sanity check vs mentions_output/
#                        for the other 6 films)
#
# Reuses the exact extraction script + prompt + model used to produce the
# pre-computed GT mentions on cluster (mentions_output/), so system and GT
# mention sets are directly comparable:
#
#   Script: ${WORK_DIR}/scripts/character_chains_qwen3_json.py
#   Model:  Qwen/Qwen3-8B (with --thinking)
#   Prompt: language-specific DE / IT (defined inside the extraction script)
#
# ─────────────────────────────────────────────────────────────────────────────
# Concurrency: --array=0-23%8 caps simultaneous GPU draws at 8.
#
# Submit (after stage2 outputs + this script are on cluster):
#   cd $REPO_ROOT
#   sbatch scripts/slurm/postprocess/run_mention_extraction_robustness.sh
#
# To resubmit only a subset (e.g. when an earlier 6-task job already
# processed the first 6 entries):
#   sbatch --array=6-23%8 scripts/slurm/postprocess/run_mention_extraction_robustness.sh
#
# Re-runs are safe: skip-on-existence inside the loop.
###############################################################################

set -eo pipefail

module load miniforge3/25.3.0-3
module load cuda/12.6.3
eval "$(conda shell.bash hook)"
conda activate AD-generation

WORK_DIR="/shares/iict-sp4.ebling.cl.uzh/data/anxu_MA_data"
REPO_ROOT="/home/anxu/shot-by-shot"
CACHE_DIR="/home/anxu/data/hf_cache"
export HF_HOME="$CACHE_DIR"

ADAPTER_SCRIPT="${REPO_ROOT}/scripts/tools/stage2_csv_to_mention_json.py"
EXTRACT_SCRIPT="${WORK_DIR}/scripts/character_chains_qwen3_json.py"
OUT_ROOT="${WORK_DIR}/mention_robustness"

mkdir -p "${OUT_ROOT}" "${WORK_DIR}/logs"

# ── Task table ──────────────────────────────────────────────────────────────
# Each line: <movie> <condition> <lang>
# IMPORTANT: tasks 0-5 must remain in their original positions — earlier job
# 2758972 was submitted with --array=0-5 against the original 6-task list, and
# its tasks pick up entries by SLURM_ARRAY_TASK_ID.  When extending to 24
# tasks, append new entries at indices 6-23 only; do not reorder the first 6.
TASKS=(
    # 0-5: pilot (Moskau + Quanto, run by job 2758972)
    "Moskau_einfach cmb_ad de"
    "Moskau_einfach baseline1_no_cmb de"
    "Moskau_einfach baseline2_always_id de"
    "Quanto_basta cmb_ad it"
    "Quanto_basta baseline1_no_cmb it"
    "Quanto_basta baseline2_always_id it"
    # 6-8: 8_Tage_im_August (DE) — GT mentions canonical from this run
    "8_Tage_im_August cmb_ad de"
    "8_Tage_im_August baseline1_no_cmb de"
    "8_Tage_im_August baseline2_always_id de"
    # 9-11: Baghdad_in_my_Shadow (DE)
    "Baghdad_in_my_Shadow cmb_ad de"
    "Baghdad_in_my_Shadow baseline1_no_cmb de"
    "Baghdad_in_my_Shadow baseline2_always_id de"
    # 12-14: Trommelwirbel (DE) — GT mentions canonical from this run
    "Trommelwirbel cmb_ad de"
    "Trommelwirbel baseline1_no_cmb de"
    "Trommelwirbel baseline2_always_id de"
    # 15-17: La_Tentazione_Di_Esistere (IT)
    "La_Tentazione_Di_Esistere cmb_ad it"
    "La_Tentazione_Di_Esistere baseline1_no_cmb it"
    "La_Tentazione_Di_Esistere baseline2_always_id it"
    # 18-20: Riders_of_justice (IT)
    "Riders_of_justice cmb_ad it"
    "Riders_of_justice baseline1_no_cmb it"
    "Riders_of_justice baseline2_always_id it"
    # 21-23: Stuerm (IT)
    "Stuerm cmb_ad it"
    "Stuerm baseline1_no_cmb it"
    "Stuerm baseline2_always_id it"
)

read -r MOVIE CONDITION LANG <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"

STAGE2_CSV="${REPO_ROOT}/results/stage2/${CONDITION}/${MOVIE}/stage2.csv"
TASK_DIR="${OUT_ROOT}/${MOVIE}__${CONDITION}"

mkdir -p "$TASK_DIR"

echo "=========================================="
echo "Task ID:    ${SLURM_ARRAY_TASK_ID}"
echo "Movie:      ${MOVIE}"
echo "Condition:  ${CONDITION}"
echo "Language:   ${LANG}"
echo "Stage2 CSV: ${STAGE2_CSV}"
echo "Output dir: ${TASK_DIR}"
echo "Node:       $(hostname)"
echo "GPU:        $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "=========================================="

if [ ! -f "$STAGE2_CSV" ]; then
    echo "[ERROR] Stage-2 CSV not found: ${STAGE2_CSV}"
    exit 1
fi

# ── Step 1: produce text_gen.json + text_gt.json on login-node-equivalent ──
# (No GPU needed; runs in <1s on a compute node anyway.)
echo
echo ">>> Step 1: stage2.csv → JSON adapter"
python "$ADAPTER_SCRIPT" \
    --stage2_csv "$STAGE2_CSV" \
    --output_dir "$TASK_DIR"

# ── Step 2: extract mentions from system output ────────────────────────────
GEN_JSON="${TASK_DIR}/text_gen.json"
GEN_OUT="${TASK_DIR}/text_gen.mentions"

if [ -f "$GEN_OUT" ]; then
    echo ">>> Step 2: SKIP (exists) ${GEN_OUT}"
else
    echo
    echo ">>> Step 2: extract mentions from system output (text_gen)"
    python "$EXTRACT_SCRIPT" \
        --AD_json    "$GEN_JSON" \
        --out_csv    "$GEN_OUT" \
        --model_name "Qwen/Qwen3-8B" \
        --batch_size 3 \
        --thinking \
        --lang       "$LANG" \
        --cache      "$CACHE_DIR"
fi

# ── Step 3: extract mentions from GT (sanity check vs mentions_output/) ────
GT_JSON="${TASK_DIR}/text_gt.json"
GT_OUT="${TASK_DIR}/text_gt.mentions"

if [ -f "$GT_OUT" ]; then
    echo ">>> Step 3: SKIP (exists) ${GT_OUT}"
elif [ ! -f "$GT_JSON" ]; then
    echo ">>> Step 3: SKIP (no text_gt.json — adapter saw no text_gt column)"
else
    echo
    echo ">>> Step 3: extract mentions from ground-truth (text_gt) for sanity"
    python "$EXTRACT_SCRIPT" \
        --AD_json    "$GT_JSON" \
        --out_csv    "$GT_OUT" \
        --model_name "Qwen/Qwen3-8B" \
        --batch_size 3 \
        --thinking \
        --lang       "$LANG" \
        --cache      "$CACHE_DIR"
fi

echo
echo "=========================================="
echo "Done: ${MOVIE} / ${CONDITION}"
echo "  text_gen.mentions lines: $(wc -l < "$GEN_OUT" 2>/dev/null || echo N/A)"
echo "  text_gt.mentions  lines: $(wc -l < "$GT_OUT"  2>/dev/null || echo N/A)"
echo "=========================================="
