#!/usr/bin/env bash
# =============================================================================
# run_pipeline_cluster.sh
#
# Phase 3 – Full two-stage Swiss AD pipeline for the cluster (GPU required).
#
# Usage:
#   ./run_pipeline_cluster.sh [language]
#   language: "German" (default) | "Italian" | "French"
#
# Environment variables (set before running, or edit defaults below):
#   REPO_ROOT        Absolute path to the repository root
#   VIDEO_DIR        Directory containing the Swiss film video files
#   ANNO_PATH        Path to the Swiss annotation CSV for Stage 1
#   CHARBANK_PATH    Path to the character bank JSON/CSV (optional)
#   OPENAI_API_KEY   Required only if using swissAD_gpt4.py
#   FEW_SHOT_CSV     Path to few-shot samples CSV (text_gt / text_gt_wo_char)
#   LR_INTERCEPT     Linear-regression intercept for word-count estimation
#   LR_SLOPE         Linear-regression slope for word-count estimation
#   LANGUAGE         Target language: German | Italian | French
# =============================================================================

set -euo pipefail

# ------------------------------------------------------------------
# Default configuration – override with environment variables
# ------------------------------------------------------------------
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
LANGUAGE="${1:-${LANGUAGE:-German}}"

# Input / output paths
ANNO_PATH="${ANNO_PATH:-${REPO_ROOT}/resources/annotations/test_sample_swiss_de.csv}"
VIDEO_DIR="${VIDEO_DIR:-/data/swiss_films/videos}"
CHARBANK_PATH="${CHARBANK_PATH:-}"          # leave empty if not available
RESULTS_DIR="${REPO_ROOT}/results"

# Stage 1 output
STAGE1_OUTPUT="${RESULTS_DIR}/stage1_real_output.csv"

# Stage 2 output
STAGE2_SAVE_DIR="${RESULTS_DIR}"
FINAL_OUTPUT="${RESULTS_DIR}/final_swiss_ads.csv"

# Few-shot examples CSV for Stage 2
FEW_SHOT_CSV="${FEW_SHOT_CSV:-${RESULTS_DIR}/oracle_few_shot_de.csv}"

# Word-count linear regression (adjust to your training-set statistics)
LR_INTERCEPT="${LR_INTERCEPT:-0.0}"
LR_SLOPE="${LR_SLOPE:-2.0}"

# ------------------------------------------------------------------
# Derived paths
# ------------------------------------------------------------------
STAGE1_SCRIPT="${REPO_ROOT}/stage1/main_qwen2vl.py"
STAGE2_SCRIPT="${REPO_ROOT}/stage2/swissAD_qwen3.py"

mkdir -p "${RESULTS_DIR}"

echo "============================================================"
echo "  Swiss AD Pipeline  |  Language: ${LANGUAGE}"
echo "============================================================"
echo "  REPO_ROOT   : ${REPO_ROOT}"
echo "  ANNO_PATH   : ${ANNO_PATH}"
echo "  VIDEO_DIR   : ${VIDEO_DIR}"
echo "  STAGE1 OUT  : ${STAGE1_OUTPUT}"
echo "  FINAL OUT   : ${FINAL_OUTPUT}"
echo "============================================================"
echo ""

# ==================================================================
# COMMAND 1 – Stage 1: Visual description with Qwen2.5-VL
# ==================================================================
echo "[Stage 1] Generating visual descriptions ..."
python "${STAGE1_SCRIPT}" \
    --dataset        swiss \
    --anno_path      "${ANNO_PATH}" \
    --video_dir      "${VIDEO_DIR}" \
    --save_dir       "${RESULTS_DIR}" \
    --num_frames     16 \
    --batch_size     1 \
    --prompt_idx     0 \
    --seed           42 \
    ${CHARBANK_PATH:+--charbank_path "${CHARBANK_PATH}"}

# Stage 1 saves a CSV in save_dir; rename to our canonical path
# (the exact filename depends on the dataset flag – adjust if needed)
STAGE1_RAW="${RESULTS_DIR}/stage1_swiss.csv"
if [ -f "${STAGE1_RAW}" ]; then
    mv "${STAGE1_RAW}" "${STAGE1_OUTPUT}"
fi

echo ""
echo "[Stage 1] Complete -> ${STAGE1_OUTPUT}"
echo ""

# ==================================================================
# COMMAND 2 – Stage 2: AD generation with Qwen3 + coherence manager
# ==================================================================
echo "[Stage 2] Generating audio descriptions (${LANGUAGE}) ..."
python "${STAGE2_SCRIPT}" \
    --pred_path            "${STAGE1_OUTPUT}" \
    --save_dir             "${STAGE2_SAVE_DIR}" \
    --dataset              swiss \
    --mode                 single \
    --language             "${LANGUAGE}" \
    --few_shot_samples_csv "${FEW_SHOT_CSV}" \
    --LR_intercept         "${LR_INTERCEPT}" \
    --LR_slope             "${LR_SLOPE}" \
    --num_examples         10 \
    --seed                 42

# Rename Stage 2 default output to canonical final path
STAGE2_RAW="${STAGE2_SAVE_DIR}/stage2_qwen3_single.csv"
if [ -f "${STAGE2_RAW}" ]; then
    mv "${STAGE2_RAW}" "${FINAL_OUTPUT}"
fi

echo ""
echo "[Stage 2] Complete -> ${FINAL_OUTPUT}"
echo ""
echo "============================================================"
echo "  Pipeline finished.  Final ADs: ${FINAL_OUTPUT}"
echo "============================================================"
