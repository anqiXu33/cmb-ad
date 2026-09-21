#!/bin/bash
#SBATCH --job-name=B_Baghdad
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --constraint="A100|H100|H200"
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/ablations/sanity_check_B/log_Baghdad_%j.log

# Sanity-check condition B: no --imdb_chars_path → CharacterMemoryBank runs without
# B condition keeps names+gender; CMB_DISABLE_DESC_HINT=1 strips per-character "appearance" hints.

set -eo pipefail
mkdir -p results/ablations/sanity_check_B/Baghdad_in_my_Shadow/
mkdir -p logs/slurm/ablations/sanity_check_B/

module load miniforge3/25.3.0-3
module load cuda/12.6.3
eval "$(conda shell.bash hook)"
conda activate AD-generation
export HF_HOME="/home/anxu/data/hf_cache"

# B condition: load names+gender (so pronoun-ambiguity stays gender-aware)
# but suppress the per-character (appearance: "...") description hint.
export CMB_DISABLE_DESC_HINT=1

cd /home/anxu/shot-by-shot

python stage2/swissAD_qwen3.py \
    --pred_path /shares/iict-sp4.ebling.cl.uzh/data/anxu_MA_data/stage1/Baghdad_in_my_Shadow/swissAD_ads/stage1_qwen2vl.csv \
    --anno_path /shares/iict-sp4.ebling.cl.uzh/data/anxu_MA_data/annotations/Baghdad_in_my_Shadow_context_faces_threads_shot_scales.csv \
    --language German \
    --batch_size 8 \
    --imdb_chars_path /shares/iict-sp4.ebling.cl.uzh/data/anxu_MA_data/annotations/imdb_chars_enriched.json \
    --LR_slope 2.8 \
    --LR_intercept 3 \
    --few_shot_samples_csv results/_legacy/phase2_oracle/oracle_few_shot_de.csv \
    --save_dir results/ablations/sanity_check_B/Baghdad_in_my_Shadow/

echo "Done: Baghdad_in_my_Shadow (B)"
