#!/bin/bash
#SBATCH --job-name=C_Moskau_Manual
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --constraint="A100|H100|H200"
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/ablations/log_C_Moskau_%j.log

# Sanity-check condition C (Manual): Stage 2 on Moskau_einfach with the
# manually-curated descriptions (imdb_chars_enriched_manual.json).
# 14 manual entries override VLM descriptions across 5 films; 2 of them
# are in this film (Viktor Schuler, Reto). All other characters keep VLM text.

set -eo pipefail
mkdir -p results/ablations/sanity_check_C_manual/Moskau_einfach/
mkdir -p logs/slurm/ablations/

module load miniforge3/25.3.0-3
module load cuda/12.6.3
eval "$(conda shell.bash hook)"
conda activate AD-generation
export HF_HOME="/home/anxu/data/hf_cache"

cd /home/anxu/shot-by-shot

python stage2/swissAD_qwen3.py \
    --pred_path /shares/iict-sp4.ebling.cl.uzh/data/anxu_MA_data/stage1/Moskau_einfach/swissAD_ads/stage1_qwen2vl.csv \
    --anno_path /shares/iict-sp4.ebling.cl.uzh/data/anxu_MA_data/annotations/Moskau_einfach_context_faces_threads_shot_scales.csv \
    --language German \
    --batch_size 8 \
    --imdb_chars_path /shares/iict-sp4.ebling.cl.uzh/data/anxu_MA_data/annotations/imdb_chars_enriched_manual.json \
    --LR_slope 2.8 \
    --LR_intercept 3 \
    --few_shot_samples_csv results/_legacy/phase2_oracle/oracle_few_shot_de.csv \
    --save_dir results/ablations/sanity_check_C_manual/Moskau_einfach/

echo "Done: Moskau_einfach (C — Manual)"
