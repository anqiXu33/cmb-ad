#!/bin/bash
#SBATCH --job-name=SC_NoCMB_Moskau
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --constraint="A100|H100|H200"
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/ablations/log_sanity_check_baseline_Moskau_%j.log

# Sanity check — baseline condition: Stage 2 with --no_cmb (multi-factor
# stripped baseline). Drops NARRATIVE CONTEXT, PRIORITY rules, and CHARACTER
# REFERENCING RULES; --imdb_chars_path is intentionally not passed because
# stage2/swissAD_qwen3.py ignores it under --no_cmb. Same model + sampling
# as the CMB condition. If CMB cannot beat this stripped baseline, the CMB
# pipeline itself has a bug.

set -eo pipefail
mkdir -p results/sanity_check_baseline/Moskau_einfach/
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
    --no_cmb \
    --LR_slope 2.8 \
    --LR_intercept 3 \
    --few_shot_samples_csv results/_legacy/phase2_oracle/oracle_few_shot_de.csv \
    --save_dir results/sanity_check_baseline/Moskau_einfach/

echo "Done: Moskau_einfach (sanity-check baseline — no CMB)"
