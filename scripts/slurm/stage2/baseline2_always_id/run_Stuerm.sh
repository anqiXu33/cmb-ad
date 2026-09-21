#!/bin/bash -l
#SBATCH --job-name=B2_Stuerm
#SBATCH --constraint="H100|H200"
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
#SBATCH --time=72:00:00
#SBATCH --output=logs/slurm/stage2/baseline2_always_id/log_b2_Stuerm_%j.log

set -eo pipefail

mkdir -p results/stage2/baseline2_always_id/Stuerm/

module load miniforge3/25.3.0-3
module load cuda/12.6.3
eval "$(conda shell.bash hook)"
conda activate AD-generation
export HF_HOME="/home/anxu/data/hf_cache"

cd /home/anxu/shot-by-shot

python stage2/stage2_qwen3_always_id.py \
    --pred_path /shares/iict-sp4.ebling.cl.uzh/data/anxu_MA_data/stage1/Stuerm/swissAD_ads/stage1_qwen2vl.csv \
    --anno_path /shares/iict-sp4.ebling.cl.uzh/data/anxu_MA_data/annotations/Stuerm_context_faces_threads_shot_scales.csv \
    --language Italian \
    --batch_size 8 \
    --LR_slope 2.8 \
    --LR_intercept 3 \
    --few_shot_samples_csv results/_legacy/phase2_oracle/oracle_few_shot_it.csv \
    --save_dir results/stage2/baseline2_always_id/Stuerm/

echo "Done: Stuerm (Italian) -> results/stage2/baseline2_always_id/Stuerm/stage2.csv"
