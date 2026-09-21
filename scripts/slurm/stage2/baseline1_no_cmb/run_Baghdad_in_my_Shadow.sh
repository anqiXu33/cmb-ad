#!/bin/bash -l
#SBATCH --job-name=B1_NoCMB_Baghdad_in_my_Shadow
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --constraint="H100|H200"
#SBATCH --time=72:00:00
#SBATCH --output=logs/slurm/stage2/baseline1_no_cmb/log_Baghdad_in_my_Shadow_%j.log

set -eo pipefail
mkdir -p results/stage2/baseline1_no_cmb/Baghdad_in_my_Shadow/
mkdir -p logs/slurm/stage2/baseline1_no_cmb/

module load miniforge3/25.3.0-3
module load cuda/12.6.3
eval "$(conda shell.bash hook)"
conda activate AD-generation
export HF_HOME="/home/anxu/data/hf_cache"

cd /home/anxu/shot-by-shot

python stage2/swissAD_qwen3.py \
    --pred_path /shares/iict-sp4.ebling.cl.uzh/data/anxu_MA_data/stage1/Baghdad_in_my_Shadow/swissAD_ads/stage1_qwen2vl.csv \
    --anno_path /shares/iict-sp4.ebling.cl.uzh/data/anxu_MA_data/annotations/Baghdad_in_my_Shadow_context_faces_threads_shot_scales.csv \
    --language German \
    --batch_size 8 \
    --no_cmb \
    --LR_slope 2.8 \
    --LR_intercept 3 \
    --few_shot_samples_csv results/_legacy/phase2_oracle/oracle_few_shot_de.csv \
    --save_dir results/stage2/baseline1_no_cmb/Baghdad_in_my_Shadow/

# Rename script default output to eval-pipeline expected name.
mv results/stage2/baseline1_no_cmb/Baghdad_in_my_Shadow/stage2_qwen3_single.csv \
   results/stage2/baseline1_no_cmb/Baghdad_in_my_Shadow/stage2.csv

echo "Done: Baghdad_in_my_Shadow (B1 — No CMB)"
