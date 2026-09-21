#!/bin/bash -l
#SBATCH --job-name=P3_27B_Riders_of_justice
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --constraint="H100|H200"
#SBATCH --time=96:00:00
#SBATCH --output=logs/slurm/stage2/cmb_ad/log_Riders_of_justice_%j.log

set -eo pipefail
mkdir -p results/stage2/cmb_ad/Riders_of_justice/
mkdir -p logs/slurm/stage2/cmb_ad/

module load miniforge3/25.3.0-3
module load cuda/12.6.3
eval "$(conda shell.bash hook)"
conda activate AD-generation
export HF_HOME="/home/anxu/data/hf_cache"

cd /home/anxu/shot-by-shot

python stage2/swissAD_qwen3.py \
    --pred_path /shares/iict-sp4.ebling.cl.uzh/data/anxu_MA_data/stage1/Riders_of_justice/swissAD_ads/stage1_qwen2vl.csv \
    --anno_path /shares/iict-sp4.ebling.cl.uzh/data/anxu_MA_data/annotations/Riders_of_justice_context_faces_threads_shot_scales.csv \
    --language Italian \
    --batch_size 8 \
    --imdb_chars_path /shares/iict-sp4.ebling.cl.uzh/data/anxu_MA_data/annotations/imdb_chars_enriched_manual.json \
    --LR_slope 2.5 \
    --LR_intercept 3 \
    --few_shot_samples_csv results/_legacy/phase2_oracle/oracle_few_shot_it.csv \
    --save_dir results/stage2/cmb_ad/Riders_of_justice/

# Rename script default output to eval-pipeline expected name.
mv results/stage2/cmb_ad/Riders_of_justice/stage2_qwen3_single.csv \
   results/stage2/cmb_ad/Riders_of_justice/stage2.csv

echo "Done: Riders_of_justice"
