#!/bin/bash
#SBATCH --job-name=SC_NoCMB_smoke
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --constraint="A100|H100|H200"
#SBATCH --time=05:00:00
#SBATCH --output=logs/slurm/ablations/log_sanity_check_baseline_smoke_Moskau_%j.log

# Smoke test — baseline (--no_cmb) condition on first 40 ADs of
# Moskau_einfach. Same model + sampling as the CMB smoke; CMB scaffolding
# stripped from the prompt. Pair with sanity_check_cmb_smoke.sh for a
# qualitative side-by-side without waiting 48h for the full sanity check.

set -eo pipefail
mkdir -p results/sanity_check_baseline_smoke/Moskau_einfach/
mkdir -p logs/slurm/ablations/

module load miniforge3/25.3.0-3
module load cuda/12.6.3
eval "$(conda shell.bash hook)"
conda activate AD-generation
export HF_HOME="/home/anxu/data/hf_cache"

cd /home/anxu/shot-by-shot

# Slice first 40 ADs from Stage-1 CSV (idempotent — see sibling script).
SRC=/shares/iict-sp4.ebling.cl.uzh/data/anxu_MA_data/stage1/Moskau_einfach/swissAD_ads/stage1_qwen2vl.csv
mkdir -p results/_smoke/Moskau_einfach
DST=results/_smoke/Moskau_einfach/stage1_qwen2vl_smoke.csv
python -c "
import pandas as pd
df = pd.read_csv('$SRC')
n = 40
df.iloc[:n].to_csv('$DST', index=False)
print(f'sliced: {len(df)} -> {n} rows -> $DST')
"

python stage2/swissAD_qwen3.py \
    --pred_path $DST \
    --anno_path /shares/iict-sp4.ebling.cl.uzh/data/anxu_MA_data/annotations/Moskau_einfach_context_faces_threads_shot_scales.csv \
    --language German \
    --batch_size 8 \
    --no_cmb \
    --LR_slope 2.8 \
    --LR_intercept 3 \
    --few_shot_samples_csv results/_legacy/phase2_oracle/oracle_few_shot_de.csv \
    --save_dir results/sanity_check_baseline_smoke/Moskau_einfach/

echo "Done: Moskau_einfach smoke (sanity-check baseline — no CMB, 40 ADs)"
