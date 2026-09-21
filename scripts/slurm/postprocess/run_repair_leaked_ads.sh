#!/bin/bash
#SBATCH --job-name=repair_leaked_ads
#SBATCH --constraint="H100|H200"
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/postprocess/log_repair_leaked_ads_%j.log

# Post-run repair pass for the residual thinking-leak class in Stage-2 outputs.
# Loads Qwen3.5-27B once and re-generates flagged anno_idx across all 24
# main-run CSVs with thinking disabled.  Writes <stage2.csv>.repaired
# alongside each input — original is left untouched.
#
# Estimated cost: ~10 min model load + ~1-3 min per leaked AD (no thinking,
# max_new_tokens=512).  At observed ~1-2% leak rate × 4911 ADs × 24 CSVs ≈
# 50-100 total leaks, total walltime ≈ 30-60 min.  4h walltime is generous.

set -eo pipefail
mkdir -p logs/slurm/postprocess/

module load miniforge3/25.3.0-3
module load cuda/12.6.3
eval "$(conda shell.bash hook)"
conda activate AD-generation
export HF_HOME="/home/anxu/data/hf_cache"

cd /home/anxu/shot-by-shot

# --batch_thesis processes all 24 main-run CSVs (3 conditions x 8 films).
# Re-runs are skipped automatically for CSVs that don't exist (jobs still
# pending) and for CSVs with zero leaks.
python scripts/tools/repair_leaked_ads.py --batch_thesis

echo "Done: repair_leaked_ads"
