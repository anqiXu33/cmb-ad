#!/bin/bash
#SBATCH --job-name=oracle_test
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100
#SBATCH --time=1:00:00
#SBATCH --output=phase2_oracle_%j.log

set -eo pipefail

# 1. load necessary modules and activate conda environment
module load miniforge3/25.3.0-3
module load cuda/12.6.3
eval "$(conda shell.bash hook)"
conda activate AD-generation

# 2. Specify the cache path for the Qwen3 model
export HF_HOME="/home/anxu/data/hf_cache"

# 3. Navigate to the project directory
cd /home/anxu/shot-by-shot

# 4. run Oracle Test
echo "Starting Oracle Test with Qwen3-8B..."
python tests/run_oracle_test.py
echo "Test Finished!"