#!/bin/bash
#SBATCH --job-name=eval_spacy
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G            
#SBATCH --time=00:10:00      
# 注意：这里我们去掉了 --gres=gpu:1，因为评估不需要显卡，这样排队会秒进！
#SBATCH --output=phase4_eval_%j.log

set -eo pipefail

# 1. load and activate conda environment
module load miniforge3/25.3.0-3
eval "$(conda shell.bash hook)"
conda activate AD-generation

# 2. enter project directory
cd /home/anxu/shot-by-shot

# 3. run evaluation


echo "--- Evaluating German ---"
python evaluate_coherence.py \
    --result_csv results/phase2_oracle_results_de.csv \
    --input_csv results/mock_oracle_input_de.csv \
    --language German

echo -e "\n--- Evaluating Italian ---"
python evaluate_coherence.py \
    --result_csv results/phase2_oracle_results_it.csv \
    --input_csv results/mock_oracle_input_it.csv \
    --language Italian


echo "Evaluation Finished!"

