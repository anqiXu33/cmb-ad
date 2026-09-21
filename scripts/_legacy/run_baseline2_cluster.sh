#!/bin/bash
# run_baseline2_cluster.sh
#
# Generates and submits one SLURM job per movie for Baseline 2 (Always-ID).
#
# Baseline 2 is stateless: no GlobalMemoryBank, no narrative context.
# The model is instructed via the system prompt to ALWAYS use explicit
# character names/descriptions and NEVER use pronouns or pro-drop.
#
# Output: results/baseline2/{movie}/stage2_qwen3_always_id.csv
# Logs:   logs/baseline2/log_b2_{movie}_{job_id}.log
#
# Usage (from the repo root on the cluster):
#   bash run_baseline2_cluster.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Movie → Language mapping
# ---------------------------------------------------------------------------
declare -A MOVIE_LANG
MOVIE_LANG["La_Tentazione_Di_Esistere"]="Italian"
MOVIE_LANG["Quanto_basta"]="Italian"
MOVIE_LANG["Stuerm"]="Italian"
MOVIE_LANG["Riders_of_justice"]="Italian"
MOVIE_LANG["8_Tage_im_August"]="German"
MOVIE_LANG["Moskau_einfach"]="German"
MOVIE_LANG["Baghdad_in_my_Shadow"]="German"
MOVIE_LANG["Trommelwirbel"]="German"

# ---------------------------------------------------------------------------
# Create shared directories
# ---------------------------------------------------------------------------
mkdir -p logs/baseline2
mkdir -p scripts/main_runners

# ---------------------------------------------------------------------------
# Generate and submit one SLURM job per movie
# ---------------------------------------------------------------------------
for MOVIE in "${!MOVIE_LANG[@]}"; do
    LANG="${MOVIE_LANG[$MOVIE]}"
    SCRIPT="scripts/main_runners/run_b2_${MOVIE}.sh"

    # Ensure per-movie result directory exists before job submission
    mkdir -p "results/baseline2/${MOVIE}"

    cat > "${SCRIPT}" << HEREDOC
#!/bin/bash
#SBATCH --job-name=B2_${MOVIE}
#SBATCH --constraint="A100|H100|H200"
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/baseline2/log_b2_${MOVIE}_%j.log

set -eo pipefail

mkdir -p results/baseline2/${MOVIE}/

module load miniforge3/25.3.0-3
module load cuda/12.6.3
eval "\$(conda shell.bash hook)"
conda activate AD-generation
export HF_HOME="/home/anxu/data/hf_cache"

cd /home/anxu/shot-by-shot

python stage2/stage2_qwen3_always_id.py \\
    --pred_path /shares/iict-sp4.ebling.cl.uzh/data/anxu_MA_data/stage1/${MOVIE}/swissAD_ads/stage1_qwen2vl.csv \\
    --anno_path /shares/iict-sp4.ebling.cl.uzh/data/anxu_MA_data/annotations/${MOVIE}_context_faces_threads_shot_scales.csv \\
    --language ${LANG} \\
    --batch_size 16 \\
    --save_dir results/baseline2/${MOVIE}/

echo "Done: ${MOVIE} (${LANG}) -> results/baseline2/${MOVIE}/stage2_qwen3_always_id.csv"
HEREDOC

    chmod +x "${SCRIPT}"
    echo "Submitting [${LANG}] ${MOVIE} ..."
    sbatch "${SCRIPT}"

done

echo ""
echo "All 8 Baseline 2 jobs submitted."
echo "Monitor with: squeue -u \$USER"
echo "Results will appear under: results/baseline2/"
