#!/bin/bash -l
#SBATCH --job-name=eval_all_models
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="A100|H100|H200"
#SBATCH --time=01:30:00
#SBATCH --output=/shares/iict-sp4.ebling.cl.uzh/data/anxu_MA_data/logs/evaluate_all_models_%j.log

###############################################################################
# Re-run scripts/eval/evaluate_all_models.py on the full 24-pair thesis matrix
# (3 system conditions × 8 films + Human GT row per film).
#
# The script computes FOUR metrics per (movie, condition):
#
#   - BLEU             (sacreBLEU corpus, 0–100)         — §6.1 main report
#   - CHRF             (sacreBLEU corpus chrF, 0–100)    — computed, NOT reported
#   - BERTScore F1     (lang-aware multilingual model)   — §6.1 main report
#   - Coherence_Score  (DE re-intro / IT pro-drop AD-level)
#                                                          — legacy L2; not used
#                                                            in §6 main table
#
# Per locked §5.3 design (2026-05-14): only BLEU + BERTScore are reported in
# the thesis §6.1.  CHRF + Coherence_Score remain in the CSV as silent extras
# (no cost to keep; useful if a reviewer requests follow-up).
#
# Output  : results/benchmarks/thesis_final_results.csv  (overwrites the
#           stale March-2026 version that still labels CMB-AD as "GMB-AD").
#
# Resource estimate
# -----------------
#   Loads BERTScore multilingual model once (~30s) + spaCy DE/IT (~5s each).
#   Per-(movie × condition): BLEU + CHRF ≈ <1s; spaCy parse ≈ 30s; BERTScore
#   ≈ 30–60s on GPU.  24 conditions × ~60s ≈ 25 min main loop + ~5 min loads.
#   1.5 h walltime is generous.
#
# Submit:
#   cd ~/shot-by-shot
#   sbatch scripts/slurm/postprocess/run_evaluate_all_models.sh
#
# Monitor:
#   squeue -u $USER --name=eval_all_models
#   tail -f /shares/iict-sp4.ebling.cl.uzh/data/anxu_MA_data/logs/evaluate_all_models_<jobid>.log
#
# After completion, sync the output CSV to your local repo:
#   scp anxu@<cluster>:~/shot-by-shot/results/benchmarks/thesis_final_results.csv \
#       ~/Documents/GOODGOODSTUDY/UZH/Master_thesis/shot-by-shot/results/benchmarks/
###############################################################################

set -eo pipefail

module load miniforge3/25.3.0-3
module load cuda/12.6.3
eval "$(conda shell.bash hook)"
conda activate AD-generation
export HF_HOME="/home/anxu/data/hf_cache"
mkdir -p /shares/iict-sp4.ebling.cl.uzh/data/anxu_MA_data/logs

cd /home/anxu/shot-by-shot

echo "=========================================="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Repo:      $(pwd)"
echo "Branch:    $(git rev-parse --abbrev-ref HEAD)"
echo "Commit:    $(git rev-parse --short HEAD)"
echo "=========================================="

# Pre-flight: spaCy DE/IT models present?
echo
echo ">>> Pre-flight: spaCy model check"
python - <<'PY'
import spacy, sys
ok = True
for m in ("de_core_news_sm", "it_core_news_sm"):
    try:
        spacy.load(m)
        print(f"  ✅ {m}")
    except OSError:
        print(f"  ❌ {m} — install with: python -m spacy download {m}")
        ok = False
sys.exit(0 if ok else 1)
PY

echo
echo ">>> Running evaluate_all_models.py"
python scripts/eval/evaluate_all_models.py

echo
echo "=========================================="
echo "Done.  Output:"
ls -la results/benchmarks/thesis_final_results.csv
echo "=========================================="
