#!/bin/bash -l
#SBATCH --job-name=pip_freeze_repro
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
#SBATCH --constraint="H100|H200"
#SBATCH --time=00:10:00
#SBATCH --output=logs/slurm/postprocess/pip_freeze_%j.log

# Dumps the AD-generation conda env that produced the 24 Stage-2 jobs to
# requirements.lock.txt at repo root, plus repro_meta.txt with python /
# CUDA / torch / transformers / flash-attn / GPU / OS context for the
# thesis reproducibility appendix.
#
# Requests 1 GPU (H100|H200) so torch.cuda + nvidia-smi reflect the same
# hardware class that actually ran Stage-2; pip freeze itself is CPU-only.

set -eo pipefail
mkdir -p logs/slurm/postprocess/

module load miniforge3/25.3.0-3
module load cuda/12.6.3
eval "$(conda shell.bash hook)"
conda activate AD-generation

cd /home/anxu/shot-by-shot

STAMP=$(date -u +%Y%m%dT%H%M%SZ)

pip freeze > requirements.lock.txt
echo "[$STAMP] pip freeze: $(wc -l < requirements.lock.txt) packages -> requirements.lock.txt"

{
  echo "# Reproducibility metadata"
  echo "# UTC:  $STAMP"
  echo "# Job:  $SLURM_JOB_ID"
  echo "# Node: $(hostname)"
  echo
  echo "## Python"
  python --version
  pip --version
  echo
  echo "## Key package versions (queried at import time)"
  python - <<'PY'
import importlib
for name in ("torch", "transformers", "tokenizers", "accelerate",
             "flash_attn", "bitsandbytes", "sentence_transformers",
             "spacy", "sacrebleu", "bert_score", "evaluate", "nltk",
             "insightface", "decord", "scenedetect", "PIL", "openpyxl",
             "num2words", "pandas", "numpy"):
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", "<no __version__>")
        print(f"{name:24s} {ver}")
    except Exception as e:
        print(f"{name:24s} <import failed: {type(e).__name__}>")

import torch
print()
print(f"torch.version.cuda          {torch.version.cuda}")
print(f"torch.cuda.is_available()   {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"torch.cuda.device_count()   {torch.cuda.device_count()}")
    print(f"torch.cuda.get_device_name  {torch.cuda.get_device_name(0)}")

# spaCy pipelines actually loaded by evaluate_coherence.py
try:
    import spacy
    for pkg in ("de_core_news_sm", "it_core_news_sm",
                "de_core_news_lg", "en_core_web_lg"):
        try:
            nlp = spacy.load(pkg)
            print(f"spacy[{pkg:18s}] {nlp.meta.get('version', '?')}")
        except Exception as e:
            print(f"spacy[{pkg:18s}] <not installed>")
except Exception:
    pass
PY
  echo
  echo "## CUDA / driver"
  nvcc --version 2>/dev/null || echo "nvcc not on PATH"
  echo
  nvidia-smi 2>/dev/null | head -20 || echo "nvidia-smi not available"
  echo
  echo "## Loaded modules"
  module list 2>&1 || true
  echo
  echo "## OS"
  uname -a
  cat /etc/os-release 2>/dev/null || true
} > repro_meta.txt

echo "Wrote requirements.lock.txt + repro_meta.txt at repo root."
ls -l requirements.lock.txt repro_meta.txt
