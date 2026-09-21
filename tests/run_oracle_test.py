"""
Phase 2 – Oracle Test (LLM Prompt Compliance via swissAD_qwen3.py).

Run this script on the Cluster (GPU required for Qwen3-8B).

Strategy:
  - Uses ground-truth text as a perfect mock Stage-1 prediction (oracle input)
    so any coherence failures are attributable to the LLM prompt, not to
    noisy visual predictions from Stage 1.
  - Runs swissAD_qwen3.py in single-sentence mode for both German and Italian.

Input files (pre-built locally, sync to cluster before running):
    results/mock_oracle_input_de.csv     <- text_gen = GT text  (German)
    results/mock_oracle_input_it.csv     <- text_gen = GT text  (Italian)
    results/oracle_few_shot_de.csv       <- few-shot pool (German,  rows 30-50)
    results/oracle_few_shot_it.csv       <- few-shot pool (Italian, rows 30-50)

Outputs (written by swissAD_qwen3.py, then renamed):
    results/phase2_oracle_results_de.csv
    results/phase2_oracle_results_it.csv

Usage (on Cluster):
    cd /path/to/shot-by-shot-char-tracking
    python tests/run_oracle_test.py
"""

import os
import sys
import subprocess

# ------------------------------------------------------------------
# Resolve repo root (one level above this script's directory)
# ------------------------------------------------------------------
REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS    = os.path.join(REPO_ROOT, "results")
STAGE2_DIR = os.path.join(REPO_ROOT, "stage2")
SCRIPT     = os.path.join(STAGE2_DIR, "swissAD_qwen3.py")

# ------------------------------------------------------------------
# Per-language configuration
# ------------------------------------------------------------------
RUNS = [
    {
        "language":       "German",
        "pred_path":      os.path.join(RESULTS, "mock_oracle_input_de.csv"),
        "few_shot_csv":   os.path.join(RESULTS, "oracle_few_shot_de.csv"),
        "default_out":    os.path.join(RESULTS, "stage2_qwen3_single.csv"),
        "final_out":      os.path.join(RESULTS, "phase2_oracle_results_de.csv"),
    },
    {
        "language":       "Italian",
        "pred_path":      os.path.join(RESULTS, "mock_oracle_input_it.csv"),
        "few_shot_csv":   os.path.join(RESULTS, "oracle_few_shot_it.csv"),
        "default_out":    os.path.join(RESULTS, "stage2_qwen3_single.csv"),
        "final_out":      os.path.join(RESULTS, "phase2_oracle_results_it.csv"),
    },
]

# ------------------------------------------------------------------
# Shared generation hyper-parameters
# Word-count estimate: rough_num_words = LR_INTERCEPT + LR_SLOPE * duration
# Adjust LR_SLOPE/LR_INTERCEPT to match your training-set statistics.
# ------------------------------------------------------------------
LR_INTERCEPT = "0.0"
LR_SLOPE     = "2.0"   # ~2 words per second
NUM_EXAMPLES = "10"
SEED         = "42"


def run_qwen3(cfg: dict) -> None:
    lang = cfg["language"]
    print(f"\n{'='*60}")
    print(f"  Phase 2 Oracle Test – {lang}")
    print(f"{'='*60}")
    print(f"  pred_path    : {cfg['pred_path']}")
    print(f"  few_shot_csv : {cfg['few_shot_csv']}")
    print(f"  final output : {cfg['final_out']}")
    print()

    # Validate input files
    for key in ("pred_path", "few_shot_csv"):
        if not os.path.exists(cfg[key]):
            print(f"[ERROR] Missing input file: {cfg[key]}")
            print("  Make sure you synced results/ from your local machine.")
            sys.exit(1)

    cmd = [
        sys.executable, SCRIPT,
        "--pred_path",            cfg["pred_path"],
        "--save_dir",             RESULTS,
        "--dataset",              "swiss",
        "--mode",                 "single",
        "--language",             lang,
        "--few_shot_samples_csv", cfg["few_shot_csv"],
        "--LR_intercept",         LR_INTERCEPT,
        "--LR_slope",             LR_SLOPE,
        "--num_examples",         NUM_EXAMPLES,
        "--seed",                 SEED,
    ]

    print(f"  Running command:\n  {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=STAGE2_DIR)

    if result.returncode != 0:
        print(f"\n[ERROR] swissAD_qwen3.py exited with code {result.returncode}")
        sys.exit(result.returncode)

    # Rename the default output file to the language-specific path
    if os.path.exists(cfg["default_out"]):
        os.rename(cfg["default_out"], cfg["final_out"])
        print(f"\n  ✓ Results saved -> {cfg['final_out']}")
    else:
        print(f"\n[WARN] Expected output not found at {cfg['default_out']}")
        print("  Check save_dir and the script's output filename.")


if __name__ == "__main__":
    os.makedirs(RESULTS, exist_ok=True)

    for cfg in RUNS:
        run_qwen3(cfg)

    print("\n" + "="*60)
    print("  Phase 2 complete.")
    print(f"  German output  : {RUNS[0]['final_out']}")
    print(f"  Italian output : {RUNS[1]['final_out']}")
    print("="*60)
