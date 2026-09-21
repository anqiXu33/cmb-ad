"""
Smoke test for CharacterMemoryBank (no LLM / GPU required).

Walks the first 30 rows of both Swiss sample CSVs and exercises the bank's
state machine (get_context / add_ad), using the GT `text` column as a mock
Stage-1 prediction. Each annotation-CSV row is treated as one AD — a
simplification of the production pipeline (which collapses multi-shot rows
per anno_idx) but adequate to surface flag-decision regressions.

Outputs:
    results/test/cmb_trace_de.csv   (German film)
    results/test/cmb_trace_it.csv   (Italian film)
"""

import ast
import os
import sys
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "stage2"))

from coherence_manager import CharacterMemoryBank  # noqa: E402

ANNOTATIONS_DIR = os.path.join(REPO_ROOT, "resources", "annotations")
RESULTS_DIR     = os.path.join(REPO_ROOT, "results", "test")
os.makedirs(RESULTS_DIR, exist_ok=True)

SAMPLES = {
    "de": {
        "csv":  os.path.join(ANNOTATIONS_DIR, "test_sample_swiss_de.csv"),
        "out":  os.path.join(RESULTS_DIR,     "cmb_trace_de.csv"),
        "lang": "German",
    },
    "it": {
        "csv":  os.path.join(ANNOTATIONS_DIR, "test_sample_swiss_it.csv"),
        "out":  os.path.join(RESULTS_DIR,     "cmb_trace_it.csv"),
        "lang": "Italian",
    },
}


def parse_pred_ids(raw) -> list:
    """Stringified dict like "{'nm001': ...}" -> ['nm001']. Empty / NaN -> []."""
    try:
        if isinstance(raw, float) or str(raw).strip() in ("", "{}"):
            return []
        parsed = ast.literal_eval(str(raw))
        if isinstance(parsed, dict):
            return list(parsed.keys())
    except (ValueError, SyntaxError):
        pass
    return []


def run_memory_test(csv_path: str, lang: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path).head(30)
    print(f"\n{'=' * 60}")
    print(f"Language: {lang}  |  Rows to process: {len(df)}")
    print(f"CSV: {csv_path}")
    print(f"{'=' * 60}")

    bank = CharacterMemoryBank()
    records = []

    for ad_idx, (_, row) in enumerate(df.iterrows()):
        text_pred_mock   = str(row["text"])         # GT text used as mock Stage-1 output
        current_char_ids = parse_pred_ids(row.get("pred_ids"))

        char_status, recent_history_str = bank.get_context(
            ad_idx=ad_idx, text_pred=text_pred_mock, char_ids=current_char_ids
        )
        bank.add_ad(text_pred_mock)

        records.append({
            "ad_idx":         ad_idx,
            "anno_idx":       row.get("anno_idx", ""),
            "original_text":  text_pred_mock,
            "parsed_ids":     str(current_char_ids),
            "char_status":    char_status,
            "recent_history": recent_history_str,
        })

        print(f"[{ad_idx:02d}] ids={current_char_ids}")

    return pd.DataFrame(records)


if __name__ == "__main__":
    for code, cfg in SAMPLES.items():
        result_df = run_memory_test(cfg["csv"], cfg["lang"])
        result_df.to_csv(cfg["out"], index=False)
        print(f"\nSaved {len(result_df)} rows -> {cfg['out']}")

    print("\nCMB trace complete. Inspect results/test/ CSVs for state-machine output.")
