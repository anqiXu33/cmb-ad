#!/usr/bin/env python3
"""
stage2_csv_to_mention_json.py
=============================
Convert a Stage-2 output CSV into the JSON input format expected by
``character_chains_qwen3_json.py`` (the cluster mention-extraction script).

Two JSON files are emitted side-by-side, both preserving anno_idx order:

  - ``text_gen.json`` — system-generated AD text (for mention-level evaluation)
  - ``text_gt.json``  — ground-truth AD text (for sanity-checking the extractor
                         against the cluster's pre-computed
                         ``mentions_output/.../<movie>.mentions``; the two
                         should agree if extraction is reproducible)

Output JSON shape (each entry):

    {"anno_idx": int, "source": str}

The extraction script reads only the ``source`` field; ``anno_idx`` is
preserved so downstream metric scripts can join the resulting ``.mentions``
file (one JSON list per line, in input order) back to the original CSV row.

Usage
-----
    python scripts/tools/stage2_csv_to_mention_json.py \\
        --stage2_csv $DATA_ROOT/stage2/cmb_ad/Moskau_einfach/stage2.csv \\
        --output_dir $DATA_ROOT/mention_robustness/Moskau_einfach__cmb_ad/

This script has no GPU dependency — runs on a login node in <1 second per CSV.
"""

import argparse
import json
import os
import sys

import pandas as pd

# Generation-column aliases used across Stage-2 scripts. Ordered by preference.
_GEN_COL_ALIASES = ("output_ads", "text_gen", "generated_ad", "summarized_AD")


def _resolve_gen_col(df: pd.DataFrame, csv_path: str) -> str:
    """Return the first matching generation column or raise."""
    for alias in _GEN_COL_ALIASES:
        if alias in df.columns:
            return alias
    raise ValueError(
        f"{csv_path}: no generation column found; expected one of "
        f"{_GEN_COL_ALIASES}, got {list(df.columns)}"
    )


def _build_entries(df: pd.DataFrame, text_col: str) -> list[dict]:
    """Build [{anno_idx, source}, ...] in stable anno_idx order."""
    return [
        {
            "anno_idx": int(row["anno_idx"]),
            "source": str(row[text_col]) if pd.notna(row[text_col]) else "",
        }
        for _, row in df.iterrows()
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage2_csv", required=True,
                    help="Path to a Stage-2 output CSV.")
    ap.add_argument("--output_dir", required=True,
                    help="Directory to write text_gen.json and text_gt.json.")
    ap.add_argument("--skip_gt", action="store_true",
                    help="Skip writing text_gt.json (only emit text_gen.json).")
    args = ap.parse_args()

    if not os.path.exists(args.stage2_csv):
        print(f"[ERROR] Not found: {args.stage2_csv}")
        sys.exit(1)

    df = pd.read_csv(args.stage2_csv)
    if "anno_idx" not in df.columns:
        print(f"[ERROR] {args.stage2_csv}: missing 'anno_idx' column.")
        sys.exit(1)

    gen_col = _resolve_gen_col(df, args.stage2_csv)
    df = df.sort_values("anno_idx").reset_index(drop=True)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── text_gen.json (always written) ──────────────────────────────────────
    gen_entries = _build_entries(df, gen_col)
    gen_path = os.path.join(args.output_dir, "text_gen.json")
    with open(gen_path, "w", encoding="utf-8") as f:
        json.dump(gen_entries, f, ensure_ascii=False, indent=2)
    print(f"  Wrote {len(gen_entries)} entries → {gen_path}  (column: {gen_col})")

    # ── text_gt.json (optional) ─────────────────────────────────────────────
    if not args.skip_gt:
        if "text_gt" not in df.columns:
            print(f"  [WARN] No 'text_gt' column — skipping text_gt.json.")
        else:
            gt_entries = _build_entries(df, "text_gt")
            gt_path = os.path.join(args.output_dir, "text_gt.json")
            with open(gt_path, "w", encoding="utf-8") as f:
                json.dump(gt_entries, f, ensure_ascii=False, indent=2)
            print(f"  Wrote {len(gt_entries)} entries → {gt_path}  (column: text_gt)")


if __name__ == "__main__":
    main()
