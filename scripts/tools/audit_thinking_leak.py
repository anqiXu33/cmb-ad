"""Audit Stage-2 CSVs for `<think>` leakage in the `text_gen` column.

Background: when Qwen3.5-27B's thinking block is truncated by max_new_tokens,
the old `_extract_ad` parser fell through and wrote the raw `<think>...`
reasoning into `text_gen`. This script scans output CSVs and reports which
rows are polluted so you can decide what to re-run.

Usage:
  python scripts/tools/audit_thinking_leak.py results/ablations/sanity_check_B/
  python scripts/tools/audit_thinking_leak.py results/ --list-rows
  python scripts/tools/audit_thinking_leak.py path/to/file.csv path/to/other.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd


THINK_OPEN = "<think>"


def audit_file(path: Path, list_rows: bool):
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[skip] {path}: {e}")
        return None

    if "text_gen" not in df.columns:
        return None

    text = df["text_gen"].fillna("").astype(str)
    mask = text.str.contains(THINK_OPEN, regex=False)
    n_bad = int(mask.sum())
    n_total = len(df)
    pct = 100.0 * n_bad / n_total if n_total else 0.0

    if n_bad == 0:
        return (n_total, 0, 0.0, [])

    bad_idx = []
    if "anno_idx" in df.columns:
        bad_idx = df.loc[mask, "anno_idx"].tolist()
    else:
        bad_idx = df.index[mask].tolist()

    if list_rows:
        idx_str = ", ".join(str(i) for i in bad_idx)
        print(f"  affected anno_idx: [{idx_str}]")

    return (n_total, n_bad, pct, bad_idx)


def gather_csvs(targets):
    csvs = []
    for t in targets:
        p = Path(t)
        if p.is_file() and p.suffix == ".csv":
            csvs.append(p)
        elif p.is_dir():
            csvs.extend(sorted(p.rglob("*.csv")))
        else:
            print(f"[warn] not found: {t}", file=sys.stderr)
    return csvs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="+", help="CSV files or directories to scan")
    ap.add_argument("--list-rows", action="store_true", help="Print affected anno_idx per file")
    ap.add_argument("--only-bad", action="store_true", help="Only print files with leakage")
    args = ap.parse_args()

    csvs = gather_csvs(args.targets)
    if not csvs:
        print("No CSV files found.")
        return

    grand_total = 0
    grand_bad = 0
    bad_files = 0

    print(f"{'rows_bad':>8} / {'rows':>5}  {'pct':>6}  file")
    print("-" * 80)

    for path in csvs:
        result = audit_file(path, args.list_rows)
        if result is None:
            continue
        n_total, n_bad, pct, _ = result
        grand_total += n_total
        if n_bad > 0:
            bad_files += 1
            grand_bad += n_bad
            print(f"{n_bad:>8} / {n_total:>5}  {pct:>5.1f}%  {path}")
        elif not args.only_bad:
            print(f"{0:>8} / {n_total:>5}  {0.0:>5.1f}%  {path}")

    print("-" * 80)
    overall_pct = 100.0 * grand_bad / grand_total if grand_total else 0.0
    print(f"TOTAL: {grand_bad} polluted / {grand_total} rows  ({overall_pct:.2f}%)  across {bad_files} file(s) with leakage")


if __name__ == "__main__":
    main()
