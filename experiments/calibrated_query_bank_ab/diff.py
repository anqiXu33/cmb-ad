"""
Summarize raw-headshot vs calibrated-centroid pred_ids on the same face
detections (output of run_ab.py). Side experiment — see README.md.

Compares three columns when present:
  - pred_ids                (original, from earlier cluster run; may differ
                             due to face-detection variance)
  - pred_ids_raw_rerun      (raw headshot, fresh face detection)
  - pred_ids_calibrated     (calibrated centroid, same fresh face detection)

The (raw_rerun vs calibrated) pair is apples-to-apples: same detected
faces, only the cosine reference differs. That diff is the calibration
effect.

Usage:
  python experiments/calibrated_query_bank_ab/diff.py \
    --csv    experiments/calibrated_query_bank_ab/results/Moskau_einfach/pred_ids_ab.csv \
    --report experiments/calibrated_query_bank_ab/results/Moskau_einfach/diff_report.txt
"""

import argparse
import ast
import pandas as pd
from collections import Counter


def parse_pred_ids(val):
    if pd.isna(val) or val == "" or val == "{}":
        return set()
    try:
        d = ast.literal_eval(val) if isinstance(val, str) else val
        if isinstance(d, dict):
            return {str(k) for k in d.keys()}
    except (ValueError, SyntaxError):
        pass
    return set()


def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def summarize_pair(df, col_a, col_b, label):
    n = len(df)
    n_changed = 0
    n_added = 0  # in B but not A
    n_removed = 0  # in A but not B
    jacc_sum = 0.0
    add_counter = Counter()
    rm_counter = Counter()
    same_count = 0

    for _, row in df.iterrows():
        a = parse_pred_ids(row[col_a])
        b = parse_pred_ids(row[col_b])
        if a == b:
            same_count += 1
        else:
            n_changed += 1
            for x in (b - a):
                n_added += 1
                add_counter[x] += 1
            for x in (a - b):
                n_removed += 1
                rm_counter[x] += 1
        jacc_sum += jaccard(a, b)

    lines = [
        f"=== {label} ({col_a} vs {col_b}) ===",
        f"  total clips:           {n}",
        f"  identical:             {same_count} ({100*same_count/n:.1f}%)",
        f"  changed:               {n_changed} ({100*n_changed/n:.1f}%)",
        f"  IDs added (B \\ A):     {n_added}  (avg {n_added/n:.2f}/clip)",
        f"  IDs removed (A \\ B):   {n_removed}  (avg {n_removed/n:.2f}/clip)",
        f"  mean Jaccard(A,B):     {jacc_sum/n:.3f}",
    ]
    if add_counter:
        top_add = add_counter.most_common(5)
        lines.append(f"  top added IDs:         {top_add}")
    if rm_counter:
        top_rm = rm_counter.most_common(5)
        lines.append(f"  top removed IDs:       {top_rm}")
    return "\n".join(lines)


def per_clip_diff(df, col_a, col_b, max_rows=20):
    rows = []
    for _, row in df.iterrows():
        a = parse_pred_ids(row[col_a])
        b = parse_pred_ids(row[col_b])
        if a == b:
            continue
        rows.append({
            "anno_idx": row.get("anno_idx", ""),
            "start": row.get("start", row.get("AD_start", "")),
            f"{col_a}_only": sorted(a - b),
            f"{col_b}_only": sorted(b - a),
            "common": sorted(a & b),
        })
        if len(rows) >= max_rows:
            break
    return rows


def main(args):
    df = pd.read_csv(args.csv)
    cols = list(df.columns)
    has_orig = "pred_ids" in cols
    has_raw = "pred_ids_raw_rerun" in cols
    has_calib = "pred_ids_calibrated" in cols

    if not (has_raw and has_calib):
        raise SystemExit(
            "[FATAL] need pred_ids_raw_rerun + pred_ids_calibrated columns"
        )

    sections = []

    if has_orig and has_raw:
        sections.append(summarize_pair(
            df, "pred_ids", "pred_ids_raw_rerun",
            "Face-detection variance (original vs fresh raw rerun)"
        ))

    sections.append(summarize_pair(
        df, "pred_ids_raw_rerun", "pred_ids_calibrated",
        "Calibration effect (apples-to-apples: same detected faces)"
    ))

    if has_orig and has_calib:
        sections.append(summarize_pair(
            df, "pred_ids", "pred_ids_calibrated",
            "Original vs calibrated (combined effect: detection + calibration)"
        ))

    sections.append("\n=== Sample of changed clips (raw_rerun vs calibrated) ===")
    samples = per_clip_diff(df, "pred_ids_raw_rerun", "pred_ids_calibrated",
                            max_rows=20)
    if not samples:
        sections.append("  (none — calibration changed nothing)")
    else:
        for r in samples:
            sections.append(
                f"  anno_idx={r['anno_idx']}  start={r['start']}\n"
                f"    raw_only:    {r['pred_ids_raw_rerun_only']}\n"
                f"    calib_only:  {r['pred_ids_calibrated_only']}\n"
                f"    common:      {r['common']}"
            )

    report = "\n\n".join(sections)
    print(report)
    if args.report:
        import os
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w") as f:
            f.write(report + "\n")
        print(f"\n[wrote report to {args.report}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True,
                        help="Output of char_recog/main_ab_compare.py")
    parser.add_argument("--report", default=None,
                        help="Optional: also write report to file")
    args = parser.parse_args()
    main(args)
