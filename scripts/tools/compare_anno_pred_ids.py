#!/usr/bin/env python3
"""
Compare pred_ids quality between two annotation CSVs of the same movie.

Use case: validate that re-running char_recog/main.py with --query_bank_path
(FocusedAD-style calibrated centroids) actually produces better pred_ids than
the original raw-headshot run.

Reports — for each side and the diff:
  • Per-AD detection rate (% ADs with non-empty pred_ids after m-* union)
  • Top-N cast recall (film-level, against imdb_chars JSON)
  • Per-character detection frequency (how many ADs each main actor appears in)
  • Per-AD agreement (% ADs where both sides identify the SAME character set)

Usage
-----
    python scripts/tools/compare_anno_pred_ids.py \\
        --raw      /path/to/Moskau_einfach_context_faces_threads_shot_scales.csv \\
        --calibrated /path/to/Moskau_einfach_..._calibrated.csv \\
        --imdb_chars_path /path/to/imdb_chars_enriched_manual.json \\
        --imdbid tt10949014
"""
import argparse
import ast
import json
import sys

import pandas as pd


def parse_ids_cell(val) -> set:
    """Parse one pred_ids cell into a set of character ID strings."""
    if pd.isna(val):
        return set()
    s = str(val).strip()
    if s in ("", "{}", "{ }", "nan"):
        return set()
    try:
        d = ast.literal_eval(s)
        if isinstance(d, dict):
            return {str(k) for k in d.keys()}
    except (ValueError, SyntaxError):
        pass
    return set()


def m_union_per_ad(df: pd.DataFrame) -> dict:
    """anno_idx → set(character IDs detected across all m-* shots of that AD)."""
    if "shot_label" not in df.columns:
        raise ValueError("annotation CSV missing 'shot_label' column.")
    m_rows = df[df["shot_label"].astype(str).str.startswith("m-", na=False)]
    out = {}
    for anno_idx, g in m_rows.groupby("anno_idx", sort=True):
        keys = set()
        for cell in g["pred_ids"]:
            keys.update(parse_ids_cell(cell))
        out[int(anno_idx)] = keys
    return out


def all_anno_ids(df: pd.DataFrame) -> set:
    """Union of every detected character ID across the whole movie."""
    s = set()
    for cell in df["pred_ids"]:
        s.update(parse_ids_cell(cell))
    return s


def fmt_pct(n: int, d: int) -> str:
    return f"{100*n/d:.1f}%" if d else "N/A"


def section(title: str):
    print("\n" + "═" * 72)
    print(f"  {title}")
    print("═" * 72)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True,
                    help="Path to original (raw-headshot) anno CSV.")
    ap.add_argument("--calibrated", required=True,
                    help="Path to calibrated (query-bank) anno CSV.")
    ap.add_argument("--imdb_chars_path", required=True,
                    help="Path to imdb_chars_enriched_manual.json or similar.")
    ap.add_argument("--imdbid", required=True,
                    help="IMDb ID of the movie (e.g. tt10949014 for Moskau).")
    args = ap.parse_args()

    raw_df = pd.read_csv(args.raw)
    cal_df = pd.read_csv(args.calibrated)
    print(f"raw:        {len(raw_df):>5} rows  ← {args.raw}")
    print(f"calibrated: {len(cal_df):>5} rows  ← {args.calibrated}")

    with open(args.imdb_chars_path) as f:
        chars = json.load(f)
    cast = chars.get(args.imdbid)
    if cast is None:
        print(f"[FATAL] No entry for imdbid={args.imdbid} in {args.imdb_chars_path}")
        sys.exit(1)
    cast_ids   = [str(c["id"]) for c in cast]                       # ordered (top-billed first)
    name_map   = {str(c["id"]): c.get("role", c.get("name", "?")) for c in cast}
    print(f"cast list:  {len(cast_ids)} characters")

    raw_ad_ids = m_union_per_ad(raw_df)
    cal_ad_ids = m_union_per_ad(cal_df)

    common_ad_idx = sorted(set(raw_ad_ids) & set(cal_ad_ids))
    if not common_ad_idx:
        print("[FATAL] No common anno_idx between the two CSVs.")
        sys.exit(1)
    print(f"AD overlap: {len(common_ad_idx)} ADs in both CSVs")

    # ── (1) Per-AD detection rate ────────────────────────────────────────────
    section("(1) Per-AD detection rate (m-* union non-empty)")
    n_raw_nonempty = sum(1 for i in common_ad_idx if raw_ad_ids[i])
    n_cal_nonempty = sum(1 for i in common_ad_idx if cal_ad_ids[i])
    n = len(common_ad_idx)
    print(f"  raw         : {n_raw_nonempty:>4}/{n}  ({fmt_pct(n_raw_nonempty, n)})")
    print(f"  calibrated  : {n_cal_nonempty:>4}/{n}  ({fmt_pct(n_cal_nonempty, n)})")
    delta = n_cal_nonempty - n_raw_nonempty
    print(f"  Δ           : {delta:+d} ADs ({fmt_pct(abs(delta), n)} of total)")

    # ── (2) Film-level cast recall (Top-5, Top-10, all) ──────────────────────
    section("(2) Film-level cast recall (Top-N by billing order)")
    raw_all = all_anno_ids(raw_df)
    cal_all = all_anno_ids(cal_df)

    def recall_at(ids_set, topn):
        target = set(cast_ids[:topn])
        return f"{len(target & ids_set)}/{len(target)}"
    print(f"  {'':<14}{'@top5':>10}{'@top10':>10}{'@all':>10}")
    print(f"  {'raw':<14}{recall_at(raw_all,5):>10}{recall_at(raw_all,10):>10}{recall_at(raw_all,len(cast_ids)):>10}")
    print(f"  {'calibrated':<14}{recall_at(cal_all,5):>10}{recall_at(cal_all,10):>10}{recall_at(cal_all,len(cast_ids)):>10}")

    # Which top-billed actors did calibrated find that raw missed (or vice versa)?
    raw_missed_top10 = set(cast_ids[:10]) - raw_all
    cal_missed_top10 = set(cast_ids[:10]) - cal_all
    new_found = raw_missed_top10 - cal_missed_top10           # raw missed, cal found
    new_lost  = cal_missed_top10 - raw_missed_top10           # cal lost, raw had

    if new_found:
        print(f"  ✚ calibrated FOUND (raw missed):", [name_map[i] for i in new_found])
    if new_lost:
        print(f"  ✗ calibrated LOST  (raw had)  :", [name_map[i] for i in new_lost])
    if not new_found and not new_lost:
        print("  (same Top-10 cast covered by both)")

    # ── (3) Per-character detection frequency ────────────────────────────────
    section("(3) Per-character detection frequency on common ADs (Top-10 only)")
    print(f"  {'character':<28}{'raw':>8}{'cal':>8}{'Δ':>8}")
    print("  " + "─" * 52)
    for cid in cast_ids[:10]:
        raw_count = sum(1 for i in common_ad_idx if cid in raw_ad_ids[i])
        cal_count = sum(1 for i in common_ad_idx if cid in cal_ad_ids[i])
        delta = cal_count - raw_count
        marker = " ⬆" if delta > 0 else (" ⬇" if delta < 0 else "")
        print(f"  {name_map[cid][:26]:<28}{raw_count:>8}{cal_count:>8}{delta:>+8}{marker}")

    # ── (4) Per-AD agreement ─────────────────────────────────────────────────
    section("(4) Per-AD agreement (do raw and calibrated identify the same characters?)")
    n_exact     = sum(1 for i in common_ad_idx if raw_ad_ids[i] == cal_ad_ids[i])
    n_both_emp  = sum(1 for i in common_ad_idx if not raw_ad_ids[i] and not cal_ad_ids[i])
    n_overlap   = sum(1 for i in common_ad_idx
                      if raw_ad_ids[i] and cal_ad_ids[i]
                      and raw_ad_ids[i] & cal_ad_ids[i])
    n_disjoint  = sum(1 for i in common_ad_idx
                      if raw_ad_ids[i] and cal_ad_ids[i]
                      and not (raw_ad_ids[i] & cal_ad_ids[i]))
    n_only_raw  = sum(1 for i in common_ad_idx if raw_ad_ids[i] and not cal_ad_ids[i])
    n_only_cal  = sum(1 for i in common_ad_idx if cal_ad_ids[i] and not raw_ad_ids[i])
    print(f"  exact-match (incl. both empty)        : {n_exact:>4}/{n}  ({fmt_pct(n_exact, n)})")
    print(f"    of which both empty                 : {n_both_emp:>4}")
    print(f"  partial overlap (≥1 shared, not equal): {n_overlap - n_exact + n_both_emp:>4}")
    print(f"  disjoint (both non-empty, no overlap) : {n_disjoint:>4}")
    print(f"  only raw  detected (cal empty)        : {n_only_raw:>4}")
    print(f"  only cal  detected (raw empty)        : {n_only_cal:>4}")

    # ── Verdict ──────────────────────────────────────────────────────────────
    section("VERDICT")
    if n_cal_nonempty > n_raw_nonempty + 0.05 * n:
        print("  ✅ Calibrated query bank improves AD-level detection coverage by >5%.")
    elif n_cal_nonempty < n_raw_nonempty - 0.05 * n:
        print("  ⚠️  Calibrated REDUCED AD-level detection coverage by >5% — investigate.")
    else:
        print("  ◯ Marginal change in AD-level detection coverage (±5%).")
    if n_disjoint > 0.1 * n:
        print(f"  ⚠️  {n_disjoint} ADs ({fmt_pct(n_disjoint, n)}) have completely DISJOINT identifications —")
        print("     calibrated and raw disagree on character identity. Worth manual spot-check.")


if __name__ == "__main__":
    main()
