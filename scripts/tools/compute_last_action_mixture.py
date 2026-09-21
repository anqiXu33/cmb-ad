"""
compute_last_action_mixture.py — what % of `Last known action:` emissions in
CMB-AD prompts carry Stage-1 proxy content vs Stage-2 sync'd content?

Context: max_chars=150 safety claim in §4.7 footnote.
- compute_truncation_safety.py measures the GT (≈Stage-2 sync'd) distribution.
- But coherence_manager.last_action is briefly a Stage-1 text_pred proxy during
  chunked inference, until sync_last_actions() overwrites it at chunk end.
- If a character reappears within the same chunk with gap > 1, the prompt
  emits `Last known action: "..."` using the Stage-1 proxy.

This script CPU-replays the chunk loop on the 8 thesis films' anno CSVs and
classifies every `Last known action:` emission as `stage1_proxy` or
`stage2_synced`. The pooled stage1_proxy rate is the headline number:

  < 1%  → Stage-1 distribution is empirically irrelevant; GT data is enough
  >= 5% → Stage-1 truncation behavior must be measured separately

Replay logic (mirrors swissAD_qwen3.py's chunk loop):
  For each chunk of `batch_size` consecutive ADs:
    For each AD in chunk:
      For each char_id appearing in this AD (m-* shot union of pred_ids):
        if char was seen before AND gap > 1:
          record (movie, ad_idx, char_id, gap, current_source)
        mark char's source := "stage1_proxy"  (get_context overwrites)
        update last_seen_ad_idx[char] := ad_idx
    At chunk end: for every char who appeared in this chunk,
      mark source := "stage2_synced"  (sync_last_actions overwrites)

Outputs (next to --output_dir):
  last_action_mixture_per_emission.csv  — one row per Last known action: event
  last_action_mixture_summary.csv       — pooled + per-movie counts
"""

import argparse
import ast
import csv
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


DEFAULT_BATCH_SIZE = 8
THESIS_FILMS = [
    "8_Tage_im_August", "Moskau_einfach", "Baghdad_in_my_Shadow", "Trommelwirbel",
    "La_Tentazione_Di_Esistere", "Quanto_basta", "Riders_of_justice", "Stuerm",
]


def _parse_keys(raw):
    if isinstance(raw, float) and pd.isna(raw):
        return []
    s = str(raw).strip()
    if s in ("", "{}", "nan"):
        return []
    try:
        d = ast.literal_eval(s)
        return list(d.keys()) if isinstance(d, dict) else []
    except (ValueError, SyntaxError):
        return []


def m_star_union(anno_df: pd.DataFrame) -> dict:
    """{anno_idx: [char_id, ...]} — mirrors swissAD_qwen3._union_pred_ids_dicts."""
    if "shot_label" not in anno_df.columns or "pred_ids" not in anno_df.columns:
        raise SystemExit("anno CSV missing shot_label/pred_ids columns")
    m_rows = anno_df[anno_df["shot_label"].astype(str).str.startswith("m-", na=False)]
    result = {}
    for ad_idx, sub in m_rows.groupby("anno_idx"):
        keys = set()
        for raw in sub["pred_ids"]:
            keys.update(_parse_keys(raw))
        result[ad_idx] = sorted(keys, key=str)
    return result


def replay_movie(anno_path: Path, batch_size: int, movie_name: str):
    """Return list of emission dicts."""
    anno_df = pd.read_csv(anno_path)
    union_map = m_star_union(anno_df)
    sorted_ad_idxs = sorted(union_map.keys())

    last_seen   = {}    # char_id -> ad_idx
    src         = {}    # char_id -> "stage1_proxy" | "stage2_synced"
    emissions   = []
    n_ads_with_chars = 0

    for chunk_start in range(0, len(sorted_ad_idxs), batch_size):
        chunk_ad_idxs = sorted_ad_idxs[chunk_start: chunk_start + batch_size]
        chunk_chars_seen = set()    # for the sync at chunk end

        for ad_idx in chunk_ad_idxs:
            char_ids = union_map.get(ad_idx, [])
            if char_ids:
                n_ads_with_chars += 1
            for char_id in char_ids:
                chunk_chars_seen.add(char_id)
                if char_id in last_seen:
                    gap = ad_idx - last_seen[char_id]
                    if gap > 1:
                        emissions.append({
                            "movie":    movie_name,
                            "ad_idx":   ad_idx,
                            "char_id":  char_id,
                            "gap":      gap,
                            "source":   src.get(char_id, "stage2_synced"),
                            "chunk_start": chunk_start,
                        })
                # get_context overwrites with Stage-1 proxy
                last_seen[char_id] = ad_idx
                src[char_id] = "stage1_proxy"

        # End of chunk: sync_last_actions overwrites for every char who appeared
        for char_id in chunk_chars_seen:
            src[char_id] = "stage2_synced"

    return emissions, n_ads_with_chars, len(sorted_ad_idxs)


def discover_anno_csvs(anno_dir: Path):
    """Match the 8 thesis films by exact *_context_faces_threads_shot_scales.csv suffix
    (the final pipeline output; CLAUDE.md §"Key Data Contracts")."""
    out = {}
    for film in THESIS_FILMS:
        candidate = anno_dir / f"{film}_context_faces_threads_shot_scales.csv"
        if candidate.exists():
            out[film] = candidate
    missing = [f for f in THESIS_FILMS if f not in out]
    if missing:
        print(f"  [WARN] no anno CSV found for: {', '.join(missing)}")
    return out


def write_csv(path: Path, rows: list):
    if not rows:
        print(f"  (no rows for {path.name})")
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"  Wrote {len(rows):>6d} rows -> {path}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--anno_dir", type=Path, required=True,
                    help="Directory containing the 8 thesis films' "
                         "<movie>_*_context_faces_threads_shot_scales.csv files.")
    ap.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
                    help=f"Chunk size in swissAD_qwen3.py (default {DEFAULT_BATCH_SIZE}).")
    ap.add_argument("--output_dir", type=Path, default=Path("."),
                    help="Output directory (default: current dir).")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    anno_csvs = discover_anno_csvs(args.anno_dir)
    if not anno_csvs:
        raise SystemExit(f"No anno CSVs found in {args.anno_dir}")

    all_emissions = []
    per_movie_rows = []
    pooled_counts = Counter()

    for film, csv_path in sorted(anno_csvs.items()):
        print(f"[{film}] replaying {csv_path.name}")
        emissions, n_ads_with_chars, n_total_ads = replay_movie(
            csv_path, args.batch_size, film
        )
        all_emissions.extend(emissions)
        c = Counter(e["source"] for e in emissions)
        n = len(emissions)
        pooled_counts.update(c)
        proxy_pct = (100 * c["stage1_proxy"] / n) if n else 0.0
        per_movie_rows.append({
            "movie": film,
            "n_ads_total":          n_total_ads,
            "n_ads_with_chars":     n_ads_with_chars,
            "n_emissions":          n,
            "n_stage1_proxy":       c["stage1_proxy"],
            "n_stage2_synced":      c["stage2_synced"],
            "pct_stage1_proxy":     round(proxy_pct, 4),
        })
        print(f"  emissions={n}, stage1_proxy={c['stage1_proxy']} "
              f"({proxy_pct:.2f}%), stage2_synced={c['stage2_synced']}")

    write_csv(args.output_dir / "last_action_mixture_per_emission.csv", all_emissions)

    # Add pooled row at top of summary
    n_pool = sum(pooled_counts.values())
    pooled_pct = (100 * pooled_counts["stage1_proxy"] / n_pool) if n_pool else 0.0
    summary_rows = [{
        "movie": "POOLED",
        "n_ads_total":      sum(r["n_ads_total"] for r in per_movie_rows),
        "n_ads_with_chars": sum(r["n_ads_with_chars"] for r in per_movie_rows),
        "n_emissions":      n_pool,
        "n_stage1_proxy":   pooled_counts["stage1_proxy"],
        "n_stage2_synced":  pooled_counts["stage2_synced"],
        "pct_stage1_proxy": round(pooled_pct, 4),
    }] + per_movie_rows
    write_csv(args.output_dir / "last_action_mixture_summary.csv", summary_rows)

    print(f"\nHEADLINE: pooled stage1_proxy rate = {pooled_pct:.2f}% "
          f"({pooled_counts['stage1_proxy']}/{n_pool} emissions)")
    if pooled_pct < 1.0:
        print("→ Stage-1 distribution is empirically irrelevant.")
        print("  GT corpus analysis (compute_truncation_safety.py) is sufficient for §4.7.")
    elif pooled_pct >= 5.0:
        print("→ Stage-1 distribution carries meaningful weight.")
        print("  Recommend running compute_truncation_safety.py with a Stage-1 mode.")
    else:
        print("→ Stage-1 distribution is borderline (1-5%).")
        print("  Up to you whether to measure separately or note as a footnote caveat.")


if __name__ == "__main__":
    main()
