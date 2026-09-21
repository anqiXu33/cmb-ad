"""Verify §6.2.4's claim that IT GT PRN is dominated by clitics + reflexives
+ demonstratives, while DE GT PRN is dominated by overt pronouns. Both should land at
similar TOTAL PRN rates (~36%) via completely different grammatical routes.

Two analyses are emitted:

  Block A — ALL PRN, ALL anno_idx (original §6.2.4 sanity computation):
    GT mention files, every AD, every PRN-labelled mention regardless of
    pred_ids cardinality. Bucket aggregates + per-token frequency.

  Block B — Continuing-thread PRN, sentinels EXCLUDED (Appendix C tables):
    Restrict to ADs whose char_continuity is True (m-* pred_ids overlap with
    the previous AD's m-* pred_ids; same derivation as
    mention_level_eval.compute_subset_metrics). Then drop PRN mentions that
    would align to a sentinel under mention_level_eval.align_pron — i.e.
    mentions in ADs with |pred_ids|==0 (STRAY) or |pred_ids|>1 (AMBIGUOUS).
    This is the denominator behind the body's "729 IT GT continuing-thread
    PRN tokens (sentinels excluded)" figure in §6.2.4.

Both blocks share DE_BUCKETS / IT_BUCKETS / classify() unchanged.

Outputs go to stdout (and optionally `--csv_out_dir` for the per-token CSVs).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# ───────────────────────────────────────────────────────────────────────────
# Path bootstrap so we can reuse mention_level_eval's anno-loading helpers
# ───────────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
for p in (_SCRIPT_DIR, _PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Try cluster mount first (local dev), then direct cluster path.
_DATA_ROOT_CANDIDATES = [
    Path("data"),
    Path("/shares/iict-sp4.ebling.cl.uzh/data/anxu_MA_data"),
]
DATA_ROOT = next((p for p in _DATA_ROOT_CANDIDATES if p.exists()), None)
if DATA_ROOT is None:
    sys.exit(f"[FATAL] No DATA_ROOT found. Tried: {_DATA_ROOT_CANDIDATES}")

MENTIONS_ROOT = DATA_ROOT / "mentions_output"
ROBUST_ROOT = DATA_ROOT / "mention_robustness"
ANNO_DIR = DATA_ROOT / "annotations"

# ───────────────────────────────────────────────────────────────────────────
# GT-mention dispatch — same as scripts/eval/mention_level_eval.py:GT_DISPATCH.
# 4 DE + 4 IT thesis-evaluation films:
#   - Moskau + Baghdad + 4 IT films → supervisor's pre-computed mentions_output
#   - 8_Tage + Trommelwirbel → our own text_gt.mentions in mention_robustness
# Same extractor pipeline (character_chains_qwen3_json.py); see chat 2026-05-10
# byte-identical verification on Moskau + Quanto.
# ───────────────────────────────────────────────────────────────────────────
GT_DISPATCH = {
    "de": {
        "8_Tage_im_August":     ROBUST_ROOT / "8_Tage_im_August__cmb_ad/text_gt.mentions",
        "Moskau_einfach":       MENTIONS_ROOT / "DE/movies/AD_Moskau_einfach.mentions",
        "Baghdad_in_my_Shadow": MENTIONS_ROOT / "DE/movies/Baghdad_in_my_Shadow_final.mentions",
        "Trommelwirbel":        ROBUST_ROOT / "Trommelwirbel__cmb_ad/text_gt.mentions",
    },
    "it": {
        "La_Tentazione_Di_Esistere": MENTIONS_ROOT / "IT/movies/La_Tentazione_Di_Esistere_ITA.mentions",
        "Quanto_basta":              MENTIONS_ROOT / "IT/movies/Quanto_basta_ITA.mentions",
        "Riders_of_justice":         MENTIONS_ROOT / "IT/movies/Riders_of_justice_ITA.mentions",
        "Stuerm":                    MENTIONS_ROOT / "IT/movies/Stuerm_ITA.mentions",
    },
}

# Bucket dictionaries — linguistic categorisation. UNCHANGED from prior version.
DE_BUCKETS = {
    "overt_subj_3rd": {"er", "sie", "es"},                 # nominative
    "overt_obj_3rd":  {"ihn", "ihm", "ihr", "ihnen"},      # accusative + dative
    "reflexive":      {"sich"},
    "possessive_pron":{"sein", "seine", "seiner", "ihrer", "ihre", "ihrem"},  # may show up
    "demonstrative":  {"der", "die", "das", "dem", "den", "dieser", "diese", "dieses", "jener"},
    "1st_2nd":        {"ich", "mich", "mir", "du", "dich", "dir", "wir", "uns", "ihr", "euch"},
    # Catch-all: everything else lands in "other"
}

IT_BUCKETS = {
    "overt_subj_3rd": {"lui", "lei", "loro"},
    "clitic_obj":     {"lo", "la", "li", "le", "gli", "ne", "ci", "vi"},  # direct + indirect + partitive
    "reflexive":      {"si"},
    "demonstrative":  {"questo", "questa", "questi", "queste", "quello", "quella",
                       "quelli", "quelle", "ciò"},
    "possessive_pron":{"suo", "sua", "suoi", "sue", "loro"},  # may overlap with "loro" subj
    "1st_2nd":        {"io", "me", "mi", "tu", "te", "ti", "noi", "ci", "voi", "vi"},
}


def classify(token: str, buckets: dict[str, set[str]]) -> str:
    t = token.lower().strip()
    t = t.strip(".,!?:;\"'«»()")
    for name, vocab in buckets.items():
        if t in vocab:
            return name
    return "other"


def load_mentions_file(path: Path) -> list[list[dict]]:
    """One JSON list per line; failed JSON → empty list."""
    out: list[list[dict]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                out.append([])
                continue
            try:
                parsed = json.loads(line)
                out.append(parsed if isinstance(parsed, list) else [])
            except json.JSONDecodeError:
                out.append([])
    return out


# ───────────────────────────────────────────────────────────────────────────
# Block A — ALL PRN, ALL anno_idx
# ───────────────────────────────────────────────────────────────────────────


def analyze_language_all_prn(lang: str, paths: dict[str, Path]) -> tuple[Counter, Counter]:
    buckets = DE_BUCKETS if lang == "de" else IT_BUCKETS
    total_mentions = 0
    total_prn = 0
    prn_tokens: Counter = Counter()
    bucket_counts: Counter = Counter()

    for film, path in paths.items():
        film_mentions = 0
        film_prn = 0
        for line_mentions in load_mentions_file(path):
            for m in line_mentions:
                film_mentions += 1
                if m.get("label") == "PRN":
                    film_prn += 1
                    tok = m["text"].lower().strip()
                    prn_tokens[tok] += 1
                    bucket_counts[classify(tok, buckets)] += 1
        total_mentions += film_mentions
        total_prn += film_prn
        rate = film_prn / film_mentions * 100 if film_mentions else 0
        print(f"  {film:40s}  PRN={film_prn:4d} / {film_mentions:5d}  ({rate:5.1f}%)")

    print(f"  TOTAL {lang.upper():38s}  PRN={total_prn:4d} / {total_mentions:5d}  ({total_prn/total_mentions*100:5.1f}%)")
    print(f"\n  PRN bucket breakdown ({lang.upper()}):")
    for b in sorted(bucket_counts, key=bucket_counts.get, reverse=True):
        pct = bucket_counts[b] / total_prn * 100
        print(f"    {b:18s}  {bucket_counts[b]:4d}  ({pct:5.1f}% of all PRN)")

    print(f"\n  Top 25 PRN tokens ({lang.upper()}):")
    for tok, c in prn_tokens.most_common(25):
        bucket = classify(tok, buckets)
        pct = c / total_prn * 100
        print(f"    {tok:20s}  {c:4d}  ({pct:5.1f}%)   [{bucket}]")

    return prn_tokens, bucket_counts


# ───────────────────────────────────────────────────────────────────────────
# Block B — Continuing-thread PRN, sentinels EXCLUDED (Appendix C tables)
# ───────────────────────────────────────────────────────────────────────────


def load_anno_continuity_and_pred_ids(movie: str) -> tuple[list[int], dict[int, bool], dict[int, set]]:
    """Return (sorted_anno_idx, char_continuity_map, pred_ids_map).

    Uses the same derivation as mention_level_eval.load_anno_pred_ids:
      - m-* shot union via evaluate_coherence._merge_pred_ids_col
      - char_continuity via evaluate_coherence._derive_char_continuity
    """
    import pandas as pd
    from evaluate_coherence import _merge_pred_ids_col, _derive_char_continuity

    anno_path = ANNO_DIR / f"{movie}_context_faces_threads_shot_scales.csv"
    if not anno_path.exists():
        raise FileNotFoundError(f"Anno CSV not found: {anno_path}")
    anno_df = pd.read_csv(anno_path)
    sorted_idx = sorted(anno_df["anno_idx"].unique())
    placeholder = pd.DataFrame({"anno_idx": sorted_idx})
    merged = _merge_pred_ids_col(placeholder, anno_df)
    merged["char_continuity"] = _derive_char_continuity(merged["pred_ids_set"])
    cc_map = dict(zip(merged["anno_idx"], merged["char_continuity"]))
    pid_map = dict(zip(merged["anno_idx"], merged["pred_ids_set"]))
    return sorted_idx, cc_map, pid_map


def analyze_language_continuing(lang: str, paths: dict[str, Path]) -> tuple[Counter, Counter, Counter, dict]:
    """Continuing-thread PRN, sentinels excluded — body-cited computation.

    For each film:
      1. Load GT mentions (one list per AD, sorted by anno_idx implicitly: the
         supervisor's extractor emits one .mentions line per AD in anno-idx order).
      2. Load char_continuity + pred_ids per anno_idx from anno CSV.
      3. Iterate ADs; restrict to char_continuity == True.
      4. For each PRN mention in those ADs, also track the sentinel-excluded subset
         (skip if |pred_ids|==0 STRAY or |pred_ids|>1 AMBIGUOUS — apply
         align_pron's sentinel filter).
      5. Bucket-classify both raw and bound surviving PRN tokens and count.

    Returns (prn_tokens_raw, prn_tokens_bound, bucket_counts_bound, per_film_stats).
    `prn_tokens_raw` is over cc==True with NO sentinel filter (the denominator
    that matches §4.4.3 / §6.2.4's "65 %" clitic claim, total IT = 542).
    """
    buckets = DE_BUCKETS if lang == "de" else IT_BUCKETS
    total_cont_mentions = 0  # all mentions (NAME+DESC+PRN) in continuing ADs
    total_cont_prn_raw = 0   # PRN before sentinel filter
    total_cont_prn_bound = 0 # PRN after sentinel filter — body's "sentinels excluded"
    prn_tokens_raw: Counter = Counter()
    prn_tokens_bound: Counter = Counter()
    bucket_counts: Counter = Counter()
    per_film: dict = {}

    for film, path in paths.items():
        mentions_per_ad = load_mentions_file(path)
        sorted_idx, cc_map, pid_map = load_anno_continuity_and_pred_ids(film)
        if len(mentions_per_ad) != len(sorted_idx):
            print(f"  [WARN] {film}: mentions lines={len(mentions_per_ad)} != anno_idx={len(sorted_idx)}; "
                  f"truncating to min for index alignment.")
        n = min(len(mentions_per_ad), len(sorted_idx))

        film_cont_mentions = 0
        film_cont_prn_raw = 0
        film_cont_prn_bound = 0
        film_tonic = 0   # for IT per-film 7–21 % range claim
        film_clitic = 0

        for i in range(n):
            aidx = sorted_idx[i]
            if cc_map.get(aidx) is not True:
                continue
            pids = pid_map.get(aidx, set())
            for m in mentions_per_ad[i]:
                lbl = m.get("label", "")
                if lbl in {"NAME", "DESC", "PRN"}:
                    film_cont_mentions += 1
                if lbl != "PRN":
                    continue
                film_cont_prn_raw += 1
                tok_raw = (m.get("text") or "").lower().strip()
                prn_tokens_raw[tok_raw] += 1
                # Track per-film tonic/clitic on the RAW (no sentinel filter)
                # population — this is the denominator behind §4.4.3's
                # "per-film range 7-21 %" tonic claim.
                if lang == "it":
                    b_raw = classify(tok_raw, buckets)
                    if b_raw == "overt_subj_3rd":
                        film_tonic += 1
                    elif b_raw == "clitic_obj":
                        film_clitic += 1
                # Sentinel filter: |pred_ids|==1 ⇒ BOUND; else STRAY/AMBIGUOUS
                if len(pids) != 1:
                    continue
                film_cont_prn_bound += 1
                prn_tokens_bound[tok_raw] += 1
                b = classify(tok_raw, buckets)
                bucket_counts[b] += 1

        total_cont_mentions += film_cont_mentions
        total_cont_prn_raw += film_cont_prn_raw
        total_cont_prn_bound += film_cont_prn_bound
        per_film[film] = {
            "cont_mentions": film_cont_mentions,
            "cont_prn_raw": film_cont_prn_raw,
            "cont_prn_bound": film_cont_prn_bound,
        }
        if lang == "it":
            per_film[film]["tonic"] = film_tonic
            per_film[film]["clitic"] = film_clitic

        line = (f"  {film:40s}  cont_mentions={film_cont_mentions:5d}  "
                f"cont_PRN_raw={film_cont_prn_raw:4d}  cont_PRN_bound={film_cont_prn_bound:4d}")
        if lang == "it":
            tonic_pct = film_tonic / film_cont_mentions * 100 if film_cont_mentions else 0.0
            clitic_pct = film_clitic / film_cont_mentions * 100 if film_cont_mentions else 0.0
            line += f"  | tonic={film_tonic:3d} ({tonic_pct:4.1f}% of cont_mentions)  clitic={film_clitic:3d} ({clitic_pct:4.1f}%)"
        print(line)

    print(f"  TOTAL {lang.upper():38s}  cont_mentions={total_cont_mentions:5d}  "
          f"cont_PRN_raw={total_cont_prn_raw:4d}  cont_PRN_bound={total_cont_prn_bound:4d}")

    # Bucket breakdown on RAW continuing-thread PRN (no sentinel filter) —
    # matches body §4.4.3 / §6.2.4's "65 %" clitic / "31 %" tonic split.
    raw_buckets: Counter = Counter()
    for tok, c in prn_tokens_raw.items():
        raw_buckets[classify(tok, buckets)] += c
    print(f"\n  PRN bucket breakdown ({lang.upper()}, continuing-thread, RAW = no sentinel filter):")
    for b in sorted(raw_buckets, key=raw_buckets.get, reverse=True):
        pct = raw_buckets[b] / total_cont_prn_raw * 100 if total_cont_prn_raw else 0.0
        print(f"    {b:18s}  {raw_buckets[b]:4d}  ({pct:5.1f}% of cont PRN raw)")

    print(f"\n  Per-token PRN frequency ({lang.upper()}, continuing-thread, RAW = no sentinel filter):")
    for tok, c in prn_tokens_raw.most_common():
        bucket = classify(tok, buckets)
        pct = c / total_cont_prn_raw * 100 if total_cont_prn_raw else 0.0
        print(f"    {tok:20s}  {c:4d}  ({pct:5.2f}%)   [{bucket}]")

    print(f"\n  PRN bucket breakdown ({lang.upper()}, continuing-thread, sentinels EXCLUDED):")
    for b in sorted(bucket_counts, key=bucket_counts.get, reverse=True):
        pct = bucket_counts[b] / total_cont_prn_bound * 100 if total_cont_prn_bound else 0.0
        print(f"    {b:18s}  {bucket_counts[b]:4d}  ({pct:5.1f}% of cont PRN bound)")

    print(f"\n  Per-token PRN frequency ({lang.upper()}, continuing-thread, sentinels EXCLUDED):")
    for tok, c in prn_tokens_bound.most_common():
        bucket = classify(tok, buckets)
        pct = c / total_cont_prn_bound * 100 if total_cont_prn_bound else 0.0
        print(f"    {tok:20s}  {c:4d}  ({pct:5.2f}%)   [{bucket}]")

    return prn_tokens_raw, prn_tokens_bound, bucket_counts, per_film


# ───────────────────────────────────────────────────────────────────────────
# Optional CSV dump for appendix table assembly
# ───────────────────────────────────────────────────────────────────────────


def dump_per_token_csv(lang: str, prn_tokens: Counter, bucket_dict: dict, total: int, out_path: Path, denom_label: str) -> None:
    rows = []
    for tok, c in prn_tokens.most_common():
        rows.append({
            "token": tok,
            "count": c,
            "pct": round(c / total * 100, 2) if total else 0.0,
            "bucket": classify(tok, bucket_dict),
        })
    rows.append({"token": "TOTAL", "count": total, "pct": 100.0 if total else 0.0, "bucket": ""})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"token,count,pct_of_{denom_label},bucket\n")
        for r in rows:
            f.write(f"{r['token']},{r['count']},{r['pct']},{r['bucket']}\n")
    print(f"  → wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_out_dir", type=Path, default=None,
                        help="If set, also write per-token CSVs (Block B) here.")
    args = parser.parse_args()

    for lang in ("de", "it"):
        print(f"\n{'='*78}")
        print(f"  BLOCK A — {lang.upper()} GT mentions, ALL PRN, ALL anno_idx")
        print(f"{'='*78}")
        analyze_language_all_prn(lang, GT_DISPATCH[lang])

        print(f"\n{'='*78}")
        print(f"  BLOCK B — {lang.upper()} GT mentions, continuing-thread, sentinels EXCLUDED")
        print(f"  (denominator behind §6.2.4 \"729 IT GT continuing-thread PRN tokens\")")
        print(f"{'='*78}")
        prn_raw, prn_bound, bucket_counts, _ = analyze_language_continuing(lang, GT_DISPATCH[lang])
        total_bound = sum(bucket_counts.values())
        total_raw = sum(prn_raw.values())
        if args.csv_out_dir is not None:
            buckets = DE_BUCKETS if lang == "de" else IT_BUCKETS
            dump_per_token_csv(
                lang, prn_raw, buckets, total_raw,
                args.csv_out_dir / f"gt_prn_per_token_{lang}_raw.csv",
                "cont_prn_raw",
            )
            dump_per_token_csv(
                lang, prn_bound, buckets, total_bound,
                args.csv_out_dir / f"gt_prn_per_token_{lang}_bound.csv",
                "cont_prn_bound",
            )


if __name__ == "__main__":
    main()
