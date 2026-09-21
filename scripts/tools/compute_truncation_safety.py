"""
Compute truncation-safety statistics for CMB-AD's last_action storage.

Used by Ch 3 §3.4 / §4.7 footnote to justify max_chars=150 in
coherence_manager._truncate_to_sentence.

For each AD text, simulate the production truncation function (NLTK
sent_tokenize with English Punkt — matching coherence_manager.py exactly)
and classify the outcome into three regimes:

  Regime A — len(text) <= 150      → whole text preserved
  Regime B — first sentence <= 150 → clean sentence-boundary truncation
  Regime C — first sentence  > 150 → hard mid-text cut at 150 chars (BAD)

Headline thesis statistic = Regime C rate, computed on TWO distributions
that together cover what the LLM actually consumes in the prompt:

  --input_source swissadt  (default) — Stage-2 sync'd distribution
      Reads SwissADT GT from all_text_data/{DE,IT}/{genre}/*.json (`source`).
      Models the post-sync state: what last_action looks like after
      sync_last_actions() overwrites the proxy with real Stage-2 output.
      This is what the LLM sees for cross-chunk character reappearances
      (~60% of `Last known action:` emissions at production bs=8 —
      see compute_last_action_mixture.py).

  --input_source stage1 — Stage-1 proxy distribution
      Reads Stage-1 outputs from stage1/<movie>/swissAD_ads/stage1_qwen2vl.csv
      (`text_gen` column) for the 8 thesis evaluation films. text_gen is
      consumed RAW (no header stripping) to match production.
      Models the proxy state: what last_action looks like during a chunk
      before sync_last_actions() runs. This is what the LLM sees for
      intra-chunk reappearances with gap>1 (~40% of emissions at bs=8).
      Stage-1 output has markdown headers ("### Description of Shot N:")
      and numbered sections that interact with NLTK tokenisation in
      non-obvious ways — this mode exposes that honestly.

The two modes together let §4.7 compute a mixture estimate:
    overall_safe_rate ≈ 0.60 * gt_safe_rate + 0.40 * stage1_safe_rate

The 8-film hardcoding in --input_source stage1 is *necessary*, not stylistic:
Stage-1 outputs exist only for the films we ran (SwissADT corpus doesn't
have Stage-1 outputs at all), and the CSV carries no language column.

Place on cluster at:
  /shares/iict-sp4.ebling.cl.uzh/data/anxu_MA_data/scripts/compute_truncation_safety.py
Outputs land in statistics/.
"""

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import sent_tokenize

DATA_ROOT_DEFAULT  = Path("/shares/iict-sp4.ebling.cl.uzh/data/anxu_MA_data")
SWISSADT_SUBDIR    = "all_text_data"
STAGE1_SUBDIR      = "stage1"
STAGE1_CSV_RELPATH = "swissAD_ads/stage1_qwen2vl.csv"
OUT_SUBDIR         = "statistics"

MAX_CHARS = 150
NLTK_LANG_MAP = {"DE": "german", "IT": "italian"}

# Thesis 8-film evaluation set + language assignment (CLAUDE.md §"Models").
# Stage-1 outputs are produced ONLY for these 8 films; hardcoding is necessary,
# not stylistic — the Stage-1 CSV carries no language column.
STAGE1_FILMS = {
    "8_Tage_im_August":          "DE",
    "Moskau_einfach":            "DE",
    "Baghdad_in_my_Shadow":      "DE",
    "Trommelwirbel":             "DE",
    "La_Tentazione_Di_Esistere": "IT",
    "Quanto_basta":              "IT",
    "Riders_of_justice":         "IT",
    "Stuerm":                    "IT",
}


def truncate_to_sentence_en(text: str, max_chars: int = MAX_CHARS) -> str:
    """Faithful mirror of coherence_manager._truncate_to_sentence."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    sents = sent_tokenize(text)
    if not sents:
        return text[:max_chars]
    if len(sents[0]) > max_chars:
        return text[:max_chars]
    result = sents[0]
    for s in sents[1:]:
        if len(result) + 1 + len(s) > max_chars:
            break
        result = result + " " + s
    return result


def classify_regime(text: str, first_sent_len: int, max_chars: int = MAX_CHARS) -> str:
    if len(text.strip()) <= max_chars:
        return "A"
    if first_sent_len <= max_chars:
        return "B"
    return "C"


def analyze_ad(text: str, lang_code: str) -> dict | None:
    text = text.strip() if text else ""
    if not text:
        return None
    sents_en     = sent_tokenize(text)
    sents_native = sent_tokenize(text, language=NLTK_LANG_MAP[lang_code])
    first_en     = len(sents_en[0])     if sents_en     else len(text)
    first_native = len(sents_native[0]) if sents_native else len(text)
    return {
        "ad_len": len(text),
        "first_sent_en_len":     first_en,
        "first_sent_native_len": first_native,
        "regime_en":     classify_regime(text, first_en),
        "regime_native": classify_regime(text, first_native),
    }


def percentile(xs_sorted, q):
    n = len(xs_sorted)
    return xs_sorted[min(int(q * n), n - 1)]


def summary_row(scope, lang, group, records, stage1_mode: bool):
    n = len(records)
    counts = Counter(r["regime_en"] for r in records)
    ad_lens       = sorted(r["ad_len"] for r in records)
    first_en_lens = sorted(r["first_sent_en_len"] for r in records)
    disagree_regime = sum(1 for r in records if r["regime_en"] != r["regime_native"])
    diff_len = [abs(r["first_sent_en_len"] - r["first_sent_native_len"]) for r in records]
    row = {
        "scope": scope, "language": lang, "group": group, "n_ads": n,
        "pct_regime_A":  round(100 * counts.get("A", 0) / n, 4),
        "pct_regime_B":  round(100 * counts.get("B", 0) / n, 4),
        "pct_regime_C":  round(100 * counts.get("C", 0) / n, 4),
        "pct_A_plus_B":  round(100 * (counts.get("A", 0) + counts.get("B", 0)) / n, 4),
        "ad_len_mean":        round(statistics.mean(ad_lens), 2),
        "ad_len_median":      statistics.median(ad_lens),
        "ad_len_p95":         percentile(ad_lens, 0.95),
        "ad_len_p99":         percentile(ad_lens, 0.99),
        "first_sent_en_mean": round(statistics.mean(first_en_lens), 2),
        "first_sent_en_p95":  percentile(first_en_lens, 0.95),
        "first_sent_en_p99":  percentile(first_en_lens, 0.99),
        "pct_native_regime_disagree": round(100 * disagree_regime / n, 4),
        "mean_abs_first_sent_diff":   round(statistics.mean(diff_len), 2),
    }
    if stage1_mode:
        # Stage-1 markdown headers ("### Description of Shot N:") often
        # tokenise as a short first "sentence" (~25-30c), leaving subsequent
        # numbered sections unreachable within max_chars.
        n_short = sum(1 for r in records if r["first_sent_en_len"] <= 30)
        row["pct_first_sent_le_30c"] = round(100 * n_short / n, 4)
    return row


def regime_c_row(language, group_label, source_id, segment_idx, src, r):
    return {
        "language": language, "group": group_label,
        "source_id": source_id, "segment_idx": segment_idx,
        "ad_len": r["ad_len"],
        "first_sent_en_len":     r["first_sent_en_len"],
        "first_sent_native_len": r["first_sent_native_len"],
        "ad_text_truncated_to_500":  src.strip()[:500],
        "what_last_action_would_be": truncate_to_sentence_en(src),
    }


def load_swissadt(data_root: Path):
    """Yield (language, genre, file_stem, segment_idx, text)."""
    root = data_root / SWISSADT_SUBDIR
    for lang_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        lang = lang_dir.name
        if lang not in NLTK_LANG_MAP:
            print(f"WARNING: unknown language {lang!r}, skipping", file=sys.stderr)
            continue
        for genre_dir in sorted(p for p in lang_dir.iterdir() if p.is_dir()):
            for json_path in sorted(genre_dir.glob("*.json")):
                try:
                    with open(json_path, encoding="utf-8") as f:
                        data = json.load(f)
                except (OSError, json.JSONDecodeError) as e:
                    print(f"WARNING: skipping {json_path} ({e})", file=sys.stderr)
                    continue
                if not isinstance(data, list):
                    print(f"WARNING: {json_path} not a list, skipping", file=sys.stderr)
                    continue
                for idx, seg in enumerate(data):
                    if not isinstance(seg, dict):
                        continue
                    src = seg.get("source") or ""
                    if not src.strip():
                        continue
                    yield lang, genre_dir.name, json_path.stem, idx, src


def load_stage1(data_root: Path):
    """Yield (language, film, film, anno_idx, text_gen) for the 8 thesis films."""
    csv.field_size_limit(sys.maxsize)
    root = data_root / STAGE1_SUBDIR
    for film, lang in sorted(STAGE1_FILMS.items()):
        path = root / film / STAGE1_CSV_RELPATH
        if not path.exists():
            print(f"WARNING: {path} not found, skipping", file=sys.stderr)
            continue
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                text = row.get("text_gen") or ""
                if not text.strip():
                    continue
                yield lang, film, film, row.get("anno_idx", ""), text


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_source", choices=["swissadt", "stage1"], default="swissadt",
                    help="swissadt: SwissADT GT (post-sync distribution). "
                         "stage1: thesis-film Stage-1 outputs (proxy distribution).")
    ap.add_argument("--data_root", type=Path, default=DATA_ROOT_DEFAULT,
                    help="Root containing all_text_data/ and stage1/ subdirs.")
    ap.add_argument("--output_dir", type=Path, default=None,
                    help="Defaults to <data_root>/statistics/.")
    args = ap.parse_args()

    stage1_mode = args.input_source == "stage1"
    out_dir = args.output_dir or (args.data_root / OUT_SUBDIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = "truncation_safety_stage1" if stage1_mode else "truncation_safety"
    per_unit_name = "per_film" if stage1_mode else "per_file"
    loader = load_stage1 if stage1_mode else load_swissadt

    per_unit_rows, regime_c_rows = [], []
    by_lang, by_lang_group = defaultdict(list), defaultdict(list)
    by_unit = defaultdict(list)

    for lang, group_label, source_id, seg_idx, text in loader(args.data_root):
        r = analyze_ad(text, lang)
        if r is None:
            continue
        by_lang[lang].append(r)
        by_lang_group[(lang, group_label)].append(r)
        by_unit[(lang, group_label, source_id)].append(r)
        if r["regime_en"] == "C":
            regime_c_rows.append(regime_c_row(lang, group_label, source_id,
                                              seg_idx, text, r))

    for (lang, group_label, source_id), recs in sorted(by_unit.items()):
        rc = Counter(r["regime_en"] for r in recs)
        n = len(recs)
        per_unit_rows.append({
            "language": lang, "group": group_label, "source_id": source_id, "n_ads": n,
            "regime_A": rc.get("A", 0),
            "regime_B": rc.get("B", 0),
            "regime_C": rc.get("C", 0),
            "pct_C":        round(100 * rc.get("C", 0) / n, 4),
            "pct_A_plus_B": round(100 * (rc.get("A", 0) + rc.get("B", 0)) / n, 4),
        })

    summary_rows = []
    pooled = [r for rs in by_lang.values() for r in rs]
    if pooled:
        summary_rows.append(summary_row("pooled", "", "", pooled, stage1_mode))
    for lang in sorted(by_lang):
        summary_rows.append(summary_row("language", lang, "", by_lang[lang], stage1_mode))
    group_scope = "lang_film" if stage1_mode else "lang_genre"
    for (lang, group_label) in sorted(by_lang_group):
        summary_rows.append(summary_row(group_scope, lang, group_label,
                                        by_lang_group[(lang, group_label)], stage1_mode))

    def write(path, rows):
        if not rows:
            print(f"(no rows for {path.name})")
            return
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {len(rows):>6d} rows -> {path}")

    write(out_dir / f"{prefix}_summary.csv",             summary_rows)
    write(out_dir / f"{prefix}_{per_unit_name}.csv",     per_unit_rows)
    write(out_dir / f"{prefix}_regime_c_examples.csv",   regime_c_rows)

    if pooled:
        n = len(pooled)
        pct_ab = 100 * sum(1 for r in pooled if r["regime_en"] in ("A", "B")) / n
        mode_lbl = "Stage-1 proxy" if stage1_mode else "Stage-2 sync'd (≈GT)"
        print(f"\nHeadline ({mode_lbl}): pooled n={n}, "
              f"Regime A+B (clean) = {pct_ab:.2f}%, "
              f"Regime C (hard cut) = {100 - pct_ab:.2f}%")


if __name__ == "__main__":
    main()
