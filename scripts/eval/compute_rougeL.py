#!/usr/bin/env python3
"""
compute_rougeL.py — additive ROUGE-L F1 over Stage-2 outputs.

Mirrors the file discovery, column resolution, and empty-row handling of
``scripts/eval/evaluate_all_models.py`` so the ROUGE-L denominator matches the
locked BLEU / BERTScore one (rows with NaN or empty ``text_gen`` / ``text_gt``
are skipped). Reads ``results/stage2/{cmb_ad,baseline1_no_cmb,baseline2_always_id}/<movie>/stage2.csv``
and writes ``results/benchmarks/rougeL_results.csv``.

Aggregation: ROUGE-L F1 per AD → mean within film → macro-mean across the 8
films (equal film weight). Per-language macro-means (DE / IT) also emitted.

Tokenizer note: ``rouge-score``'s default tokenizer applies an ASCII-only regex
and strips DE/IT accents. We pass a custom Unicode-aware ``\\w+`` tokenizer
with lowercasing so umlauts / accents survive; supplying ``tokenizer=`` also
disables the English Porter stemmer (the right default for German/Italian).

Run:  python scripts/eval/compute_rougeL.py
"""

from __future__ import annotations

import os
import re
import sys
from statistics import mean

import pandas as pd
from rouge_score import rouge_scorer


# ── Paths / config (mirror evaluate_all_models.py) ────────────────────────────
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))

CONDITIONS: dict[str, str] = {
    "CMB-AD (Ours)":          "cmb_ad",
    "Baseline 1 (No-CMB)":    "baseline1_no_cmb",
    "Baseline 2 (Always-ID)": "baseline2_always_id",
}

MOVIES: dict[str, list[str]] = {
    "de": [
        "8_Tage_im_August",
        "Moskau_einfach",
        "Baghdad_in_my_Shadow",
        "Trommelwirbel",
    ],
    "it": [
        "La_Tentazione_Di_Esistere",
        "Quanto_basta",
        "Riders_of_justice",
        "Stuerm",
    ],
}

_GEN_COL_ALIASES = ("output_ads", "text_gen", "generated_ad", "summarized_AD")

OUTPUT_CSV = os.path.join(_PROJECT_ROOT, "results", "benchmarks", "rougeL_results.csv")


# ── Unicode-aware tokenizer (replaces ASCII default that strips ä ö ü ß è é …) ─
_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)


class UnicodeWordTokenizer:
    def tokenize(self, text: str) -> list[str]:
        return _WORD_RE.findall(text.lower())


def _stage2_path(condition_dir: str, movie: str) -> str:
    return os.path.join(_PROJECT_ROOT, "results", "stage2", condition_dir, movie, "stage2.csv")


def _resolve_gen_col(df: pd.DataFrame) -> str | None:
    for alias in _GEN_COL_ALIASES:
        if alias in df.columns:
            return alias
    return None


def _film_rougeL(csv_path: str, scorer: rouge_scorer.RougeScorer) -> tuple[float | None, int]:
    """Return (mean ROUGE-L F1 over kept pairs, n_pairs). None if CSV missing/empty."""
    if not os.path.exists(csv_path):
        return None, 0

    df = pd.read_csv(csv_path)
    gen_col = _resolve_gen_col(df)
    if gen_col is None or "text_gt" not in df.columns:
        return None, 0

    # Mirror get_aligned_pairs: drop NaN, drop empty-after-strip.
    pair = df[["text_gt", gen_col]].copy()
    pair = pair.dropna(subset=["text_gt", gen_col])
    pair["text_gt"] = pair["text_gt"].astype(str)
    pair[gen_col]    = pair[gen_col].astype(str)
    pair = pair[pair["text_gt"].str.strip() != ""]
    pair = pair[pair[gen_col].str.strip() != ""]

    if pair.empty:
        return None, 0

    scores = [
        scorer.score(ref, hyp)["rougeL"].fmeasure
        for ref, hyp in zip(pair["text_gt"].tolist(), pair[gen_col].tolist())
    ]
    return mean(scores), len(scores)


def main() -> None:
    scorer = rouge_scorer.RougeScorer(
        ["rougeL"], use_stemmer=False, tokenizer=UnicodeWordTokenizer()
    )

    rows: list[dict] = []
    per_film: dict[str, dict[tuple[str, str], float]] = {label: {} for label in CONDITIONS}

    print(f"{'Condition':24s} {'Film':30s} {'Lang':5s} {'N':>6s}  rougeL_f1")
    print("─" * 75)

    for label, cond_dir in CONDITIONS.items():
        for lang, movies in MOVIES.items():
            for movie in movies:
                path = _stage2_path(cond_dir, movie)
                score, n = _film_rougeL(path, scorer)
                if score is None:
                    print(f"{label:24s} {movie:30s} {lang:5s} {'N/A':>6s}  N/A  (path: {path})")
                    rows.append({
                        "condition": label, "film": movie, "lang": lang,
                        "rougeL_f1": None,
                    })
                    continue
                print(f"{label:24s} {movie:30s} {lang:5s} {n:>6d}  {score:.4f}")
                per_film[label][(movie, lang)] = score
                rows.append({
                    "condition": label, "film": movie, "lang": lang,
                    "rougeL_f1": round(score, 4),
                })

    # Macro-means (overall / DE / IT) per condition.
    summary_rows: list[dict] = []
    for label in CONDITIONS:
        films = per_film[label]
        all_vals = list(films.values())
        de_vals  = [v for (m, l), v in films.items() if l == "de"]
        it_vals  = [v for (m, l), v in films.items() if l == "it"]
        summary_rows.append({
            "condition": label, "film": "MACRO_MEAN", "lang": "all",
            "rougeL_f1": round(mean(all_vals), 4) if all_vals else None,
        })
        summary_rows.append({
            "condition": label, "film": "MACRO_MEAN_DE", "lang": "de",
            "rougeL_f1": round(mean(de_vals), 4) if de_vals else None,
        })
        summary_rows.append({
            "condition": label, "film": "MACRO_MEAN_IT", "lang": "it",
            "rougeL_f1": round(mean(it_vals), 4) if it_vals else None,
        })

    print("\n" + "═" * 75)
    print(f"{'Condition':24s} {'overall':>10s} {'DE':>10s} {'IT':>10s}")
    print("─" * 75)
    for label in CONDITIONS:
        films = per_film[label]
        all_vals = list(films.values())
        de_vals  = [v for (m, l), v in films.items() if l == "de"]
        it_vals  = [v for (m, l), v in films.items() if l == "it"]
        ov = mean(all_vals) if all_vals else float("nan")
        de = mean(de_vals)  if de_vals  else float("nan")
        it = mean(it_vals)  if it_vals  else float("nan")
        print(f"{label:24s} {ov:>10.4f} {de:>10.4f} {it:>10.4f}")
    print("═" * 75)

    out_df = pd.DataFrame(
        rows + summary_rows, columns=["condition", "film", "lang", "rougeL_f1"]
    )
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved → {os.path.relpath(OUTPUT_CSV, _PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
