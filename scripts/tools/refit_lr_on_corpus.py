#!/usr/bin/env python3
"""
Refit word_count = intercept + slope * duration_seconds on the full Swiss-AD
corpus (214 scripts across DE/IT × {movies, docu_series, documentaries,
gameshows, series}) and compare against the currently-used values
(LR_intercept=3, LR_slope=2.8).

The currently-used coefficients were fitted at project inception when only
the 8 evaluation films were available. This script validates whether they
remain representative on the broader 214-file corpus, addressing the soft
contamination flagged in thesis_notes.md §12.

Three slices reported:
  * Full corpus                      — for the §12 leak-validation answer
  * Per-language and per-genre       — to surface DE-vs-IT or genre drift
  * 8 evaluation films               — to recover the original fit

Output: stdout table; exit code 0 if full-corpus coefficients are within
±10% of (3.0, 2.8), 1 otherwise.

Usage (cluster login node, mount, or local rsync of all_text_data/):
    python3 scripts/tools/refit_lr_on_corpus.py \\
        --corpus_dir /shares/iict-sp4.ebling.cl.uzh/data/anxu_MA_data/all_text_data
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

# 8 evaluation films (filename substrings to match against the JSON filenames
# in all_text_data/{DE,IT}/movies/). Matched case-insensitively.
EVAL_FILMS = [
    "8_Tage_im_August", "Moskau_einfach", "Baghdad_in_my_Shadow", "Trommelwirbel",
    "La_Tentazione_Di_Esistere", "Quanto_basta", "Riders_of_justice", "Stuerm",
]

CURRENT_INTERCEPT = 3.0
CURRENT_SLOPE = 2.8
TOLERANCE_PCT = 10.0  # ±10% defines "within tolerance" for the exit-code gate


def parse_timestamp(ts: str) -> float:
    """Parse '0:00:10.560000' or '0:00:01' to seconds."""
    parts = ts.split(":")
    h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
    return h * 3600 + m * 60 + s


def is_eval_film(fname: str) -> bool:
    low = fname.lower()
    return any(name.lower() in low for name in EVAL_FILMS)


def collect_segments(corpus_dir: str):
    """Walk all_text_data/{DE,IT}/{genre}/*.json. Yield (lang, genre, fname,
    is_eval, duration_s, word_count) per AD segment."""
    pattern = os.path.join(corpus_dir, "*", "*", "*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        sys.exit(f"No JSONs found under {pattern}; check --corpus_dir")
    for path in files:
        rel = os.path.relpath(path, corpus_dir)
        parts = rel.split(os.sep)
        if len(parts) != 3:
            continue
        lang, genre, fname = parts
        if lang not in ("DE", "IT"):
            continue
        eval_match = is_eval_film(fname)
        try:
            with open(path, encoding="utf-8") as f:
                segments = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  [skip] {rel}: {e}", file=sys.stderr)
            continue
        if not isinstance(segments, list):
            continue
        for seg in segments:
            try:
                dur = parse_timestamp(seg["end"]) - parse_timestamp(seg["start"])
                wc = len(str(seg.get("source", "")).split())
            except (KeyError, ValueError):
                continue
            if dur <= 0 or wc == 0:
                continue
            yield lang, genre, fname, eval_match, dur, wc


def fit_lr(durations, words):
    """Return (intercept, slope, r2) for word_count = a + b * duration."""
    durations = np.asarray(durations, dtype=float)
    words = np.asarray(words, dtype=float)
    if len(durations) < 2:
        return None
    slope, intercept = np.polyfit(durations, words, 1)
    pred = slope * durations + intercept
    ss_res = ((words - pred) ** 2).sum()
    ss_tot = ((words - words.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(intercept), float(slope), float(r2)


def fmt_row(label: str, n: int, fit):
    if fit is None:
        return f"  {label:<40} {n:>7}   (n<2, skipped)"
    intercept, slope, r2 = fit
    return f"  {label:<40} {n:>7}   {intercept:>7.2f}   {slope:>5.2f}   {r2:>4.2f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--corpus_dir",
        default="/shares/iict-sp4.ebling.cl.uzh/data/anxu_MA_data/all_text_data",
        help="Root of all_text_data/{DE,IT}/{genre}/*.json.",
    )
    args = p.parse_args()

    print(f"Walking {args.corpus_dir} ...")
    rows = list(collect_segments(args.corpus_dir))
    if not rows:
        sys.exit("no usable segments found; aborting")
    print(f"Loaded {len(rows)} segments.\n")

    durs_full = [r[4] for r in rows]
    words_full = [r[5] for r in rows]
    durs_de = [r[4] for r in rows if r[0] == "DE"]
    words_de = [r[5] for r in rows if r[0] == "DE"]
    durs_it = [r[4] for r in rows if r[0] == "IT"]
    words_it = [r[5] for r in rows if r[0] == "IT"]
    durs_eval = [r[4] for r in rows if r[3]]
    words_eval = [r[5] for r in rows if r[3]]

    by_genre = defaultdict(list)
    for lang, genre, *_, dur, wc in rows:
        by_genre[(lang, genre)].append((dur, wc))

    print("=== LR fit: word_count = intercept + slope * duration_s ===\n")
    print(f"  {'Slice':<40} {'n':>7}   {'intcpt':>7}   {'slope':>5}   {'R²':>4}")
    print("  " + "-" * 70)
    print(fmt_row("Full corpus (DE+IT, all genres)", len(rows), fit_lr(durs_full, words_full)))
    print(fmt_row("DE only", len(durs_de), fit_lr(durs_de, words_de)))
    print(fmt_row("IT only", len(durs_it), fit_lr(durs_it, words_it)))
    print(fmt_row("8 evaluation films (subset)", len(durs_eval), fit_lr(durs_eval, words_eval)))
    print()
    print("  Per-genre:")
    for (lang, genre), pairs in sorted(by_genre.items()):
        durs, words = zip(*pairs)
        print(fmt_row(f"  {lang}/{genre}", len(pairs), fit_lr(durs, words)))

    print(
        f"\n=== Comparison to currently-used values "
        f"(LR_intercept={CURRENT_INTERCEPT}, LR_slope={CURRENT_SLOPE}) ===\n"
    )
    full_fit = fit_lr(durs_full, words_full)
    eval_fit = fit_lr(durs_eval, words_eval)

    def pct_delta(new, old):
        if old == 0:
            return float("inf")
        return 100.0 * (new - old) / old

    full_intercept_d = pct_delta(full_fit[0], CURRENT_INTERCEPT)
    full_slope_d = pct_delta(full_fit[1], CURRENT_SLOPE)
    eval_intercept_d = pct_delta(eval_fit[0], CURRENT_INTERCEPT)
    eval_slope_d = pct_delta(eval_fit[1], CURRENT_SLOPE)

    print(
        f"  Full corpus    : intercept Δ = {full_intercept_d:+6.1f}%, "
        f"slope Δ = {full_slope_d:+6.1f}%"
    )
    print(
        f"  8 eval films   : intercept Δ = {eval_intercept_d:+6.1f}%, "
        f"slope Δ = {eval_slope_d:+6.1f}%"
    )

    full_within = (
        abs(full_intercept_d) <= TOLERANCE_PCT and abs(full_slope_d) <= TOLERANCE_PCT
    )
    print()
    if full_within:
        print(
            f"VERDICT: full-corpus coefficients are within ±{TOLERANCE_PCT:.0f}% of "
            f"the currently-used (3.0, 2.8). Document in thesis_notes.md §12 "
            f"as 'validated against full 214-film corpus' and close the leak concern."
        )
        sys.exit(0)
    else:
        print(
            f"VERDICT: full-corpus coefficients deviate by more than ±{TOLERANCE_PCT:.0f}% "
            f"from the currently-used values. Decide between:\n"
            f"  (a) keep current values + record deviation as a Limitations item, OR\n"
            f"  (b) refit and re-run (expensive — affects all 24 main-run word_limits)."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
