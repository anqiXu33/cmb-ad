"""
Sanity-check comparison for CMB vs --no_cmb baseline on a single movie
(default: Moskau_einfach, German).

Reports per condition:
  - AD-length stats (words): mean / median / min / max / non-empty count
  - German re-introduction precision on cross-cut shots (main coherence)
  - German pronoun rate on continuing shots (reverse coherence — guards
    against rigid always-ID models trivially scoring 100%)

Plus 3 sample shots where the two conditions differ (cross-cut preferred —
that is where coherence rules matter most for CMB).

Usage:
    python scripts/eval/compare_sanity_check.py \\
        --cmb_csv      results/sanity_check_cmb/Moskau_einfach/stage2_qwen3_single.csv \\
        --baseline_csv results/sanity_check_baseline/Moskau_einfach/stage2_qwen3_single.csv \\
        --movie        Moskau_einfach \\
        --language     German
"""

import argparse
import os
import re
import sys

# Path setup so we can import sibling modules.
# This script lives at scripts/eval/, so project root is two levels up.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
for _p in (_SCRIPT_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd  # noqa: E402

from evaluate_coherence import (  # noqa: E402
    _merge_pred_ids_col,
    _derive_char_continuity,
    evaluate_german,
    evaluate_italian,
)
from scripts.eval.evaluate_reverse_coherence import (  # noqa: E402
    german_pronoun_rate_on_continuing,
    get_nlp as get_de_nlp,
)


_ANNO_PATH_TEMPLATES = [
    os.path.join(
        _PROJECT_ROOT, "resources", "annotations",
        "{movie}_context_faces_threads_shot_scales.csv",
    ),
    "/shares/iict-sp4.ebling.cl.uzh/data/anxu_MA_data/annotations/"
    "{movie}_context_faces_threads_shot_scales.csv",
    "data/annotations/"
    "{movie}_context_faces_threads_shot_scales.csv",
]

_GEN_COL_ALIASES = ("text_gen", "output_ads", "generated_ad", "summarized_AD")


def _load_anno(movie: str) -> pd.DataFrame:
    for tmpl in _ANNO_PATH_TEMPLATES:
        path = tmpl.format(movie=movie)
        if os.path.exists(path):
            print(f"  Annotation: {path}")
            return pd.read_csv(path)
    raise FileNotFoundError(f"No annotation CSV found for movie='{movie}'.")


def _load_and_prepare(csv_path: str, anno_df: pd.DataFrame, label: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Standardise generation column to `output_ads` (what the evaluators expect).
    for alias in _GEN_COL_ALIASES:
        if alias in df.columns:
            if alias != "output_ads":
                df = df.rename(columns={alias: "output_ads"})
            break
    else:
        raise ValueError(
            f"[{label}] {csv_path}: no generation column found among {_GEN_COL_ALIASES}"
        )

    df = _merge_pred_ids_col(df, anno_df)
    df["char_continuity"] = _derive_char_continuity(df["pred_ids_set"])
    return df


def _word_count(text) -> int:
    if not isinstance(text, str) or not text.strip():
        return 0
    return len(re.findall(r"\S+", text))


def _length_stats(df: pd.DataFrame) -> dict:
    lengths = df["output_ads"].fillna("").map(_word_count)
    nonempty = lengths[lengths > 0]
    if len(nonempty) == 0:
        return {"n_total": len(df), "n_nonempty": 0,
                "mean": 0.0, "median": 0.0, "min": 0, "max": 0}
    return {
        "n_total": len(df),
        "n_nonempty": int((lengths > 0).sum()),
        "mean": round(float(nonempty.mean()), 1),
        "median": float(nonempty.median()),
        "min": int(nonempty.min()),
        "max": int(nonempty.max()),
    }


def _coherence(df: pd.DataFrame, language: str, nlp) -> dict:
    if language == "German":
        return evaluate_german(df, nlp)
    if language == "Italian":
        return evaluate_italian(df, nlp)
    raise ValueError(f"Unsupported language: {language}")


def _sample_diffs(cmb_df: pd.DataFrame, base_df: pd.DataFrame, n: int = 3) -> list[dict]:
    """Pick `n` shots where outputs differ. Prefer cross-cut (char_continuity=False)
    rows since those are where CMB's re-introduction logic should kick in."""
    merged = cmb_df.merge(
        base_df[["anno_idx", "output_ads"]].rename(columns={"output_ads": "base_ad"}),
        on="anno_idx",
        how="inner",
        suffixes=("", "_base"),
    )
    differing = merged[
        merged["output_ads"].astype(str).str.strip()
        != merged["base_ad"].astype(str).str.strip()
    ]

    cross_cut = differing[differing["char_continuity"] == False]  # noqa: E712
    pool = cross_cut if len(cross_cut) >= n else differing
    if len(pool) == 0:
        return []
    sample = pool.head(n) if len(pool) >= n else pool
    return [
        {
            "anno_idx": int(row["anno_idx"]),
            # Preserve None (undetermined) — bool(None) would silently flip it
            # to False and mis-label as "cross-cut" in the display.
            "char_continuity": row["char_continuity"],
            "cmb": str(row["output_ads"]).strip(),
            "baseline": str(row["base_ad"]).strip(),
            "gt": str(row.get("text_gt", "")).strip(),
        }
        for _, row in sample.iterrows()
    ]


def _print_block(title: str, rows: list[tuple[str, object]]) -> None:
    print(f"\n  {title}")
    print(f"  {'-' * len(title)}")
    for label, val in rows:
        print(f"    {label:<30} {val}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cmb_csv", required=True)
    ap.add_argument("--baseline_csv", required=True)
    ap.add_argument("--movie", required=True,
                    help="Movie folder name, e.g. Moskau_einfach (used to locate annotation CSV).")
    ap.add_argument("--language", choices=["German", "Italian"], default="German")
    ap.add_argument("--n_diffs", type=int, default=3,
                    help="Number of differing shots to print (default 3).")
    args = ap.parse_args()

    print("=" * 72)
    print(f"  SANITY-CHECK COMPARISON  |  movie={args.movie}  language={args.language}")
    print("=" * 72)

    anno_df = _load_anno(args.movie)
    cmb_df = _load_and_prepare(args.cmb_csv, anno_df, "CMB")
    base_df = _load_and_prepare(args.baseline_csv, anno_df, "Baseline")

    if args.language == "German":
        nlp = get_de_nlp()
    else:
        # Aligned with evaluate_coherence.py / evaluate_all_models.py: pin to `_sm`.
        import spacy
        try:
            nlp = spacy.load("it_core_news_sm")
        except OSError as e:
            raise RuntimeError(
                "spaCy model 'it_core_news_sm' not installed. "
                "Run: python -m spacy download it_core_news_sm"
            ) from e

    # ── per-condition stats ───────────────────────────────────────────────
    print(f"\n  CMB CSV     : {args.cmb_csv}")
    print(f"  Baseline CSV: {args.baseline_csv}")

    cmb_len  = _length_stats(cmb_df)
    base_len = _length_stats(base_df)
    cmb_coh  = _coherence(cmb_df,  args.language, nlp)
    base_coh = _coherence(base_df, args.language, nlp)

    if args.language == "German":
        cmb_pron  = german_pronoun_rate_on_continuing(cmb_df,  nlp)
        base_pron = german_pronoun_rate_on_continuing(base_df, nlp)
    else:
        cmb_pron = base_pron = None  # reverse coherence is German-only for now

    # ── summary table ─────────────────────────────────────────────────────
    print("\n" + "─" * 72)
    print(f"  {'METRIC':<46} {'CMB':>10} {'BASELINE':>12}")
    print("─" * 72)
    rows = [
        ("AD count (non-empty / total)",
         f"{cmb_len['n_nonempty']}/{cmb_len['n_total']}",
         f"{base_len['n_nonempty']}/{base_len['n_total']}"),
        ("AD length — mean (words)",        cmb_len["mean"],   base_len["mean"]),
        ("AD length — median",              cmb_len["median"], base_len["median"]),
        ("AD length — min / max",
         f"{cmb_len['min']}/{cmb_len['max']}", f"{base_len['min']}/{base_len['max']}"),
    ]
    coh_key = "reintr_precision" if args.language == "German" else "prodrop_precision"
    rows.extend([
        (f"Coherence ({coh_key}) %",
         cmb_coh.get(coh_key), base_coh.get(coh_key)),
        ("  cross-cut shots evaluated"
         if args.language == "German"
         else "  continuing shots evaluated",
         cmb_coh.get("n_cross_cut" if args.language == "German" else "n_continuing"),
         base_coh.get("n_cross_cut" if args.language == "German" else "n_continuing")),
    ])
    if args.language == "German":
        rows.append(("Reverse coherence — pronoun rate %", cmb_pron, base_pron))
    for label, g, b in rows:
        print(f"  {label:<46} {str(g):>10} {str(b):>12}")
    print("─" * 72)

    # ── interpretation hint ───────────────────────────────────────────────
    print("\n  INTERPRETATION:")
    if args.language == "German":
        print("    - reintr_precision: HIGHER is better (correct re-intro on cross-cut)")
        print("    - pronoun_rate:     >0% expected for CMB; ~0% expected for")
        print("                        baseline w/o CMB if it tends to use names always.")
    else:
        print("    - prodrop_precision: HIGHER is better (no explicit subject pronoun")
        print("                         on continuing shots).")

    # ── sample diffs ──────────────────────────────────────────────────────
    diffs = _sample_diffs(cmb_df, base_df, n=args.n_diffs)
    print(f"\n  SAMPLE DIFFS ({len(diffs)} of {args.n_diffs} requested, cross-cut prioritised)")
    print("  " + "─" * 70)
    for i, d in enumerate(diffs, 1):
        if d["char_continuity"] is True:
            thread_tag = "continuing"
        elif d["char_continuity"] is False:
            thread_tag = "cross-cut"
        else:
            thread_tag = "undetermined"
        print(f"\n  [{i}] anno_idx={d['anno_idx']}  ({thread_tag})")
        print(f"      GT      : {d['gt']}")
        print(f"      CMB     : {d['cmb']}")
        print(f"      Baseline: {d['baseline']}")

    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
