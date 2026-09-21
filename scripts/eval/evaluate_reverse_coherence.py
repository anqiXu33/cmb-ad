#!/usr/bin/env python3
"""
evaluate_reverse_coherence.py  —  German Reverse Coherence Test
================================================================
Complements evaluate_all_models.py by measuring the side of German
inter-shot coherence that the main Coherence_Score leaves unmeasured.

Why this matters
----------------
The main Coherence_Score (German) only tests cross-cut shots:
  → Re-introduction Precision: on char_continuity == False, does the model
    use a NOUN/PROPN subject instead of a PRON?

A model that NEVER uses pronouns (Baseline 2 / Always-ID) scores ~100%
trivially — not because it is coherent, but because it is uniformly rigid.

This script measures the complementary side:

  German (continuing shots, char_continuity == True)
    → Pronoun Usage Rate:
        % of continuing shots where the root-verb subject IS a pronoun.
        High → model correctly uses pronouns when referent is in context.
        Low  → model always uses names regardless of context (rigid).

Together the two metrics form a complete picture for German:

              | Cross-cut                   | Continuing               |
  ────────────┼─────────────────────────────┼──────────────────────────┤
  German      | reintr_precision (main) ✓   | pronoun_rate (here) ← NEW|

Usage
-----
  python scripts/evaluate_reverse_coherence.py

Output
------
  results/Reverse_Coherence_Results.csv
  Pretty-printed table to stdout.
"""

import os
import sys
import warnings
import logging

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.disable(logging.WARNING)

import pandas as pd
import spacy

# ── path setup ────────────────────────────────────────────────────────────────
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))

for _p in (_SCRIPT_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluate_coherence import (
    _merge_pred_ids_col,
    _derive_char_continuity,
    _root_subject,
)

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

GERMAN_MOVIES = [
    "8_Tage_im_August",
    "Moskau_einfach",
    "Baghdad_in_my_Shadow",
    "Trommelwirbel",
]

BASELINE1_PATH = os.path.join(
    _PROJECT_ROOT, "results", "stage2", "baseline1_no_cmb", "{movie}", "stage2.csv"
)
BASELINE2_PATH = os.path.join(
    _PROJECT_ROOT, "results", "stage2", "baseline2_always_id", "{movie}", "stage2.csv"
)
CMBAD_PATH = os.path.join(
    _PROJECT_ROOT, "results", "stage2", "cmb_ad", "{movie}", "stage2.csv"
)
ANNO_PATHS: list[str] = [
    os.path.join(
        _PROJECT_ROOT, "resources", "annotations",
        "{movie}_context_faces_threads_shot_scales.csv",
    ),
    (
        "/shares/iict-sp4.ebling.cl.uzh/data/anxu_MA_data/annotations"
        "/{movie}_context_faces_threads_shot_scales.csv"
    ),
    (
        "data/annotations"
        "/{movie}_context_faces_threads_shot_scales.csv"
    ),
]

MODELS: dict[str, str] = {
    "Baseline 1 (Single-shot)": BASELINE1_PATH,
    "Baseline 2 (Always-ID)":   BASELINE2_PATH,
    "CMB-AD (Ours)":            CMBAD_PATH,
}

_GEN_COL_ALIASES = ("output_ads", "text_gen", "generated_ad", "summarized_AD")

OUTPUT_CSV = os.path.join(_PROJECT_ROOT, "results", "benchmarks", "reverse_coherence.csv")

# ── spaCy ─────────────────────────────────────────────────────────────────────
# Aligned with evaluate_coherence.py and evaluate_all_models.py: use the `_sm`
# model so the main + reverse coherence numbers come from the same parser.
def get_nlp():
    model_name = "de_core_news_sm"
    try:
        nlp = spacy.load(model_name)
        print(f"  Loaded spaCy model: {model_name}")
        return nlp
    except OSError:
        print(f"  [WARN] spaCy model '{model_name}' not installed.\n"
              f"         Run:  python -m spacy download {model_name}")
        return None


# ═════════════════════════════════════════════════════════════════════════════
# I/O HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def load_annotation_csv(movie: str) -> "pd.DataFrame | None":
    for tmpl in ANNO_PATHS:
        path = tmpl.format(movie=movie)
        if os.path.exists(path):
            return pd.read_csv(path)
    print(f"  [WARN] Annotation CSV not found for '{movie}'.")
    return None


def load_gen_csv(path: str, label: str) -> "pd.DataFrame | None":
    if not os.path.exists(path):
        print(f"  [SKIP] Not found: {path}")
        return None
    df = pd.read_csv(path)
    for alias in _GEN_COL_ALIASES:
        if alias in df.columns:
            if alias != "output_ads":
                df = df.rename(columns={alias: "output_ads"})
            return df
    print(f"  [WARN] No generated-text column found for {label}.")
    return None


def prepare_df(gen_df: pd.DataFrame, anno_df: "pd.DataFrame | None",
               label: str) -> "pd.DataFrame | None":
    if anno_df is None:
        print(f"  [SKIP] No annotation CSV — skipped for {label}.")
        return None
    try:
        merged = _merge_pred_ids_col(gen_df.copy(), anno_df)
    except (ValueError, KeyError) as e:
        print(f"  [WARN] pred_ids merge failed for {label}: {e}")
        return None
    merged["char_continuity"] = _derive_char_continuity(merged["pred_ids_set"])
    return merged


# ═════════════════════════════════════════════════════════════════════════════
# METRIC: German Pronoun Usage Rate on Continuing Shots
# ═════════════════════════════════════════════════════════════════════════════

def german_pronoun_rate_on_continuing(df: pd.DataFrame, nlp) -> "float | None":
    """
    Of all shots where char_continuity == True (scene continues),
    what fraction uses a pronoun as the root-verb's grammatical subject?

    High → model correctly uses pronouns when referent is already in context.
    Low  → model always uses names regardless of context (rigid like Baseline 2).
    """
    continuing = df[df["char_continuity"] == True].copy()
    n = len(continuing)
    if n == 0:
        return None

    pron_count = 0
    for _, row in continuing.iterrows():
        text = str(row["output_ads"]).strip() if pd.notna(row["output_ads"]) else ""
        if not text:
            continue
        subj = _root_subject(nlp(text))
        if subj is not None and subj.pos_ == "PRON":
            pron_count += 1

    return round(pron_count / n * 100, 1)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═" * 70)
    print("  German Reverse Coherence: Pronoun Usage Rate on Continuing Shots")
    print("═" * 70)
    print()
    print("  Metric: of all char_continuity==True shots, % where root subject")
    print("          is a pronoun (Er/Sie/Es/...).")
    print()
    print("  Expected:")
    print("    Baseline 2 ≈  0%  (never uses pronouns — rigid by design)")
    print("    CMB-AD     > 0%   (uses pronouns when context warrants — smart)")
    print("    Human GT   > 0%   (natural human pronoun usage)")
    print("═" * 70)

    nlp = get_nlp()

    rows: list[dict] = []

    for movie in GERMAN_MOVIES:
        print(f"\n{'─' * 70}")
        print(f"  ▶  {movie}  [German]")
        print(f"{'─' * 70}")

        anno_df = load_annotation_csv(movie)

        # ── Human GT ──────────────────────────────────────────────────────────
        print("\n  ┌─ Human (Ground Truth)")
        gt_score = None
        if anno_df is not None and "text" in anno_df.columns and nlp is not None:
            gt_df = (
                anno_df[["anno_idx", "text"]]
                .drop_duplicates(subset="anno_idx")
                .rename(columns={"text": "output_ads"})
                .sort_values("anno_idx")
                .reset_index(drop=True)
            )
            merged_gt = prepare_df(gt_df, anno_df, "Human GT")
            if merged_gt is not None:
                gt_score = german_pronoun_rate_on_continuing(merged_gt, nlp)
        print(f"  └─ Pronoun Rate on Continuing: "
              f"{f'{gt_score:.1f}%' if gt_score is not None else 'N/A'}")
        rows.append({"Movie": movie, "Model": "Human (Ground Truth)",
                     "Pronoun_Rate_Continuing": gt_score})

        # ── Models ────────────────────────────────────────────────────────────
        for model_label, path_tmpl in MODELS.items():
            print(f"\n  ┌─ {model_label}")
            model_path = path_tmpl.format(movie=movie)
            gen_df = load_gen_csv(model_path, model_label)
            score  = None
            if gen_df is not None and nlp is not None:
                merged = prepare_df(gen_df, anno_df, model_label)
                if merged is not None:
                    score = german_pronoun_rate_on_continuing(merged, nlp)
            print(f"  └─ Pronoun Rate on Continuing: "
                  f"{f'{score:.1f}%' if score is not None else 'N/A'}")
            rows.append({"Movie": movie, "Model": model_label,
                         "Pronoun_Rate_Continuing": score})

    # ── save & print ──────────────────────────────────────────────────────────
    results_df = pd.DataFrame(rows, columns=["Movie", "Model",
                                              "Pronoun_Rate_Continuing"])
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n\nResults saved → {OUTPUT_CSV}")

    display_df = results_df.copy()
    display_df["Pronoun_Rate_Continuing"] = display_df["Pronoun_Rate_Continuing"].apply(
        lambda v: f"{v:.1f}%" if v is not None else "N/A"
    )

    pd.set_option("display.max_colwidth", 35)
    pd.set_option("display.width", 100)

    print("\n" + "═" * 70)
    print("  GERMAN PRONOUN USAGE RATE ON CONTINUING SHOTS")
    print("═" * 70)
    print(display_df.to_string(index=False))
    print("═" * 70)


if __name__ == "__main__":
    main()
