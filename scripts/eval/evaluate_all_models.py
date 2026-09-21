#!/usr/bin/env python3
"""
evaluate_all_models.py  —  Master Thesis Final Evaluation Script
=================================================================
Evaluates three audio-description models across eight movies and writes
a single aggregated CSV table ready for inclusion in the thesis.

Models  (all read results/stage2/<model>/<movie>/stage2.csv)
------
  Baseline 1  (No CMB ablation, Qwen3.5-27B + thinking, --no_cmb)
  Baseline 2  (Always-ID,        Qwen3.5-27B + thinking)
  CMB-AD      (Memory-augmented, Qwen3.5-27B + thinking + full CMB) — ours

Metrics
-------
  BLEU          sacrebleu corpus BLEU (0–100 scale)
  CHRF          sacrebleu corpus chrF (character-level F-score, 0–100
                scale) — added as a complement to BLEU for morpho-
                logically rich languages (German compounds, Italian
                inflection) where word-level n-gram overlap is brittle.
  BERTScore F1  mean BERTScore F1, language-aware (0–1 scale)
  Coherence     Pro-drop Precision (Italian) or Re-introduction
                Precision (German)  — delegated entirely to the
                existing evaluate_coherence.py functions.

Output
------
  results/benchmarks/thesis_final_results.csv
  Pretty-printed table to stdout.

Usage
-----
  # from the project root:
  python scripts/eval/evaluate_all_models.py

  # or make it executable and run directly:
  chmod +x scripts/evaluate_all_models.py
  ./scripts/evaluate_all_models.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# Standard-library imports  (always first, no side-effects)
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys
import warnings
import logging

# ─────────────────────────────────────────────────────────────────────────────
# Silence noisy third-party warnings BEFORE importing heavy libraries.
# This prevents the HuggingFace tokenizers fork-safety warning, the
# transformers "model not on GPU" note, and similar noise.
# ─────────────────────────────────────────────────────────────────────────────
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.disable(logging.WARNING)

# ─────────────────────────────────────────────────────────────────────────────
# Third-party imports
# ─────────────────────────────────────────────────────────────────────────────
import pandas as pd
import spacy
import evaluate as hf_evaluate   # HuggingFace evaluate; aliased to avoid
                                  # shadowing the built-in evaluate()

# ─────────────────────────────────────────────────────────────────────────────
# Make evaluate_coherence.py importable regardless of where the script is
# invoked from.  We add both the script's own directory (scripts/) and the
# project root (shot-by-shot/) to sys.path so Python can find the module.
# ─────────────────────────────────────────────────────────────────────────────
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))               # …/scripts/eval/
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))             # …/shot-by-shot/

for _candidate in (_SCRIPT_DIR, _PROJECT_ROOT):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

# Import only the four public functions we need from evaluate_coherence.py.
# The module performs no work on import — all heavy lifting is inside functions.
from evaluate_coherence import (
    evaluate_german,
    evaluate_italian,
    _merge_pred_ids_col,
    _derive_char_continuity,
)

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION  — edit these constants to adapt to a different experiment
# ═════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = _PROJECT_ROOT

# ── Movies and their BERTScore / spaCy language codes ────────────────────────
# Keys are ISO-639-1 codes accepted by HuggingFace evaluate and spaCy.
MOVIES: dict[str, list[str]] = {
    "it": [
        "La_Tentazione_Di_Esistere",
        "Quanto_basta",
        "Stuerm",
        "Riders_of_justice",
    ],
    "de": [
        "8_Tage_im_August",
        "Moskau_einfach",
        "Baghdad_in_my_Shadow",
        "Trommelwirbel",
    ],
}

# Flat ordered list of (movie, lang) — determines table row order
MOVIE_LANG: list[tuple[str, str]] = [
    (movie, lang)
    for lang, movies in MOVIES.items()
    for movie in movies
]

# ── File-path templates  ({movie} is expanded via str.format) ─────────────────

# Baseline 1 (No CMB ablation, Qwen3.5-27B + thinking, --no_cmb).
# Local run produced by scripts/slurm/stage2/baseline1_no_cmb/.
BASELINE1_PATH = os.path.join(
    PROJECT_ROOT, "results", "stage2", "baseline1_no_cmb", "{movie}", "stage2.csv"
)

# Baseline 2 (Always-ID, Qwen3.5-27B + thinking).
BASELINE2_PATH = os.path.join(
    PROJECT_ROOT, "results", "stage2", "baseline2_always_id", "{movie}", "stage2.csv"
)

# CMB-AD (our method, Qwen3.5-27B + thinking + full CMB).
CMBAD_PATH = os.path.join(
    PROJECT_ROOT, "results", "stage2", "cmb_ad", "{movie}", "stage2.csv"
)

# Annotation CSVs supply both the ground-truth AD text and the character-thread
# IDs that coherence evaluation requires.
# ① Try cluster (canonical location) first, then ② local resources/ copy.
ANNO_PATHS: list[str] = [
    os.path.join(
        PROJECT_ROOT, "resources", "annotations",
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

# All models to evaluate: display-label → path template.
# "GT" is a sentinel value — the Human (Ground Truth) row has no separate
# CSV; it reads directly from the annotation CSV already loaded per movie.
_GT_SENTINEL = "GT"

MODELS: dict[str, str] = {
    "Baseline 1 (No-CMB)":    BASELINE1_PATH,
    "Baseline 2 (Always-ID)": BASELINE2_PATH,
    "CMB-AD (Ours)":          CMBAD_PATH,
    "Human (Ground Truth)":   _GT_SENTINEL,
}

# Where to write the final aggregated table
OUTPUT_CSV = os.path.join(PROJECT_ROOT, "results", "benchmarks", "thesis_final_results.csv")

# Column aliases that Stage-2 scripts use for the generated AD text.
# The coherence functions (and our internal logic) always use "output_ads".
_GEN_COL_ALIASES = ("output_ads", "text_gen", "generated_ad", "summarized_AD")

# ═════════════════════════════════════════════════════════════════════════════
# ONE-TIME METRIC OBJECT INITIALISATION
# Loading these once at module level avoids re-downloading model weights on
# every call and lets us reuse the same tokenizer across all 24 evaluations.
# ═════════════════════════════════════════════════════════════════════════════

print("Initialising BLEU metric …")
_bleu_metric = hf_evaluate.load("sacrebleu")

print("Initialising CHRF metric …")
_chrf_metric = hf_evaluate.load("chrf")

print("Initialising BERTScore metric …")
_bertscore_metric = hf_evaluate.load("bertscore")

# ─────────────────────────────────────────────────────────────────────────────
# spaCy model cache — each language model is loaded at most once per run
# ─────────────────────────────────────────────────────────────────────────────
_NLP_CACHE: dict[str, spacy.language.Language] = {}

_SPACY_MODELS = {"de": "de_core_news_sm", "it": "it_core_news_sm"}


def get_nlp(lang: str) -> spacy.language.Language | None:
    """
    Return a (possibly cached) spaCy Language object for *lang*.

    Parameters
    ----------
    lang : "de" or "it"

    Returns
    -------
    spacy.language.Language, or None if the model is not installed.
    Prints an installation hint when the model is missing so the user
    knows exactly what command to run.
    """
    if lang in _NLP_CACHE:
        return _NLP_CACHE[lang]

    model_name = _SPACY_MODELS.get(lang)
    if model_name is None:
        print(f"  [WARN] No spaCy model configured for lang='{lang}'.")
        return None

    try:
        nlp = spacy.load(model_name)
        _NLP_CACHE[lang] = nlp
        print(f"  Loaded spaCy model: {model_name}")
        return nlp
    except OSError:
        print(
            f"  [WARN] spaCy model '{model_name}' not installed.\n"
            f"         Run:  python -m spacy download {model_name}"
        )
        return None


# ═════════════════════════════════════════════════════════════════════════════
# I/O HELPERS
# ═════════════════════════════════════════════════════════════════════════════


def load_annotation_csv(movie: str) -> pd.DataFrame | None:
    """
    Load the annotation CSV for *movie*, trying multiple candidate paths.

    The annotation CSV contains:
      - ``anno_idx``  (shot identifier shared with generated CSVs)
      - ``thread``    (character-thread ID needed for coherence)
      - ``text``      (ground-truth AD, used as a reference fallback)

    Returns
    -------
    pd.DataFrame on success, None if no candidate path exists.
    """
    for tmpl in ANNO_PATHS:
        path = tmpl.format(movie=movie)
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"  Annotation CSV loaded ({len(df)} rows): {os.path.relpath(path, PROJECT_ROOT)}")
            return df

    print(
        f"  [WARN] Annotation CSV not found for '{movie}'. "
        "Coherence evaluation will be skipped for this movie."
    )
    return None


def resolve_gen_col(df: pd.DataFrame, label: str) -> pd.DataFrame | None:
    """
    Normalise the generated-text column to the canonical name ``output_ads``.

    Stage-2 scripts use different column names (text_gen, generated_ad, …).
    The coherence functions in evaluate_coherence.py all expect ``output_ads``,
    so we rename whichever alias is present.

    Returns
    -------
    DataFrame with the column renamed, or None if no alias is found.
    """
    for alias in _GEN_COL_ALIASES:
        if alias in df.columns:
            if alias != "output_ads":
                df = df.rename(columns={alias: "output_ads"})
            return df

    print(
        f"  [WARN] Could not find a generated-text column in the CSV "
        f"for {label}. Expected one of: {_GEN_COL_ALIASES}. Skipping."
    )
    return None


# ═════════════════════════════════════════════════════════════════════════════
# METRIC COMPUTATION
# ═════════════════════════════════════════════════════════════════════════════


def get_aligned_pairs(
    gen_df: pd.DataFrame,
    anno_df: pd.DataFrame | None,
    label: str,
) -> tuple[list[str], list[str]] | tuple[None, None]:
    """
    Build aligned (prediction, reference) lists suitable for BLEU / BERTScore.

    Primary strategy  – use ``text_gt`` embedded in the generated CSV.
      Stage-2 scripts copy the ground-truth text into a ``text_gt`` column, so
      we don't need to re-join with the annotation CSV in the common case.

    Fallback strategy – join against the annotation CSV on ``anno_idx``.
      Used when ``text_gt`` is absent (e.g. a custom pipeline that didn't
      propagate the column).

    In both cases we drop rows where either side is NaN or empty to prevent
    sacrebleu / BERTScore from receiving blank strings, which would inflate or
    deflate scores unpredictably.

    Returns
    -------
    (predictions, references) as plain Python ``list[str]``, or (None, None).
    """
    # ── Primary: text_gt is already in the generated CSV ─────────────────────
    if "text_gt" in gen_df.columns:
        pair_df = gen_df[["output_ads", "text_gt"]].copy()
        pair_df = pair_df.dropna(subset=["output_ads", "text_gt"])
        pair_df = pair_df[pair_df["output_ads"].str.strip() != ""]
        pair_df = pair_df[pair_df["text_gt"].str.strip() != ""]

        if not pair_df.empty:
            return pair_df["output_ads"].tolist(), pair_df["text_gt"].tolist()

    # ── Fallback: join anno_df on anno_idx ───────────────────────────────────
    if anno_df is None:
        print(f"  [WARN] No text_gt column and no annotation CSV for {label}; "
              "BLEU/BERTScore skipped.")
        return None, None

    if "text" not in anno_df.columns:
        print(f"  [WARN] Annotation CSV has no 'text' column for {label}.")
        return None, None

    if "anno_idx" not in gen_df.columns:
        print(f"  [WARN] Generated CSV has no 'anno_idx' column for {label}.")
        return None, None

    # Deduplicate annotation rows (one row per face → one row per shot)
    gt_map = (
        anno_df[["anno_idx", "text"]]
        .drop_duplicates(subset="anno_idx")
        .rename(columns={"text": "_ref"})
    )

    merged = gen_df[["anno_idx", "output_ads"]].merge(gt_map, on="anno_idx", how="inner")
    merged = merged.dropna(subset=["output_ads", "_ref"])
    merged = merged[merged["output_ads"].str.strip() != ""]
    merged = merged[merged["_ref"].str.strip() != ""]

    if merged.empty:
        print(f"  [WARN] No valid prediction-reference pairs for {label}.")
        return None, None

    return merged["output_ads"].tolist(), merged["_ref"].tolist()


def compute_bleu(predictions: list[str], references: list[str]) -> float:
    """
    Compute corpus-level BLEU via sacrebleu.

    sacrebleu (as wrapped by HuggingFace ``evaluate``) expects *references*
    as a list-of-lists: each element is a list of reference strings for the
    corresponding prediction (multi-reference setup).  We only have one
    reference per prediction, so we wrap each string in a list.

    Returns
    -------
    float on the 0–100 scale (sacrebleu convention).
    """
    refs_wrapped = [[r] for r in references]   # sacrebleu format
    result = _bleu_metric.compute(predictions=predictions, references=refs_wrapped)
    return round(result["score"], 2)


def compute_chrf(predictions: list[str], references: list[str]) -> float:
    """
    Compute corpus-level chrF via sacrebleu.

    chrF is a character-n-gram F-score (Popović, 2015) that complements BLEU
    on morphologically rich languages: German compound nouns and Italian
    inflectional variants both fragment word-level n-gram overlap, but their
    shared character substrings are still captured.

    Same reference-wrapping convention as BLEU (list-of-lists per prediction).

    Returns
    -------
    float on the 0–100 scale (sacrebleu convention).
    """
    refs_wrapped = [[r] for r in references]
    result = _chrf_metric.compute(predictions=predictions, references=refs_wrapped)
    return round(result["score"], 2)


def compute_bertscore(predictions: list[str], references: list[str], lang: str) -> float:
    """
    Compute mean BERTScore F1.

    Parameters
    ----------
    lang : ISO code ("it" or "de") — selects the pre-trained BERT model that
           matches the target language, which matters for sub-word tokenisation.

    Returns
    -------
    float on the 0–1 scale, rounded to 4 decimal places.
    """
    result = _bertscore_metric.compute(
        predictions=predictions,
        references=references,
        lang=lang,
        verbose=False,
    )
    f1_scores = result["f1"]
    mean_f1 = sum(f1_scores) / len(f1_scores)
    return round(mean_f1, 4)


def compute_coherence(
    gen_df: pd.DataFrame,
    anno_df: pd.DataFrame | None,
    lang: str,
    nlp: spacy.language.Language | None,
    label: str,
) -> float | None:
    """
    Compute the language-specific coherence score by delegating to the
    functions in evaluate_coherence.py.

    Pipeline (mirrors evaluate_coherence.py's ``main()`` logic)
    ──────────────────────────────────────────────────────────────
    1. Merge per-AD pred_ids (m-* union) from *anno_df* into *gen_df* using
       ``_merge_pred_ids_col`` (adds a ``pred_ids_set`` column; output is
       sorted by anno_idx).
    2. Derive ``char_continuity`` with ``_derive_char_continuity`` —
       True/False/None per AD based on pred_ids overlap with the previous AD.
    3. Call ``evaluate_german`` or ``evaluate_italian`` depending on *lang*.
       ADs with char_continuity is None are excluded from the precision calc.
    4. Extract the scalar metric:
         Italian → ``prodrop_precision``
         German  → ``reintr_precision``

    Returns
    -------
    float (score as a percentage, e.g. 78.3) or None when unavailable.
    """
    # ── Guard: prerequisites ─────────────────────────────────────────────────
    if anno_df is None:
        return None   # no thread info → coherence impossible

    if nlp is None:
        return None   # spaCy model missing → coherence impossible

    if "output_ads" not in gen_df.columns:
        print(f"  [WARN] 'output_ads' column missing for coherence in {label}.")
        return None

    # ── Step 1: merge pred_ids (m-* union) from annotation CSV ───────────────
    try:
        merged_df = _merge_pred_ids_col(gen_df.copy(), anno_df)
    except (ValueError, KeyError) as exc:
        print(f"  [WARN] pred_ids merge failed for {label}: {exc}")
        return None

    # ── Step 2: derive char_continuity by character overlap ───────────────────
    # char_continuity == True   → at least one detected character continues
    # char_continuity == False  → detected character sets are disjoint
    # char_continuity is None   → either side has no detected character → N/A
    merged_df["char_continuity"] = _derive_char_continuity(
        merged_df["pred_ids_set"]
    )

    # ── Step 3 & 4: call the appropriate evaluator and extract scalar ─────────
    try:
        if lang == "de":
            metrics = evaluate_german(merged_df, nlp)
            score = metrics.get("reintr_precision")   # % or None
        else:  # "it"
            metrics = evaluate_italian(merged_df, nlp)
            score = metrics.get("prodrop_precision")  # % or None
    except Exception as exc:
        print(f"  [WARN] Coherence evaluation error for {label}: {exc}")
        return None

    return score   # already rounded to 1 d.p. by evaluate_coherence.py


# ═════════════════════════════════════════════════════════════════════════════
# GROUND-TRUTH UPPER-BOUND EVALUATOR
# ═════════════════════════════════════════════════════════════════════════════


def evaluate_gt_for_movie(
    anno_df: pd.DataFrame | None,
    lang: str,
    nlp: spacy.language.Language | None,
) -> dict[str, float | None]:
    """
    Evaluate Human (Ground Truth) coherence using the annotation CSV directly.

    BLEU and BERTScore are intentionally skipped — comparing ground truth to
    itself would always yield a perfect score and provides no useful signal.

    Coherence pipeline
    ------------------
    1. Build a per-shot DataFrame from the annotation CSV.
       The annotation CSV has multiple rows per shot (one per detected face /
       bounding-box), so we deduplicate by ``anno_idx``, keeping one ``text``
       value per shot (the GT AD text is identical for all face-rows of a shot).
    2. Rename ``text`` → ``output_ads`` so the coherence functions can find it.
    3. Delegate to ``compute_coherence``, which will derive per-AD pred_ids
       (m-* union) from the full ``anno_df`` via ``_merge_pred_ids_col`` and
       compute char_continuity by character overlap.

    Returns
    -------
    dict with BLEU=None, BERTScore_F1=None, Coherence_Score=float|None.
    """
    result: dict[str, float | None] = {
        "BLEU": None,           # N/A — GT vs GT is trivially perfect
        "CHRF": None,           # N/A — same reason
        "BERTScore_F1": None,   # N/A — same reason
        "Coherence_Score": None,
    }

    if anno_df is None:
        print("  [SKIP] No annotation CSV — GT coherence cannot be computed.")
        return result

    if "text" not in anno_df.columns:
        print("  [WARN] Annotation CSV has no 'text' column — GT coherence skipped.")
        return result

    # Build a clean per-shot DataFrame (one row per unique anno_idx).
    # drop_duplicates keeps the first occurrence; the GT text is the same across
    # all face-level rows for the same shot, so any row suffices.
    gt_df = (
        anno_df[["anno_idx", "text"]]
        .drop_duplicates(subset="anno_idx")
        .copy()
        .rename(columns={"text": "output_ads"})
        .sort_values("anno_idx")
        .reset_index(drop=True)
    )

    print(f"  Using annotation CSV as GT ({len(gt_df)} shots) for coherence …")

    # compute_coherence handles _merge_pred_ids_col + _derive_char_continuity internally.
    result["Coherence_Score"] = compute_coherence(
        gt_df, anno_df, lang, nlp, label="Human (Ground Truth)"
    )

    return result


# ═════════════════════════════════════════════════════════════════════════════
# PER-MODEL EVALUATOR  (BLEU + BERTScore + Coherence in one call)
# ═════════════════════════════════════════════════════════════════════════════


def evaluate_model_for_movie(
    model_label: str,
    model_path: str,
    movie: str,
    lang: str,
    anno_df: pd.DataFrame | None,
    nlp: spacy.language.Language | None,
) -> dict[str, float | None]:
    """
    Evaluate a single model on a single movie.

    Parameters
    ----------
    model_label : human-readable model name (for warning messages)
    model_path  : fully-expanded path to the model's output CSV
    movie       : movie folder name (for logging)
    lang        : "de" or "it"
    anno_df     : annotation DataFrame (may be None if unavailable)
    nlp         : spaCy Language object (may be None if not installed)

    Returns
    -------
    dict with keys BLEU, BERTScore_F1, Coherence_Score.
    Values are floats when successfully computed, None otherwise.
    The caller will convert None → "N/A" in the display table.
    """
    result: dict[str, float | None] = {
        "BLEU": None,
        "CHRF": None,
        "BERTScore_F1": None,
        "Coherence_Score": None,
    }

    # ── Guard: file existence check ───────────────────────────────────────────
    # os.path.exists() is used (as specified) so missing cluster files or
    # not-yet-generated Baseline 2 files produce a friendly warning, not a crash.
    if not os.path.exists(model_path):
        print(f"  [SKIP] File not found — results will be N/A:\n"
              f"         {model_path}")
        return result

    # ── Load the generated CSV ────────────────────────────────────────────────
    try:
        gen_df = pd.read_csv(model_path)
    except Exception as exc:
        print(f"  [WARN] Failed to read '{model_path}': {exc}")
        return result

    print(
        f"  Loaded {len(gen_df):4d} rows  ← "
        f"{os.path.relpath(model_path, PROJECT_ROOT)}"
    )

    # ── Normalise generated-text column to "output_ads" ──────────────────────
    gen_df = resolve_gen_col(gen_df, model_label)
    if gen_df is None:
        return result   # warning already printed inside resolve_gen_col

    # ── BLEU and BERTScore ───────────────────────────────────────────────────
    # get_aligned_pairs handles: primary (text_gt in CSV) and fallback (anno_df).
    # It drops NaN / empty rows so metrics are never computed on blank strings.
    predictions, references = get_aligned_pairs(gen_df, anno_df, model_label)

    if predictions is not None and len(predictions) > 0:
        print(f"  Computing BLEU on {len(predictions)} sentence pairs …")
        try:
            result["BLEU"] = compute_bleu(predictions, references)
        except Exception as exc:
            print(f"  [WARN] BLEU failed: {exc}")

        print(f"  Computing CHRF on {len(predictions)} sentence pairs …")
        try:
            result["CHRF"] = compute_chrf(predictions, references)
        except Exception as exc:
            print(f"  [WARN] CHRF failed: {exc}")

        print(f"  Computing BERTScore (lang={lang}) on {len(predictions)} pairs …")
        try:
            result["BERTScore_F1"] = compute_bertscore(predictions, references, lang)
        except Exception as exc:
            print(f"  [WARN] BERTScore failed: {exc}")
    else:
        print("  [SKIP] No valid prediction-reference pairs — BLEU/CHRF/BERTScore skipped.")

    # ── Coherence ─────────────────────────────────────────────────────────────
    # compute_coherence internally calls _merge_pred_ids_col + _derive_char_continuity
    # and then the language-specific evaluator from evaluate_coherence.py.
    print("  Computing coherence …")
    result["Coherence_Score"] = compute_coherence(
        gen_df, anno_df, lang, nlp, model_label
    )

    return result


# ═════════════════════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ═════════════════════════════════════════════════════════════════════════════

_LANG_LABEL = {"it": "Italian", "de": "German"}


def _fmt_bleu(v) -> str:
    return f"{v:.2f}" if v is not None else "N/A"

def _fmt_chrf(v) -> str:
    return f"{v:.2f}" if v is not None else "N/A"

def _fmt_bert(v) -> str:
    return f"{v:.4f}" if v is not None else "N/A"

def _fmt_coh(v) -> str:
    return f"{v:.1f}%" if v is not None else "N/A"


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """
    Outer loop: iterate over (movie, language) pairs and then over each model.
    Collect all results into a list of dicts, build a DataFrame, save CSV, and
    pretty-print the final table.
    """
    print("\n" + "═" * 70)
    print("  Master Thesis — Comprehensive Model Evaluation")
    print("  Models:   Baseline 1 | Baseline 2 | CMB-AD (Ours) | Human (GT)")
    print(f"  Movies:   {len(MOVIE_LANG)} ({sum(len(v) for v in MOVIES.values())} total)")
    print("═" * 70)

    rows: list[dict] = []

    for movie, lang in MOVIE_LANG:
        lang_label = _LANG_LABEL[lang]

        print(f"\n{'─' * 70}")
        print(f"  ▶  {movie}  [{lang_label}]")
        print(f"{'─' * 70}")

        # Annotation CSV is shared by all three models for this movie
        anno_df = load_annotation_csv(movie)

        # spaCy model — loaded once per language thanks to _NLP_CACHE
        nlp = get_nlp(lang)

        for model_label, path_tmpl in MODELS.items():
            print(f"\n  ┌─ {model_label}")

            if path_tmpl == _GT_SENTINEL:
                # Ground Truth: no generation CSV — use annotation CSV directly.
                metrics = evaluate_gt_for_movie(anno_df, lang, nlp)
            else:
                model_path = path_tmpl.format(movie=movie)
                metrics = evaluate_model_for_movie(
                    model_label=model_label,
                    model_path=model_path,
                    movie=movie,
                    lang=lang,
                    anno_df=anno_df,
                    nlp=nlp,
                )
            print(
                f"  └─ BLEU={_fmt_bleu(metrics['BLEU'])}  "
                f"CHRF={_fmt_chrf(metrics['CHRF'])}  "
                f"BERTScore={_fmt_bert(metrics['BERTScore_F1'])}  "
                f"Coherence={_fmt_coh(metrics['Coherence_Score'])}"
            )

            rows.append(
                {
                    "Movie":           movie,
                    "Language":        lang_label,
                    "Model":           model_label,
                    "BLEU":            metrics["BLEU"],
                    "CHRF":            metrics["CHRF"],
                    "BERTScore_F1":    metrics["BERTScore_F1"],
                    "Coherence_Score": metrics["Coherence_Score"],
                }
            )

    # ── Assemble final DataFrame ──────────────────────────────────────────────
    results_df = pd.DataFrame(
        rows,
        columns=["Movie", "Language", "Model", "BLEU", "CHRF", "BERTScore_F1", "Coherence_Score"],
    )

    # ── Save to CSV ───────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n\nResults saved  →  {OUTPUT_CSV}")

    # ── Pretty-print the final table ──────────────────────────────────────────
    # Build a display copy that converts None/NaN to "N/A" and formats floats.
    display_df = results_df.copy()
    display_df["BLEU"]            = display_df["BLEU"].apply(_fmt_bleu)
    display_df["CHRF"]            = display_df["CHRF"].apply(_fmt_chrf)
    display_df["BERTScore_F1"]    = display_df["BERTScore_F1"].apply(_fmt_bert)
    display_df["Coherence_Score"] = display_df["Coherence_Score"].apply(_fmt_coh)

    pd.set_option("display.max_colwidth", 28)
    pd.set_option("display.width", 130)

    print("\n" + "═" * 70)
    print("  THESIS FINAL RESULTS TABLE")
    print("═" * 70)
    print(display_df.to_string(index=False))
    print("═" * 70)

    # ── Quick sanity summary: how many cells are unexpectedly N/A? ───────────
    # Human (GT) rows always have BLEU=None and BERTScore_F1=None by design,
    # so we exclude those two columns from the GT rows when counting.
    n_gt_rows   = len(results_df[results_df["Model"] == "Human (Ground Truth)"])
    expected_na = n_gt_rows * 3   # BLEU + CHRF + BERTScore for each GT row
    actual_na   = int(results_df[["BLEU", "CHRF", "BERTScore_F1", "Coherence_Score"]].isna().sum().sum())
    unexpected_na = actual_na - expected_na

    if unexpected_na > 0:
        print(
            f"\n  ⚠  {unexpected_na} unexpected N/A values detected. "
            "This typically means some model files are not yet generated "
            "(Baseline 2) or reside only on the cluster (Baseline 1)."
        )
    else:
        print("\n  ✓  All metric values computed successfully (GT BLEU/CHRF/BERTScore intentionally N/A).")


if __name__ == "__main__":
    main()
