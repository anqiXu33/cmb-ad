"""
Phase 4 – Inter-shot Coherence Evaluation Script (spaCy dependency parsing).

Usage:
    python evaluate_coherence.py --result_csv <path> --language <German|Italian>
                                 [--input_csv <path>]  [--output_dir <path>]

    --input_csv   Path to the annotation CSV that carries 'pred_ids' and
                  'shot_label' (e.g. <movie>_context_faces_threads_shot_scales.csv).
                  Required when result_csv has no pre-computed 'char_continuity'.

Example:
    python evaluate_coherence.py \\
        --result_csv results/stage2/cmb_ad/Moskau_einfach/stage2.csv \\
        --input_csv  resources/annotations/Moskau_einfach_context_faces_threads_shot_scales.csv \\
        --language   German

Dependencies:
    pip install spacy
    python -m spacy download de_core_news_sm   # German
    python -m spacy download it_core_news_sm   # Italian

Metric definition
-----------------
``char_continuity`` is derived from CHARACTER OVERLAP between consecutive ADs:
the m-* (center) shots' pred_ids are unioned per AD, and adjacent ADs are
compared by set intersection.  This directly captures "is the previous AD's
character still on screen, so a pronoun would be unambiguous?" — the question
the metric is supposed to answer.

  char_continuity = True   prev ∩ curr ≠ ∅  (continuing — pronoun OK)
  char_continuity = False  prev ∩ curr = ∅  (cross-cut — must reintroduce)
  char_continuity = None   either side empty (no detected character →
                          metric N/A; AD is excluded from precision calc)

The previous thread-string-equality logic was abandoned because the thread
column carries per-AD-window LOCAL cluster topology (indices 0..N-1), not
cross-AD identifiers, so string equality between adjacent ADs was a structural-
pattern match, not scene continuity.  Manual video inspection on Moskau_einfach
confirmed both directions of disagreement.  See P2 audit for the data.

Language-specific evaluators (unchanged):

  German – Descriptive Re-introduction
      After a cross-cut (char_continuity == False), the root verb's subject must
      be a NOUN or PROPN.  PRON (Er/Sie/Es/…) signals a failed re-introduction.
      Subject dependency labels checked: 'sb' (TIGER) and 'nsubj' (UD).
      Metric: Re-introduction Precision = % cross-cut shots whose root subject
              is NOUN or PROPN.

  Italian – Pro-drop Rate
      On continuing threads (char_continuity == True), the AD must not contain
      any token where pos_ == 'PRON' AND dep_ == 'nsubj'.  A verb-first or
      name-first start are both acceptable.
      Metric: Pro-drop Precision = % continuing shots with no PRON nsubj.

Input CSV columns expected:
    output_ads      – generated AD sentence (Stage-2 output)
                      Aliases accepted: text_gen, generated_ad, summarized_AD
    anno_idx        – AD identifier shared with the annotation CSV
    char_continuity  – optional pre-computed flag; otherwise re-derived from
                      the annotation CSV's pred_ids + shot_label columns

Outputs:
    Plain-text summary report printed to stdout and saved to
    results/<parent>_<file>_coherence.txt
"""

import argparse
import ast
import os
import sys

import pandas as pd
import spacy
from spacy.language import Language


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

# German subject dependency labels (TIGER + Universal Dependencies)
DE_SUBJ_DEPS = {"sb", "nsubj", "nsubj:pass", "nsubjpass"}

# Italian nominal-subject dependency label
IT_SUBJ_DEP = "nsubj"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _normalise(text) -> str:
    """Return stripped string, or empty string if null/NaN."""
    if pd.isna(text):
        return ""
    return str(text).strip()


def _parse_pred_ids_str(val) -> set:
    """
    Parse a single per-shot pred_ids cell (e.g. "{'16658215': [(26, 1), ...]}")
    into a set of character ID strings.  Empty / NaN / unparseable → empty set.
    """
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


def _derive_char_continuity(pred_ids_series: pd.Series) -> pd.Series:
    """
    Derive char_continuity by character overlap between consecutive ADs.

    pred_ids_series: per-AD set of character ID strings (the m-* union from
    _merge_pred_ids_col).  Caller is responsible for sorting by anno_idx.

    Returns:
      True   curr ∩ prev is non-empty            (continuing — pronoun OK)
      False  both non-empty but disjoint          (cross-cut — must reintroduce)
      None   either side empty                    (undetermined — N/A;
             excluded from precision calculation by the language evaluators)

    Rationale: pred_ids overlap is the direct signal for "can the AD writer
    pronoun a previously-seen character?".  When no character is detected in
    one of the ADs, the metric question is not meaningful — the AD is about
    environment / objects / abstract content — and contributing it to either
    subset would contaminate the precision numerator/denominator.
    """
    out: list = [None]  # first AD has no predecessor
    for i in range(1, len(pred_ids_series)):
        prev = pred_ids_series.iloc[i - 1]
        curr = pred_ids_series.iloc[i]
        if not isinstance(prev, set) or not isinstance(curr, set):
            out.append(None)
        elif not prev or not curr:
            out.append(None)
        else:
            out.append(bool(prev & curr))
    return pd.Series(out, index=pred_ids_series.index, dtype=object)


def _root_subject(doc) -> spacy.tokens.Token | None:
    """
    Return the direct subject child of the ROOT token, or None if absent.
    Checks all ROOT tokens (handles rare cases with multiple roots).
    """
    for token in doc:
        if token.dep_ == "ROOT":
            for child in token.children:
                if child.dep_ in DE_SUBJ_DEPS:
                    return child
    return None


# ------------------------------------------------------------------
# German: Descriptive Re-introduction  (spaCy de_core_news_sm)
# ------------------------------------------------------------------


def evaluate_german(df: pd.DataFrame, nlp: Language) -> dict:
    """
    For each cross-cut shot (char_continuity == False) — ADs whose detected
    characters are disjoint from the previous AD's:
      - Parse with spaCy.
      - Find the root verb's subject via dep_ in DE_SUBJ_DEPS.
      - VIOLATION  : subject pos_ == PRON
      - SUCCESS    : subject pos_ in {NOUN, PROPN}
      - UNDETERMINED: no grammatical subject found (e.g. scene-setting shots)

    ADs with char_continuity is None (either side of the boundary has no
    detected character) are excluded from the cross-cut subset entirely —
    they are reported as ``n_excluded_undetermined_thread`` but do not
    affect the precision calculation.
    """
    n_excluded = int(df["char_continuity"].isna().sum())
    cross_cut = df[df["char_continuity"] == False].copy()
    n_cross = len(cross_cut)
    if n_cross == 0:
        return {
            "n_cross_cut": 0,
            "n_excluded_undetermined_thread": n_excluded,
            "reintr_precision": None,
            "note": "No cross-cut shots found.",
        }

    violations = []  # subject is PRON
    successes = []  # subject is NOUN / PROPN
    undetermined = []  # no subject token found, or empty AD

    for _, row in cross_cut.iterrows():
        ad = _normalise(row["output_ads"])
        if not ad:
            undetermined.append({"text": ad, "subj_token": None, "subj_pos": None})
            continue

        doc = nlp(ad)
        subj = _root_subject(doc)

        if subj is None:
            undetermined.append({"text": ad, "subj_token": None, "subj_pos": None})
        elif subj.pos_ == "PRON":
            violations.append(
                {"text": ad, "subj_token": subj.text, "subj_pos": subj.pos_}
            )
        elif subj.pos_ in {"NOUN", "PROPN"}:
            successes.append(
                {"text": ad, "subj_token": subj.text, "subj_pos": subj.pos_}
            )
        else:
            # e.g. subject is NUM, ADJ, etc. — not a pronoun violation
            undetermined.append(
                {"text": ad, "subj_token": subj.text, "subj_pos": subj.pos_}
            )

    n_viol = len(violations)
    n_succ = len(successes)
    # Precision: of all cross-cut shots, what fraction avoided a PRON subject?
    precision = (n_cross - n_viol) / n_cross

    return {
        "n_cross_cut": n_cross,
        "n_excluded_undetermined_thread": n_excluded,
        "n_pron_subject": n_viol,
        "n_noun_subject": n_succ,
        "n_undetermined": len(undetermined),
        "reintr_precision": round(precision * 100, 1),
        "violations": violations[:5],
        "successes": successes[:5],
    }


# ------------------------------------------------------------------
# Italian: Pro-drop Rate  (spaCy it_core_news_sm)
# ------------------------------------------------------------------


def evaluate_italian(df: pd.DataFrame, nlp: Language) -> dict:
    """
    For each continuing-thread shot (char_continuity == True) — ADs that share
    at least one detected character with the previous AD:
      - Parse with spaCy.
      - Scan ALL tokens for pos_ == 'PRON' AND dep_ == 'nsubj'.
      - VIOLATION  : at least one such token found.
      - SUCCESS    : no such token found anywhere in the sentence.

    ADs with char_continuity is None (either side has no detected character)
    are excluded from the continuing subset.
    """
    n_excluded = int(df["char_continuity"].isna().sum())
    continuing = df[df["char_continuity"] == True].copy()
    n_cont = len(continuing)
    if n_cont == 0:
        return {
            "n_continuing": 0,
            "n_excluded_undetermined_thread": n_excluded,
            "prodrop_precision": None,
            "note": "No continuing-thread shots found.",
        }

    violations = []  # explicit PRON nsubj found
    successes = []  # no PRON nsubj

    for _, row in continuing.iterrows():
        ad = _normalise(row["output_ads"])
        if not ad:
            # Empty output counts as a success (no pronoun used)
            successes.append({"text": ad, "pron_token": None})
            continue

        doc = nlp(ad)
        pron_subj = next(
            (t for t in doc if t.pos_ == "PRON" and t.dep_ == IT_SUBJ_DEP),
            None,
        )

        if pron_subj is not None:
            violations.append({"text": ad, "pron_token": pron_subj.text})
        else:
            successes.append({"text": ad, "pron_token": None})

    n_viol = len(violations)
    precision = (n_cont - n_viol) / n_cont

    return {
        "n_continuing": n_cont,
        "n_excluded_undetermined_thread": n_excluded,
        "n_pron_nsubj": n_viol,
        "n_no_pron_nsubj": len(successes),
        "prodrop_precision": round(precision * 100, 1),
        "violations": violations[:5],
        "successes": successes[:5],
    }


# ------------------------------------------------------------------
# Thread-column merge helper
# ------------------------------------------------------------------


def _merge_pred_ids_col(result_df: pd.DataFrame, anno_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'pred_ids_set' column to result_df: per-AD union of pred_ids over
    the AD's m-* (center) shots, parsed into a Python set of character IDs.

    Why the m-* union (vs. just m-0 or all 5 shots): the AD's narration window
    runs from AD_start to AD_end and is covered by the m-* shots; l-* / r-*
    are past / future context.  Stage 1's current behaviour takes only the
    first m-* shot, dropping characters that appear only in m-1 or m-2 (~9 %
    of ADs across the 8 thesis films).  Re-deriving the union here makes the
    coherence metric independent of that Stage-1 limitation.

    The annotation CSV has one row per (AD × context shot); we group by
    anno_idx and union pred_ids across rows whose shot_label starts with
    "m-".  ADs with no detected face in any m-* shot get an empty set, which
    _derive_char_continuity maps to "undetermined" (excluded from precision).

    The returned DataFrame is sorted by anno_idx and re-indexed, so callers
    can rely on iloc[i-1] being the immediately preceding AD.
    """
    if "pred_ids" not in anno_df.columns:
        raise ValueError("annotation CSV does not contain a 'pred_ids' column.")
    if "shot_label" not in anno_df.columns:
        raise ValueError("annotation CSV does not contain a 'shot_label' column.")
    if "anno_idx" not in result_df.columns or "anno_idx" not in anno_df.columns:
        raise ValueError("Both result_df and anno_df must have an 'anno_idx' column.")

    m_rows = anno_df[anno_df["shot_label"].astype(str).str.startswith("m-", na=False)]
    if len(m_rows) == 0:
        raise ValueError("No m-* shots found in annotation CSV — cannot derive pred_ids.")

    def _union(group):
        keys: set = set()
        for s in group["pred_ids"]:
            keys.update(_parse_pred_ids_str(s))
        return keys

    pred_map = m_rows.groupby("anno_idx").apply(_union)

    result_df = result_df.copy()
    result_df["pred_ids_set"] = result_df["anno_idx"].map(pred_map)
    # ADs not present in anno (shouldn't happen) → empty set, not NaN
    result_df["pred_ids_set"] = result_df["pred_ids_set"].apply(
        lambda x: x if isinstance(x, set) else set()
    )
    # Sort so consecutive-row comparison in _derive_char_continuity is correct
    result_df = result_df.sort_values("anno_idx").reset_index(drop=True)

    n_total = len(result_df)
    n_with_chars = sum(1 for s in result_df["pred_ids_set"] if s)
    print(
        f"  Merged pred_ids (m-* union) from anno_df via anno_idx "
        f"({n_with_chars}/{n_total} ADs with detected chars; "
        f"{n_total - n_with_chars} empty → undetermined)."
    )
    return result_df


# ------------------------------------------------------------------
# Report formatter
# ------------------------------------------------------------------


def format_report(language: str, metrics: dict, csv_path: str) -> str:
    lines = []
    sep = "=" * 66
    lines += [
        sep,
        f"  Inter-shot Coherence Report  –  {language}  (spaCy)",
        f"  Input : {os.path.basename(csv_path)}",
        sep,
    ]

    if language == "German":
        n = metrics["n_cross_cut"]
        prec = metrics["reintr_precision"]
        n_excluded = metrics.get("n_excluded_undetermined_thread", 0)
        lines += [
            "",
            "  Metric : Descriptive Re-introduction Precision",
            "  Method : pred_ids overlap → char_continuity; spaCy root-subject POS",
            "  -----------------------------------------------",
            f"  Cross-cut shots evaluated       : {n}",
            f"  Excluded (undetermined thread)  : {n_excluded}",
        ]
        if prec is None:
            lines.append(f"  Result : N/A  ({metrics.get('note', '')})")
        else:
            lines += [
                f"  Subject is PRON  (violation)    : {metrics['n_pron_subject']}",
                f"  Subject is NOUN/PROPN (correct) : {metrics['n_noun_subject']}",
                f"  No subject found (undetermined) : {metrics['n_undetermined']}",
                "",
                f"  RE-INTRODUCTION PRECISION : {prec}%",
                "  (= % cross-cut shots where root subject is NOT a pronoun)",
            ]
            if metrics["violations"]:
                lines += [
                    "",
                    "  --- Violations: PRON subject on cross-cut (first 5) ---",
                ]
                for v in metrics["violations"]:
                    lines.append(
                        f"    * [{v['subj_token']} / {v['subj_pos']}]  {v['text']}"
                    )
            if metrics["successes"]:
                lines += [
                    "",
                    "  --- Correct re-introductions: NOUN/PROPN subject (first 5) ---",
                ]
                for s in metrics["successes"]:
                    lines.append(
                        f"    + [{s['subj_token']} / {s['subj_pos']}]  {s['text']}"
                    )

    elif language == "Italian":
        n = metrics["n_continuing"]
        prec = metrics["prodrop_precision"]
        n_excluded = metrics.get("n_excluded_undetermined_thread", 0)
        lines += [
            "",
            "  Metric : Pro-drop Precision (Italian)",
            "  Method : pred_ids overlap → char_continuity; spaCy PRON+nsubj scan",
            "  -----------------------------------------------",
            f"  Continuing-thread shots evaluated : {n}",
            f"  Excluded (undetermined thread)    : {n_excluded}",
        ]
        if prec is None:
            lines.append(f"  Result : N/A  ({metrics.get('note', '')})")
        else:
            lines += [
                f"  Explicit PRON nsubj (violation)  : {metrics['n_pron_nsubj']}",
                f"  No PRON nsubj found (correct)    : {metrics['n_no_pron_nsubj']}",
                "",
                f"  PRO-DROP PRECISION : {prec}%",
                "  (= % continuing shots with no explicit subject pronoun)",
            ]
            if metrics["violations"]:
                lines += [
                    "",
                    "  --- Violations: PRON nsubj on continuing thread (first 5) ---",
                ]
                for v in metrics["violations"]:
                    lines.append(f"    * [{v['pron_token']}]  {v['text']}")
            if metrics["successes"]:
                lines += ["", "  --- Correct pro-drop sentences (first 5) ---"]
                for s in metrics["successes"]:
                    lines.append(f"    +  {s['text']}")

    lines += ["", sep]
    return "\n".join(lines)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate inter-shot coherence of generated Swiss ADs (spaCy)."
    )
    parser.add_argument(
        "--result_csv", required=True, help="Path to a Stage-2 output CSV."
    )
    parser.add_argument(
        "--language",
        required=True,
        choices=["German", "Italian"],
        help="Target language of the generated ADs.",
    )
    parser.add_argument(
        "--input_csv",
        default=None,
        help="Annotation CSV with 'pred_ids' and 'shot_label' columns "
        "(e.g. <movie>_context_faces_threads_shot_scales.csv). "
        "When provided, char_continuity is re-derived from pred_ids overlap. "
        "Required unless result_csv already has an 'char_continuity' column.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to save the report (default: same dir as result_csv).",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load spaCy model
    # ------------------------------------------------------------------
    model_name = "de_core_news_sm" if args.language == "German" else "it_core_news_sm"
    try:
        nlp = spacy.load(model_name)
    except OSError:
        print(f"[ERROR] spaCy model '{model_name}' not found.")
        print(f"  Install it with:  python -m spacy download {model_name}")
        sys.exit(1)
    print(f"Loaded spaCy model: {model_name}")

    # ------------------------------------------------------------------
    # Load result CSV
    # ------------------------------------------------------------------
    if not os.path.exists(args.result_csv):
        print(f"[ERROR] File not found: {args.result_csv}")
        sys.exit(1)

    df = pd.read_csv(args.result_csv)
    print(f"Loaded {len(df)} rows from {args.result_csv}")
    print(f"Columns: {df.columns.tolist()}\n")

    # ------------------------------------------------------------------
    # Resolve output_ads column
    # ------------------------------------------------------------------
    if "output_ads" not in df.columns:
        for candidate in ("text_gen", "generated_ad", "summarized_AD"):
            if candidate in df.columns:
                df = df.rename(columns={candidate: "output_ads"})
                print(f"  Mapped column '{candidate}' -> 'output_ads'")
                break
        else:
            print(
                "[ERROR] Could not find an output AD column. "
                "Expected one of: output_ads, text_gen, generated_ad, summarized_AD"
            )
            sys.exit(1)

    # ------------------------------------------------------------------
    # Resolve char_continuity column
    #   Strategy: if --input_csv (annotation CSV) is provided, always re-derive
    #   char_continuity from pred_ids overlap (m-* union per AD) — this is the
    #   trustworthy signal.  Pre-computed char_continuity from old runs is
    #   ignored unless no annotation CSV is available.
    # ------------------------------------------------------------------
    if args.input_csv:
        if not os.path.exists(args.input_csv):
            print(f"[ERROR] --input_csv not found: {args.input_csv}")
            sys.exit(1)
        input_df = pd.read_csv(args.input_csv)
        print(f"Loaded annotation CSV ({len(input_df)} rows): {args.input_csv}")
        try:
            df = _merge_pred_ids_col(df, input_df)
        except (ValueError, KeyError) as e:
            print(f"[ERROR] pred_ids merge failed: {e}")
            sys.exit(1)
        df["char_continuity"] = _derive_char_continuity(df["pred_ids_set"])
    elif "char_continuity" in df.columns or "is_same_thread" in df.columns:
        # Fallback: trust the pre-computed flag (old runs / oracle eval).
        # Pre-rename CSVs use the legacy column name 'is_same_thread'; the
        # value semantics are identical.
        src_col = "char_continuity" if "char_continuity" in df.columns else "is_same_thread"
        if src_col == "is_same_thread":
            print("  [INFO] Using legacy column 'is_same_thread' (pre-rename CSV).")
        df["char_continuity"] = df[src_col].map(
            lambda x: (
                None if pd.isna(x) else
                (str(x).strip().lower() == "true" if isinstance(x, str) else bool(x))
            )
        )
    else:
        print(
            "[ERROR] Result CSV has no 'char_continuity' (or legacy "
            "'is_same_thread') column and no --input_csv was given to "
            "derive one from pred_ids."
        )
        print("  Pass --input_csv <path-to-annotation-CSV>.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    if args.language == "German":
        metrics = evaluate_german(df, nlp)
    else:
        metrics = evaluate_italian(df, nlp)

    report = format_report(args.language, metrics, args.result_csv)
    print(report)

    # ------------------------------------------------------------------
    # Save report
    # ------------------------------------------------------------------
    # out_dir  = args.output_dir or os.path.dirname(os.path.abspath(args.result_csv))
    # os.makedirs(out_dir, exist_ok=True)
    # lang_tag = args.language.lower()
    # out_path = os.path.join(out_dir, f"coherence_report_{lang_tag}.txt")
    # with open(out_path, "w", encoding="utf-8") as fh:
    #     fh.write(report + "\n")
    # out_dir  = args.output_dir or os.path.dirname(os.path.abspath(args.result_csv))
    # os.makedirs(out_dir, exist_ok=True)
    # lang_tag = args.language.lower()

    # # 获取当前 csv 的名字，比如 "La_Tentazione_Di_Esistere_final_AD"
    # base_name = os.path.splitext(os.path.basename(args.result_csv))[0]

    # # 把电影名字拼到报告文件名里！
    # out_path = os.path.join(out_dir, f"{base_name}_coherence_report.txt")

    # with open(out_path, "w", encoding="utf-8") as fh:
    #     fh.write(report + "\n")
    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.result_csv))
    os.makedirs(out_dir, exist_ok=True)

    # 1. take the file name without extension (e.g. stage2_qwen3_single)
    file_name = os.path.splitext(os.path.basename(args.result_csv))[0]

    # 2. take the parent directory name (e.g. results) to distinguish from other files with the same name in different folders
    parent_dir = os.path.basename(os.path.dirname(args.result_csv))

    # 3. combine parent directory and file name to create a unique report name
    out_path = os.path.join(out_dir, f"{parent_dir}_{file_name}_coherence.txt")

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    print(f"\nReport saved -> {out_path}")


if __name__ == "__main__":
    main()
