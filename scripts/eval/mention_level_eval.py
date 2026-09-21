#!/usr/bin/env python3
"""
mention_level_eval.py — Mention-level evaluation for thesis §6 main metric.
==========================================================================
Reads NAME / DESC / PRN mentions extracted from system Stage-2 outputs and
ground-truth AD text, aligns each mention to a character via the pred_ids
m-* shot union and the curated character metadata, and computes:

  L3 metrics (mention-level)
  ──────────────────────────
   1. Reference Decision Accuracy (RDA)        — per-(AD, char) form match
   2. Form distribution L1 distance            — overall NAME/DESC/PRN ratio
   3. Mention-level Re-introduction Precision  — cross-cut subset, % non-PRN
   4. Continuing PRN rate (mention-level)      — continuing subset, % PRN
   5. Pronoun-Rate Distance to Human (PRD-H)   — |sys − GT| of (4)

Plus the four exclusion counters (matched / mismatched / ambiguous / stray)
so §6 can quote how much of the per-mention data entered the RDA denominator.

Inputs (per movie × condition)
──────────────────────────────
  System mentions:  ${WORK_DIR}/mention_robustness/<movie>__<condition>/text_gen.mentions
  System anno-idx:  ${WORK_DIR}/mention_robustness/<movie>__<condition>/text_gen.json
  GT mentions:      dispatched (see GT_PATHS) — supervisor's pre-computed
                    mentions_output/ where available, else our own
                    text_gt.mentions in mention_robustness/
  Annotation CSV:   ${WORK_DIR}/annotations/<movie>_*.csv  (for pred_ids)
  Char metadata:    ${WORK_DIR}/annotations/imdb_chars_enriched_manual.json

Outputs
───────
  results/benchmarks/mention_level_results.csv          (per movie × condition)
  results/benchmarks/mention_alignment_<movie>__<cond>.csv  (per (AD, char) detail)

Conservative alignment heuristic (per design discussion 2026-05-08)
───────────────────────────────────────────────────────────────────
  NAME : token-level match against char.role (then char.name fallback);
         best-overlap wins, ties → first
  DESC : substring containment between mention text and char.description
         (lowercase); best-overlap wins
  PRN  : if |pred_ids[anno_idx]| == 1 → bind to that char
         if |pred_ids| > 1            → "AMBIGUOUS" (excluded from RDA)
         if |pred_ids| == 0           → "STRAY"     (excluded)

Mentions matching a char NOT in pred_ids[anno_idx] are tagged "OFFSCREEN":
  they enter form distribution but are excluded from RDA.
Mentions matching no char at all are tagged "STRAY".

RDA = matched / (matched + mismatched), per (AD, char) pair where:
  - char ∈ pred_ids[anno_idx]
  - both GT and SYS have a mention bound to char in this AD
  - first-mention label of each is compared (sufficient per design 2b)

Usage
─────
  python scripts/eval/mention_level_eval.py
  # or limit to one movie for fast iteration:
  python scripts/eval/mention_level_eval.py --movie Moskau_einfach
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

# ───────────────────────────────────────────────────────────────────────────
# Path bootstrap so we can import _merge_pred_ids_col from evaluate_coherence
# ───────────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
for p in (_SCRIPT_DIR, _PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from evaluate_coherence import _merge_pred_ids_col, _derive_char_continuity  # noqa: E402

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

# Try cluster mount first (local dev), then direct cluster path.
_DATA_ROOT_CANDIDATES = [
    Path("data"),
    Path("/shares/iict-sp4.ebling.cl.uzh/data/anxu_MA_data"),
]
DATA_ROOT = next((p for p in _DATA_ROOT_CANDIDATES if p.exists()), None)
if DATA_ROOT is None:
    sys.exit(f"[FATAL] No DATA_ROOT found. Tried: {_DATA_ROOT_CANDIDATES}")

ROBUST_ROOT = DATA_ROOT / "mention_robustness"
MENTIONS_ROOT = DATA_ROOT / "mentions_output"
ANNO_DIR = DATA_ROOT / "annotations"
CHAR_JSON = ANNO_DIR / "imdb_chars_enriched_manual.json"

# Movie → IMDb id (for char metadata lookup) and language
THESIS_FILMS = {
    # German
    "8_Tage_im_August":          {"imdbid": "tt6864088", "lang": "de"},
    "Moskau_einfach":            {"imdbid": "tt10949014", "lang": "de"},
    "Baghdad_in_my_Shadow":      {"imdbid": "tt7117552", "lang": "de"},
    "Trommelwirbel":             {"imdbid": "tt9558602", "lang": "de"},
    # Italian (placeholder imdbids; corrected at runtime from anno CSV's imdbid column)
    "La_Tentazione_Di_Esistere": {"imdbid": None, "lang": "it"},
    "Quanto_basta":              {"imdbid": None, "lang": "it"},
    "Riders_of_justice":         {"imdbid": None, "lang": "it"},
    "Stuerm":                    {"imdbid": None, "lang": "it"},
}

CONDITIONS = ["cmb_ad", "baseline1_no_cmb", "baseline2_always_id"]

# GT mentions dispatch:
#   - Supervisor's pre-computed mentions where available (verified byte-identical
#     to our re-extraction on Moskau + Quanto cases — see chat log 2026-05-10).
#   - Our text_gt.mentions for 8_Tage_im_August + Trommelwirbel (no supervisor file).
GT_DISPATCH = {
    "Moskau_einfach":            MENTIONS_ROOT / "DE/movies/AD_Moskau_einfach.mentions",
    "Baghdad_in_my_Shadow":      MENTIONS_ROOT / "DE/movies/Baghdad_in_my_Shadow_final.mentions",
    "8_Tage_im_August":          ROBUST_ROOT  / "8_Tage_im_August__cmb_ad/text_gt.mentions",
    "Trommelwirbel":             ROBUST_ROOT  / "Trommelwirbel__cmb_ad/text_gt.mentions",
    "La_Tentazione_Di_Esistere": MENTIONS_ROOT / "IT/movies/La_Tentazione_Di_Esistere_ITA.mentions",
    "Quanto_basta":              MENTIONS_ROOT / "IT/movies/Quanto_basta_ITA.mentions",
    "Riders_of_justice":         MENTIONS_ROOT / "IT/movies/Riders_of_justice_ITA.mentions",
    "Stuerm":                    MENTIONS_ROOT / "IT/movies/Stuerm_ITA.mentions",
}

OUTPUT_CSV = _PROJECT_ROOT / "results" / "benchmarks" / "mention_level_results.csv"
DIAG_DIR   = _PROJECT_ROOT / "results" / "benchmarks" / "mention_alignment"

# ═══════════════════════════════════════════════════════════════════════════
# I/O helpers
# ═══════════════════════════════════════════════════════════════════════════


def load_mentions_file(path: Path) -> list[list[dict]]:
    """One JSON list per line; failed JSON → empty list."""
    out = []
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


def load_anno_pred_ids(movie: str) -> tuple[pd.DataFrame, str]:
    """Load anno CSV; return (df with pred_ids_set + char_continuity, imdbid)."""
    anno_path = ANNO_DIR / f"{movie}_context_faces_threads_shot_scales.csv"
    if not anno_path.exists():
        raise FileNotFoundError(f"Anno CSV not found: {anno_path}")
    anno_df = pd.read_csv(anno_path)
    imdbid = str(anno_df["imdbid"].iloc[0])
    # Build a one-row-per-anno_idx frame to feed _merge_pred_ids_col, then derive
    # char_continuity (= renamed is_same_thread, char-overlap based).
    placeholder = pd.DataFrame({"anno_idx": sorted(anno_df["anno_idx"].unique())})
    merged = _merge_pred_ids_col(placeholder, anno_df)
    merged["char_continuity"] = _derive_char_continuity(merged["pred_ids_set"])
    return merged, imdbid


def load_char_meta(imdbid: str) -> list[dict]:
    """Return list of char dicts for this movie from imdb_chars_enriched_manual.json."""
    with open(CHAR_JSON, encoding="utf-8") as f:
        all_chars = json.load(f)
    return all_chars.get(imdbid, [])


# ═══════════════════════════════════════════════════════════════════════════
# Alignment heuristic
# ═══════════════════════════════════════════════════════════════════════════


def _toks(s) -> set[str]:
    """Lowercase whitespace-tokenised set.

    Accepts str or list[str]; flattens lists by space-joining (handles cases
    where char.role is a list because one actor plays two roles, e.g.
    ['Palle Olesen', 'Aharon Nahas Shadid'] in Riders_of_justice).
    """
    if not s:
        return set()
    if isinstance(s, list):
        s = " ".join(str(x) for x in s)
    elif not isinstance(s, str):
        s = str(s)
    return {t.strip(",.!?;:'\"()") for t in s.lower().split() if t.strip(",.!?;:'\"()")}


def align_name(text: str, chars: list[dict]) -> str | None:
    """Token-overlap match against char.role (preferred) and char.name."""
    mtoks = _toks(text)
    if not mtoks:
        return None
    best, best_score = None, 0
    for c in chars:
        for fld in ("role", "name"):
            ctoks = _toks(c.get(fld, ""))
            score = len(mtoks & ctoks)
            if score > best_score:
                best, best_score = c["id"], score
    return best


def align_desc(text: str, chars: list[dict]) -> str | None:
    """Substring containment between mention text and char.description (lowercase).

    Picks the char whose description shares the most characters with the mention.
    """
    if not text:
        return None
    mlow = text.lower()
    best, best_score = None, 0
    for c in chars:
        d = (c.get("description") or "").lower().strip()
        if not d:
            continue
        # Score: longest contiguous substring overlap (heuristic via token intersection)
        # Using token overlap is more robust for German compounds than naive substring.
        score = len(_toks(d) & _toks(mlow))
        if score > best_score:
            best, best_score = c["id"], score
    return best


def align_pron(pred_ids: set) -> str:
    """Return char_id (single), 'AMBIGUOUS' (multi), or 'STRAY' (empty)."""
    if not pred_ids:
        return "STRAY"
    if len(pred_ids) == 1:
        return next(iter(pred_ids))
    return "AMBIGUOUS"


def assign_mention_to_char(
    mention: dict,
    pred_ids: set,
    chars: list[dict],
) -> str:
    """Return char_id, or one of: 'AMBIGUOUS', 'STRAY', 'OFFSCREEN'.

    'OFFSCREEN' = matched a char_id that is NOT in pred_ids for this AD.
    """
    label = mention.get("label", "")
    text = mention.get("text", "")
    if label == "PRN":
        return align_pron(pred_ids)
    if label == "NAME":
        cid = align_name(text, chars)
    elif label == "DESC":
        cid = align_desc(text, chars)
    else:
        return "STRAY"
    if cid is None:
        return "STRAY"
    return cid if cid in pred_ids else "OFFSCREEN"


# ═══════════════════════════════════════════════════════════════════════════
# Per-AD mention assignment
# ═══════════════════════════════════════════════════════════════════════════


def assign_ad_mentions(
    mentions: list[dict], pred_ids: set, chars: list[dict],
) -> list[tuple[dict, str]]:
    """Assign each mention to a char_id or sentinel; preserves order."""
    return [(m, assign_mention_to_char(m, pred_ids, chars)) for m in mentions]


def first_label_per_char(assigned: list[tuple[dict, str]]) -> dict[str, str]:
    """For each char_id (real, not sentinel), return label of FIRST mention."""
    out: dict[str, str] = {}
    for m, cid in assigned:
        if cid in {"AMBIGUOUS", "STRAY", "OFFSCREEN"}:
            continue
        if cid not in out:
            out[cid] = m.get("label", "")
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════


def compute_rda(
    sys_assigned_per_ad: list[list[tuple[dict, str]]],
    gt_assigned_per_ad: list[list[tuple[dict, str]]],
    pred_ids_per_ad: list[set],
) -> dict:
    matched, mismatched = 0, 0
    sys_only, gt_only = 0, 0
    for sys_ass, gt_ass, pids in zip(sys_assigned_per_ad, gt_assigned_per_ad, pred_ids_per_ad):
        sys_first = first_label_per_char(sys_ass)
        gt_first  = first_label_per_char(gt_ass)
        for cid in pids:
            in_sys = cid in sys_first
            in_gt  = cid in gt_first
            if in_sys and in_gt:
                if sys_first[cid] == gt_first[cid]:
                    matched += 1
                else:
                    mismatched += 1
            elif in_sys and not in_gt:
                sys_only += 1
            elif in_gt and not in_sys:
                gt_only += 1
    n_pairs = matched + mismatched
    return {
        "RDA": (matched / n_pairs) if n_pairs > 0 else None,
        "n_matched": matched,
        "n_mismatched": mismatched,
        "n_sys_only": sys_only,
        "n_gt_only":  gt_only,
        "n_pairs":    n_pairs,
    }


def compute_form_distribution(
    sys_per_ad: list[list[dict]], gt_per_ad: list[list[dict]],
) -> dict:
    def _counts(per_ad: list[list[dict]]) -> dict:
        c = Counter()
        for ms in per_ad:
            for m in ms:
                lab = m.get("label", "")
                if lab in {"NAME", "DESC", "PRN"}:
                    c[lab] += 1
        return c
    sys_c = _counts(sys_per_ad)
    gt_c  = _counts(gt_per_ad)

    def _pct(c):
        tot = sum(c.values())
        if tot == 0:
            return {"NAME": 0.0, "DESC": 0.0, "PRN": 0.0}
        return {k: c.get(k, 0) / tot for k in ("NAME", "DESC", "PRN")}

    sys_p = _pct(sys_c)
    gt_p  = _pct(gt_c)
    l1 = 0.5 * sum(abs(sys_p[k] - gt_p[k]) for k in ("NAME", "DESC", "PRN"))
    return {
        "sys_NAME_pct": sys_p["NAME"], "sys_DESC_pct": sys_p["DESC"], "sys_PRN_pct": sys_p["PRN"],
        "gt_NAME_pct":  gt_p["NAME"],  "gt_DESC_pct":  gt_p["DESC"],  "gt_PRN_pct":  gt_p["PRN"],
        "form_dist_L1": l1,
        "n_sys_mentions": sum(sys_c.values()),
        "n_gt_mentions":  sum(gt_c.values()),
    }


def compute_subset_metrics(
    per_ad: list[list[dict]], char_continuity: list,
) -> dict:
    """Re-intro precision (cross-cut, % non-PRN) + Continuing PRN rate."""
    # Label audit (2026-05-22, n=32,136 mentions, sys+GT, all films×conditions):
    # every extracted mention carries exactly one of {NAME, DESC, PRN}, so no
    # label filter is needed here — the denominators below equal the
    # NAME/DESC/PRN total. (compute_form_distribution filters for the same reason.)
    cc_mentions = [m.get("label", "") for ms, cont in zip(per_ad, char_continuity)
                   if cont is False for m in ms]
    cont_mentions = [m.get("label", "") for ms, cont in zip(per_ad, char_continuity)
                     if cont is True for m in ms]
    reintro = (sum(1 for l in cc_mentions if l != "PRN") / len(cc_mentions)
               if cc_mentions else None)
    cont_prn = (sum(1 for l in cont_mentions if l == "PRN") / len(cont_mentions)
                if cont_mentions else None)
    return {
        "Reintro_Prec_M":          reintro,
        "Continuing_PRN_rate_M":   cont_prn,
        "n_cc_mentions":           len(cc_mentions),
        "n_cont_mentions":         len(cont_mentions),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Per-(movie, condition) evaluator
# ═══════════════════════════════════════════════════════════════════════════


def evaluate_one(movie: str, condition: str) -> dict | None:
    info = THESIS_FILMS[movie]
    sys_dir = ROBUST_ROOT / f"{movie}__{condition}"

    sys_mentions_file = sys_dir / "text_gen.mentions"
    sys_json_file     = sys_dir / "text_gen.json"
    gt_mentions_file  = GT_DISPATCH[movie]

    if not sys_mentions_file.exists():
        return {"Movie": movie, "Condition": condition, "status": "SKIP_NO_SYS"}
    if not gt_mentions_file.exists():
        return {"Movie": movie, "Condition": condition, "status": "SKIP_NO_GT"}

    sys_per_ad = load_mentions_file(sys_mentions_file)
    gt_per_ad  = load_mentions_file(gt_mentions_file)

    if len(sys_per_ad) != len(gt_per_ad):
        return {"Movie": movie, "Condition": condition,
                "status": f"SKIP_LEN_MISMATCH sys={len(sys_per_ad)} gt={len(gt_per_ad)}"}

    # anno_idx for each line — from text_gen.json (sorted by anno_idx)
    with open(sys_json_file) as f:
        anno_idx_list = [e["anno_idx"] for e in json.load(f)]
    if len(anno_idx_list) != len(sys_per_ad):
        return {"Movie": movie, "Condition": condition,
                "status": f"SKIP_JSON_MISMATCH json={len(anno_idx_list)} mentions={len(sys_per_ad)}"}

    # pred_ids per AD + char_continuity
    pred_df, imdbid = load_anno_pred_ids(movie)
    pred_map = dict(zip(pred_df["anno_idx"], pred_df["pred_ids_set"]))
    cc_map   = dict(zip(pred_df["anno_idx"], pred_df["char_continuity"]))
    pred_ids_per_ad = [pred_map.get(a, set()) for a in anno_idx_list]
    cc_per_ad       = [cc_map.get(a, None)    for a in anno_idx_list]

    # Char metadata
    chars = load_char_meta(imdbid)
    if not chars:
        return {"Movie": movie, "Condition": condition,
                "status": f"SKIP_NO_CHAR_META imdbid={imdbid}"}

    # Assign mentions
    sys_assigned = [assign_ad_mentions(ms, pids, chars)
                    for ms, pids in zip(sys_per_ad, pred_ids_per_ad)]
    gt_assigned  = [assign_ad_mentions(ms, pids, chars)
                    for ms, pids in zip(gt_per_ad, pred_ids_per_ad)]

    # Sentinel counters (across all mentions)
    def _sent_counts(assigned):
        c = Counter()
        for ass in assigned:
            for _, cid in ass:
                if cid in {"AMBIGUOUS", "STRAY", "OFFSCREEN"}:
                    c[cid] += 1
                else:
                    c["BOUND"] += 1
        return c
    sys_sent = _sent_counts(sys_assigned)
    gt_sent  = _sent_counts(gt_assigned)

    # Metrics
    rda  = compute_rda(sys_assigned, gt_assigned, pred_ids_per_ad)
    fdist = compute_form_distribution(sys_per_ad, gt_per_ad)
    subm = compute_subset_metrics(sys_per_ad, cc_per_ad)
    subm_gt = compute_subset_metrics(gt_per_ad, cc_per_ad)

    prd_h_m = (abs(subm["Continuing_PRN_rate_M"] - subm_gt["Continuing_PRN_rate_M"])
               if subm["Continuing_PRN_rate_M"] is not None
               and subm_gt["Continuing_PRN_rate_M"] is not None
               else None)

    out = {
        "Movie":     movie,
        "Language":  info["lang"],
        "Condition": condition,
        "status":    "OK",
        # form distribution
        **{k: v for k, v in fdist.items()},
        # RDA + counters
        **rda,
        # Subset metrics
        **{f"{k}_sys": v for k, v in subm.items() if k.startswith(("Reintro", "Continuing"))},
        **{f"{k}_gt":  v for k, v in subm_gt.items() if k.startswith(("Reintro", "Continuing"))},
        "PRD_H_M": prd_h_m,
        # Sentinel counts
        "sys_n_AMBIGUOUS": sys_sent.get("AMBIGUOUS", 0),
        "sys_n_STRAY":     sys_sent.get("STRAY", 0),
        "sys_n_OFFSCREEN": sys_sent.get("OFFSCREEN", 0),
        "gt_n_AMBIGUOUS":  gt_sent.get("AMBIGUOUS", 0),
        "gt_n_STRAY":      gt_sent.get("STRAY", 0),
        "gt_n_OFFSCREEN":  gt_sent.get("OFFSCREEN", 0),
    }

    # Diagnostic CSV: per-(AD, char) RDA judgment
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    diag_rows = []
    for sys_ass, gt_ass, pids, aidx in zip(
        sys_assigned, gt_assigned, pred_ids_per_ad, anno_idx_list
    ):
        sys_first = first_label_per_char(sys_ass)
        gt_first  = first_label_per_char(gt_ass)
        for cid in pids:
            sys_l = sys_first.get(cid)
            gt_l  = gt_first.get(cid)
            if sys_l is None and gt_l is None:
                continue
            judgment = (
                "match" if (sys_l == gt_l) and sys_l is not None
                else "mismatch" if (sys_l and gt_l)
                else "sys_only" if sys_l
                else "gt_only"
            )
            diag_rows.append({
                "anno_idx": aidx, "char_id": cid,
                "sys_label": sys_l, "gt_label": gt_l,
                "judgment": judgment,
                "char_continuity": cc_map.get(aidx),
            })
    if diag_rows:
        pd.DataFrame(diag_rows).to_csv(
            DIAG_DIR / f"{movie}__{condition}.csv", index=False
        )

    return out


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--movie",     default=None, help="Limit to one movie.")
    ap.add_argument("--condition", default=None, help="Limit to one condition.")
    args = ap.parse_args()

    movies = [args.movie] if args.movie else list(THESIS_FILMS.keys())
    conds  = [args.condition] if args.condition else CONDITIONS

    print(f"DATA_ROOT = {DATA_ROOT}")
    print(f"Films     = {len(movies)}, Conditions = {len(conds)} → {len(movies)*len(conds)} runs")
    print()

    rows = []
    for movie in movies:
        for cond in conds:
            print(f"  ▶  {movie} / {cond}", end=" ... ", flush=True)
            try:
                res = evaluate_one(movie, cond)
            except Exception as e:
                res = {"Movie": movie, "Condition": cond, "status": f"ERROR: {e}"}
                print(f"ERROR: {e}")
            else:
                if res is None:
                    print("SKIP (None)")
                    continue
                if res.get("status") == "OK":
                    rda_str = f"{res['RDA']:.3f}" if res.get("RDA") is not None else "N/A"
                    fdist_str = f"{res.get('form_dist_L1', 0):.3f}"
                    print(f"RDA={rda_str}  L1={fdist_str}  pairs={res.get('n_pairs', 0)}")
                else:
                    print(res["status"])
            rows.append(res)

    df = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults saved → {OUTPUT_CSV}")
    print(f"Per-(AD, char) diagnostics in → {DIAG_DIR}/")


if __name__ == "__main__":
    main()
