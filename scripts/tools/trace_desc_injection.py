#!/usr/bin/env python3
"""
trace_desc_injection.py — CPU-only replay of CharacterMemoryBank.

Goal: verify which descriptions were actually injected into the Stage-2
prompt per AD, without needing GPU or German comprehension.

Outputs:
  --output          desc_trace_<movie>.csv  (long form, one row per char per AD)
  --review_output   review_<movie>.csv      (wide form, optional, requires --stage2_csv)

The script does NOT depend on GPU or any running SLURM job. It reproduces
exactly the CMB state machine that swissAD_qwen3.py runs (m-* shot union for
pred_ids; same flag decision logic) by importing CharacterMemoryBank and
char_utils from stage2/.

Typical usage on cluster login node:

  # Per-movie trace (10 seconds CPU)
  python3 scripts/tools/trace_desc_injection.py \\
      --pred_path       results/stage1/Moskau_einfach/swissAD_ads/stage1_qwen2vl.csv \\
      --anno_path       $DATA_ROOT/annotations/Moskau_einfach_*.csv \\
      --imdb_chars_path $DATA_ROOT/annotations/imdb_chars_enriched_manual.json \\
      --output          desc_trace_Moskau_einfach.csv

  # Review CSV joined with completed stage2.csv (or its .partial)
  python3 scripts/tools/trace_desc_injection.py \\
      --pred_path       ... --anno_path ... --imdb_chars_path ... \\
      --output          desc_trace_Moskau_einfach.csv \\
      --stage2_csv      results/stage2/cmb_ad/Moskau_einfach/stage2.csv \\
      --review_output   review_Moskau_einfach.csv \\
      --rep_crops_dir   results/preprocess/rep_crops
"""

import argparse
import ast
import glob
import os
import re
import sys
from pathlib import Path

import pandas as pd

# Allow `from coherence_manager import ...` regardless of CWD.
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "stage2"))

from coherence_manager import CharacterMemoryBank  # noqa: E402
from char_utils import build_char_maps             # noqa: E402


# ---------------------------------------------------------------------------
# m-* union (mirrors swissAD_qwen3._union_pred_ids_dicts so the trace sees
# the same pred_ids as production runs)
# ---------------------------------------------------------------------------

def _union_pred_ids_dicts(series):
    keys = set()
    for raw in series:
        if isinstance(raw, float) and pd.isna(raw):
            continue
        s = str(raw).strip()
        if s in ("", "{}", "nan"):
            continue
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, dict):
                keys.update(parsed.keys())
        except (ValueError, SyntaxError):
            continue
    if not keys:
        return "{}"
    return "{" + ", ".join(f"{repr(k)}: 0" for k in sorted(keys, key=str)) + "}"


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


# ---------------------------------------------------------------------------
# char_status parsing
# Mirror of coherence_manager._display_name + status_parts format. The lines
# are stable because they go straight into the LLM prompt.
# Examples matched:
#   "  - Berti [1716855]: last seen 3 AD(s) ago. [REINTRODUCE — use name or description] (appearance: \"...\")"
#   "  - [9999999]: first appearance. [REINTRODUCE — use name if known]"
#   "  - Carl Heymann [0535421]: last seen 1 AD(s) ago. [CONTINUING — pronoun ok]"
# ---------------------------------------------------------------------------

CHAR_LINE_RE = re.compile(
    r"""^\s*-\s+
        (?:(?P<name>.+?)\s+)?              # optional role name (greedy until "[")
        \[(?P<char_id>[A-Za-z0-9_-]+)\]:\s+
        (?P<seen>last\ seen\ \d+\ AD\(s\)\ ago|first\ appearance)\.\s+
        \[(?P<flag>[^\]]+)\]
        (?:\s+\(appearance:\s+"(?P<desc>[^"]*)"\))?
        \s*$
    """,
    re.VERBOSE,
)
EMPTY_SHOT_RE = re.compile(r"^\s*-\s+\[EMPTY SHOT\]")


def parse_char_status(char_status: str):
    rows = []
    for line in char_status.splitlines():
        if EMPTY_SHOT_RE.match(line):
            rows.append({
                "char_id":     "",
                "char_name":   "",
                "flag":        "EMPTY_SHOT",
                "description": "",
            })
            continue
        m = CHAR_LINE_RE.match(line)
        if not m:
            continue  # skip "Last known action" trailers and any unmatched lines
        rows.append({
            "char_id":     m.group("char_id"),
            "char_name":   (m.group("name") or "").strip(),
            "flag":        m.group("flag").strip(),
            "description": m.group("desc") or "",
        })
    return rows


# ---------------------------------------------------------------------------
# Load + replay
# ---------------------------------------------------------------------------

def build_pred_df(anno_path, pred_path):
    anno_df = pd.read_csv(anno_path)
    if "shot_label" not in anno_df.columns or "pred_ids" not in anno_df.columns:
        raise SystemExit(f"--anno_path missing shot_label/pred_ids: {anno_path}")

    m_rows = anno_df[anno_df["shot_label"].astype(str).str.startswith("m-", na=False)]
    if len(m_rows) == 0:
        raise SystemExit(f"--anno_path has no m-* rows: {anno_path}")
    union_map = m_rows.groupby("anno_idx")["pred_ids"].apply(_union_pred_ids_dicts)

    if pred_path is not None:
        pred_df = pd.read_csv(pred_path)
        pred_df["pred_ids"] = pred_df["anno_idx"].map(union_map).fillna("{}")
    else:
        # No Stage-1 CSV: synthesise from anno_df. Keys per AD are still
        # correct (m-* union); only the text_pred proxy differs (anno GT
        # text instead of Stage-1 text_gen). This does NOT affect which
        # flag fires or which desc is injected — only the per-character
        # `Last known action: "..."` proxy text shown to the LLM.
        first = (
            anno_df.drop_duplicates("anno_idx")
                   .set_index("anno_idx")
                   .sort_index()
        )
        pred_df = pd.DataFrame({
            "anno_idx": first.index.tolist(),
            "imdbid":   first["imdbid"].tolist(),
            "text_gen": first["text"].fillna("").tolist(),
            "text_gt":  first["text"].fillna("").tolist(),
            "start":    first["AD_start"].tolist() if "AD_start" in first.columns
                        else first.get("start", pd.Series([0]*len(first))).tolist(),
            "end":      first["AD_end"].tolist()   if "AD_end"   in first.columns
                        else first.get("end",   pd.Series([0]*len(first))).tolist(),
        })
        pred_df["pred_ids"] = pred_df["anno_idx"].map(union_map).fillna("{}")
        print("  [INFO] --pred_path not provided; synthesised pred_df from anno_df.")

    n_nonempty = (
        ~pred_df["pred_ids"].astype(str).str.strip().isin(["", "{}", "nan"])
    ).sum()
    print(f"  pred_ids m-* union: {n_nonempty}/{len(pred_df)} ADs non-empty")
    return pred_df.sort_values("anno_idx").reset_index(drop=True)


def run_trace(pred_df, char_gender, char_names, char_descriptions):
    bank = CharacterMemoryBank(
        char_gender=char_gender,
        char_names=char_names,
        char_descriptions=char_descriptions,
    )
    trace_rows = []
    for _, row in pred_df.iterrows():
        ad_idx = int(row["anno_idx"])
        text_pred = str(row.get("text_gen", "") or "")
        char_ids = _parse_keys(row.get("pred_ids", "{}"))
        char_status, _ = bank.get_context(ad_idx, text_pred, char_ids)
        for r in parse_char_status(char_status):
            trace_rows.append({
                "anno_idx":      ad_idx,
                "char_id":       r["char_id"],
                "char_name":     r["char_name"],
                "gender":        char_gender.get(r["char_id"], "") or "",
                "flag":          r["flag"],
                "has_desc_hint": bool(r["description"]),
                "description":   r["description"],
            })
        bank.add_ad(text_pred)
    return pd.DataFrame(trace_rows)


# ---------------------------------------------------------------------------
# Review (merge with stage2.csv)
# ---------------------------------------------------------------------------

def fmt_time(seconds):
    if pd.isna(seconds):
        return ""
    try:
        total = int(round(float(seconds)))
    except (TypeError, ValueError):
        return ""
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def find_rep_crops(rep_crops_dir, imdbid, char_id):
    pattern = os.path.join(rep_crops_dir, imdbid, f"{char_id}_rep*.jpg")
    return ";".join(sorted(glob.glob(pattern)))


def build_review(trace_df, stage2_csv, rep_crops_dir, output_path):
    s2 = pd.read_csv(stage2_csv)
    if "anno_idx" not in s2.columns:
        raise SystemExit(f"--stage2_csv missing 'anno_idx' column: {stage2_csv}")

    must_name_mask = trace_df["flag"].str.contains(
        "REINTRODUCE|USE NAME", regex=True, na=False
    )

    # Per-AD aggregations
    flag_summary = (
        trace_df[trace_df["flag"] != "EMPTY_SHOT"]
        .groupby("anno_idx", group_keys=True)
        .apply(
            lambda g: " | ".join(
                f"{(r.char_name or r.char_id)} [{r.flag}]" for r in g.itertuples()
            ),
            include_groups=False,
        )
        .rename("flags")
    )
    desc_summary = (
        trace_df[trace_df["has_desc_hint"]]
        .groupby("anno_idx", group_keys=True)
        .apply(
            lambda g: " | ".join(
                f"{(r.char_name or r.char_id)}: {r.description}"
                for r in g.itertuples()
            ),
            include_groups=False,
        )
        .rename("desc_injected")
    )
    must_chars = (
        trace_df[must_name_mask]
        .groupby("anno_idx", group_keys=True)
        .apply(
            lambda g: [
                (r.char_name or r.char_id, r.char_id) for r in g.itertuples()
            ],
            include_groups=False,
        )
        .rename("_must_name_pairs")
    )

    review = (
        s2.merge(flag_summary,   on="anno_idx", how="left")
          .merge(desc_summary,   on="anno_idx", how="left")
          .merge(must_chars,     on="anno_idx", how="left")
    )

    review["time"] = review.apply(
        lambda r: f"{fmt_time(r['start'])}→{fmt_time(r['end'])}"
                  if "start" in review.columns and "end" in review.columns else "",
        axis=1,
    )

    # name_hits: one column summarising whether each must-name char's first-token
    # appears in text_gen. Hits give a quick language-agnostic confirmation that
    # the model used the name; misses are the candidates worth translating to see
    # whether the description's visual feature was used instead.
    text_gen_lc = review["text_gen"].fillna("").astype(str).str.lower()

    def _hits(must_pairs, text_lc):
        if not isinstance(must_pairs, list) or not must_pairs:
            return ""
        out = []
        for name, _cid in must_pairs:
            token = (name.split()[0] if name else "").lower()
            if not token:
                continue
            mark = "✓" if token in text_lc else "✗"
            out.append(f"{token}{mark}")
        return " ".join(out)

    review["name_hits"] = [
        _hits(p, t) for p, t in zip(review["_must_name_pairs"], text_gen_lc)
    ]

    # Optional rep_crop links per AD (must-name chars only)
    if rep_crops_dir:
        imdbid = str(s2["imdbid"].iloc[0]) if "imdbid" in s2.columns else ""
        def _crops(must_pairs):
            if not isinstance(must_pairs, list) or not must_pairs:
                return ""
            paths = []
            for _name, cid in must_pairs:
                if cid:
                    p = find_rep_crops(rep_crops_dir, imdbid, cid)
                    if p:
                        paths.append(p)
            return ";".join(paths)
        review["rep_crops"] = review["_must_name_pairs"].apply(_crops)

    review = review.drop(columns=["_must_name_pairs"])

    # Reorder: human-readable columns first
    front = ["anno_idx", "time", "text_gt", "text_gen",
             "flags", "desc_injected", "name_hits"]
    if "rep_crops" in review.columns:
        front.append("rep_crops")
    front = [c for c in front if c in review.columns]
    rest = [c for c in review.columns if c not in front]
    review = review[front + rest]

    review.to_csv(output_path, index=False)
    n_with_desc = review["desc_injected"].notna().sum()
    print(f"  Review written: {output_path} "
          f"({len(review)} ADs, {n_with_desc} with desc injected)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--anno_path",       required=True,
                   help="Annotation CSV (provides m-* shot rows for pred_ids union).")
    p.add_argument("--imdb_chars_path", required=True,
                   help="imdb_chars_enriched_manual.json (or _enriched.json).")
    p.add_argument("--output",          required=True,
                   help="Desc-trace CSV path (long form, one row per char per AD).")
    p.add_argument("--pred_path",       default=None,
                   help="Stage-1 CSV (optional). Without it the GT text from "
                        "anno_df is used as text_pred proxy. This does NOT "
                        "change which flag fires or which desc gets injected.")
    p.add_argument("--stage2_csv",      default=None,
                   help="Optional stage2.csv (or .partial) for the merged review CSV.")
    p.add_argument("--review_output",   default=None,
                   help="Review CSV path (defaults to <output>_review.csv).")
    p.add_argument("--rep_crops_dir",   default=None,
                   help="Optional rep_crops/ root for review CSV link column.")
    args = p.parse_args()

    pred_df = build_pred_df(args.anno_path, args.pred_path)

    if "imdbid" not in pred_df.columns:
        raise SystemExit("pred_df has no 'imdbid' column")
    movie_imdb_id = str(pred_df["imdbid"].iloc[0])

    char_gender, char_names, char_descriptions = build_char_maps(
        args.imdb_chars_path, movie_imdb_id
    )

    trace_df = run_trace(pred_df, char_gender, char_names, char_descriptions)
    trace_df.to_csv(args.output, index=False)
    n_desc = int(trace_df["has_desc_hint"].sum())
    print(f"  Desc trace written: {args.output} "
          f"({len(trace_df)} rows, {n_desc} with desc injected)")

    if args.stage2_csv:
        review_path = args.review_output or args.output.replace(".csv", "_review.csv")
        build_review(trace_df, args.stage2_csv, args.rep_crops_dir, review_path)


if __name__ == "__main__":
    main()
