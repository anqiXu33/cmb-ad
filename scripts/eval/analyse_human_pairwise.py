"""Analyze paired-blind human-eval CSVs returned by raters.

Reads:
  results/human_eval/paired_blind_<movie>_<rater>.csv      (vote, comment)
  results/human_eval/private/unblind_key_<movie>.csv       (A/B → model, pred_roles, lengths)
  results/benchmarks/thesis_final_results.csv              (auto-metric per-film)

Writes:
  results/human_eval/analysis_summary.md     human-readable tables for §6
  results/human_eval/analysis_breakdown.csv  long-form: one row per (film, bucket, vote)

Rater convention: vote ∈ {A, B, E}. `E` is treated as Tie (the README says Tie;
Annette wrote E for "equivalent").
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "results" / "human_eval"
PRIVATE_DIR = EVAL_DIR / "private"
SURFACE_PATH  = ROOT / "results" / "benchmarks" / "thesis_final_results.csv"
MENTION_PATH  = ROOT / "results" / "benchmarks" / "mention_level_results.csv"

# Condition slug used in mention_level_results.csv → thesis display name.
COND_CMB = "cmb_ad"
COND_B1  = "baseline1_no_cmb"
SURFACE_MODEL_CMB = "CMB-AD (Ours)"
SURFACE_MODEL_B1  = "Baseline 1 (No-CMB)"

FILMS = [
    ("Moskau_einfach", "German",  "Annette"),
    ("Quanto_basta",   "Italian", "Annette"),
]

VOTE_MAP = {"A": "A", "B": "B", "E": "Tie", "Tie": "Tie", "tie": "Tie"}


# ----- comment bucketing ----------------------------------------------------

# Order matters: first match wins. Patterns are case-insensitive substring tests.
BUCKET_PATTERNS = [
    ("credits",        [r"\bcredits?\b", r"\bemp?ty\b"]),                 # title-card / empty-output rows
    ("wrong_action",   [r"wrong action", r"doesn.?t happen", r"wrong actions"]),
    ("both_wrong",     [r"both wrong"]),                                   # AFTER wrong_action so "both wrong action" lands there
    ("not_recognized", [r"not recognized", r"^not [A-Z]", r"\bnot [A-Z][a-z]+"]),  # "Viktor not recognized", "not Odile"
    ("name_vs_desc",   [r"name used", r"names? from", r"read names"]),
    ("pronoun_issue",  [r"pronoun", r"pro.?drop"]),
]


def bucket_for(comment: str) -> str:
    if not isinstance(comment, str) or not comment.strip():
        return "_no_comment"
    text = comment.strip()
    for name, pats in BUCKET_PATTERNS:
        for p in pats:
            if re.search(p, text, flags=re.IGNORECASE):
                return name
    return "other"


# ----- per-film analysis ----------------------------------------------------

def load_film(movie: str, rater: str) -> pd.DataFrame:
    rated = pd.read_csv(EVAL_DIR / f"paired_blind_{movie}_{rater}.csv")
    key   = pd.read_csv(PRIVATE_DIR / f"unblind_key_{movie}.csv")
    df = rated.merge(key, on="anno_idx", how="left", validate="one_to_one")

    # normalise vote → {A, B, Tie, (none)}
    df["vote_norm"] = (
        df["vote"].fillna("").astype(str).str.strip()
                 .map(lambda v: VOTE_MAP.get(v, "(none)" if v == "" else f"INVALID:{v}"))
    )

    # winner in {CMB-AD, B1, Tie, (none)}
    def winner(row):
        v = row["vote_norm"]
        if v == "Tie":   return "Tie"
        if v == "(none)": return "(none)"
        if v.startswith("INVALID"): return v
        return row["model_A"] if v == "A" else row["model_B"]
    df["winner"] = df.apply(winner, axis=1)
    df["winner"] = df["winner"].replace({"CMB_AD": "CMB-AD", "B1_no_cmb": "B1"})

    df["bucket"] = df["comment"].apply(bucket_for)
    df["len_diff_cmb_minus_b1"] = df["len_cmb"] - df["len_b1"]
    return df


def vote_table(df: pd.DataFrame) -> dict:
    """Return CMB/B1/Tie/(none) counts + binomial test on decisive votes."""
    counts = df["winner"].value_counts().to_dict()
    cmb  = counts.get("CMB-AD", 0)
    b1   = counts.get("B1", 0)
    tie  = counts.get("Tie", 0)
    none = counts.get("(none)", 0)
    decisive = cmb + b1
    if decisive > 0:
        bt = binomtest(cmb, decisive, p=0.5, alternative="two-sided")
        cmb_win_rate = cmb / decisive
        ci_low, ci_high = bt.proportion_ci(confidence_level=0.95)
        p_value = bt.pvalue
    else:
        cmb_win_rate = float("nan"); ci_low = ci_high = float("nan"); p_value = float("nan")
    return dict(
        n_total=len(df), n_rated=len(df) - none, n_decisive=decisive,
        cmb=cmb, b1=b1, tie=tie, none=none,
        cmb_win_rate=cmb_win_rate, ci_low=ci_low, ci_high=ci_high, p_value=p_value,
    )


def slice_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """vote counts × col-value, plus CMB-win rate on decisive."""
    rows = []
    for val, sub in df.groupby(col, dropna=False):
        t = vote_table(sub)
        rows.append({col: val, **t})
    return pd.DataFrame(rows)


def length_corr(df: pd.DataFrame) -> dict:
    """Is rater preference correlated with length difference?"""
    dec = df[df["winner"].isin({"CMB-AD", "B1"})].copy()
    dec["cmb_won"] = (dec["winner"] == "CMB-AD").astype(int)
    if len(dec) < 5:
        return dict(n=len(dec), mean_diff_when_cmb_won=None, mean_diff_when_b1_won=None, pearson_r=None)
    corr = dec[["cmb_won", "len_diff_cmb_minus_b1"]].corr().iloc[0, 1]
    return dict(
        n=len(dec),
        mean_diff_when_cmb_won=dec.loc[dec.cmb_won == 1, "len_diff_cmb_minus_b1"].mean(),
        mean_diff_when_b1_won=dec.loc[dec.cmb_won == 0, "len_diff_cmb_minus_b1"].mean(),
        pearson_r=corr,
    )


def quintile_drift(df: pd.DataFrame) -> pd.DataFrame:
    """vote distribution across anno_idx quintiles — checks rater fatigue/drift."""
    q = df.copy()
    q["quintile"] = pd.qcut(q["anno_idx"], 5, labels=[f"Q{i+1}" for i in range(5)])
    return slice_by(q, "quintile")


def surface_rows(movie: str) -> pd.DataFrame:
    """§6.1 surface metrics: BLEU + BERTScore_F1 (CHRF + Coherence_Score dropped from thesis)."""
    if not SURFACE_PATH.exists():
        return pd.DataFrame()
    am = pd.read_csv(SURFACE_PATH)
    return am[(am["Movie"] == movie) & (am["Model"].isin([SURFACE_MODEL_CMB, SURFACE_MODEL_B1]))][
        ["Model", "BLEU", "BERTScore_F1"]
    ]


def mention_rows(movie: str) -> pd.DataFrame:
    """§6.2 mention-level battery: RDA, Form L1, Re-intro Precision, Continuing PRN Rate, PRD-H."""
    if not MENTION_PATH.exists():
        return pd.DataFrame()
    am = pd.read_csv(MENTION_PATH)
    return am[(am["Movie"] == movie) & (am["Condition"].isin([COND_CMB, COND_B1]))][
        ["Condition", "RDA", "form_dist_L1", "Reintro_Prec_M_sys",
         "Continuing_PRN_rate_M_sys", "PRD_H_M"]
    ]


# ----- formatting -----------------------------------------------------------

def fmt_pct(x): return "—" if pd.isna(x) else f"{x*100:.1f}%"
def fmt_num(x): return "—" if pd.isna(x) else (f"{x:.3f}" if isinstance(x, float) else str(x))


def md_main_table(per_film: dict) -> str:
    lines = ["| Film | n rated | CMB-AD | B1 | Tie | CMB win-rate (decisive) | 95% CI | sign-test p |",
             "|---|---:|---:|---:|---:|---:|---|---:|"]
    for film, t in per_film.items():
        ci = f"[{fmt_pct(t['ci_low'])}, {fmt_pct(t['ci_high'])}]"
        p  = "<1e-12" if t["p_value"] is not None and t["p_value"] < 1e-12 else f"{t['p_value']:.3g}"
        lines.append(f"| {film} | {t['n_rated']} | {t['cmb']} | {t['b1']} | {t['tie']} | "
                     f"{fmt_pct(t['cmb_win_rate'])} | {ci} | {p} |")
    return "\n".join(lines)


def md_slice_table(slice_df: pd.DataFrame, slice_col: str) -> str:
    cols = [slice_col, "n_rated", "cmb", "b1", "tie", "cmb_win_rate"]
    out = slice_df[cols].copy()
    out["cmb_win_rate"] = out["cmb_win_rate"].apply(fmt_pct)
    return out.to_markdown(index=False)


def md_auto_vs_human(per_film: dict) -> str:
    """Match thesis §6.1 (BLEU + BERTScore_F1) and §6.2 (5 mention-level metrics).
    CHRF and the old surface Coherence_Score are intentionally excluded."""
    SURFACE_METRICS = [("BLEU", "BLEU"), ("BERTScore_F1", "BERTScore F1")]
    MENTION_METRICS = [
        ("RDA",                       "RDA",                                 "higher"),
        ("form_dist_L1",              "Form L1 (lower=better)",              "lower"),
        ("Reintro_Prec_M_sys",        "Re-intro Precision (mention-level)",  "higher"),
        ("Continuing_PRN_rate_M_sys", "Continuing PRN Rate",                 "higher_until_GT"),
        ("PRD_H_M",                   "PRD-H (lower=better)",                "lower"),
    ]
    lines = ["| Film | Metric | CMB-AD | B1 | Δ (CMB−B1) |", "|---|---|---:|---:|---:|"]
    for movie, language, _ in FILMS:
        # §6.1 surface block
        am = surface_rows(movie)
        if not am.empty:
            cmb_row = am[am["Model"] == SURFACE_MODEL_CMB].iloc[0]
            b1_row  = am[am["Model"] == SURFACE_MODEL_B1].iloc[0]
            for col, display in SURFACE_METRICS:
                d = cmb_row[col] - b1_row[col]
                lines.append(f"| {movie} | {display} | {cmb_row[col]:.4g} | {b1_row[col]:.4g} | {d:+.4g} |")
        # §6.2 mention-level block
        mm = mention_rows(movie)
        if not mm.empty:
            cmb_row = mm[mm["Condition"] == COND_CMB].iloc[0]
            b1_row  = mm[mm["Condition"] == COND_B1].iloc[0]
            for col, display, _direction in MENTION_METRICS:
                d = cmb_row[col] - b1_row[col]
                lines.append(f"| {movie} | {display} | {cmb_row[col]:.4g} | {b1_row[col]:.4g} | {d:+.4g} |")
        # Human row (decisive win-rate as the anchor)
        t = per_film[movie]
        lines.append(f"| {movie} | **Human win-rate (decisive)** | {fmt_pct(t['cmb_win_rate'])} | "
                     f"{fmt_pct(1 - t['cmb_win_rate'])} | "
                     f"{(2 * t['cmb_win_rate'] - 1) * 100:+.1f}pp |")
    return "\n".join(lines)


# ----- main -----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_md",  default=str(EVAL_DIR / "analysis_summary.md"))
    ap.add_argument("--out_csv", default=str(EVAL_DIR / "analysis_breakdown.csv"))
    args = ap.parse_args()

    per_film = {}
    breakdown_rows = []
    md_parts = ["# Human-Eval Analysis — CMB-AD vs Baseline 1 (single rater: Annette)",
                "", "Vote convention: rater used `E` for Tie (README says Tie).", ""]

    md_parts.append("## 1. Main result — vote distribution per film\n")
    md_parts.append("Decisive = CMB-AD wins + B1 wins (Tie excluded from win-rate denominator).\n")

    for movie, language, rater in FILMS:
        df = load_film(movie, rater)
        t = vote_table(df)
        per_film[movie] = t
        # breakdown: overall row
        breakdown_rows.append(dict(film=movie, slice="overall", slice_value="all", **t))

    md_parts.append(md_main_table(per_film))

    md_parts += ["", "## 2. Conditioned on face-detection signal (`pred_empty`)\n",
                 "Expectation: CMB advantage concentrates in `pred_empty=False` rows; `pred_empty=True` rows should approach 50/50 because both models fall back to Stage-1 paraphrase.\n"]
    for movie, language, rater in FILMS:
        df = load_film(movie, rater)
        s = slice_by(df, "pred_empty")
        md_parts.append(f"### {movie}\n")
        md_parts.append(md_slice_table(s, "pred_empty") + "\n")
        for _, row in s.iterrows():
            breakdown_rows.append(dict(film=movie, slice="pred_empty",
                                       slice_value=str(row["pred_empty"]), **row.drop("pred_empty").to_dict()))

    md_parts += ["", "## 3. Conditioned on comment bucket\n",
                 "Buckets are mutually exclusive; `_no_comment` means rater left the comment field blank.\n",
                 "`both_wrong` should be excluded when claiming a model 'won' — both outputs are flagged unusable.\n",
                 "`not_recognized` is the dominant CMB-failure mode (face-detection FP/FN → CMB confidently injects wrong name).\n"]
    for movie, language, rater in FILMS:
        df = load_film(movie, rater)
        s = slice_by(df, "bucket").sort_values("n_total", ascending=False)
        md_parts.append(f"### {movie}\n")
        md_parts.append(md_slice_table(s, "bucket") + "\n")
        for _, row in s.iterrows():
            breakdown_rows.append(dict(film=movie, slice="bucket",
                                       slice_value=str(row["bucket"]), **row.drop("bucket").to_dict()))

    md_parts += ["", "## 4. Length confound check\n",
                 "Per-row `len_diff = len_cmb − len_b1` (word counts). If rater systematically prefers shorter (or longer) outputs, the Pearson r between `cmb_won ∈ {0,1}` and `len_diff` will be far from 0.\n",
                 "| Film | n decisive | mean Δlen when CMB won | mean Δlen when B1 won | Pearson r |",
                 "|---|---:|---:|---:|---:|"]
    for movie, language, rater in FILMS:
        df = load_film(movie, rater)
        lc = length_corr(df)
        md_parts.append(f"| {movie} | {lc['n']} | {fmt_num(lc['mean_diff_when_cmb_won'])} | "
                        f"{fmt_num(lc['mean_diff_when_b1_won'])} | {fmt_num(lc['pearson_r'])} |")

    md_parts += ["", "## 5. Rater drift across anno_idx quintiles\n",
                 "Fatigue / learning check. If the CMB win-rate trends monotonically across Q1→Q5, the rater changed how they scored as they went.\n"]
    for movie, language, rater in FILMS:
        df = load_film(movie, rater)
        s = quintile_drift(df)
        md_parts.append(f"### {movie}\n")
        md_parts.append(md_slice_table(s, "quintile") + "\n")
        for _, row in s.iterrows():
            breakdown_rows.append(dict(film=movie, slice="quintile",
                                       slice_value=str(row["quintile"]), **row.drop("quintile").to_dict()))

    md_parts += ["", "## 6. Cross-film consistency\n"]
    rates = [per_film[m]["cmb_win_rate"] for m, *_ in FILMS]
    md_parts.append(f"CMB-AD win-rate (decisive): Moskau {fmt_pct(rates[0])}, Quanto {fmt_pct(rates[1])} — "
                    f"direction agrees in both films; magnitudes within {abs(rates[0]-rates[1])*100:.1f}pp of each other.\n")

    md_parts += ["", "## 7. Auto-metric vs human preference\n",
                 "Aligned with thesis metric set: §6.1 surface (BLEU + BERTScore_F1) and §6.2 mention-level battery (RDA, Form L1, Re-intro Precision, Continuing PRN Rate, PRD-H). CHRF and the surface Coherence_Score are not reported in the thesis and are excluded here.\n",
                 "For Continuing PRN Rate, both 'higher than B1' and 'closer to human GT' are interpretable signals; the §6.2 reading is that closer-to-GT is what counts, and PRD-H quantifies that distance directly.\n"]
    md_parts.append(md_auto_vs_human(per_film))

    # save
    out_md = Path(args.out_md)
    out_md.write_text("\n".join(md_parts) + "\n", encoding="utf-8")
    pd.DataFrame(breakdown_rows).to_csv(args.out_csv, index=False)
    print(f"wrote {out_md}")
    print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
