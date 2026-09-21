"""
Validate CMB-AD Stage 2 sanity-check results for Moskau_einfach (conditions A & C).

Checks the 6 success criteria from the sanity-check spec:
  1. Both CSVs exist and are non-empty
  2. Row count matches the number of shots in the film (compared via len(df))
  3. Empty AD outputs < 5%
  4. AD length distribution reasonable (mean 5–25 words, max < 100)
  5. A and C differ on at least some shots (else: char DB not actually being read)
  6. Manual-curated chars (Viktor: Schnurrbart; Reto: Brille / lange Haare)
     surface in C diff samples — visual inspection required, sample diffs printed.

Run AFTER both jobs complete:
    python scripts/validate_sanity_check.py
"""

import sys
from pathlib import Path

import pandas as pd

BASE = Path("results") / "ablations"
FILM = "Moskau_einfach"
AD_COL = "text_gen"   # column in stage2.csv

CONDS = [
    ("A_VLM",    "sanity_check_A_vlm"),
    ("C_Manual", "sanity_check_C_manual"),
]

MANUAL_CUE_TOKENS = [
    "schnauzbart", "schnurrbart", "moustache", "mustache",          # Viktor
    "brille", "langhaarig", "lange haare", "long hair", "glasses",  # Reto
]


def load(cond_dir: str) -> pd.DataFrame | None:
    p = BASE / cond_dir / FILM / "stage2.csv"
    if not p.exists():
        print(f"  [MISSING] {p}")
        return None
    return pd.read_csv(p)


def summarize(label: str, df: pd.DataFrame) -> dict:
    lens = df[AD_COL].dropna().astype(str).str.split().str.len()
    empty = df[AD_COL].isna().sum() + (df[AD_COL].astype(str).str.strip() == "").sum()
    stats = {
        "rows":          len(df),
        "empty_ad":      int(empty),
        "empty_pct":     100 * empty / max(len(df), 1),
        "len_mean":      float(lens.mean()) if len(lens) else 0.0,
        "len_median":    float(lens.median()) if len(lens) else 0.0,
        "len_max":       int(lens.max()) if len(lens) else 0,
    }
    print(f"\n[{label}]")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k:12s}: {v:.2f}")
        else:
            print(f"  {k:12s}: {v}")
    return stats


def main():
    print(f"Validating CMB-AD sanity check for {FILM}")
    print(f"  AD output column: {AD_COL}")

    dfs = {}
    for label, dirname in CONDS:
        df = load(dirname)
        if df is None:
            print(f"\n[FAIL] CSV missing for {label} — job may not have finished.")
            sys.exit(1)
        dfs[label] = df

    stats = {label: summarize(label, df) for label, df in dfs.items()}

    df_a, df_c = dfs["A_VLM"], dfs["C_Manual"]

    # --- Criterion 5: A vs C diff
    print("\n[Diff: A vs C]")
    if len(df_a) != len(df_c):
        print(f"  [WARN] row count mismatch  A={len(df_a)} vs C={len(df_c)} "
              f"— cannot align row-by-row.")
        diff_count = -1
    else:
        diff_mask = df_a[AD_COL].fillna("") != df_c[AD_COL].fillna("")
        diff_count = int(diff_mask.sum())
        diff_pct = 100 * diff_count / len(df_a)
        print(f"  Shots where A and C differ: {diff_count} / {len(df_a)} ({diff_pct:.1f}%)")

        if diff_count > 0:
            print("\n[Sample diffs — first 5]")
            for idx in df_a.index[diff_mask][:5]:
                print(f"\n  Shot anno_idx={df_a.loc[idx, 'anno_idx']}:")
                print(f"    A (VLM):    {df_a.loc[idx, AD_COL]}")
                print(f"    C (Manual): {df_c.loc[idx, AD_COL]}")

            # --- Criterion 6: probe for manual-curated visual cues in C
            cue_hits_a = sum(
                any(tok in str(t).lower() for tok in MANUAL_CUE_TOKENS)
                for t in df_a[AD_COL]
            )
            cue_hits_c = sum(
                any(tok in str(t).lower() for tok in MANUAL_CUE_TOKENS)
                for t in df_c[AD_COL]
            )
            print(f"\n[Manual-cue probe] tokens={MANUAL_CUE_TOKENS}")
            print(f"  Hits in A: {cue_hits_a}")
            print(f"  Hits in C: {cue_hits_c}  (expected: >= A; if 0 in both, "
                  "supervisor should spot-check Viktor/Reto shots manually)")

    # --- Criteria summary
    print("\n" + "=" * 60)
    print("Success criteria")
    print("=" * 60)

    def status(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    c1 = all(s["rows"] > 0 for s in stats.values())
    print(f"  1. Both CSVs non-empty                                  : {status(c1)}")

    c2 = (stats["A_VLM"]["rows"] == stats["C_Manual"]["rows"]) and stats["A_VLM"]["rows"] > 0
    print(f"  2. Row counts match between A and C  ({stats['A_VLM']['rows']} vs "
          f"{stats['C_Manual']['rows']})                : {status(c2)}")

    c3 = all(s["empty_pct"] < 5.0 for s in stats.values())
    print(f"  3. Empty AD < 5%                                        : {status(c3)}")

    c4 = all(5.0 <= s["len_mean"] <= 25.0 and s["len_max"] < 100 for s in stats.values())
    print(f"  4. AD length: mean 5–25 words, max < 100                : {status(c4)}")

    c5 = diff_count > 0
    print(f"  5. A and C differ on at least some shots                : {status(c5)}")

    c6_auto = (diff_count > 0)
    print(f"  6. Manual cues in sample diffs                          : "
          f"{'manual review' if c6_auto else 'N/A'} "
          f"(see [Sample diffs] and [Manual-cue probe] above)")

    overall = c1 and c2 and c3 and c4 and c5
    print("\n" + ("=" * 60))
    print(f"Overall (criteria 1–5 auto):  {status(overall)}")
    print("Criterion 6 requires manual review of sample diffs.")
    print("=" * 60)

    sys.exit(0 if overall else 2)


if __name__ == "__main__":
    main()
