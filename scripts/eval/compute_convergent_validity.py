#!/usr/bin/env python3
"""
compute_convergent_validity.py — Per-AD directional agreement between the
mention-level L3 metric (RDA judgements aggregated to AD-level) and the
human rater's pairwise vote.  Reports per-film + combined.

Purpose (§6.4.4 / §7.6.3 of the thesis)
───────────────────────────────────────
Reframes what used to be loosely called "human vs. auto cross-validation"
as **convergent validity** in the construct-validity sense: two
methodologies (token-level mention alignment vs. holistic AD-level
preference) measuring the same construct (character-reference decision
quality) should agree directionally.

Method (per design 2026-05-14)
──────────────────────────────
For each anno_idx with human vote in {A, B}:

  L3 aggregation to AD level:
    cmb_net = (matches of CMB-AD per char in this AD) - (mismatches)
    b1_net  = (matches of B1     per char in this AD) - (mismatches)
    L3_prefer ← "CMB"  if cmb_net > b1_net
                "B1"   if b1_net > cmb_net
                "tie"  otherwise

  Human direction:
    human ← "CMB" if vote=A on CMB-keyed film  OR vote=B on B1-keyed film
            "B1"  if vote=B on CMB-keyed film  OR vote=A on B1-keyed film

  Counts a 2×3 contingency: human direction × L3 direction.
  Reports per-direction "non-conflict" rate
  (= directly agree OR L3 tie).

Inputs
──────
  results/human_eval/paired_blind_<movie>_Annette.csv
  results/human_eval/private/unblind_key_<movie>.csv
  results/benchmarks/mention_alignment/<movie>__cmb_ad.csv
  results/benchmarks/mention_alignment/<movie>__baseline1_no_cmb.csv

Outputs
───────
  results/benchmarks/convergent_validity.csv     (per film + combined)
  Pretty-printed table to stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]

# (movie, language) — films with completed Annette ratings as of 2026-05-17
FILMS = [
    ("Moskau_einfach", "DE"),
    ("Quanto_basta",   "IT"),
]


def load_per_ad_l3_direction(movie: str) -> dict[int, str]:
    """For each anno_idx where pred_ids has chars, return 'CMB' / 'B1' / 'tie'.

    Net match score per (AD, condition) = (#matches) - (#mismatches).
    """
    cmb_path = _REPO / f"results/benchmarks/mention_alignment/{movie}__cmb_ad.csv"
    b1_path  = _REPO / f"results/benchmarks/mention_alignment/{movie}__baseline1_no_cmb.csv"

    def _net(p: Path) -> pd.Series:
        df = pd.read_csv(p)
        return df.groupby("anno_idx").apply(
            lambda g: int((g["judgment"] == "match").sum()
                          - (g["judgment"] == "mismatch").sum())
        )

    cmb_net = _net(cmb_path)
    b1_net  = _net(b1_path)

    all_aidx = sorted(set(cmb_net.index) | set(b1_net.index))
    out = {}
    for a in all_aidx:
        c = int(cmb_net.get(a, 0))
        b = int(b1_net.get(a, 0))
        if c > b:
            out[a] = "CMB"
        elif b > c:
            out[a] = "B1"
        else:
            out[a] = "tie"
    return out


def load_human_direction(movie: str) -> dict[int, str]:
    """Return {anno_idx: 'CMB' | 'B1' | 'tie' | 'exclude'} from human votes."""
    votes_path = _REPO / f"results/human_eval/paired_blind_{movie}_Annette.csv"
    key_path   = _REPO / f"results/human_eval/private/unblind_key_{movie}.csv"
    votes = pd.read_csv(votes_path)
    key   = pd.read_csv(key_path)
    df = votes.merge(key[["anno_idx", "model_A", "model_B"]],
                     on="anno_idx", how="left")

    out = {}
    for _, r in df.iterrows():
        v = r["vote"]
        aidx = int(r["anno_idx"])
        if pd.isna(v) or v == "":
            continue
        if v == "E":
            comm = str(r.get("comment") or "").lower()
            if "credit" in comm:
                out[aidx] = "exclude"
            else:
                out[aidx] = "tie"
        elif v == "Tie":
            out[aidx] = "tie"
        elif v in ("A", "B"):
            # Decode via unblind key (model_A vs model_B vary per film)
            model_for_vote = r["model_A"] if v == "A" else r["model_B"]
            out[aidx] = "CMB" if model_for_vote == "CMB_AD" else "B1"
    return out


def cross_tab_and_metrics(human: dict, l3: dict) -> dict:
    """Build the 3x3 contingency on aidx ∈ both; compute non-conflict rates."""
    common = sorted(set(human.keys()) & set(l3.keys()))
    # Exclude credits-flagged human rows
    common = [a for a in common if human[a] != "exclude"]

    table = pd.DataFrame(0, index=["CMB", "B1", "tie"], columns=["CMB", "B1", "tie"])
    for a in common:
        h, ll = human[a], l3[a]
        if h in table.index and ll in table.columns:
            table.loc[h, ll] += 1

    n_cmb = int(table.loc["CMB"].sum())
    n_b1  = int(table.loc["B1"].sum())
    n_tie = int(table.loc["tie"].sum())

    # "non-conflict" = same direction OR L3 tie (we use L3_tie as inconclusive)
    cmb_nonconflict = (table.loc["CMB", "CMB"] + table.loc["CMB", "tie"]) / max(n_cmb, 1)
    b1_nonconflict  = (table.loc["B1",  "B1"]  + table.loc["B1",  "tie"])  / max(n_b1, 1)

    # Directional conflict only (excluding ties on either side)
    cmb_conflict = table.loc["CMB", "B1"] / max(n_cmb, 1)
    b1_conflict  = table.loc["B1",  "CMB"] / max(n_b1, 1)

    return {
        "table":  table,
        "n":      len(common),
        "n_human_CMB": n_cmb,
        "n_human_B1":  n_b1,
        "n_human_tie": n_tie,
        "cmb_nonconflict_rate": round(float(cmb_nonconflict), 4),
        "b1_nonconflict_rate":  round(float(b1_nonconflict),  4),
        "cmb_directional_conflict_rate": round(float(cmb_conflict), 4),
        "b1_directional_conflict_rate":  round(float(b1_conflict),  4),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_csv",
                    default=str(_REPO / "results/benchmarks/convergent_validity.csv"))
    args = ap.parse_args()

    rows = []
    # Per-film + combined
    combined_human, combined_l3 = {}, {}

    for movie, lang in FILMS:
        try:
            human = load_human_direction(movie)
            l3    = load_per_ad_l3_direction(movie)
        except FileNotFoundError as e:
            print(f"  [SKIP] {movie}: {e}")
            continue

        res = cross_tab_and_metrics(human, l3)

        print(f"\n{'='*82}\n  {movie}  ({lang})\n{'='*82}")
        print("  Contingency  (rows = human direction, cols = L3 direction):")
        print(res["table"].to_string())
        print(f"\n  n compared (after credits exclusion): {res['n']}")
        print(f"  Human CMB-preferred ADs : {res['n_human_CMB']}  →  "
              f"non-conflict {res['cmb_nonconflict_rate']:.4f}  "
              f"(directional conflict {res['cmb_directional_conflict_rate']:.4f})")
        print(f"  Human B1-preferred ADs  : {res['n_human_B1']}  →  "
              f"non-conflict {res['b1_nonconflict_rate']:.4f}  "
              f"(directional conflict {res['b1_directional_conflict_rate']:.4f})")

        rows.append({
            "movie": movie, "language": lang,
            "n_compared": res["n"],
            "n_human_CMB": res["n_human_CMB"],
            "n_human_B1":  res["n_human_B1"],
            "n_human_tie": res["n_human_tie"],
            "cmb_nonconflict_rate":           res["cmb_nonconflict_rate"],
            "b1_nonconflict_rate":            res["b1_nonconflict_rate"],
            "cmb_directional_conflict_rate":  res["cmb_directional_conflict_rate"],
            "b1_directional_conflict_rate":   res["b1_directional_conflict_rate"],
        })

        # Accumulate for combined; prefix anno_idx with film to avoid collision
        # (anno_idx ranges overlap between films)
        for a, d in human.items():
            combined_human[f"{movie}_{a}"] = d
        for a, d in l3.items():
            combined_l3[f"{movie}_{a}"] = d

    # Combined report
    if rows:
        res_all = cross_tab_and_metrics(combined_human, combined_l3)
        print(f"\n{'='*82}\n  COMBINED (DE + IT)\n{'='*82}")
        print("  Contingency  (rows = human direction, cols = L3 direction):")
        print(res_all["table"].to_string())
        print(f"\n  n compared (after credits exclusion): {res_all['n']}")
        print(f"  Human CMB-preferred ADs : {res_all['n_human_CMB']}  →  "
              f"non-conflict {res_all['cmb_nonconflict_rate']:.4f}  "
              f"(directional conflict {res_all['cmb_directional_conflict_rate']:.4f})")
        print(f"  Human B1-preferred ADs  : {res_all['n_human_B1']}  →  "
              f"non-conflict {res_all['b1_nonconflict_rate']:.4f}  "
              f"(directional conflict {res_all['b1_directional_conflict_rate']:.4f})")

        rows.append({
            "movie": "COMBINED", "language": "DE+IT",
            "n_compared": res_all["n"],
            "n_human_CMB": res_all["n_human_CMB"],
            "n_human_B1":  res_all["n_human_B1"],
            "n_human_tie": res_all["n_human_tie"],
            "cmb_nonconflict_rate":           res_all["cmb_nonconflict_rate"],
            "b1_nonconflict_rate":            res_all["b1_nonconflict_rate"],
            "cmb_directional_conflict_rate":  res_all["cmb_directional_conflict_rate"],
            "b1_directional_conflict_rate":   res_all["b1_directional_conflict_rate"],
        })

    df = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"\nResults saved → {args.out_csv}")


if __name__ == "__main__":
    main()
