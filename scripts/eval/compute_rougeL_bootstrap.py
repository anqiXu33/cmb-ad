#!/usr/bin/env python3
"""
compute_rougeL_bootstrap.py — Paired bootstrap 95% CI on per-film ROUGE-L
deltas, mirroring the L3 routine in compute_l3_bootstrap_ci.py.

Methodology (§5.4 locked design 2026-05-14, identical to the L3 battery)
────────────────────────────────────────────────────────────────────────
For each baseline b ∈ {Baseline 1, Baseline 2}:
  delta_per_film[f] = CMB-AD[f] − b[f]      for f in 8 films  (higher is better)
  Bootstrap N = 10_000 resamples of size 8 with replacement; mean each time;
  95% CI = (2.5%, 97.5%) percentiles of the resample-mean distribution.
  direction_consistency = fraction of resample means strictly > 0
  (defined per §5.4 wording, not the stricter "CI excludes 0" used by the
  L3 script's sign_consistent column).

The bootstrap routine ``paired_bootstrap_ci`` is imported verbatim from
compute_l3_bootstrap_ci.py for procedural identity with BLEU/BERTScore/L3.

Inputs
──────
  results/benchmarks/rougeL_results.csv  (24 per-film rows + 9 MACRO_* rows)

Outputs
───────
  results/benchmarks/rougeL_bootstrap.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO       = _SCRIPT_DIR.parents[1]

# Reuse the L3 routine verbatim. main() in that module is __main__-guarded so
# the import has no side effects beyond `paired_bootstrap_ci`'s definition.
sys.path.insert(0, str(_SCRIPT_DIR))
from compute_l3_bootstrap_ci import paired_bootstrap_ci  # noqa: E402


ROUGEL_CSV = _REPO / "results" / "benchmarks" / "rougeL_results.csv"
OUT_CSV    = _REPO / "results" / "benchmarks" / "rougeL_bootstrap.csv"

CMB_LABEL = "CMB-AD (Ours)"
CONTRASTS = [
    ("CMB-AD vs Baseline 1", "Baseline 1 (No-CMB)"),
    ("CMB-AD vs Baseline 2", "Baseline 2 (Always-ID)"),
]


def _load_per_film_pivot() -> pd.DataFrame:
    df = pd.read_csv(ROUGEL_CSV)
    df = df[~df["film"].astype(str).str.startswith("MACRO_")].copy()
    piv = df.pivot(index="film", columns="condition", values="rougeL_f1")
    # Order films deterministically: DE first (4), then IT (4) — same as MOVIES dict.
    de = ["8_Tage_im_August", "Moskau_einfach", "Baghdad_in_my_Shadow", "Trommelwirbel"]
    it = ["La_Tentazione_Di_Esistere", "Quanto_basta", "Riders_of_justice", "Stuerm"]
    piv = piv.reindex(de + it)
    return piv


def main() -> None:
    piv = _load_per_film_pivot()
    assert piv.shape == (8, 3), f"unexpected pivot shape {piv.shape}"
    assert CMB_LABEL in piv.columns

    cmb = piv[CMB_LABEL].values

    rows: list[dict] = []
    print("\n" + "═" * 92)
    print("  ROUGE-L paired bootstrap on per-film deltas (n_films=8, N=10000)")
    print("  Positive Δ = CMB-AD better")
    print("═" * 92)
    print(f"  {'Contrast':<26} {'n':>3} {'mean Δ':>9} {'95% CI':>22} {'dir-cons.':>12}")
    print("─" * 92)

    for label, base_col in CONTRASTS:
        base = piv[base_col].values
        deltas = cmb - base

        # Verification assertions: all per-film deltas strictly positive.
        assert (deltas > 0).all(), (
            f"{label}: expected all positive deltas, got {deltas.tolist()}"
        )

        # Reuse L3 routine verbatim (N=10000, alpha=0.05, seed=20260517).
        mean_d, ci_lo, ci_hi, boot_means = paired_bootstrap_ci(deltas)

        # §5.4 direction-consistency = fraction of resample means > 0.
        dir_cons = float((boot_means > 0).mean())

        # Sanity: CI lower bound must be > 0, and ≥ min per-film delta is
        # NOT a strict mathematical guarantee (resamples can over-represent
        # small deltas; CI lo can dip below min(delta)). The §5.4 invariants
        # actually needed are dir_cons == 1.0 and ci_lo > 0 — assert those.
        assert dir_cons == 1.0, (
            f"{label}: direction-consistency {dir_cons} != 1.0 "
            f"despite strictly positive deltas — bug."
        )
        assert ci_lo > 0, (
            f"{label}: CI lower bound {ci_lo} ≤ 0 despite strictly positive "
            f"deltas — bug."
        )

        ci_str = f"[{ci_lo:+.4f}, {ci_hi:+.4f}]"
        print(f"  {label:<26} {len(deltas):>3d} {mean_d:>+9.4f} {ci_str:>22} {dir_cons:>12.4f}")

        rows.append({
            "contrast":             label,
            "scope":                "all",
            "n_films":              int(len(deltas)),
            "mean_delta":           round(float(mean_d), 4),
            "ci_lo":                round(float(ci_lo),  4),
            "ci_hi":                round(float(ci_hi),  4),
            "direction_consistency": round(dir_cons,     4),
        })

    print("═" * 92)

    out_df = pd.DataFrame(
        rows,
        columns=["contrast", "scope", "n_films", "mean_delta", "ci_lo", "ci_hi", "direction_consistency"],
    )
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved → {OUT_CSV.relative_to(_REPO)}")


if __name__ == "__main__":
    main()
