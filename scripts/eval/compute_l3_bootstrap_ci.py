#!/usr/bin/env python3
"""
compute_l3_bootstrap_ci.py — Paired bootstrap 95% CIs on per-film deltas
for the L3 mention-level metrics, ready to cite in §5.4 / §6.2-6.3.

Methodology (per §5.3 / §5.4 locked design 2026-05-14)
──────────────────────────────────────────────────────
For each metric m ∈ {RDA, form_dist_L1, Reintro_Prec_M_sys,
Continuing_PRN_rate_M_sys, PRD_H_M}:

  For each baseline b ∈ {Baseline 1, Baseline 2}:

    delta_per_film[f] = sign_factor * (CMB[f] - b[f])     for f in 8 films
      where sign_factor = +1 if "higher is better", -1 if "lower is better"
      so a positive delta always means CMB is better.

    Bootstrap (N = 10000 resamples):
      For each iteration: sample 8 films WITH replacement, take their mean.
      Collect mean_delta over iterations.

    Report: point estimate (= mean of delta_per_film),
            95% CI = (2.5%, 97.5%) percentiles of bootstrap distribution,
            "sign-consistent" = does CI exclude 0 in the favourable direction?

n = 8 films → no p-values reported for the main table (too small for any
classical NHST to be informative); only effect sizes + CIs, per the
locked statistical-testing decision in §5.4.

Inputs
──────
  results/benchmarks/mention_level_results.csv      (24 rows, full 24 pairs)

Outputs
───────
  results/benchmarks/l3_bootstrap_ci.csv            (per metric × baseline)
  Pretty-printed table to stdout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]

# ── Locked metric set (matches §5.3.2 outline 2026-05-14) ──────────────────
# (column_name, "higher_is_better"|"lower_is_better", display_label)
METRICS = [
    ("RDA", "higher", "RDA"),
    ("form_dist_L1", "lower", "Form distribution L1"),
    ("Reintro_Prec_M_sys", "higher", "Mention-level Re-intro Prec"),
    (
        "Continuing_PRN_rate_M_sys",
        "matches_human",
        "Continuing PRN Rate (target = GT)",
    ),  # special: closeness to gt rate
    ("PRD_H_M", "lower", "PRD-H"),
]

CONDITIONS = {
    "cmb_ad": "CMB-AD",
    "baseline1_no_cmb": "Baseline 1 (No-CMB)",
    "baseline2_always_id": "Baseline 2 (Always-ID)",
}


def paired_bootstrap_ci(
    deltas: np.ndarray,
    n_boot: int = 10000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float, np.ndarray]:
    """Return (mean, ci_low, ci_high, boot_means).

    Deltas should already be oriented so positive = CMB better.
    """
    if rng is None:
        rng = np.random.default_rng(20260517)  # deterministic seed for reproducibility
    n = len(deltas)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), np.array([])
    # Sample with replacement N_BOOT times, compute mean each time
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = deltas[idx].mean(axis=1)
    mean = float(deltas.mean())
    ci_low = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_high = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return mean, ci_low, ci_high, boot_means


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--results_csv",
        default=str(_REPO / "results/benchmarks/mention_level_results.csv"),
    )
    ap.add_argument(
        "--out_csv", default=str(_REPO / "results/benchmarks/l3_bootstrap_ci.csv")
    )
    ap.add_argument("--n_boot", type=int, default=10000)
    args = ap.parse_args()

    df = pd.read_csv(args.results_csv)
    df = df[df["status"] == "OK"].copy()

    # Pivot by (Movie, Condition) → per (Movie, Condition) row
    pivots = {}
    for col, _, _ in METRICS:
        pivots[col] = df.pivot(index="Movie", columns="Condition", values=col)

    # ── Compute per-(metric × baseline) delta, bootstrap CI ────────────────
    rows = []
    print("\n" + "=" * 92)
    print(
        f"  Paired bootstrap on per-film deltas (n_films = "
        f"{pivots['RDA'].shape[0]}, n_boot = {args.n_boot})"
    )
    print(f"  Positive delta = CMB-AD better")
    print("=" * 92)
    print(
        f"  {'Metric':<35} {'vs':<6} {'mean_Δ':>9} {'95% CI':>22} {'sign-consistent?':>20}"
    )
    print("-" * 92)

    for col, direction, label in METRICS:
        p = pivots[col]
        if "cmb_ad" not in p.columns:
            continue
        cmb = p["cmb_ad"].values
        for base_col in ("baseline1_no_cmb", "baseline2_always_id"):
            if base_col not in p.columns:
                continue
            base = p[base_col].values
            # Orient: positive = CMB better
            if direction == "higher":
                deltas = cmb - base
            elif direction == "lower":
                deltas = base - cmb  # smaller is better, so flip
            elif direction == "matches_human":
                # Closeness to GT: |sys - gt| — smaller is better
                gt_col = "Continuing_PRN_rate_M_gt"
                if gt_col not in df.columns:
                    continue
                gt_p = df.pivot(index="Movie", columns="Condition", values=gt_col)
                gt = gt_p["cmb_ad"].values  # GT same regardless of condition
                cmb_dist = np.abs(cmb - gt)
                base_dist = np.abs(base - gt)
                deltas = base_dist - cmb_dist  # smaller dist for CMB → positive
            else:
                continue

            mean, lo, hi, _ = paired_bootstrap_ci(deltas, n_boot=args.n_boot)
            sign_consistent = lo > 0  # CI excludes 0 on the CMB-favourable side
            ci_str = f"[{lo:+.4f}, {hi:+.4f}]"
            label_short = label[:33]
            base_label = "B1" if base_col == "baseline1_no_cmb" else "B2"
            mark = "✅" if sign_consistent else " "
            print(
                f"  {label_short:<35} {base_label:<6} {mean:>+9.4f} {ci_str:>22} {mark:>14}"
            )

            rows.append(
                {
                    "metric": col,
                    "metric_label": label,
                    "baseline": CONDITIONS[base_col],
                    "n_films": len(deltas),
                    "mean_delta": round(mean, 5),
                    "ci95_low": round(lo, 5),
                    "ci95_high": round(hi, 5),
                    "sign_consistent": bool(sign_consistent),
                    "per_film_deltas": [round(float(d), 5) for d in deltas],
                    "orientation": direction,
                }
            )

    print("=" * 92)

    # ── Save ───────────────────────────────────────────────────────────────
    out_df = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"\nResults saved → {args.out_csv}")

    # Print a concise summary line
    n_pass = sum(r["sign_consistent"] for r in rows)
    n_total = len(rows)
    print(
        f"\n  Summary: {n_pass}/{n_total} CMB-vs-baseline deltas have a "
        f"95% CI that excludes 0 in the CMB-favourable direction."
    )


if __name__ == "__main__":
    main()
