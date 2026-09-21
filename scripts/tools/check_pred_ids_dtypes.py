#!/usr/bin/env python3
"""
Pre-flight check before submitting the 8-film CMB-AD batch with the m-* union
patch in stage2/swissAD_qwen3.py.

For each of the 8 thesis movies, verify:
  1. Stage-1 CSV and annotation CSV both readable.
  2. anno_idx dtype matches between the two (else .map() silently misses).
  3. anno_df has 'shot_label' and 'pred_ids' columns (else patch falls back).
  4. All pred_df anno_idx values are present in anno_df.
  5. Simulate the patch's rewrite — print pre/post non-empty pred_ids counts
     and the "+N recovered" delta the patched script will print at startup.

Prints one block per movie + a final PASS/FAIL summary.  Exits non-zero on
any structural failure so it can gate the submit step.

Usage (from repo root, on the login node — no GPU, no torch needed):
    python3 scripts/tools/check_pred_ids_dtypes.py

Optional env var override (e.g. for a local mounted view):
    DATA_ROOT=data \
        python3 scripts/tools/check_pred_ids_dtypes.py
"""
import ast
import os
import sys
from pathlib import Path

import pandas as pd


DATA_ROOT = Path(os.environ.get(
    "DATA_ROOT",
    "/shares/iict-sp4.ebling.cl.uzh/data/anxu_MA_data",
))

MOVIES = [
    # German (4)
    "8_Tage_im_August",
    "Baghdad_in_my_Shadow",
    "Moskau_einfach",
    "Trommelwirbel",
    # Italian (4)
    "La_Tentazione_Di_Esistere",
    "Quanto_basta",
    "Riders_of_justice",
    "Stuerm",
]


def _union_pred_ids_dicts(series):
    """Mirror of the helper in stage2/swissAD_qwen3.py.  Kept inline so this
    script is torch-free and runnable on the login node."""
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


def _nonempty_count(series):
    return int(
        (~series.astype(str).str.strip().isin(["", "{}", "nan"])).sum()
    )


def _safe_exists(path):
    """Path.exists() raises OSError on partially mounted FUSE/SSHFS volumes
    (e.g. local cluster mount where only some subtrees are visible).  Treat
    any OS-level error as 'not present' so the script keeps walking the
    remaining movies instead of crashing on the first one."""
    try:
        return path.exists()
    except OSError:
        return False


def check_movie(movie):
    fails = []
    pred_path = DATA_ROOT / "stage1" / movie / "swissAD_ads" / "stage1_qwen2vl.csv"
    anno_path = DATA_ROOT / "annotations" / f"{movie}_context_faces_threads_shot_scales.csv"

    print(f"\n=== {movie} ===")

    if not _safe_exists(pred_path):
        print(f"  [FAIL] pred_path missing or unreachable: {pred_path}")
        return ["pred_path missing or unreachable"]
    if not _safe_exists(anno_path):
        print(f"  [FAIL] anno_path missing or unreachable: {anno_path}")
        return ["anno_path missing or unreachable"]

    pred = pd.read_csv(pred_path)
    anno = pd.read_csv(anno_path)
    print(f"  pred_df rows: {len(pred):4d}    anno_df rows: {len(anno):5d}")

    # ---- Check 1: anno_idx dtype must match -------------------------------
    pred_dtype = pred["anno_idx"].dtype
    anno_dtype = anno["anno_idx"].dtype
    dtype_match = pred_dtype == anno_dtype
    flag = "OK  " if dtype_match else "FAIL"
    print(f"  [{flag}] anno_idx dtype: pred={pred_dtype}  anno={anno_dtype}")
    if not dtype_match:
        fails.append(
            f"anno_idx dtype mismatch ({pred_dtype} vs {anno_dtype}) "
            "— .map() will silently miss every row"
        )

    # ---- Check 2: required anno columns -----------------------------------
    has_shot_label = "shot_label" in anno.columns
    has_pred_ids = "pred_ids" in anno.columns
    cols_ok = has_shot_label and has_pred_ids
    flag = "OK  " if cols_ok else "FAIL"
    print(
        f"  [{flag}] anno cols: shot_label={has_shot_label}  "
        f"pred_ids={has_pred_ids}"
    )
    if not has_shot_label:
        fails.append("anno_df missing 'shot_label' column")
    if not has_pred_ids:
        fails.append("anno_df missing 'pred_ids' column")

    # ---- Check 3: anno_idx overlap ---------------------------------------
    pred_idx_set = set(pred["anno_idx"])
    anno_idx_set = set(anno["anno_idx"])
    missing = pred_idx_set - anno_idx_set
    flag = "OK  " if not missing else "FAIL"
    print(
        f"  [{flag}] anno_idx overlap: pred_df={len(pred_idx_set)}  "
        f"intersect={len(pred_idx_set & anno_idx_set)}  "
        f"missing_in_anno={len(missing)}"
    )
    if missing:
        fails.append(
            f"{len(missing)} pred_df anno_idx values missing from anno_df "
            f"(first 3: {sorted(missing)[:3]})"
        )

    # If any structural check failed, skip the simulation — its numbers
    # would be misleading.
    if fails:
        return fails

    # ---- Check 4: simulate the patch's rewrite ---------------------------
    m_rows = anno[anno["shot_label"].astype(str).str.startswith("m-", na=False)]
    if len(m_rows) == 0:
        print("  [FAIL] no m-* rows in anno_df")
        return ["no m-* rows"]

    union_map = m_rows.groupby("anno_idx")["pred_ids"].apply(_union_pred_ids_dicts)

    pre_nonempty = _nonempty_count(pred["pred_ids"]) if "pred_ids" in pred.columns else 0
    rewritten = pred["anno_idx"].map(union_map).fillna("{}")
    post_nonempty = _nonempty_count(rewritten)
    recovered = post_nonempty - pre_nonempty

    pct_pre = 100.0 * pre_nonempty / len(pred) if len(pred) else 0.0
    pct_post = 100.0 * post_nonempty / len(pred) if len(pred) else 0.0
    print(
        f"  [SIM ] pred_ids non-empty: "
        f"{pre_nonempty}/{len(pred)} ({pct_pre:.1f}%) "
        f"→ {post_nonempty}/{len(pred)} ({pct_post:.1f}%)  "
        f"+{recovered} recovered"
    )

    if recovered < 0:
        fails.append(
            f"recovered count is negative ({recovered}) — patch would lose ADs"
        )
    elif recovered == 0:
        # Not a structural failure — but worth flagging so the user notices
        # that this film won't change at all.
        print("  [WARN] recovered=0; patch will not change anything for this film.")

    return fails


def main():
    print("Pre-flight check for m-* union patch in stage2/swissAD_qwen3.py")
    print(f"DATA_ROOT: {DATA_ROOT}")
    print(f"{len(MOVIES)} movies to check")

    all_fails = {}
    total_recovered = 0
    total_ads = 0
    for m in MOVIES:
        fails = check_movie(m)
        if fails:
            all_fails[m] = fails

    print("\n" + "=" * 60)
    if all_fails:
        print(f"FAIL: {len(all_fails)}/{len(MOVIES)} movies have problems:")
        for m, fails in all_fails.items():
            print(f"  - {m}:")
            for f in fails:
                print(f"      * {f}")
        print("\nDo NOT submit until these are resolved.")
        sys.exit(1)
    else:
        print(f"PASS: all {len(MOVIES)} movies look good.")
        print("Safe to submit:")
        print("  bash scripts/slurm/stage2/cmb_ad/submit_all.sh")
        sys.exit(0)


if __name__ == "__main__":
    main()
