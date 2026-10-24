"""Discourse-state features for the reference-form selector.

Input : results/followup/ads.csv  (one row per AD, incl. ADs with no face)
        results/followup/silver_labels.csv (one row per (AD, on-screen char))
Output: one feature row per silver-label row.

All features are computable at inference time from the pipeline's own state:
timestamps, pred_ids history, gender map, the forms chosen for earlier ADs
(here: the gold forms, i.e. teacher forcing), and film-grammar columns.

Feature groups
--------------
timing       gap_ads, gap_sec, log_gap_sec, first_in_seq, consecutive, ad_dur
competition  n_chars_cur, n_chars_prev, n_same_gender_cur, n_same_gender_prev,
             n_unknown_gender_cur, competitor_named_prev, prev_ad_empty
history      prev_form_* (one-hot of this char's form at its last appearance),
             n_prior_name, n_prior_prn, n_prior_appear, name_ratio_prior
film         thread_switch, shot_scale (NaN where the dataset has none)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "followup"

FORMS = ["NAME", "DESC", "PRN", "OMIT", "AMBIG", "none"]


def _split(s) -> list[str]:
    return [] if (pd.isna(s) or s == "") else str(s).split("|")


def build_features(ads: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    ads = ads.sort_values(["dataset", "seq_id", "start", "anno_idx"]).reset_index(drop=True)
    rows = rows.copy()
    gender = dict(zip(rows["char_id"], rows["char_gender"].fillna("")))
    # label lookup: (dataset, seq, anno_idx, char) -> form
    form_of = {(d, s, a, c): l for d, s, a, c, l in
               zip(rows.dataset, rows.seq_id, rows.anno_idx, rows.char_id, rows.label)}

    feats = []
    for (ds, seq), g in ads.groupby(["dataset", "seq_id"], sort=False):
        state: dict[str, dict] = {}       # char -> last_idx, last_end, consecutive, counts
        prev_pids: set[str] = set()
        prev_named: set[str] = set()
        prev_end = None
        for i, ad in enumerate(g.itertuples(index=False)):
            pids = set(_split(ad.pred_ids))
            named = set(_split(ad.named_ids))
            for c in sorted(pids):
                st = state.get(c)
                gc = gender.get(c, "")
                same_cur = sum(1 for o in pids if o != c and gender.get(o, "") == gc and gc)
                unk_cur = sum(1 for o in pids if o != c and not gender.get(o, ""))
                same_prev = sum(1 for o in prev_pids if o != c and gender.get(o, "") == gc and gc)
                unk_prev = sum(1 for o in prev_pids if o != c and not gender.get(o, ""))
                f = {
                    "dataset": ds, "seq_id": seq, "anno_idx": ad.anno_idx, "char_id": c,
                    "gap_ads": (i - st["last_idx"]) if st else -1,
                    "gap_sec": (ad.start - st["last_end"]) if st else -1.0,
                    "first_in_seq": int(st is None),
                    "consecutive": (st["consecutive"] if st and st["last_idx"] == i - 1 else 0),
                    "ad_dur": ad.end - ad.start,
                    "n_chars_cur": len(pids),
                    "n_chars_prev": len(prev_pids),
                    "n_same_gender_cur": same_cur,
                    "n_same_gender_prev": same_prev,
                    "n_unknown_gender_cur": unk_cur + unk_prev,
                    "gender_known": int(bool(gc)),
                    "competitor_named_prev": int(bool(prev_named - {c})),
                    "self_named_prev": int(c in prev_named),
                    "prev_ad_empty": int(len(prev_pids) == 0),
                    "in_prev_ad": int(c in prev_pids),
                    "n_prior_appear": st["appear"] if st else 0,
                    "n_prior_name": st["n_name"] if st else 0,
                    "n_prior_prn": st["n_prn"] if st else 0,
                    "thread_switch": ad.thread_switch,
                    "shot_scale": ad.shot_scale,
                    "n_words": len(str(ad.text).split()),
                }
                f["log_gap_sec"] = np.log1p(f["gap_sec"]) if f["gap_sec"] >= 0 else -1.0
                f["name_ratio_prior"] = (f["n_prior_name"] / f["n_prior_appear"]) if f["n_prior_appear"] else -1.0
                pf = st["last_form"] if st else "none"
                for form in FORMS:
                    f[f"prev_form_{form}"] = int(pf == form)
                feats.append(f)
            # update state after the AD (teacher forcing on gold forms)
            for c in pids:
                lab = form_of.get((ds, seq, ad.anno_idx, c), "none")
                st = state.setdefault(c, {"last_idx": -99, "last_end": 0.0, "consecutive": 0,
                                           "appear": 0, "n_name": 0, "n_prn": 0, "last_form": "none"})
                st["consecutive"] = st["consecutive"] + 1 if st["last_idx"] == i - 1 else 1
                st["last_idx"] = i
                st["last_end"] = ad.end
                st["appear"] += 1
                st["n_name"] += int(lab == "NAME")
                st["n_prn"] += int(lab == "PRN")
                st["last_form"] = lab
            prev_pids, prev_named, prev_end = pids, named, ad.end
    F = pd.DataFrame(feats)
    rows = rows.drop(columns=[c for c in ("thread_switch", "shot_scale") if c in rows.columns])
    return rows.merge(F, on=["dataset", "seq_id", "anno_idx", "char_id"], how="left")


FEATURE_COLS = [
    "gap_ads", "gap_sec", "log_gap_sec", "first_in_seq", "consecutive", "ad_dur",
    "n_chars_cur", "n_chars_prev", "n_same_gender_cur", "n_same_gender_prev",
    "n_unknown_gender_cur", "gender_known", "competitor_named_prev", "self_named_prev",
    "prev_ad_empty", "in_prev_ad", "n_prior_appear", "n_prior_name", "n_prior_prn",
    "name_ratio_prior", "thread_switch", "shot_scale",
] + [f"prev_form_{f}" for f in FORMS]


if __name__ == "__main__":
    ads = pd.read_csv(OUT / "ads.csv")
    rows = pd.read_csv(OUT / "silver_labels.csv")
    df = build_features(ads, rows)
    df.to_csv(OUT / "features.csv", index=False)
    print(df.shape)
    print(df[FEATURE_COLS].describe().T[["mean", "min", "max"]].round(2).to_string())
