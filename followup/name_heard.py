"""Does the audience having HEARD a character's name shape how AD refers to them?

Uses REFRAMED (AD + timed subtitles). Two analyses:

A. Descriptive. For every character that the AD names at least once:
     t_ad   = time of the first AD line that names them
     t_dial = time of the first subtitle line that contains their name
   and the share of characters whose name was spoken BEFORE the AD first used it,
   plus the delay between the two. If professional describers wait for the
   film to reveal a name, t_dial < t_ad should dominate.

B. Predictive. Text-only chains (followup.text_chains) give PRN vs NAME/DESC
   labels; LightGBM with the 24 discourse features is compared against the
   same model plus "name heard" features, on 5-fold GroupKFold by movie.

New features (all computed from the subtitles before the AD line starts):
    heard_dialogue_before     name spoken in dialogue at any earlier time
    n_heard_dialogue          how many earlier subtitle lines contain it
    log_sec_since_heard       log1p(seconds since last spoken), -1 if never
    heard_but_not_named_ad    spoken in dialogue, but never named in AD yet

Run (after `python -m followup.reframed --export results/followup/reframed.parquet`):
    python -m followup.name_heard --data results/followup/reframed.parquet
Out: results/followup/name_heard_summary.md, name_heard_results.json  (numbers only, no AD text)
"""
from __future__ import annotations

import argparse
import bisect
import json
import re
import sys
from collections import Counter
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold

from followup.mentions_en import _name_tokens
from followup.reframed import OUT, load
from followup.text_chains import TEXT_FEATURE_COLS, build_features, resolve_sequence

HEARD_COLS = ["heard_dialogue_before", "n_heard_dialogue", "log_sec_since_heard",
              "heard_but_not_named_ad"]


def _name_regex(role: str):
    toks = [t for t in _name_tokens(role) if len(t) > 2]
    if not toks:
        return None
    return re.compile(r"\b(" + "|".join(re.escape(t) for t in toks) + r")\b", re.I)


def dialogue_index(dialogue: pd.DataFrame, charbanks: dict) -> dict:
    """(seq, char_id) -> sorted list of start times of subtitle lines naming the char."""
    idx = {}
    for seq, g in dialogue.groupby("seq_id"):
        times = g["start"].to_numpy()
        texts = g["text"].astype(str).tolist()
        for c in charbanks.get(seq, []):
            rx = _name_regex(c["role"])
            if rx is None:
                continue
            idx[(seq, c["id"])] = [t for t, s in zip(times, texts) if rx.search(s)]
    return idx


def add_heard_features(df: pd.DataFrame, didx: dict) -> pd.DataFrame:
    vals = {k: [] for k in HEARD_COLS}
    for r in df.itertuples(index=False):
        ts = didx.get((r.seq_id, r.char_id), [])
        k = bisect.bisect_left(ts, r.start)
        heard = k > 0
        vals["heard_dialogue_before"].append(int(heard))
        vals["n_heard_dialogue"].append(k)
        vals["log_sec_since_heard"].append(float(np.log1p(r.start - ts[k - 1])) if heard else -1.0)
        vals["heard_but_not_named_ad"].append(int(heard and r.n_prior_name == 0))
    for k, v in vals.items():
        df[k] = v
    return df


# ----------------------------------------------------------------------------
# A. descriptive
# ----------------------------------------------------------------------------
def first_naming_vs_dialogue(rows: pd.DataFrame, didx: dict) -> dict:
    named = rows[rows.label == "NAME"].groupby(["seq_id", "char_id"]).start.min()
    cats, delays = Counter(), []
    for (seq, cid), t_ad in named.items():
        ts = didx.get((seq, cid), [])
        if not ts:
            cats["never spoken in subtitles"] += 1
        elif ts[0] < t_ad:
            cats["spoken before first AD naming"] += 1
            delays.append(t_ad - ts[0])
        else:
            cats["AD names before it is spoken"] += 1
    n = sum(cats.values())
    out = {"n_characters_named_in_AD": n,
           "shares": {k: round(v / n, 3) for k, v in cats.items()} if n else {},
           "counts": dict(cats)}
    if delays:
        d = np.array(delays)
        out["delay_sec_spoken_to_AD_name"] = {"median": float(np.median(d)),
                                              "p25": float(np.percentile(d, 25)),
                                              "p75": float(np.percentile(d, 75))}
    return out


# ----------------------------------------------------------------------------
# B. predictive
# ----------------------------------------------------------------------------
def _lgbm():
    return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31,
                              min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
                              verbose=-1, random_state=0)


def _scores(y, p):
    return {"acc_bin": float(accuracy_score(y, p)),
            "f1_prn": float(f1_score(y, p, zero_division=0)),
            "pred_prn_rate": float(np.mean(p)), "gold_prn_rate": float(np.mean(y))}


def predictive(df: pd.DataFrame) -> dict:
    df = df[df.label.isin(["NAME", "DESC", "PRN"])].reset_index(drop=True)
    y = (df.label == "PRN").astype(int).to_numpy()
    base_cols = [c for c in TEXT_FEATURE_COLS if c not in ("thread_switch", "shot_scale")]
    res = {"n": int(len(df)), "n_movies": int(df.seq_id.nunique())}
    gkf = GroupKFold(n_splits=5)
    preds = {}
    for name, cols in [("discourse", base_cols), ("discourse+heard", base_cols + HEARD_COLS)]:
        X = df[cols].fillna(-1).to_numpy()
        p = np.empty(len(df), int)
        for tr, te in gkf.split(X, y, groups=df.seq_id):
            m = _lgbm()
            m.fit(X[tr], y[tr])
            p[te] = m.predict(X[te])
        preds[name] = p
        res[name] = _scores(y, p)
    sub = (df.heard_dialogue_before == 1).to_numpy()
    res["subset_heard"] = {k: _scores(y[sub], v[sub]) for k, v in preds.items()} | {"n": int(sub.sum())}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(OUT / "reframed.parquet"))
    ap.add_argument("--ad_column", default="audio_description_us")
    a = ap.parse_args()

    ads, dialogue, charbanks = load(a.data, ad_column=a.ad_column)
    all_rows, stats = [], Counter()
    for seq, g in ads.groupby("seq_id", sort=False):
        rows, st = resolve_sequence(g, charbanks.get(seq, []))
        all_rows.extend(rows)
        stats.update(st)
    rows = pd.DataFrame(all_rows)
    print(f"[name_heard] {len(rows)} labelled (AD, char) rows; resolution {dict(stats)}", file=sys.stderr)
    feats, _ = build_features(rows, ads)
    didx = dialogue_index(dialogue, charbanks)
    feats = add_heard_features(feats, didx)

    result = {"ad_column": a.ad_column,
              "n_movies": int(ads.seq_id.nunique()), "n_ad_lines": int(len(ads)),
              "n_subtitle_lines": int(len(dialogue)),
              "label_counts": feats.label.value_counts().to_dict(),
              "A_first_naming_vs_dialogue": first_naming_vs_dialogue(feats, didx),
              "B_predictive_prn_vs_explicit": predictive(feats)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "name_heard_results.json").write_text(json.dumps(result, indent=2))

    A, B = result["A_first_naming_vs_dialogue"], result["B_predictive_prn_vs_explicit"]
    L = ["# Name heard in dialogue vs AD reference (REFRAMED, US AD, train+validation)", "",
         f"{result['n_movies']} movies, {result['n_ad_lines']} AD lines, {result['n_subtitle_lines']} subtitle lines, "
         f"labels {result['label_counts']}", "",
         "## A. First AD naming vs first time the name is spoken", "",
         f"characters named in AD: {A['n_characters_named_in_AD']}", ""]
    for k, v in A.get("shares", {}).items():
        L.append(f"- {k}: {100 * v:.1f}% ({A['counts'][k]})")
    if "delay_sec_spoken_to_AD_name" in A:
        d = A["delay_sec_spoken_to_AD_name"]
        L.append(f"- delay spoken -> first AD naming: median {d['median']:.0f}s (IQR {d['p25']:.0f}-{d['p75']:.0f}s)")
    L += ["", "## B. PRN vs explicit, 5-fold GroupKFold by movie", "",
          "| features | rows | acc PRN/non | F1 PRN | pred PRN% | gold PRN% |", "| --- | --- | --- | --- | --- | --- |"]
    for k in ("discourse", "discourse+heard"):
        s = B[k]
        L.append(f"| {k} | {B['n']} | {s['acc_bin']:.3f} | {s['f1_prn']:.3f} | {100*s['pred_prn_rate']:.1f} | {100*s['gold_prn_rate']:.1f} |")
    for k in ("discourse", "discourse+heard"):
        s = B["subset_heard"][k]
        L.append(f"| {k} (name already spoken) | {B['subset_heard']['n']} | {s['acc_bin']:.3f} | {s['f1_prn']:.3f} | "
                 f"{100*s['pred_prn_rate']:.1f} | {100*s['gold_prn_rate']:.1f} |")
    (OUT / "name_heard_summary.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
