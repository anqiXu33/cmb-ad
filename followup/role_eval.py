"""Does the mention's syntactic position help, and where are the hard cases?

Same rows and the same 5-fold GroupKFold split as followup/text_models.py.
Compares, with and without the role_* features:
    lgbm      LightGBM on the discourse features
    stack     LightGBM on discourse features + out-of-fold TF-IDF text probabilities
and reports every model overall and per syntactic position (subject, object,
possessive, prepositional object, other).

A planner that decides BEFORE the sentence is written does not know the
position. It can still use a role-aware model by asking it once per position
("as subject: NAME, as possessive: PRN") and passing both recommendations to the
generator; evaluating that planner at the position the describer actually used
gives exactly the "+role" numbers below.

Run:  python -m followup.role_eval
Out:  results/followup/role_eval_summary.md, role_eval_results.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.model_selection import GroupKFold

from followup.text_chains import ROLE_COLS, ROLES, TEXT_FEATURE_COLS
from followup.text_models import OUT, _lgbm, load, scores, tfidf_lr

BASE_COLS = [c for c in TEXT_FEATURE_COLS if c not in ROLE_COLS]
ROLE_NAMES = {"subj": "subject", "obj": "object", "poss": "possessive",
              "pobj": "prep. object", "other": "other"}


def lgbm_pred(tr, te, cols):
    m = _lgbm()
    m.fit(tr[cols].values, tr.y)
    return m.predict(te[cols].values)


def stack_pred(tr, te, cols):
    inner = GroupKFold(n_splits=3)
    oof = np.zeros((len(tr), 3))
    for a, b in inner.split(tr, tr.y, groups=tr.seq_id):
        _, p = tfidf_lr(tr.iloc[a], tr.iloc[b])
        oof[b] = p
    _, p_te = tfidf_lr(tr, te)
    m = _lgbm()
    m.fit(np.hstack([tr[cols].values, oof]), tr.y)
    return m.predict(np.hstack([te[cols].values, p_te]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["cv", "lodo"], default="cv",
                    help="cv: 5-fold GroupKFold by sequence; lodo: train on two datasets, test on the third")
    a = ap.parse_args()
    df = load()
    print(f"{len(df)} rows; role counts {df.role.value_counts().to_dict()}", flush=True)
    if a.protocol == "lodo":
        ds = df.dataset.to_numpy()
        splits = [(np.where(ds != d)[0], np.where(ds == d)[0]) for d in sorted(df.dataset.unique())]
    else:
        splits = list(GroupKFold(n_splits=5).split(df, df.y, groups=df.seq_id))
    suffix = "_lodo" if a.protocol == "lodo" else ""
    models = {"lgbm": lambda tr, te: lgbm_pred(tr, te, BASE_COLS),
              "lgbm+role": lambda tr, te: lgbm_pred(tr, te, BASE_COLS + ROLE_COLS),
              "stack": lambda tr, te: stack_pred(tr, te, BASE_COLS),
              "stack+role": lambda tr, te: stack_pred(tr, te, BASE_COLS + ROLE_COLS)}
    preds = {k: np.empty(len(df), int) for k in models}
    for k, (tr, te) in enumerate(splits):
        for name, fn in models.items():
            preds[name][te] = fn(df.iloc[tr], df.iloc[te])
        print(f"fold {k} done", flush=True)
    preds = {"rule": np.where(df.rule_pred == "PRN", 2, 0), **preds}

    y = df.y.values
    res = {}
    parts = ["ALL"] + ROLES + [f"ds:{d}" for d in sorted(df.dataset.unique())]
    for part in parts:
        if part == "ALL":
            mask = np.ones(len(df), bool)
        elif part.startswith("ds:"):
            mask = (df.dataset == part[3:]).values
        else:
            mask = (df.role == part).values
        res[part] = {"n": int(mask.sum()), "gold_prn_rate": float((y[mask] == 2).mean()),
                     **{m: scores(y[mask], p[mask]) for m, p in preds.items()}}
    (OUT / f"role_eval{suffix}_results.json").write_text(json.dumps(res, indent=2))

    proto = ("leave one dataset out: each dataset scored by models trained on the other two"
             if a.protocol == "lodo" else "5-fold by sequence")
    L = [f"# Syntactic position: does it help, and where is it hard? (text-only chains, {proto})", "",
         "| position | n | gold PRN% | model | acc PRN/non | F1 PRN | pred PRN% |",
         "| --- | --- | --- | --- | --- | --- | --- |"]
    for part, r in res.items():
        label = "all" if part == "ALL" else (part if part.startswith("ds:") else ROLE_NAMES[part])
        for m in preds:
            s = r[m]
            L.append(f"| {label} | {r['n']} | {100 * r['gold_prn_rate']:.1f} | {m} | {s['acc_bin']:.3f} | "
                     f"{s['f1_prn']:.3f} | {100 * s['pred_prn_rate']:.1f} |")
    (OUT / f"role_eval{suffix}_summary.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
