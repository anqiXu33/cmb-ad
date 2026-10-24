"""Does reading the text help? Text classifiers vs. rule vs. discourse-feature LightGBM.

All models use the SAME rows (text_features.csv, labels NAME/DESC/PRN) and the
SAME 5-fold GroupKFold split by sequence as followup/train.py --variant text.

Models
------
rule          thesis rule planner replayed on the text stream
lgbm_feat     LightGBM on the 24 discourse features (= train.py text variant)
tfidf_lr      logistic regression on word 1-2-gram TF-IDF of the masked
              current AD and of the masked 3-AD context (separate vocabularies)
combo_lr      tfidf_lr + standardised discourse features in one model
stack_lgbm    LightGBM on discourse features + out-of-fold tfidf_lr
              probabilities (nested 3-fold inside each training fold)

Run:  python -m followup.text_models
Out:  results/followup/text_models_summary.md, text_models_results.json
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from followup.text_chains import TEXT_FEATURE_COLS
from followup.text_inputs import build_inputs

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "followup"
CLASSES = ["NAME", "DESC", "PRN"]


def load() -> pd.DataFrame:
    df = pd.read_csv(OUT / "text_features.csv")
    rule = pd.read_csv(OUT / "text_rule_preds.csv")
    df = df.merge(rule, on=["dataset", "seq_id", "anno_idx", "char_id"], how="left")
    df = df[df.label.isin(CLASSES)].copy()
    df["y"] = df.label.map({c: i for i, c in enumerate(CLASSES)})
    df[TEXT_FEATURE_COLS] = df[TEXT_FEATURE_COLS].fillna(-1)
    ads = pd.read_csv(OUT / "text_ads.csv")
    df = build_inputs(df, ads)
    return df.reset_index(drop=True)


def scores(y, pred) -> dict:
    y, pred = np.asarray(y), np.asarray(pred)
    yb, pb = (y == 2).astype(int), (pred == 2).astype(int)
    return {"acc3": accuracy_score(y, pred),
            "f1_macro3": f1_score(y, pred, average="macro"),
            "acc_bin": accuracy_score(yb, pb),
            "f1_prn": f1_score(yb, pb, zero_division=0),
            "f1_desc": f1_score((y == 1).astype(int), (pred == 1).astype(int), zero_division=0),
            "pred_prn_rate": float(pb.mean())}


class TextVec:
    def __init__(self):
        kw = dict(ngram_range=(1, 2), min_df=3, max_features=50000, sublinear_tf=True,
                  lowercase=True, token_pattern=r"(?u)\b\w+\b")
        self.cur = TfidfVectorizer(**kw)
        self.ctx = TfidfVectorizer(**kw)

    def fit_transform(self, d):
        return sparse.hstack([self.cur.fit_transform(d.current), self.ctx.fit_transform(d.context)]).tocsr()

    def transform(self, d):
        return sparse.hstack([self.cur.transform(d.current), self.ctx.transform(d.context)]).tocsr()


def tfidf_lr(train, test, C=2.0):
    v = TextVec()
    Xtr, Xte = v.fit_transform(train), v.transform(test)
    clf = LogisticRegression(max_iter=3000, C=C)
    clf.fit(Xtr, train.y)
    return clf.predict(Xte), clf.predict_proba(Xte)


def combo_lr(train, test, C=2.0):
    v = TextVec()
    sc = StandardScaler()
    Ftr = sc.fit_transform(train[TEXT_FEATURE_COLS].values)
    Fte = sc.transform(test[TEXT_FEATURE_COLS].values)
    Xtr = sparse.hstack([v.fit_transform(train), sparse.csr_matrix(Ftr)]).tocsr()
    Xte = sparse.hstack([v.transform(test), sparse.csr_matrix(Fte)]).tocsr()
    clf = LogisticRegression(max_iter=3000, C=C)
    clf.fit(Xtr, train.y)
    return clf.predict(Xte)


def _lgbm():
    return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31,
                              min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
                              verbose=-1, random_state=0)


def lgbm_feat(train, test):
    m = _lgbm()
    m.fit(train[TEXT_FEATURE_COLS].values, train.y)
    return m.predict(test[TEXT_FEATURE_COLS].values)


def stack_lgbm(train, test):
    # nested out-of-fold text probabilities for the training fold
    inner = GroupKFold(n_splits=3)
    oof = np.zeros((len(train), 3))
    for tr, te in inner.split(train, train.y, groups=train.seq_id):
        _, p = tfidf_lr(train.iloc[tr], train.iloc[te])
        oof[te] = p
    _, p_test = tfidf_lr(train, test)
    Xtr = np.hstack([train[TEXT_FEATURE_COLS].values, oof])
    Xte = np.hstack([test[TEXT_FEATURE_COLS].values, p_test])
    m = _lgbm()
    m.fit(Xtr, train.y)
    return m.predict(Xte)


MODELS = {"lgbm_feat": lgbm_feat,
          "tfidf_lr": lambda tr, te: tfidf_lr(tr, te)[0],
          "combo_lr": combo_lr,
          "stack_lgbm": stack_lgbm}


def main():
    df = load()
    print(f"{len(df)} rows", flush=True)
    preds = {m: np.empty(len(df), dtype=int) for m in MODELS}
    gkf = GroupKFold(n_splits=5)
    for k, (tr, te) in enumerate(gkf.split(df, df.y, groups=df.seq_id)):
        for name, fn in MODELS.items():
            preds[name][te] = fn(df.iloc[tr], df.iloc[te])
        print(f"fold {k} done", flush=True)
    rule_pred = np.where(df.rule_pred == "PRN", 2, 0)

    res = {}
    for ds in ["ALL"] + sorted(df.dataset.unique()):
        mask = np.ones(len(df), bool) if ds == "ALL" else (df.dataset == ds).values
        y = df.y.values[mask]
        r = {"n": int(mask.sum()), "gold_prn_rate": float((y == 2).mean())}
        r["majority"] = scores(y, np.zeros_like(y))
        rs = scores(y, rule_pred[mask])
        # rule has no DESC output: count NAME-for-DESC as correct in acc3
        rs["acc3"] = float(np.mean([(p == t) or (p == 0 and t == 1) for p, t in zip(rule_pred[mask], y)]))
        rs["f1_macro3"] = float("nan")
        rs["f1_desc"] = float("nan")
        r["rule"] = rs
        for name in MODELS:
            r[name] = scores(y, preds[name][mask])
        res[ds] = r
    (OUT / "text_models_results.json").write_text(json.dumps(res, indent=2))

    order = ["majority", "rule", "lgbm_feat", "tfidf_lr", "combo_lr", "stack_lgbm"]
    L = ["# Does reading the text help? (text-only chains, 5-fold GroupKFold by sequence)", "",
         "| split | model | acc3 | F1 macro3 | acc PRN/non | F1 PRN | F1 DESC | pred PRN% | gold PRN% |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    fmt = lambda v: "" if (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"
    for ds, r in res.items():
        for m in order:
            s = r[m]
            L.append(f"| {ds} (n={r['n']}) | {m} | {fmt(s['acc3'])} | {fmt(s['f1_macro3'])} | "
                     f"{fmt(s['acc_bin'])} | {fmt(s['f1_prn'])} | {fmt(s['f1_desc'])} | "
                     f"{100*s['pred_prn_rate']:.1f} | {100*r['gold_prn_rate']:.1f} |")
    (OUT / "text_models_summary.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
