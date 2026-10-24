"""Learned reference-form selector vs. the thesis rule planner.

Protocols
---------
A  pooled 5-fold GroupKFold by sequence (movie / episode / clip), all datasets
B  leave-one-dataset-out (train on two, test on the third)

Models: majority, rule planner (replayed), logistic regression, LightGBM.
Targets: 3-class {NAME, DESC, PRN} on rows with a bound silver label;
         binary PRN vs non-PRN is also reported because the rule planner only
         decides pronoun-vs-not.

Run:  python -m followup.train
Out:  results/followup/eval_summary.md, eval_results.json, lgbm_importance.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from followup.features import FEATURE_COLS

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "followup"
CLASSES = ["NAME", "DESC", "PRN"]


def load() -> pd.DataFrame:
    df = pd.read_csv(OUT / "features.csv")
    rule = pd.read_csv(OUT / "rule_preds.csv")
    df = df.merge(rule, on=["dataset", "seq_id", "anno_idx", "char_id"], how="left")
    df = df[df.label.isin(CLASSES)].copy()
    df["y"] = df.label.map({c: i for i, c in enumerate(CLASSES)})
    df["y_bin"] = (df.label == "PRN").astype(int)
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(-1)
    return df


def _scores(y_true, y_pred, y_true_lbl, pred_lbl) -> dict:
    """3-class and binary metrics. pred_lbl may be 'NAME' for the rule, which is
    scored leniently against DESC too (the rule has no DESC output)."""
    out = {}
    out["acc3"] = accuracy_score(y_true, y_pred)
    out["f1_macro3"] = f1_score(y_true, y_pred, average="macro")
    yb = (np.asarray(y_true_lbl) == "PRN").astype(int)
    pb = (np.asarray(pred_lbl) == "PRN").astype(int)
    out["acc_bin"] = accuracy_score(yb, pb)
    out["f1_prn"] = f1_score(yb, pb, zero_division=0)
    out["pred_prn_rate"] = float(pb.mean())
    out["gold_prn_rate"] = float(yb.mean())
    return out


def rule_scores(d: pd.DataFrame) -> dict:
    pred = d.rule_pred.values
    # lenient 3-class: rule NAME counts as correct for NAME or DESC
    y3 = d.label.values
    acc3 = np.mean([(p == t) or (p == "NAME" and t == "DESC") for p, t in zip(pred, y3)])
    s = _scores(d.y, [0 if p == "NAME" else 2 for p in pred], y3, pred)
    s["acc3"] = float(acc3)
    s["f1_macro3"] = float("nan")   # not meaningful for a 2-output planner
    return s


def majority_scores(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    maj = train.label.mode()[0]
    pred = np.array([maj] * len(test))
    return _scores(test.y, [CLASSES.index(maj)] * len(test), test.label.values, pred)


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, model: str):
    X, y = train[FEATURE_COLS].values, train.y.values
    Xt = test[FEATURE_COLS].values
    if model == "lr":
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"))
        clf.fit(X, y)
        return clf.predict(Xt), None
    clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31,
                             min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
                             verbose=-1, random_state=0)
    clf.fit(X, y)
    return clf.predict(Xt), clf.feature_importances_


def evaluate(train: pd.DataFrame, test: pd.DataFrame) -> tuple[dict, np.ndarray | None]:
    res = {"n_test": int(len(test)),
           "majority": majority_scores(train, test),
           "rule": rule_scores(test)}
    imp = None
    for m in ("lr", "lgbm"):
        pred, imp_m = fit_predict(train, test, m)
        res[m] = _scores(test.y, pred, test.label.values, np.array(CLASSES)[pred])
        if imp_m is not None:
            imp = imp_m
    return res, imp


def main():
    df = load()
    results = {"protocol_A_pooled_cv": {}, "protocol_B_leave_one_dataset_out": {}}
    imps = []

    # A: pooled grouped CV, scored per dataset and overall
    gkf = GroupKFold(n_splits=5)
    oof_pred = np.empty(len(df), dtype=int)
    for tr, te in gkf.split(df, df.y, groups=df.seq_id):
        pred, imp = fit_predict(df.iloc[tr], df.iloc[te], "lgbm")
        oof_pred[te] = pred
        imps.append(imp)
    oof_lr = np.empty(len(df), dtype=int)
    for tr, te in gkf.split(df, df.y, groups=df.seq_id):
        oof_lr[te], _ = fit_predict(df.iloc[tr], df.iloc[te], "lr")
    for ds in ["ALL"] + sorted(df.dataset.unique()):
        mask = np.ones(len(df), bool) if ds == "ALL" else (df.dataset == ds).values
        d = df[mask]
        results["protocol_A_pooled_cv"][ds] = {
            "n": int(mask.sum()),
            "majority": majority_scores(df, d),
            "rule": rule_scores(d),
            "lr": _scores(d.y, oof_lr[mask], d.label.values, np.array(CLASSES)[oof_lr[mask]]),
            "lgbm": _scores(d.y, oof_pred[mask], d.label.values, np.array(CLASSES)[oof_pred[mask]]),
        }

    # B: leave one dataset out
    for ds in sorted(df.dataset.unique()):
        tr, te = df[df.dataset != ds], df[df.dataset == ds]
        res, _ = evaluate(tr, te)
        results["protocol_B_leave_one_dataset_out"][ds] = res

    imp = pd.DataFrame({"feature": FEATURE_COLS, "importance": np.mean(imps, axis=0)})
    imp.sort_values("importance", ascending=False).to_csv(OUT / "lgbm_importance.csv", index=False)
    (OUT / "eval_results.json").write_text(json.dumps(results, indent=2))

    # markdown summary
    lines = ["# Learned selector vs rule planner (English silver labels)", ""]
    for proto, block in results.items():
        lines += [f"## {proto}", "",
                  "| split | n | model | acc3 | F1 macro3 | acc PRN/non | F1 PRN | pred PRN% | gold PRN% |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for split, res in block.items():
            n = res.get("n", res.get("n_test"))
            for m in ("majority", "rule", "lr", "lgbm"):
                s = res[m]
                f1m = "" if np.isnan(s["f1_macro3"]) else f"{s['f1_macro3']:.3f}"
                lines.append(f"| {split} | {n} | {m} | {s['acc3']:.3f} | {f1m} | {s['acc_bin']:.3f} | "
                             f"{s['f1_prn']:.3f} | {100*s['pred_prn_rate']:.1f} | {100*s['gold_prn_rate']:.1f} |")
        lines.append("")
    lines += ["## LightGBM feature importance (top 12, mean split count over folds)", ""]
    for _, r in imp.sort_values("importance", ascending=False).head(12).iterrows():
        lines.append(f"- {r.feature}: {r.importance:.0f}")
    (OUT / "eval_summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
