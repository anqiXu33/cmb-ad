"""Shared pieces for the LLM experiments (zero-shot prompting and LoRA fine-tuning).

Task: binary, PRONOUN vs explicit reference (NAME or DESC), on the cleaned
text-only labels (followup.text_chains). Every LLM is compared on the SAME rows
against:
    rule        the thesis rule planner replayed on the text stream
    lgbm+role   LightGBM on discourse features + syntactic position (5-fold OOF)
    stack+role  lgbm+role + out-of-fold TF-IDF text probabilities (5-fold OOF)
Folds are the same 5-fold GroupKFold by sequence used everywhere else.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold

from followup.text_inputs import build_inputs
from followup.text_models import OUT, load

ROLE_NAMES = {"subj": "subject", "obj": "object", "poss": "possessive",
              "pobj": "prep. object", "other": "other"}


def load_rows(k: int = 10, mask_name: bool = True) -> pd.DataFrame:
    """Labelled rows with `context` = the previous k AD lines (newline-separated)."""
    df = load()
    ads = pd.read_csv(OUT / "text_ads.csv")
    df = build_inputs(df, ads, k=k, mask_name=mask_name, sep="\n")
    df["y_bin"] = (df.y == 2).astype(int)
    return df.reset_index(drop=True)


def folds(df: pd.DataFrame):
    return list(GroupKFold(n_splits=5).split(df, df.y, groups=df.seq_id))


def baseline_preds(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Binary OOF predictions of the reference models on all rows (cached)."""
    key = hashlib.md5("|".join(df.dataset + ":" + df.seq_id.astype(str) + ":" +
                               df.anno_idx.astype(str) + ":" + df.char_id).encode()).hexdigest()[:12]
    cache = OUT / f"baselines_oof_{key}.npz"
    if cache.exists():
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    from followup.role_eval import BASE_COLS, lgbm_pred, stack_pred
    from followup.text_chains import ROLE_COLS
    cols = BASE_COLS + ROLE_COLS
    out = {"rule": (df.rule_pred == "PRN").astype(int).to_numpy(),
           "lgbm+role": np.empty(len(df), int), "stack+role": np.empty(len(df), int)}
    for tr, te in folds(df):
        out["lgbm+role"][te] = (lgbm_pred(df.iloc[tr], df.iloc[te], cols) == 2).astype(int)
        out["stack+role"][te] = (stack_pred(df.iloc[tr], df.iloc[te], cols) == 2).astype(int)
    np.savez(cache, **out)
    return out


def _s(y, p) -> dict:
    return {"acc": float(accuracy_score(y, p)), "f1_prn": float(f1_score(y, p, zero_division=0)),
            "pred_prn": float(np.mean(p)), "gold_prn": float(np.mean(y)), "n": int(len(y))}


def report(df: pd.DataFrame, mask: np.ndarray, preds: dict[str, np.ndarray],
           title: str, notes: list[str], tag: str) -> str:
    """Markdown + JSON with overall, per-position and per-dataset scores on `mask` rows."""
    y = df.y_bin.to_numpy()
    res = {"overall": {m: _s(y[mask], p[mask]) for m, p in preds.items()}}
    for r, name in ROLE_NAMES.items():
        k = mask & (df.role == r).to_numpy()
        if k.sum() >= 20:
            res[name] = {m: _s(y[k], p[k]) for m, p in preds.items()}
    for d in sorted(df.dataset.unique()):
        k = mask & (df.dataset == d).to_numpy()
        if k.sum() >= 20:
            res["dataset " + d] = {m: _s(y[k], p[k]) for m, p in preds.items()}
    L = [f"# {title}", "", *notes, "",
         "| subset | n | gold PRN% | model | acc PRN/non | F1 PRN | pred PRN% |",
         "| --- | --- | --- | --- | --- | --- | --- |"]
    for part, r in res.items():
        for m, s in r.items():
            L.append(f"| {part} | {s['n']} | {100 * s['gold_prn']:.1f} | {m} | {s['acc']:.3f} | "
                     f"{s['f1_prn']:.3f} | {100 * s['pred_prn']:.1f} |")
    md = "\n".join(L)
    (OUT / f"{tag}_summary.md").write_text(md)
    (OUT / f"{tag}_results.json").write_text(json.dumps(res, indent=2))
    return md


def tiny_tokenizer(texts: list[str]):
    """Offline word-level tokenizer with a chat-free API, for --dry_run only."""
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast
    t = Tokenizer(models.WordLevel(unk_token="<unk>"))
    t.pre_tokenizer = pre_tokenizers.Whitespace()
    t.train_from_iterator(texts, trainers.WordLevelTrainer(
        vocab_size=6000, special_tokens=["<pad>", "<unk>", "<eos>", "NAME", "PRONOUN"]))
    tok = PreTrainedTokenizerFast(tokenizer_object=t, pad_token="<pad>", unk_token="<unk>", eos_token="<eos>")
    return tok
