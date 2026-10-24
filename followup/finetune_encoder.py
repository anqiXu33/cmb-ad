"""Fine-tune a small pretrained encoder for reference-form selection (runs on a Mac).

Same rows, same 5-fold GroupKFold split and same masked inputs as
followup/text_models.py, so the numbers drop straight into that table.
The encoder reads the pair (context, current) built by followup/text_inputs.py
and predicts NAME / DESC / PRN for the masked TARGET slot.

Then a stacked model (LightGBM on the 24 discourse features + the encoder's
out-of-fold probabilities) is evaluated on the same folds.

Usage on the MacBook (Apple GPU via MPS):

    pip install torch transformers lightgbm scikit-learn pandas
    # from the repo root, after followup.text_chains has been run once:
    python -m followup.finetune_encoder                          # distilroberta-base, 5 folds
    python -m followup.finetune_encoder --folds 1                # quick look: fold 0 only
    python -m followup.finetune_encoder --model microsoft/deberta-v3-small   # needs: pip install sentencepiece
    python -m followup.finetune_encoder --dry_run                # tiny random model, no download

Rough timing on an M1 Pro (16 GB), distilroberta-base, max_len 192, 3 epochs:
~5-8 min per fold. Close other heavy apps; memory peaks around 4 GB.

Outputs (results/followup/):
    encoder_<tag>_oof.npy           out-of-fold class probabilities
    encoder_<tag>_summary.md        metrics, same columns as text_models_summary.md
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from followup.text_models import CLASSES, OUT, _lgbm, load, scores
from followup.text_chains import TEXT_FEATURE_COLS
from sklearn.model_selection import GroupKFold


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


# ----------------------------------------------------------------------------
# model / tokenizer
# ----------------------------------------------------------------------------
def build_model_and_tokenizer(name: str, dry_run: bool, texts: list[str]):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    if not dry_run:
        tok = AutoTokenizer.from_pretrained(name)
        tok.truncation_side = "left"          # drop the OLDEST context first
        model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=len(CLASSES))
        return model, tok

    # offline smoke test: word-level tokenizer trained on our text + tiny random RoBERTa
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors
    from transformers import PreTrainedTokenizerFast, RobertaConfig, RobertaForSequenceClassification
    t = Tokenizer(models.WordLevel(unk_token="<unk>"))
    t.pre_tokenizer = pre_tokenizers.Whitespace()
    t.train_from_iterator(texts, trainers.WordLevelTrainer(
        vocab_size=8000, special_tokens=["<s>", "<pad>", "</s>", "<unk>"]))
    t.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>", pair="<s> $A </s> </s> $B </s>",
        special_tokens=[("<s>", t.token_to_id("<s>")), ("</s>", t.token_to_id("</s>"))])
    tok = PreTrainedTokenizerFast(tokenizer_object=t, bos_token="<s>", eos_token="</s>",
                                  sep_token="</s>", cls_token="<s>", pad_token="<pad>", unk_token="<unk>")
    tok.truncation_side = "left"
    cfg = RobertaConfig(vocab_size=t.get_vocab_size(), hidden_size=64, num_hidden_layers=2,
                        num_attention_heads=2, intermediate_size=128, max_position_embeddings=260,
                        num_labels=len(CLASSES), pad_token_id=tok.pad_token_id)
    return RobertaForSequenceClassification(cfg), tok


class PairDS(Dataset):
    def __init__(self, df):
        self.ctx = df.context.tolist()
        self.cur = df.current.tolist()
        self.y = df.y.tolist()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.ctx[i], self.cur[i], self.y[i]


def make_collate(tok, max_len):
    def collate(batch):
        ctx, cur, y = zip(*batch)
        enc = tok(list(ctx), list(cur), truncation="only_first", max_length=max_len,
                  padding=True, return_tensors="pt")
        enc.pop("token_type_ids", None)
        enc["labels"] = torch.tensor(y)
        return enc
    return collate


# ----------------------------------------------------------------------------
# train / predict one fold
# ----------------------------------------------------------------------------
def run_fold(args, train_df, test_df, device, all_texts):
    set_seed(args.seed)
    model, tok = build_model_and_tokenizer(args.model, args.dry_run, all_texts)
    model.to(device)
    collate = make_collate(tok, args.max_len)
    tr = DataLoader(PairDS(train_df), batch_size=args.bs, shuffle=True, collate_fn=collate)
    te = DataLoader(PairDS(test_df), batch_size=args.bs * 2, shuffle=False, collate_fn=collate)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = args.epochs * len(tr)
    warm = int(0.06 * total)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, warm)) * max(0.0, (total - s) / max(1, total - warm)))

    step, t0 = 0, time.time()
    for ep in range(args.epochs):
        model.train()
        for batch in tr:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % 100 == 0 or step == total:
                print(f"    step {step}/{total}  loss {loss.item():.3f}  {time.time() - t0:.0f}s", flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break

    model.eval()
    probs = []
    with torch.no_grad():
        for batch in te:
            batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            probs.append(torch.softmax(model(**batch).logits.float(), -1).cpu().numpy())
    return np.concatenate(probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilroberta-base")
    ap.add_argument("--folds", type=int, default=5, help="run only the first N of the 5 folds")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max_len", type=int, default=192)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_steps", type=int, default=0, help="debug: stop each fold after N steps")
    ap.add_argument("--dry_run", action="store_true", help="tiny random model, no download")
    args = ap.parse_args()
    if args.dry_run:
        args.lr = max(args.lr, 1e-3)

    device = pick_device()
    tag = "dryrun" if args.dry_run else args.model.split("/")[-1]
    print(f"device={device}  model={'tiny-random' if args.dry_run else args.model}", flush=True)

    df = load()
    all_texts = (df.context + " " + df.current).tolist()
    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(df, df.y, groups=df.seq_id))[: args.folds]

    oof = np.full((len(df), len(CLASSES)), np.nan)
    for k, (tr, te) in enumerate(splits):
        print(f"fold {k}: train {len(tr)}  test {len(te)}", flush=True)
        oof[te] = run_fold(args, df.iloc[tr], df.iloc[te], device, all_texts)
    np.save(OUT / f"encoder_{tag}_oof.npy", oof)

    done = ~np.isnan(oof[:, 0])
    y = df.y.values

    # stacking: LightGBM on discourse features + encoder OOF probabilities,
    # evaluated on the same folds (only over rows that have OOF predictions)
    stack_pred = np.full(len(df), -1)
    if args.folds == 5:
        X = np.hstack([df[TEXT_FEATURE_COLS].values, oof])
        for tr, te in splits:
            m = _lgbm()
            m.fit(X[tr], y[tr])
            stack_pred[te] = m.predict(X[te])

    # baselines on the same evaluated rows
    feat_pred = np.full(len(df), -1)
    for tr, te in splits:
        m = _lgbm()
        m.fit(df[TEXT_FEATURE_COLS].values[tr], y[tr])
        feat_pred[te] = m.predict(df[TEXT_FEATURE_COLS].values[te])
    rule_pred = np.where(df.rule_pred == "PRN", 2, 0)

    rows = {"rule": scores(y[done], rule_pred[done]),
            "lgbm_feat": scores(y[done], feat_pred[done]),
            f"encoder ({tag})": scores(y[done], oof[done].argmax(1))}
    if args.folds == 5:
        rows[f"stack_lgbm (features + {tag})"] = scores(y[done], stack_pred[done])
    rows["rule"]["acc3"] = float(np.mean([(p == t) or (p == 0 and t == 1)
                                          for p, t in zip(rule_pred[done], y[done])]))

    L = [f"# Encoder fine-tuning ({tag}), {done.sum()} evaluated rows, folds={args.folds}", "",
         f"device={device}, epochs={args.epochs}, bs={args.bs}, lr={args.lr}, max_len={args.max_len}", "",
         "| model | acc3 | F1 macro3 | acc PRN/non | F1 PRN | F1 DESC | pred PRN% |",
         "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, s in rows.items():
        L.append(f"| {name} | {s['acc3']:.3f} | {s['f1_macro3']:.3f} | {s['acc_bin']:.3f} | "
                 f"{s['f1_prn']:.3f} | {s['f1_desc']:.3f} | {100 * s['pred_prn_rate']:.1f} |")
    L.append(f"\ngold PRN% = {100 * (y[done] == 2).mean():.1f}")
    if args.dry_run:
        L.append("\nDRY RUN: tiny random model, numbers are meaningless; this only checks the pipeline.")
    (OUT / f"encoder_{tag}_summary.md").write_text("\n".join(L))
    (OUT / f"encoder_{tag}_results.json").write_text(json.dumps(rows, indent=2))
    print("\n".join(L))


if __name__ == "__main__":
    main()
