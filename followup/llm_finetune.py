"""QLoRA fine-tuning of a small LLM as a PRONOUN-vs-explicit classifier.

The model reads the previous k AD lines and the current line (the target's
name masked as TGTNAME everywhere, the reference slot as TARGET, as for the
encoder) and a classification head predicts whether the describer used a
pronoun. Only LoRA adapters and the head are trained; the base model is loaded
in 4 bit on CUDA.

Same 5-fold GroupKFold by sequence as every other model; by default only
fold 0 is run (about an hour on a Kaggle T4 for a 1.5B model), and the
reference models are scored on the same rows.

Kaggle (GPU T4, Internet on):
    python -m followup.llm_finetune --model Qwen/Qwen2.5-1.5B --load_4bit --k 10 --folds 1
Offline smoke test:
    python -m followup.llm_finetune --dry_run --folds 1 --max_steps 10
"""
from __future__ import annotations

import argparse
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from followup.llm_common import baseline_preds, folds, load_rows, report, tiny_tokenizer


def to_text(r) -> str:
    ctx = r.context if r.context != "NOCONTEXT" else "(none)"
    return f"Previous AD lines:\n{ctx}\n\nCurrent AD line:\n{r.current}\n\nHow is TARGET (= TGTNAME) referred to?"


def build(args, texts):
    from peft import LoraConfig, get_peft_model
    if args.dry_run:
        from transformers import Qwen2Config, Qwen2ForSequenceClassification
        tok = tiny_tokenizer(texts)
        cfg = Qwen2Config(vocab_size=max(tok.get_vocab().values()) + 1, hidden_size=64, intermediate_size=128,
                          num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                          max_position_embeddings=2048, pad_token_id=tok.pad_token_id, num_labels=2)
        model = Qwen2ForSequenceClassification(cfg)
    else:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        kw = {"num_labels": 2, "torch_dtype": torch.float16}
        if args.load_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=torch.float16)
            kw["device_map"] = {"": 0}
        model = AutoModelForSequenceClassification.from_pretrained(args.model, **kw)
        model.config.pad_token_id = tok.pad_token_id
        if args.load_4bit:
            from peft import prepare_model_for_kbit_training
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    tok.truncation_side = "left"          # keep the current line, drop the oldest context
    lora = LoraConfig(task_type="SEQ_CLS", r=16, lora_alpha=32, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                      modules_to_save=["score"])
    model = get_peft_model(model, lora)
    # trainable params (LoRA + head) in fp32 for stable fp16 mixed precision
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    return model, tok


def run_fold(args, tr, te, device, all_texts):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    model, tok = build(args, all_texts)
    if not args.load_4bit:
        model.to(device)
    collate = lambda b: (tok([x for x, _ in b], truncation=True, max_length=args.max_len,
                             padding=True, return_tensors="pt"), torch.tensor([y for _, y in b]))
    dl_tr = DataLoader(list(zip(tr.text.tolist(), tr.y_bin.tolist())), batch_size=args.bs, shuffle=True, collate_fn=collate)
    dl_te = DataLoader(list(zip(te.text.tolist(), te.y_bin.tolist())), batch_size=args.bs * 2, shuffle=False, collate_fn=collate)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps = args.epochs * len(dl_tr) // args.accum
    if args.max_steps:
        steps = min(steps, args.max_steps)
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (steps - s) / max(1, steps - warm)))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    model.train()
    step, t0, micro = 0, time.time(), 0
    for ep in range(args.epochs):
        for enc, y in dl_tr:
            enc = {k: v.to(device) for k, v in enc.items() if k != "token_type_ids"}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss = model(**enc, labels=y.to(device)).loss / args.accum
            scaler.scale(loss).backward()
            micro += 1
            if micro % args.accum:
                continue
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad()
            step += 1
            if step % 50 == 0 or step == steps:
                print(f"    step {step}/{steps}  loss {loss.item() * args.accum:.3f}  {time.time() - t0:.0f}s", flush=True)
            if step >= steps:
                break
        if step >= steps:
            break
    model.eval()
    probs = []
    with torch.no_grad():
        for enc, _ in dl_te:
            enc = {k: v.to(device) for k, v in enc.items() if k != "token_type_ids"}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(**enc).logits.float()
            probs.append(torch.softmax(logits, -1)[:, 1].cpu().numpy())
    return np.concatenate(probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--load_4bit", action="store_true")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--folds", type=int, default=1, help="run the first N of the 5 folds")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_rows(k=a.k, mask_name=True)
    df["text"] = [to_text(r) for r in df.itertuples(index=False)]
    print(f"device={device}; {len(df)} rows; example input:\n{df.text.iloc[0]}\n", flush=True)
    splits = folds(df)[: a.folds]
    p = np.full(len(df), np.nan)
    pred_prior = np.zeros(len(df), int)
    for k, (tr, te) in enumerate(splits):
        print(f"fold {k}: train {len(tr)}  test {len(te)}", flush=True)
        p[te] = run_fold(a, df.iloc[tr], df.iloc[te], device, df.text.tolist())
        # threshold so the predicted pronoun rate equals the TRAINING fold's rate
        prior = float(df.y_bin.to_numpy()[tr].mean())
        pred_prior[te] = (p[te] > np.quantile(p[te], 1 - prior)).astype(int)
    tag = "llm_finetune_" + ("dryrun" if a.dry_run else a.model.split("/")[-1])
    np.save(f"results/followup/{tag}_probs.npy", p)
    mask = ~np.isnan(p)
    preds = baseline_preds(df)
    preds["finetuned@0.5"] = (np.nan_to_num(p) > 0.5).astype(int)
    preds["finetuned@prior"] = pred_prior
    notes = [f"model: {'tiny random (dry run)' if a.dry_run else a.model}{' 4-bit QLoRA' if a.load_4bit else ' LoRA'}; "
             f"{a.k} previous AD lines, target name masked; folds run: {a.folds} of 5; epochs {a.epochs}; "
             f"lr {a.lr}; batch {a.bs}x{a.accum}; max_len {a.max_len}.",
             "finetuned@prior: threshold set per fold so the predicted pronoun rate equals the training fold's rate.",
             "Reference models are 5-fold out-of-fold predictions on the same rows."]
    if a.dry_run:
        notes.append("DRY RUN: numbers are meaningless; this only checks the pipeline.")
    print(report(df, mask, preds, "Fine-tuned LLM vs rule vs learned models (PRONOUN vs explicit)", notes, tag))


if __name__ == "__main__":
    main()
