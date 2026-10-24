"""Zero-shot LLM baseline: ask an instruction-tuned model NAME or PRONOUN.

No training. For each sampled row the model reads the previous k AD lines
(names visible) and the current line with the reference slot shown as [___],
and is asked how a professional describer would refer to the character there.
The answer is read from the next-token logits of "NAME" vs "PRONOUN", so there
is no free-form generation and no parsing.

Two decision rules are reported:
    zeroshot@0.5        P(PRONOUN) > 0.5
    zeroshot@prior      threshold set so the predicted pronoun rate equals the
                        pronoun rate of all OTHER rows (no labels of the scored
                        rows are used)

Kaggle (GPU T4, Internet on):
    python -m followup.llm_zeroshot --model Qwen/Qwen2.5-7B-Instruct --load_4bit --n 3000 --k 10
Offline smoke test:
    python -m followup.llm_zeroshot --dry_run --n 40
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from followup.llm_common import baseline_preds, load_rows, report, tiny_tokenizer

PROMPT = """Below are consecutive audio description (AD) lines from a film or TV episode, written for blind and partially sighted viewers.

Previous lines:
{context}

Current line (one reference to {name} is replaced by [___]):
{current}

How should [___] refer to {name}: by NAME or by PRONOUN (he, she, him, her, his)? Choose what a professional audio describer would write, so that a listener who cannot see the screen always knows who is meant.
Answer with one word: NAME or PRONOUN."""


def make_prompt(r, name_field: str = "char_role") -> str:
    name = str(getattr(r, name_field)) or "the character"
    ctx = r.context if r.context != "NOCONTEXT" else "(none, this is the first line)"
    return PROMPT.format(context=ctx, name=name, current=r.current.replace("TARGET", "[___]"))


def load_model(args, prompts):
    if args.dry_run:
        from transformers import Qwen2Config, Qwen2ForCausalLM
        tok = tiny_tokenizer(prompts)
        cfg = Qwen2Config(vocab_size=max(tok.get_vocab().values()) + 1, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048,
                          pad_token_id=tok.pad_token_id)
        return Qwen2ForCausalLM(cfg).eval(), tok, False
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = {"torch_dtype": torch.float16, "device_map": "auto"}
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.float16)
    return AutoModelForCausalLM.from_pretrained(args.model, **kw).eval(), tok, True


def answer_ids(tok):
    a = tok.encode("NAME", add_special_tokens=False)
    b = tok.encode("PRONOUN", add_special_tokens=False)
    assert a and b and a[0] != b[0], f"answer tokens collide: {a} {b}"
    return a[0], b[0]


@torch.no_grad()
def score(model, tok, prompts, chat: bool, bs: int) -> np.ndarray:
    id_name, id_prn = answer_ids(tok)
    dev = next(model.parameters()).device
    out, t0 = [], time.time()
    for i in range(0, len(prompts), bs):
        batch = prompts[i:i + bs]
        if chat:
            batch = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                             add_generation_prompt=True) for p in batch]
        enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=not chat).to(dev)
        enc.pop("token_type_ids", None)
        logits = model(**enc).logits[:, -1, :].float()      # left padding: last position is real
        two = torch.stack([logits[:, id_name], logits[:, id_prn]], dim=1)
        out.append(torch.softmax(two, dim=1)[:, 1].cpu().numpy())
        if (i // bs) % 25 == 0:
            print(f"  {i + len(batch)}/{len(prompts)}  {time.time() - t0:.0f}s", flush=True)
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--load_4bit", action="store_true")
    ap.add_argument("--k", type=int, default=10, help="previous AD lines shown")
    ap.add_argument("--n", type=int, default=3000, help="rows scored (random sample, seed 0)")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()

    df = load_rows(k=a.k, mask_name=False)
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(len(df), size=min(a.n, len(df)), replace=False))
    mask = np.zeros(len(df), bool)
    mask[idx] = True
    prompts = [make_prompt(r) for r in df.iloc[idx].itertuples(index=False)]
    print(f"{len(df)} rows, scoring {len(idx)}; example prompt:\n{prompts[0]}\n", flush=True)

    model, tok, chat = load_model(a, prompts)
    p = np.full(len(df), np.nan)
    p[idx] = score(model, tok, prompts, chat, a.bs)
    tag = "llm_zeroshot_" + ("dryrun" if a.dry_run else a.model.split("/")[-1])
    np.save(f"results/followup/{tag}_probs.npy", p)

    prior = float(df.y_bin.to_numpy()[~mask].mean())
    thr = float(np.quantile(p[idx], 1 - prior))
    preds = baseline_preds(df)
    preds["zeroshot@0.5"] = (np.nan_to_num(p) > 0.5).astype(int)
    preds["zeroshot@prior"] = (np.nan_to_num(p) > thr).astype(int)
    notes = [f"model: {'tiny random (dry run)' if a.dry_run else a.model}{' 4-bit' if a.load_4bit else ''}; "
             f"{a.k} previous AD lines with names visible; {len(idx)} random rows of {len(df)}.",
             f"zeroshot@prior threshold = {thr:.3f} (pronoun rate of the unscored rows = {100 * prior:.1f}%).",
             "Reference models are 5-fold out-of-fold predictions on the same rows."]
    if a.dry_run:
        notes.append("DRY RUN: numbers are meaningless; this only checks the pipeline.")
    print(report(df, mask, preds, "Zero-shot LLM vs rule vs learned models (PRONOUN vs explicit)", notes, tag))


if __name__ == "__main__":
    main()
