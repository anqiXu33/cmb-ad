"""Smoke-test Qwen3.5-27B chat template behavior.

Run on cluster BEFORE submitting any Stage-2 jobs:

    python scripts/tools/smoke_test_chat_template.py

What it verifies (tokenizer-only, no GPU required by default):
  1. enable_thinking=True ends the prompt at <|im_start|>assistant\\n
     (no <think> token present in the prompt itself).
  2. enable_thinking=False prepends an empty <think>\\n\\n</think>\\n\\n
     block at the start of the assistant turn.
  3. Our manual `prompt + "<think>\\n\\n</think>\\n\\n"` append (used in the
     fallback path of swissAD_qwen3.py and stage2_qwen3_always_id.py)
     produces a string IDENTICAL to enable_thinking=False rendering.

Add --gpu to additionally run one tiny generation for each setting and check
that the model actually skips reasoning when thinking is disabled. Slow.
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_id", default="Qwen/Qwen3.5-27B",
                    help="HF model id (default: Qwen/Qwen3.5-27B)")
    ap.add_argument("--gpu", action="store_true",
                    help="Also load the model and generate one short output per setting "
                         "(slow; needs the model already in HF cache)")
    args = ap.parse_args()

    print(f"Loading tokenizer for {args.model_id} ...")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_id)
    print(f"  pad_token_id={tok.pad_token_id}  eos_token_id={tok.eos_token_id}")

    msgs = [
        {"role": "system", "content": "You summarize movie audio descriptions."},
        {"role": "user", "content": "Description: 'A man enters a kitchen.'"},
    ]

    p_thinking = tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    p_no_thinking = tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )

    print()
    print("================== enable_thinking=True ==================")
    print("Last 200 chars (repr):")
    print(repr(p_thinking[-200:]))

    print()
    print("================== enable_thinking=False =================")
    print("Last 200 chars (repr):")
    print(repr(p_no_thinking[-200:]))

    # ----- our manual fallback append -----
    empty_think = "<think>\n\n</think>\n\n"
    p_manual = p_thinking.rstrip() + "\n" + empty_think
    print()
    print("=========== thinking=True + manual append ===============")
    print("Last 200 chars (repr):")
    print(repr(p_manual[-200:]))

    # ----- equivalence check -----
    print()
    print("================== EQUIVALENCE CHECK =====================")
    if p_manual == p_no_thinking:
        print("  PASS: manual append == enable_thinking=False rendering")
    else:
        print("  WARN: manual append differs from enable_thinking=False")
        # Show diff at the suffix where they diverge
        for i in range(min(len(p_manual), len(p_no_thinking))):
            if p_manual[-(i+1)] != p_no_thinking[-(i+1)]:
                print(f"    first divergence from end: char {i}")
                print(f"    manual ends:  {repr(p_manual[-(i+50):])}")
                print(f"    no_think ends:{repr(p_no_thinking[-(i+50):])}")
                break
        # Show length difference
        print(f"    len(manual)={len(p_manual)}  len(no_thinking)={len(p_no_thinking)}")

    # ----- presence checks -----
    print()
    has_think_in_thinking = "<think>" in p_thinking
    has_think_in_no_thinking = "<think>" in p_no_thinking
    has_close_in_no_thinking = "</think>" in p_no_thinking
    print("Token presence:")
    print(f"  enable_thinking=True  has '<think>' in prompt: {has_think_in_thinking}")
    print(f"  enable_thinking=False has '<think>' in prompt: {has_think_in_no_thinking}")
    print(f"  enable_thinking=False has '</think>' in prompt: {has_close_in_no_thinking}")
    if not has_think_in_thinking and has_think_in_no_thinking and has_close_in_no_thinking:
        print("  PASS: token layout matches the expected Qwen3 pattern")
    else:
        print("  WARN: unexpected token layout — fallback patch assumptions may not hold")

    if not args.gpu:
        print()
        print("Skipped GPU generation test (use --gpu to enable).")
        return

    # ----- optional GPU generation test -----
    print()
    print("================== GPU GENERATION TEST ===================")
    print("Loading model ... (this can take a minute)")
    import torch
    import transformers
    pipe = transformers.pipeline(
        "text-generation",
        model=args.model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    pipe.tokenizer.padding_side = "left"
    if pipe.tokenizer.pad_token_id is None:
        pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id

    for label, prompt in [
        ("enable_thinking=True (expect <think>...</think> in output)", p_thinking),
        ("enable_thinking=False (expect direct answer, no <think>)", p_no_thinking),
        ("manual fallback append (expect direct answer, no <think>)", p_manual),
    ]:
        print()
        print(f"--- {label} ---")
        out = pipe(
            [prompt],
            max_new_tokens=256,
            do_sample=False,
            return_full_text=False,
            pad_token_id=pipe.tokenizer.eos_token_id,
        )
        gen = out[0][0]["generated_text"]
        print(f"  generated (first 200 chars): {repr(gen[:200])}")
        if "<think>" in gen and "</think>" not in gen:
            print("  WARN: <think> opened but never closed — fallback bug pattern")
        elif "<think>" in gen:
            print("  Note: model emitted its own <think>...</think> block")
        else:
            print("  OK: model skipped thinking entirely")


if __name__ == "__main__":
    main()
