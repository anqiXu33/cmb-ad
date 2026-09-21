#!/usr/bin/env python3
"""
Post-run repair for the residual thinking-leak class in Stage-2 outputs.

A small fraction of CMB-AD / Baseline-1 ADs fall into a third failure mode
the inline fallback pass does not catch: the model exhausts
``max_new_tokens=8192`` entirely on the thinking section, never writes
``</think>`` and never reaches the JSON output.  The whole reasoning text
(typically English meta-commentary like *"The input describes…"*) ends up
in ``text_gen``.

This script:

1. Detects such leaked ADs by regex over a fixed pattern set.
2. Re-generates each flagged anno_idx with ``enable_thinking=False`` and a
   tighter ``max_new_tokens`` budget on a minimal "summarize this Stage-1
   description into one sentence" prompt — no CMB scaffolding, no
   preceding-AD context.  This is a best-effort recovery: the resulting AD
   is locally valid German / Italian, but does not benefit from the
   chunk-level CMB state of its neighbours.
3. Writes ``<stage2.csv>.repaired`` alongside each input — never overwrites
   the original.

Usage
-----
Detect-only across the 24 main-run CSVs (no GPU, login node)::

    python scripts/tools/repair_leaked_ads.py --batch_thesis --dry_run

Repair a single CSV::

    python scripts/tools/repair_leaked_ads.py \\
        --stage2_csv  results/stage2/cmb_ad/Moskau_einfach/stage2.csv \\
        --pred_csv    /shares/.../stage1/Moskau_einfach/swissAD_ads/stage1_qwen2vl.csv \\
        --language    German

Repair all 24 main-run CSVs in one job (loads the 27B model once)::

    python scripts/tools/repair_leaked_ads.py --batch_thesis

Conventions
-----------
- Detection patterns live in ``LEAK_PATTERNS`` and are intentionally
  English-only — Stage-2 production output is short German / Italian, so
  any English keyword is a strong leak signal.
- ``--batch_thesis`` resolves Stage-1 paths via ``DATA_ROOT`` (env var, or
  the cluster default).  Per-film language assignments are hard-coded in
  ``THESIS_FILMS`` (4 German + 4 Italian).
"""
import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

# Strong signals that text_gen is leaked thinking, not a German/Italian AD.
# Production AD output is short and language-pure; English keywords or
# meta-commentary phrases are unambiguous markers.
LEAK_PATTERNS = [
    "The input",
    "Character Tracking",
    "Status indicates",
    "summarized_AD",
    "I should",
    "I need to",
    "I'll",
    "I will",
    "Let me",
    "the description",
    "The character",
    "PRIORITY",
    "flag indicates",
    "<think>",
    "tracking status",
]
LEAK_RE = re.compile(
    "|".join(re.escape(p) for p in LEAK_PATTERNS), re.IGNORECASE
)


def detect_leaks(df):
    """Return rows whose text_gen contains any LEAK_PATTERN."""
    if "text_gen" not in df.columns or len(df) == 0:
        return df.iloc[0:0]
    mask = df["text_gen"].astype(str).apply(lambda s: bool(LEAK_RE.search(s)))
    return df[mask].copy()


# ---------------------------------------------------------------------------
# Thesis-batch enumeration
# ---------------------------------------------------------------------------

THESIS_FILMS = {
    # German (4)
    "8_Tage_im_August":          "German",
    "Baghdad_in_my_Shadow":      "German",
    "Moskau_einfach":            "German",
    "Trommelwirbel":             "German",
    # Italian (4)
    "La_Tentazione_Di_Esistere": "Italian",
    "Quanto_basta":              "Italian",
    "Riders_of_justice":         "Italian",
    "Stuerm":                    "Italian",
}

DATA_ROOT = Path(os.environ.get(
    "DATA_ROOT",
    "/shares/iict-sp4.ebling.cl.uzh/data/anxu_MA_data",
))


def thesis_input_pairs():
    """Yield (stage2_csv, pred_csv, language, label) for all 24 main-run outputs."""
    for cond in ("cmb_ad", "baseline1_no_cmb", "baseline2_always_id"):
        for movie, lang in THESIS_FILMS.items():
            stage2 = Path("results/stage2") / cond / movie / "stage2.csv"
            pred = (
                DATA_ROOT / "stage1" / movie / "swissAD_ads" / "stage1_qwen2vl.csv"
            )
            yield stage2, pred, lang, f"{cond}/{movie}"


# ---------------------------------------------------------------------------
# Repair prompt + output parsing
# ---------------------------------------------------------------------------

# Minimal system prompt — explicitly NOT asking for any reasoning.  Paired
# with enable_thinking=False in the chat template.
SYS_PROMPT_REPAIR = (
    "You are an intelligent chatbot designed for summarizing movie audio descriptions. "
    "Convert the predicted description into ONE succinct sentence in {language}. "
    'Output ONLY the JSON: {{"summarized_AD": "..."}}. '
    "Do not include any explanation, reasoning, or text outside the JSON."
)


def build_repair_prompts(tokenizer, leaked_df, pred_df, language):
    """Build clean repair prompts: Stage-1 description in, AD out, no thinking."""
    pred_by_idx = pred_df.set_index("anno_idx")
    formatted = []
    for _, row in leaked_df.iterrows():
        anno_idx = int(row["anno_idx"])
        if anno_idx not in pred_by_idx.index:
            formatted.append(None)
            continue
        text_pred = str(pred_by_idx.loc[anno_idx]["text_gen"])
        duration = round(row["end"] - row["start"], 2)
        word_limit = max(int(2.8 * duration + 3) + 1, 8)
        user = (
            f"Convert this movie clip description into ONE concise audio "
            f"description sentence in {language} (~{word_limit} words).\n\n"
            f'Description: "{text_pred}"\n\n'
            'Output ONLY the JSON object: {"summarized_AD": "..."}'
        )
        msgs = [
            {"role": "system", "content": SYS_PROMPT_REPAIR.format(language=language)},
            {"role": "user",   "content": user},
        ]
        prompt = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        formatted.append(prompt)
    return formatted


JSON_RE = re.compile(r'"summarized_AD"\s*:\s*"([^"]*)"', re.DOTALL)


def parse_repair_output(raw):
    """Extract the AD text from the model's JSON output."""
    if not raw:
        return ""
    m = JSON_RE.search(raw)
    if m:
        return m.group(1).strip()
    # Last-ditch fallback: strip JSON wrappers
    cleaned = (
        raw.strip()
           .removeprefix('{"summarized_AD":')
           .removeprefix('{"summarized_AD" :')
           .strip()
           .removesuffix("}")
           .strip()
           .strip('"')
           .strip()
    )
    return cleaned


# ---------------------------------------------------------------------------
# Repair pipeline
# ---------------------------------------------------------------------------


def repair_one(stage2_csv, pred_csv, language, pipeline,
               batch_size=8, max_new_tokens=512):
    """Detect and repair leaks in a single stage2.csv.

    Returns a result dict; on success writes ``<stage2_csv>.repaired``."""
    if not stage2_csv.exists():
        return {"label": str(stage2_csv), "error": "stage2.csv not found"}

    df = pd.read_csv(stage2_csv)
    leaked = detect_leaks(df)
    if len(leaked) == 0:
        return {"stage2": str(stage2_csv), "n_total": len(df),
                "n_leaks": 0, "repaired": 0, "output": None}

    if not pred_csv.exists():
        return {"stage2": str(stage2_csv), "n_total": len(df),
                "n_leaks": len(leaked), "repaired": 0,
                "error": f"pred_csv missing: {pred_csv}"}
    pred_df = pd.read_csv(pred_csv)
    if "anno_idx" not in pred_df.columns or "text_gen" not in pred_df.columns:
        return {"stage2": str(stage2_csv), "n_total": len(df),
                "n_leaks": len(leaked), "repaired": 0,
                "error": "pred_csv lacks anno_idx or text_gen column"}

    # Build prompts (skip leaked anno_idx that have no Stage-1 entry)
    prompts = build_repair_prompts(pipeline.tokenizer, leaked, pred_df, language)
    valid_local = [i for i, p in enumerate(prompts) if p is not None]
    valid_prompts = [prompts[i] for i in valid_local]
    n_skipped = len(leaked) - len(valid_local)

    if not valid_prompts:
        return {"stage2": str(stage2_csv), "n_total": len(df),
                "n_leaks": len(leaked), "repaired": 0,
                "error": f"no valid Stage-1 mapping for any leaked anno_idx "
                         f"({n_skipped} skipped)"}

    # Batched inference, no thinking, deterministic
    repaired_texts = []
    for start in range(0, len(valid_prompts), batch_size):
        batch = valid_prompts[start:start + batch_size]
        out = pipeline(
            batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_full_text=False,
            num_beams=1,
        )
        for o in out:
            text = o[0]["generated_text"] if isinstance(o, list) else o["generated_text"]
            repaired_texts.append(parse_repair_output(text))

    # Write back: replace text_gen at each repaired anno_idx
    out_df = df.copy()
    n_actually_repaired = 0
    for local_i, new_text in zip(valid_local, repaired_texts):
        if not new_text:
            continue
        anno_idx = int(leaked.iloc[local_i]["anno_idx"])
        out_df.loc[out_df["anno_idx"] == anno_idx, "text_gen"] = new_text
        n_actually_repaired += 1

    repaired_path = stage2_csv.with_suffix(".csv.repaired")
    out_df.to_csv(repaired_path, index=False)
    return {
        "stage2": str(stage2_csv),
        "n_total": len(df),
        "n_leaks": len(leaked),
        "repaired": n_actually_repaired,
        "skipped_no_stage1": n_skipped,
        "output": str(repaired_path),
    }


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


def dry_run(targets):
    print(f"Dry run — detecting leaks across {len(targets)} CSVs:\n")
    total_total = 0
    total_leaks = 0
    missing = 0
    for stage2_csv, _, _, label in targets:
        if not stage2_csv.exists():
            print(f"  [skip] {label}: {stage2_csv} not found")
            missing += 1
            continue
        df = pd.read_csv(stage2_csv)
        leaked = detect_leaks(df)
        total_total += len(df)
        total_leaks += len(leaked)
        pct = 100.0 * len(leaked) / max(len(df), 1)
        flag = "OK  " if len(leaked) == 0 else "LEAK"
        print(f"  [{flag}] {label:50s} {len(leaked):3d}/{len(df):4d} "
              f"({pct:.1f}%)")
        for _, r in leaked.head(2).iterrows():
            print(f"         [{int(r['anno_idx'])}] "
                  f"{str(r['text_gen'])[:120]}")

    print()
    if total_total > 0:
        pct = 100.0 * total_leaks / total_total
        print(f"Total: {total_leaks}/{total_total} ADs leaked ({pct:.2f}%)")
    if missing:
        print(f"Missing: {missing} CSV(s) — likely jobs not yet completed.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def initialise_pipeline(model_id="Qwen/Qwen3.5-27B"):
    """Load Qwen3.5-27B for repair generation (thinking disabled at call time)."""
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto",
    )
    pipe = transformers.pipeline(
        "text-generation", model=model, tokenizer=tokenizer, device_map="auto",
    )
    pipe.tokenizer.padding_side = "left"
    if pipe.tokenizer.pad_token_id is None:
        pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
    return pipe


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--batch_thesis", action="store_true",
        help="Operate on all 24 main-run CSVs (3 conditions x 8 films).",
    )
    g.add_argument("--stage2_csv", type=str, default=None,
                   help="Path to a single stage2.csv to repair.")
    ap.add_argument("--pred_csv", type=str, default=None,
                    help="Stage-1 CSV (required with --stage2_csv).")
    ap.add_argument("--language", type=str, default=None,
                    help="'German' or 'Italian' (required with --stage2_csv).")
    ap.add_argument("--dry_run", action="store_true",
                    help="Detect-only; print leaks, don't load the model.")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--model_id", type=str, default="Qwen/Qwen3.5-27B")
    args = ap.parse_args()

    if args.batch_thesis:
        targets = list(thesis_input_pairs())
    else:
        if not (args.pred_csv and args.language):
            ap.error("--stage2_csv requires --pred_csv and --language")
        targets = [(
            Path(args.stage2_csv),
            Path(args.pred_csv),
            args.language,
            Path(args.stage2_csv).parent.name,
        )]

    if args.dry_run:
        dry_run(targets)
        return

    pipeline = initialise_pipeline(args.model_id)

    summary = []
    for stage2_csv, pred_csv, lang, label in targets:
        print(f"\n=== {label} ===")
        result = repair_one(
            stage2_csv, pred_csv, lang, pipeline,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
        )
        summary.append({"label": label, **result})
        n_leaks = result.get("n_leaks", 0)
        n_total = result.get("n_total", 0)
        repaired = result.get("repaired", 0)
        if "error" in result:
            print(f"  [ERROR] {result['error']}")
        else:
            print(f"  {n_leaks}/{n_total} leaks → {repaired} repaired"
                  + (f" -> {result['output']}" if result.get("output") else ""))

    # Final summary line
    print("\n=== Summary ===")
    total_leaks = sum(r.get("n_leaks", 0) for r in summary)
    total_repaired = sum(r.get("repaired", 0) for r in summary)
    total_total = sum(r.get("n_total", 0) for r in summary)
    print(f"Across {len(summary)} CSVs: {total_leaks} leaks / {total_total} ADs"
          f" → {total_repaired} repaired")
    errored = [s["label"] for s in summary if "error" in s]
    if errored:
        print(f"Errored: {len(errored)} ({', '.join(errored[:3])}{'…' if len(errored)>3 else ''})")


if __name__ == "__main__":
    main()
