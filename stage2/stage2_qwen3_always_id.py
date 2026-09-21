import os
import re
import json
import torch
import random
import argparse
import numpy as np
import transformers
import pandas as pd
from tqdm import tqdm
from promptloader import get_user_prompt


# ---------------------------------------------------------------------------
# Baseline 2: Always-ID (absolutist-rule ablation)
# ---------------------------------------------------------------------------
# This script isolates the rule-philosophy axis vs B1 (--no_cmb):
#
#   • NO CharacterMemoryBank — no per-character flags, no recent_history,
#     no PRIORITY rule, no Status block.
#   • preceding_ad IS chained from the prior chunk's real Stage-2 output —
#     identical cross-shot context as B1, so the only remaining difference
#     vs B1 is the prompt's identification policy.
#   • The system prompt is augmented with a language-specific CRITICAL RULE
#     that forces the model to ALWAYS use explicit character names or
#     descriptions and NEVER use pronouns or pro-drop.
#
# Architecture mirrors swissAD_qwen3.chunked_inference:
#
#   Per chunk of batch_size ADs:
#     1. Assemble preceding_ad list — AD 0 of chunk gets real Stage-2 output
#        from previous chunk; AD 1+ gets Stage-1 text_pred proxy of the
#        just-previous intra-chunk AD (same proxy strategy as CMB-AD/B1).
#     2. Build formatted prompts via the base promptloader (with always-ID
#        CRITICAL RULE in the system prompt).
#     3. Batched GPU inference + fallback (truncated </think> or generic
#        placeholders trigger a thinking-disabled retry with anti-generic
#        injection — same logic as CMB-AD's _infer_single_batch).
#     4. Update preceding_ad from the chunk's last non-empty output.
#
# Output is saved as `stage2.csv` under --save_dir.  A
# `stage2.csv.partial` checkpoint is written every 10 chunks and the
# script auto-resumes from it on restart (SLURM TIMEOUT recovery).
# The checkpoint is deleted on successful completion.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Language-specific CRITICAL RULE (appended to base system prompt)
# ---------------------------------------------------------------------------

_ALWAYS_ID_RULES: dict[str, str] = {
    "Italian": (
        "\nCRITICAL RULE: 1. You MUST explicitly use the character's exact name "
        "provided in the context for EVERY action. "
        "2. NEVER use personal pronouns (like 'lui', 'lei'). "
        "3. NEVER omit the subject (ABSOLUTELY NO pro-drop). "
        "Example: If the character is Elvezio, write 'Elvezio si siede', "
        "NEVER 'Lui si siede' and NEVER 'Si siede'."
    ),
    "German": (
        "\nCRITICAL RULE: 1. You MUST explicitly use the character's exact name "
        "provided in the context for EVERY action. "
        "2. NEVER use personal pronouns (like 'er', 'sie', 'es'). "
        "Example: If the character is Anna, write 'Anna setzt sich', "
        "NEVER 'Sie setzt sich'."
    ),
}

_BASE_SYS_PROMPT = (
    "You are an intelligent chatbot designed for summarizing movie audio descriptions."
    "Here's how you accomplish the task: convert the predicted descriptions into one sentence. "
    "First, you will reflect in a <think>...</think> section. Then, you will provide the final "
    "result in JSON format: {{'summarized_AD': '...' }}\n"
    "Your thinking must be brief, limit the chain to 4-5 sentences and you MUST end the "
    "thinking with a </think>.\n"
    "After </think>, directly output the final sentence in the required JSON format. "
    "Do not include explanations or anything outside the JSON after </think>."
)


def get_sys_prompt(language: str) -> str:
    """Return the base system prompt with the language-specific always-ID rule appended."""
    return _BASE_SYS_PROMPT + _ALWAYS_ID_RULES.get(language, "")


# ---------------------------------------------------------------------------
# Model initialisation
# ---------------------------------------------------------------------------

def initialise_model():
    model_id = "Qwen/Qwen3.5-27B"
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    # Left-padding is required for correct batched causal-LM inference.
    pipeline.tokenizer.padding_side = "left"
    if pipeline.tokenizer.pad_token_id is None:
        pipeline.tokenizer.pad_token_id = pipeline.tokenizer.eos_token_id
    return pipeline


# ---------------------------------------------------------------------------
# Per-chunk prompt builder (CPU, no LLM)
# ---------------------------------------------------------------------------

def _build_chunk_prompts(
    pipeline, chunk_df, chunk_preceding_ads, args,
    all_gts_wo_char, all_gts_num_words,
):
    """
    Build user prompts for one chunk of pred_df rows and apply the Qwen3
    chat template (thinking mode).

    `chunk_preceding_ads` is a list of strings parallel to chunk_df rows,
    giving the preceding_ad for each AD: AD 0 of the chunk gets the real
    Stage-2 output from the previous chunk; AD 1+ gets the Stage-1
    text_pred proxy of the just-previous intra-chunk AD (matches the
    swissAD_qwen3.chunked_inference proxy strategy).
    """
    sys_prompt = get_sys_prompt(args.language)
    formatted_prompts = []

    for (_, row), preceding_ad in zip(chunk_df.iterrows(), chunk_preceding_ads):
        duration        = round(row["end"] - row["start"], 2)
        rough_num_words = args.LR_intercept + args.LR_slope * duration

        # Few-shot examples: sample GT ADs of matching length (empty if pool absent)
        if all_gts_wo_char:
            candid = [i for i, s in enumerate(all_gts_num_words)
                      if rough_num_words - 1 <= s <= rough_num_words + 1]
            if len(candid) < args.num_examples:
                candid = list(range(len(all_gts_num_words)))
            examples = [all_gts_wo_char[i]
                        for i in random.choices(candid, k=args.num_examples)]
        else:
            examples = []

        user_prompt = get_user_prompt(
            mode=args.mode,
            prompt_idx=args.prompt_idx,
            verb_list=None,
            text_pred=str(row["text_gen"]),
            word_limit=max(int(rough_num_words) + 1, 8),
            examples=examples,
            language=args.language,
            preceding_ad=preceding_ad,
        )

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user",   "content": user_prompt},
        ]
        formatted = pipeline.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        formatted_prompts.append(formatted)

    return formatted_prompts


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def _extract_ad(generated_text, mode):
    """
    Parse a single model output (generated tokens only, no input echo).
    Strips the <think>...</think> block if present, then extracts the AD.
    Returns an empty string on any parse failure.
    """
    # If thinking opened but never closed (truncated by max_new_tokens), the
    # split-on-</think> fallback would dump the raw reasoning into the CSV.
    if "<think>" in generated_text and "</think>" not in generated_text:
        return ""
    answer = generated_text.split("</think>")[-1].strip()
    try:
        if mode == "single":
            json_match = re.search(r'\{[^{}]*"summarized_AD"[^{}]*\}', answer)
            if json_match:
                try:
                    result = json.loads(json_match.group()).get("summarized_AD", "")
                except Exception:
                    result = ""
            else:
                try:
                    result = json.loads(answer).get("summarized_AD", "")
                except Exception:
                    result = (
                        answer.replace('{"summarized_AD": "', "")
                        .replace('"}', "")
                        .strip()
                    )
            if result and result[-1] != ".":
                result += "."
            return result
        else:  # assistant mode — 5 candidates
            ad_list = []
            for idx in range(1, 6):
                tmp = answer.split(f'"summarized_AD_{idx}":')[-1]
                tmp = tmp.split(",\n")[0].split("\n")[0]
                tmp = (tmp.replace('"', "")
                          .replace("{", "")
                          .replace("}", "")
                          .strip())
                if tmp and tmp[-1] != ".":
                    tmp += "."
                ad_list.append(tmp)
            return str(ad_list)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Per-chunk GPU inference (with fallback) — used by chunked_inference()
# ---------------------------------------------------------------------------

def _infer_single_batch(pipeline, batch_prompts, mode, terminators):
    """
    Run a single GPU batch and return a list of AD strings (one per prompt).

    Fallback: items where the thinking block was truncated (no </think>) or
    where the output contains generic placeholder terms are re-run with
    thinking disabled and an anti-generic instruction injected.

    Mirrors swissAD_qwen3._infer_single_batch byte-for-byte (same generic
    pattern, same generation params, same fallback prompt construction).
    """
    generic_pattern = (
        r'\b(?:(?:ein[enrms]?)\s+(?:mann|frau|andere[rnms]?)'
        r'|(?:un|una|un\')\s*(?:uomo|donna|ragazzo|ragazza|altro|altra))\b'
    )

    # ---- Main pass -------------------------------------------------------
    batch_outputs = pipeline(
        batch_prompts,
        max_new_tokens=8192,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0,
        pad_token_id=pipeline.tokenizer.eos_token_id,
        return_full_text=False,
    )

    ads = [""] * len(batch_prompts)
    need_retry = []

    for i, item_out in enumerate(batch_outputs):
        generated = item_out[0]["generated_text"]
        if "</think>" not in generated:
            need_retry.append(i)
        else:
            ad_text = _extract_ad(generated, mode)
            if re.search(generic_pattern, ad_text, re.IGNORECASE):
                need_retry.append(i)
            else:
                ads[i] = ad_text

    # ---- Fallback pass (truncated / generic) -----------------------------
    if need_retry:
        # Qwen3 chat template with enable_thinking=True ends the prompt at
        # "<|im_start|>assistant\n" and lets the model emit its own <think>.
        # To disable thinking on the retry we pre-fill an empty think block at
        # the start of the assistant turn — mirroring what the template would
        # render for enable_thinking=False.
        empty_think = "<think>\n\n</think>\n\n"
        anti_generic_rule = (
            "\nCRITICAL RULE: DO NOT use generic terms like 'ein Mann', 'eine Frau', "
            "'un uomo', or 'una donna'. Rewrite using a vivid physical description "
            "from the Context!"
        )
        assistant_marker = "<|im_end|>\n<|im_start|>assistant"

        fallback_prompts = []
        for idx in need_retry:
            fp = batch_prompts[idx]
            # Inject the anti-generic rule at the end of the user turn.
            fp = fp.replace(
                assistant_marker,
                anti_generic_rule + assistant_marker,
            )
            # Pre-fill empty thinking so the model jumps straight to the JSON.
            if empty_think not in fp.rsplit(assistant_marker, 1)[-1]:
                fp = fp.rstrip() + "\n" + empty_think
            fallback_prompts.append(fp)

        fallback_outputs = pipeline(
            fallback_prompts,
            max_new_tokens=512,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            min_p=0,
            pad_token_id=pipeline.tokenizer.eos_token_id,
            return_full_text=False,
        )
        for local_idx, item_out in zip(need_retry, fallback_outputs):
            ads[local_idx] = _extract_ad(item_out[0]["generated_text"], mode)

    return ads


# ---------------------------------------------------------------------------
# Chunked inference — preceding_ad chained from prior chunk's real output
# ---------------------------------------------------------------------------

def chunked_inference(
    pipeline, pred_df, args,
    all_gts_wo_char, all_gts_num_words,
    save_path=None, checkpoint_every=10,
):
    """
    Process ADs in windows of args.batch_size, threading the real Stage-2
    preceding_ad output from each chunk into the next chunk's prompts.

    Within a chunk: AD 0 sees the real preceding_ad from the previous chunk;
    intra-chunk ADs (AD 1+) use Stage-1 text_pred as a proxy preceding_ad —
    same proxy strategy as swissAD_qwen3.chunked_inference (which CMB-AD
    later overwrites via sync_history; here there is no sync because B2 has
    no CharacterMemoryBank).

    Resume: if `save_path + '.partial'` exists, load completed ADs and find
    the last non-empty AD as the resume preceding_ad.  Inference resumes
    from the next batch boundary.
    """
    terminators = [pipeline.tokenizer.eos_token_id]
    endoftext_id = pipeline.tokenizer.convert_tokens_to_ids("<|endoftext|>")
    if endoftext_id is not None:
        terminators.append(endoftext_id)

    n_shots = len(pred_df)

    # ---- Resume from partial checkpoint if present ---------------------------
    all_ads = []
    preceding_ad = ""
    partial_path = save_path + ".partial" if save_path else None
    if partial_path is not None and os.path.exists(partial_path):
        try:
            partial_df = pd.read_csv(partial_path)
            partial_ads = partial_df["text_gen"].fillna("").astype(str).tolist()
            n_done = len(partial_ads)
            n_aligned = (n_done // args.batch_size) * args.batch_size
            if n_aligned == 0:
                print(f"  [resume] partial has {n_done} rows < batch_size; ignoring")
            else:
                if n_aligned < n_done:
                    print(f"  [resume] partial has {n_done} rows; "
                          f"truncating to {n_aligned} (batch-aligned)")
                    partial_ads = partial_ads[:n_aligned]
                # Find last non-empty AD as resume preceding_ad
                for ad in reversed(partial_ads):
                    if ad:
                        preceding_ad = ad
                        break
                all_ads = list(partial_ads)
                print(f"  [resume] loaded {len(all_ads)}/{n_shots} ADs from "
                      f"{partial_path}; resuming at chunk "
                      f"{len(all_ads) // args.batch_size}.")
        except Exception as e:
            print(f"  [WARN] failed to load partial {partial_path} ({e}); "
                  f"starting fresh")
            all_ads = []
            preceding_ad = ""

    batch_count = len(all_ads) // args.batch_size

    for chunk_start in tqdm(
        range(len(all_ads), n_shots, args.batch_size),
        desc="Chunked inference (preceding_ad chained)",
        initial=batch_count,
        total=(n_shots + args.batch_size - 1) // args.batch_size,
    ):
        chunk_df = pred_df.iloc[chunk_start: chunk_start + args.batch_size]

        # Step 1 — assemble preceding_ad list for this chunk.
        # AD 0 gets the real Stage-2 output from the previous chunk.
        # AD 1+ gets Stage-1 text_pred proxy of the just-previous intra-chunk AD.
        chunk_preceding_ads = []
        intra_preceding = preceding_ad
        for _, row in chunk_df.iterrows():
            chunk_preceding_ads.append(intra_preceding)
            intra_preceding = str(row["text_gen"])

        # Step 2 — build formatted prompts for this chunk.
        chunk_prompts = _build_chunk_prompts(
            pipeline, chunk_df, chunk_preceding_ads, args,
            all_gts_wo_char, all_gts_num_words,
        )

        # Step 3 — GPU inference for this chunk (with fallback).
        chunk_ads = _infer_single_batch(pipeline, chunk_prompts, args.mode, terminators)
        all_ads.extend(chunk_ads)

        # Step 4 — update preceding_ad from chunk's last non-empty output.
        for ad in reversed(chunk_ads):
            if ad:
                preceding_ad = ad
                break

        # Checkpoint
        batch_count += 1
        if save_path is not None and batch_count % checkpoint_every == 0:
            n_done = len(all_ads)
            partial_df = pd.DataFrame({
                "imdbid":   pred_df["imdbid"].iloc[:n_done].tolist(),
                "start":    pred_df["start"].iloc[:n_done].tolist(),
                "end":      pred_df["end"].iloc[:n_done].tolist(),
                "text_gt":  pred_df["text_gt"].iloc[:n_done].tolist(),
                "text_gen": all_ads,
                "anno_idx": pred_df["anno_idx"].iloc[:n_done].tolist(),
            })
            partial_df.to_csv(save_path + ".partial", index=False)
            print(f"  [ckpt] {n_done}/{n_shots} rows -> {save_path}.partial")

    return all_ads


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    pipeline = initialise_model()

    pred_df = pd.read_csv(args.pred_path)

    # --anno_path is accepted for CLI consistency but only pred_ids is pulled
    # in (if present).  thread/char tracking is intentionally NOT used —
    # Baseline 2 has no CharacterMemoryBank.
    if args.anno_path is not None:
        anno_df   = pd.read_csv(args.anno_path)
        keep_cols = ["anno_idx"] + [
            c for c in ("pred_ids",)
            if c in anno_df.columns and c not in pred_df.columns
        ]
        if len(keep_cols) > 1:
            anno_subset = anno_df[keep_cols].drop_duplicates(subset=["anno_idx"])
            pred_df = pd.merge(pred_df, anno_subset, on="anno_idx", how="left")
            print(f"  Merged pred_ids from --anno_path ({len(pred_df)} rows retained).")
        else:
            print("  [INFO] --anno_path provided; no new columns to merge.")

    if args.few_shot_samples_csv is not None:
        gt_df             = pd.read_csv(args.few_shot_samples_csv)
        all_gts_wo_char   = gt_df["text_gt_wo_char"].tolist()
        all_gts_num_words = [len(str(e).strip().split()) for e in all_gts_wo_char]
        print(f"  Loaded {len(all_gts_wo_char)} few-shot examples from {args.few_shot_samples_csv}")
    else:
        all_gts_wo_char   = []
        all_gts_num_words = []
        print("  [INFO] --few_shot_samples_csv not provided; running without few-shot examples.")

    print(f"Baseline 2 (Always-ID): NO CMB, preceding_ad chained — absolutist-rule ablation.")
    print(f"Shots: {len(pred_df)} | batch_size: {args.batch_size} | language: {args.language}")

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "stage2.csv")

    output_ads_list = chunked_inference(
        pipeline, pred_df, args, all_gts_wo_char, all_gts_num_words,
        save_path=save_path,
    )

    output_df = pd.DataFrame({
        "imdbid":   pred_df["imdbid"].tolist(),
        "start":    pred_df["start"].tolist(),
        "end":      pred_df["end"].tolist(),
        "text_gt":  pred_df["text_gt"].tolist(),
        "text_gen": output_ads_list,
        "anno_idx": pred_df["anno_idx"].tolist(),
    })
    output_df.to_csv(save_path, index=False)
    print(f"Saved {len(output_df)} rows -> {save_path}")

    partial_path = save_path + ".partial"
    if os.path.exists(partial_path):
        os.remove(partial_path)
        print(f"  Removed checkpoint: {partial_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Baseline 2 (Always-ID): Stage-2 AD generation with Qwen3.5-27B. "
            "No CharacterMemoryBank; preceding_ad IS chained (matches B1) so the "
            "only difference vs B1 is the always-ID CRITICAL RULE in the system "
            "prompt that forces explicit character names and forbids pronouns."
        )
    )
    parser.add_argument("--pred_path",            default=None,     type=str)
    parser.add_argument("--anno_path",            default=None,     type=str,
                        help="Annotation CSV (pred_ids merged if present; "
                             "thread/CMB NOT used — Baseline 2 has no bank).")
    parser.add_argument("--save_dir",             default=None,     type=str)
    parser.add_argument("--mode",                 default="single", type=str)
    parser.add_argument("--seed",                 default=42,       type=int)
    parser.add_argument("--prompt_idx",           default=0,        type=int)
    parser.add_argument("--num_examples",         default=10,       type=int)
    parser.add_argument("--LR_intercept",         default=0.0,      type=float)
    parser.add_argument("--LR_slope",             default=0.0,      type=float)
    parser.add_argument("--language",             default="German", type=str,
                        help="Target language: 'German' or 'Italian'. "
                             "'French' is accepted but has no always-ID rule.")
    parser.add_argument("--few_shot_samples_csv",                   type=str)
    parser.add_argument(
        "--batch_size", default=8, type=int,
        help="ADs per chunk (= GPU batch size). Matches CMB-AD/B1 default.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    main(args)
