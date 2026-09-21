import os
import ast
import json
import torch
import random
import argparse
import numpy as np
import re
import transformers
import pandas as pd
from tqdm import tqdm
from swiss_promptloader import get_user_prompt
from coherence_manager import CharacterMemoryBank
from char_utils import build_char_maps


# ---------------------------------------------------------------------------
# Batching strategy — chunked inference with real memory write-back
# ---------------------------------------------------------------------------
# CharacterMemoryBank context has two tiers:
#
#   Tier A — metadata-derived (no LLM needed):
#     • char_status  (from 'pred_ids' + Stage-1 text_gen)
#       Each character carries its own continuity flag:
#       [CONTINUING], [REINTRODUCE], [USE NAME — ...]
#
#   Tier B — output-derived (sequential dependency on the LLM):
#     • recent_history  (last 3 *generated* ADs)
#     • preceding_ad    (immediately preceding *generated* AD)
#
# Chunked inference eliminates whole-movie proxy contamination:
#   Step 1 (CPU): Build contexts. First AD of each chunk sees real history
#     from the previous chunk; intra-chunk ADs use text_gen proxy.
#   Step 2 (CPU): Apply Qwen3 chat template.
#   Step 3 (GPU): Batched inference with fallback pass.
#   Step 4 (CPU): sync_history() + sync_last_actions() replace proxy with real LLM outputs.
# ---------------------------------------------------------------------------

SYS_PROMPT = (
    "You are an intelligent chatbot designed for summarizing movie audio descriptions."
    "Here's how you accomplish the task: convert the predicted descriptions into one sentence. "
    "First, you will reflect in a <think>...</think> section. Then, you will provide the final "
    "result in JSON format: {{'summarized_AD': '...' }}\n"
    "Your thinking must be brief, limit the chain to 4-5 sentences and you MUST end the "
    "thinking with a </think>.\n"
    "After </think>, directly output the final sentence in the required JSON format. "
    "Do not include explanations or anything outside the JSON after </think>."
)


# ---------------------------------------------------------------------------
# Model initialisation
# ---------------------------------------------------------------------------


def initialise_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "Qwen/Qwen3.5-27B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto"
    )
    pipe = transformers.pipeline(
        "text-generation", model=model, tokenizer=tokenizer, device_map="auto"
    )
    # Left-padding is required for correct batched causal-LM inference:
    # all sequences in a batch share the same right-aligned generation start.
    pipe.tokenizer.padding_side = "left"
    if pipe.tokenizer.pad_token_id is None:
        pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
    return pipe


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------



def _parse_pred_ids(row, has_pred_ids_col):
    """Return the list of actor ID strings from the pred_ids column."""
    if not has_pred_ids_col:
        return []
    raw = row["pred_ids"]
    try:
        if isinstance(raw, float) and pd.isna(raw):
            return []
        if str(raw).strip() in ("", "{}"):
            return []
        parsed = ast.literal_eval(str(raw))
        if isinstance(parsed, dict):
            return list(parsed.keys())
    except (ValueError, SyntaxError):
        pass
    return []


def _union_pred_ids_dicts(series):
    """
    Union pred_ids dict-strings across rows (one per m-* shot of an AD).
    Returns a dict-string compatible with _parse_pred_ids: keys are the
    union of actor IDs, values are placeholder zeros.  Empty union → "{}".
    """
    keys = set()
    for raw in series:
        if isinstance(raw, float) and pd.isna(raw):
            continue
        s = str(raw).strip()
        if s in ("", "{}", "nan"):
            continue
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, dict):
                keys.update(parsed.keys())
        except (ValueError, SyntaxError):
            continue
    if not keys:
        return "{}"
    return "{" + ", ".join(f"{repr(k)}: 0" for k in sorted(keys, key=str)) + "}"


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def build_formatted_prompts(pipeline, pred_df, contexts, args, all_gts_wo_char, all_gts_num_words):
    """
    For each shot, build the user prompt from pre-computed context and apply
    the Qwen3 chat template (thinking mode).  Returns a list of formatted
    prompt strings ready for batched pipeline inference.
    """
    formatted_prompts = []

    for shot_idx, ((_, row), ctx) in enumerate(zip(pred_df.iterrows(), contexts)):
        duration = round(row["end"] - row["start"], 2)
        rough_num_words = args.LR_intercept + args.LR_slope * duration

        # Few-shot examples: sample GT ADs of matching length (empty if pool absent)
        if all_gts_wo_char:
            candid = [
                i for i, s in enumerate(all_gts_num_words)
                if rough_num_words - 1 <= s <= rough_num_words + 1
            ]
            if len(candid) < args.num_examples:
                candid = list(range(len(all_gts_num_words)))
            examples = [all_gts_wo_char[i] for i in random.choices(candid, k=args.num_examples)]
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
            preceding_ad=ctx["preceding_ad"],
            char_status=ctx["char_status"],
            recent_history=ctx["recent_history_str"],
            enable_cmb=not getattr(args, "no_cmb", False),
        )

        messages = [
            {"role": "system", "content": SYS_PROMPT},
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
    # Bail out early so the caller can decide what to do.
    if "<think>" in generated_text and "</think>" not in generated_text:
        return ""
    # Strip thinking block; works whether </think> is present or not
    answer = generated_text.split("</think>")[-1].strip()
    try:
        if mode == "single":
            # Try to find the last JSON object in the answer
            # (model sometimes emits reasoning text before the JSON)
            import re as _re

            json_match = _re.search(r'\{[^{}]*"summarized_AD"[^{}]*\}', answer)
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
                tmp = tmp.replace('"', "").replace("{", "").replace("}", "").strip()
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
# Chunked inference — real memory write-back
# ---------------------------------------------------------------------------


def _replay_completed_chunks(
    pred_df, partial_ads, has_pred_ids_col, memory_bank, batch_size,
):
    """Replay CMB state-modifying operations for shots 0..len(partial_ads)-1
    using the real LLM outputs stored in `partial_ads` (loaded from a previous
    job's `<save>.partial`). Mirrors the per-chunk operations of the main loop
    but skips GPU inference. Returns the last non-empty AD as `preceding_ad`."""
    n_done = len(partial_ads)
    for chunk_start in range(0, n_done, batch_size):
        chunk_end = min(chunk_start + batch_size, n_done)
        chunk_df = pred_df.iloc[chunk_start:chunk_end]
        chunk_char_map = {}
        for ad_idx, (_, row) in enumerate(chunk_df.iterrows(), start=chunk_start):
            text_pred = str(row["text_gen"])
            current_char_ids = _parse_pred_ids(row, has_pred_ids_col)
            chunk_char_map[ad_idx] = current_char_ids
            # get_context updates last_seen_ad_idx + consecutive_ads
            memory_bank.get_context(
                ad_idx=ad_idx, text_pred=text_pred, char_ids=current_char_ids
            )
            # add_ad pushes onto recent_history (proxy, will be overwritten by sync below)
            memory_bank.add_ad(text_pred)
        # End of chunk: overwrite proxy history + last_actions with real outputs
        memory_bank.sync_history(partial_ads[:chunk_end])
        chunk_real_ads = partial_ads[chunk_start:chunk_end]
        chunk_ad_texts = {chunk_start + i: ad for i, ad in enumerate(chunk_real_ads)}
        memory_bank.sync_last_actions(chunk_char_map, chunk_ad_texts)

    preceding_ad = ""
    for ad in reversed(partial_ads):
        if ad:
            preceding_ad = ad
            break
    return preceding_ad


def chunked_inference(
    pipeline,
    pred_df,
    has_pred_ids_col,
    args,
    all_gts_wo_char,
    all_gts_num_words,
    char_gender,
    char_names=None,
    char_descriptions=None,
    save_path=None,
    checkpoint_every=10,
):
    """
    Process ADs in windows of `args.batch_size`, writing real LLM outputs
    back to the CharacterMemoryBank after each chunk so that cross-chunk
    recent_history is authentic (not Stage-1 proxy text).

    Within a chunk the first AD sees fully real history; subsequent ADs
    see a mix of real (from the previous chunk) and proxy (from earlier in
    this chunk).  This is an acceptable trade-off that eliminates the
    whole-movie proxy contamination of the old 3-pass approach.

    Resume: if `save_path + '.partial'` exists, the previously-completed
    ADs are loaded and CMB state is rebuilt by replaying the per-chunk
    state-machine operations (no GPU). Inference resumes from the next
    batch boundary. Useful for SLURM TIMEOUT recovery.
    """
    terminators = [pipeline.tokenizer.eos_token_id]
    endoftext_id = pipeline.tokenizer.convert_tokens_to_ids("<|endoftext|>")
    if endoftext_id is not None:
        terminators.append(endoftext_id)

    memory_bank = CharacterMemoryBank(
        char_gender=char_gender or {},
        char_names=char_names or {},
        char_descriptions=char_descriptions or {},
    )
    n_shots = len(pred_df)

    # ---- Resume from partial checkpoint if present ----------------------------
    all_ads = []
    preceding_ad = ""
    partial_path = save_path + ".partial" if save_path else None
    if partial_path is not None and os.path.exists(partial_path):
        try:
            partial_df = pd.read_csv(partial_path)
            partial_ads = partial_df["text_gen"].fillna("").astype(str).tolist()
            n_done = len(partial_ads)
            # Truncate to nearest batch_size boundary (defensive — checkpoints are
            # always aligned, but a corrupted/partial write must not desync CMB).
            n_aligned = (n_done // args.batch_size) * args.batch_size
            if n_aligned == 0:
                print(f"  [resume] partial has {n_done} rows < batch_size; ignoring")
            else:
                if n_aligned < n_done:
                    print(f"  [resume] partial has {n_done} rows; "
                          f"truncating to {n_aligned} (batch-aligned)")
                    partial_ads = partial_ads[:n_aligned]
                preceding_ad = _replay_completed_chunks(
                    pred_df, partial_ads, has_pred_ids_col,
                    memory_bank, args.batch_size,
                )
                all_ads = list(partial_ads)
                print(f"  [resume] loaded {len(all_ads)}/{n_shots} ADs from "
                      f"{partial_path}; CMB state replayed; "
                      f"resuming at chunk {len(all_ads) // args.batch_size}.")
        except Exception as e:
            print(f"  [WARN] failed to load partial {partial_path} ({e}); "
                  f"starting fresh")
            all_ads = []
            preceding_ad = ""

    batch_count = len(all_ads) // args.batch_size

    for chunk_start in tqdm(
        range(len(all_ads), n_shots, args.batch_size),
        desc="Chunked inference (real memory)",
        initial=batch_count,
        total=(n_shots + args.batch_size - 1) // args.batch_size,
    ):
        chunk_df = pred_df.iloc[chunk_start: chunk_start + args.batch_size]

        # ----------------------------------------------------------------
        # Step 1 — Build CMB contexts for this chunk.
        # memory_bank.recent_history already holds real outputs from the
        # previous chunk, so AD 0 of this chunk gets authentic history.
        # Intra-chunk ADs use text_gen proxy for preceding_ad (AD 1+).
        # ----------------------------------------------------------------
        chunk_contexts = []
        chunk_char_map = {}          # {ad_idx: char_ids} — for sync_last_actions
        intra_preceding = preceding_ad   # real for AD 0 of chunk

        for ad_idx, (_, row) in enumerate(chunk_df.iterrows(), start=chunk_start):
            text_pred = str(row["text_gen"])
            current_char_ids = _parse_pred_ids(row, has_pred_ids_col)
            chunk_char_map[ad_idx] = current_char_ids

            char_status, recent_history_str = memory_bank.get_context(
                ad_idx=ad_idx,
                text_pred=text_pred,
                char_ids=current_char_ids,
            )

            chunk_contexts.append({
                "char_status":        char_status,
                "recent_history_str": recent_history_str,
                "preceding_ad":       intra_preceding,
            })

            # Proxy update — keeps char_status (last_action, consecutive_ads)
            # accurate for subsequent ADs within this chunk.
            memory_bank.add_ad(text_pred)
            intra_preceding = text_pred   # proxy preceding_ad for next intra-chunk AD

        # ----------------------------------------------------------------
        # Step 2 — Build formatted prompts for this chunk.
        # ----------------------------------------------------------------
        chunk_prompts = build_formatted_prompts(
            pipeline, chunk_df, chunk_contexts, args, all_gts_wo_char, all_gts_num_words
        )

        # ----------------------------------------------------------------
        # Step 3 — GPU inference for this chunk (with fallback).
        # ----------------------------------------------------------------
        chunk_ads = _infer_single_batch(pipeline, chunk_prompts, args.mode, terminators)
        all_ads.extend(chunk_ads)

        # ----------------------------------------------------------------
        # Step 4 — Write real outputs back to CMB.
        # Replace proxy recent_history and last_action with genuine LLM outputs.
        # ----------------------------------------------------------------
        memory_bank.sync_history(all_ads)
        chunk_ad_texts = {
            chunk_start + i: ad for i, ad in enumerate(chunk_ads)
        }
        memory_bank.sync_last_actions(chunk_char_map, chunk_ad_texts)
        for ad in reversed(chunk_ads):
            if ad:
                preceding_ad = ad
                break

        # ----------------------------------------------------------------
        # Checkpoint
        # ----------------------------------------------------------------
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

    # If --anno_path is provided:
    #   (1) FORCE-REWRITE pred_ids as the union over the AD's m-* shots.
    #       Stage-1 dataloader passes only the first m-* shot's pred_ids;
    #       ~9% of ADs across the 8 thesis films have an empty m-0 but a
    #       non-empty m-1/m-2.  Re-deriving the union here keeps CMB's
    #       input symmetric with the evaluation pipeline (which does the
    #       same union via _merge_pred_ids_col in evaluate_coherence.py).
    #   (2) Merge `thread` only when missing — never overwrite Stage-1.
    if args.anno_path is not None:
        anno_df = pd.read_csv(args.anno_path)

        if "shot_label" in anno_df.columns and "pred_ids" in anno_df.columns:
            m_rows = anno_df[
                anno_df["shot_label"].astype(str).str.startswith("m-", na=False)
            ]
            if len(m_rows) > 0:
                union_map = (
                    m_rows.groupby("anno_idx")["pred_ids"]
                          .apply(_union_pred_ids_dicts)
                )
                if "pred_ids" in pred_df.columns:
                    pre_nonempty = int(
                        (~pred_df["pred_ids"].astype(str).str.strip()
                            .isin(["", "{}", "nan"])).sum()
                    )
                else:
                    pre_nonempty = 0
                pred_df["pred_ids"] = (
                    pred_df["anno_idx"].map(union_map).fillna("{}")
                )
                post_nonempty = int(
                    (~pred_df["pred_ids"].astype(str).str.strip()
                        .isin(["", "{}", "nan"])).sum()
                )
                print(
                    f"  Rewrote pred_ids with m-* union from --anno_path: "
                    f"{post_nonempty}/{len(pred_df)} ADs non-empty "
                    f"(was {pre_nonempty} pre-rewrite, "
                    f"+{post_nonempty - pre_nonempty} recovered)."
                )
            else:
                print("  [WARN] --anno_path has no m-* shot rows; "
                      "skipping pred_ids rewrite.")
        else:
            print("  [WARN] --anno_path lacks 'shot_label' or 'pred_ids' "
                  "columns; skipping pred_ids rewrite.")

        if "thread" in anno_df.columns and "thread" not in pred_df.columns:
            anno_subset = anno_df[["anno_idx", "thread"]].drop_duplicates(
                subset=["anno_idx"]
            )
            pred_df = pd.merge(pred_df, anno_subset, on="anno_idx", how="left")
            print(
                f"  Merged 'thread' from --anno_path via left join on anno_idx "
                f"({len(pred_df)} rows retained)."
            )

    has_pred_ids_col = "pred_ids" in pred_df.columns

    if args.few_shot_samples_csv is not None:
        gt_df = pd.read_csv(args.few_shot_samples_csv)
        all_gts_wo_char   = gt_df["text_gt_wo_char"].tolist()
        all_gts_num_words = [len(str(e).strip().split()) for e in all_gts_wo_char]
        print(f"  Loaded {len(all_gts_wo_char)} few-shot examples from {args.few_shot_samples_csv}")
    else:
        all_gts_wo_char   = []
        all_gts_num_words = []
        print("  [INFO] --few_shot_samples_csv not provided; running without few-shot examples.")

    print(f"Shots: {len(pred_df)} | batch_size: {args.batch_size} | language: {args.language}")

    # Build gender + name + description maps from imdb_chars_ext.json (single read).
    char_gender = {}
    char_names = {}
    char_descriptions = {}
    if args.no_cmb:
        # B1 ablation: prompt drops all CMB scaffolding regardless of metadata.
        # Skip char_maps load to keep this branch self-contained.
        if args.imdb_chars_path:
            print("  [INFO] --no_cmb is set; ignoring --imdb_chars_path for the B1 ablation.")
        print("  [INFO] Running in --no_cmb mode (Baseline 1 ablation): "
              "CMB scaffolding stripped from prompt.")
    elif args.imdb_chars_path and "imdbid" in pred_df.columns:
        movie_imdb_id = str(pred_df["imdbid"].iloc[0])
        char_gender, char_names, char_descriptions = build_char_maps(
            args.imdb_chars_path,
            movie_imdb_id,
        )
    else:
        print(
            "  [INFO] --imdb_chars_path not provided; "
            "using conservative multi-character pronoun flagging."
        )

    # Chunked inference: context-build + GPU inference + real memory write-back
    # interleaved per batch_size window.
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"stage2_qwen3_{args.mode}.csv")
    output_ads_list = chunked_inference(
        pipeline,
        pred_df,
        has_pred_ids_col,
        args,
        all_gts_wo_char,
        all_gts_num_words,
        char_gender,
        char_names=char_names,
        char_descriptions=char_descriptions,
        save_path=save_path,
    )

    # Save output
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path",            default=None,      type=str)
    parser.add_argument("--anno_path",            default=None,      type=str,
                        help="Path to the original annotation CSV containing thread and "
                             "pred_ids. Merged into pred_path on anno_idx (or row index).")
    parser.add_argument("--save_dir",             default=None,      type=str)
    parser.add_argument("--dataset",              default=None,      type=str)
    parser.add_argument("--mode",                 default="single",  type=str)
    parser.add_argument("--seed",                 default=42,        type=int)
    parser.add_argument("--prompt_idx",           default=0,         type=int)
    parser.add_argument("--num_examples",         default=10,        type=int)
    parser.add_argument("--LR_intercept",         default=0.0,       type=float)
    parser.add_argument("--LR_slope",             default=0.0,       type=float)
    parser.add_argument("--language",             default="English", type=str)
    parser.add_argument("--few_shot_samples_csv",                    type=str)
    parser.add_argument("--imdb_chars_path",      default=None,      type=str,
                        help="Path to imdb_chars_ext.json for gender-aware pronoun-ambiguity "
                             "detection. When omitted, any shot with 2+ characters is flagged "
                             "conservatively.")
    parser.add_argument("--batch_size",           default=8,         type=int,
                        help="Prompts per GPU batch. Start with 8 on A100-40 GB; raise to 16 "
                             "if VRAM allows, lower if OOM.")
    parser.add_argument("--no_cmb", action="store_true",
                        help="Baseline 1 ablation: disable CharacterMemoryBank entirely. "
                             "Prompt drops the NARRATIVE CONTEXT block, PRIORITY rules, and "
                             "CHARACTER REFERENCING RULES; the simplified coherence block is "
                             "used instead. Same model, same chat template, same sampling — "
                             "isolates the contribution of CMB.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    main(args)
