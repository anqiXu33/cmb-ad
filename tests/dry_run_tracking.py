import sys
import os
import argparse
import pandas as pd
import ast

# --- Setup paths to import from stage2 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
stage2_dir = os.path.join(parent_dir, "stage2")
sys.path.append(stage2_dir)

from coherence_manager import CharacterMemoryBank
from char_utils import build_char_gender_map


def parse_pred_ids(raw):
    """Safe parsing of the pred_ids column."""
    try:
        if pd.isna(raw) or str(raw).strip() in ("", "{}"):
            return []
        parsed = ast.literal_eval(str(raw))
        if isinstance(parsed, dict):
            return list(parsed.keys())
    except:
        pass
    return []


def parse_thread_to_set(raw):
    """Parses thread column '[[0], [1]]' into a set {0, 1}."""
    try:
        if pd.isna(raw) or str(raw).strip() == "":
            return set()
        val = ast.literal_eval(str(raw))
        flat_set = set()
        if isinstance(val, list):
            for item in val:
                if isinstance(item, list):
                    flat_set.update(item)
                else:
                    flat_set.add(item)
        return flat_set
    except:
        return set()


def main(args):
    print(f"Loading predictions from: {args.pred_path}")
    if not os.path.exists(args.pred_path):
        print(f"Error: File not found at {args.pred_path}")
        return
    df = pd.read_csv(args.pred_path)

    # 1. Merge thread/pred_ids from anno_path if provided (Logic same as swissAD_qwen3.py)
    if args.anno_path:
        print(f"Merging annotations from: {args.anno_path}")
        if not os.path.exists(args.anno_path):
            print(f"Error: Annotation file not found at {args.anno_path}")
            return
        anno_df = pd.read_csv(args.anno_path)
        cols_to_merge = ["anno_idx"]
        if "thread" in anno_df.columns and "thread" not in df.columns:
            cols_to_merge.append("thread")
        if "pred_ids" in anno_df.columns and "pred_ids" not in df.columns:
            cols_to_merge.append("pred_ids")

        if len(cols_to_merge) > 1:
            anno_subset = anno_df[cols_to_merge].drop_duplicates(subset=["anno_idx"])
            df = pd.merge(df, anno_subset, on="anno_idx", how="left")

    # 2. Setup Gender Map (Optional but recommended for accurate Ambiguity checks)
    char_gender = {}
    if not args.imdb_chars_path:
        print(
            "[INFO] No gender map provided. Ambiguity check will be conservative (any 2+ chars)."
        )
    elif not os.path.exists(args.imdb_chars_path):
        print(f"Error: imdb_chars_path file not found at {args.imdb_chars_path}")
        print(
            "[WARN] Continuing without gender map. Ambiguity check will be conservative."
        )
    elif "imdbid" not in df.columns:
        print(
            "[WARN] --imdb_chars_path provided, but 'imdbid' column not found in CSV. Cannot build gender map."
        )
    else:
        movie_id = str(df["imdbid"].iloc[0])
        print(f"Building gender map for movie: {movie_id}")
        char_gender = build_char_gender_map(args.imdb_chars_path, movie_id)

    # 3. Initialize Bank
    memory_bank = CharacterMemoryBank(char_gender=char_gender)

    # 4. Dry Run Loop
    print("\n" + "=" * 120)
    print(
        f"{'Idx':<4} | {'Thread':<8} | {'Pred IDs':<20} | {'Old: NewThread?':<15} | {'New: NewThread?':<15} | {'Flags (Old)'}"
    )
    print("=" * 120)

    results_rows = []
    prev_thread_raw = None

    for i, row in df.iterrows():
        # Prepare Inputs
        text_pred = str(row.get("text_gen", ""))
        current_thread = row.get("thread", None)
        current_ids = parse_pred_ids(row.get("pred_ids", {}))

        # --- EXECUTE LOGIC ---
        char_status, _, is_same_thread = memory_bank.update_and_get_context(
            i, text_pred, current_thread, current_ids
        )
        # ---------------------

        # Extract Internal State for Debugging View
        internal_state = []
        flags_triggered = []

        # --- OLD LOGIC SIMULATION (Strict Equality) ---
        if i == 0:
            is_new_thread_old_sim = True
        else:
            # Old logic: strictly NEW if thread string is different
            is_new_thread_old_sim = current_thread != prev_thread_raw

        prev_thread_raw = current_thread

        # CURRENT LOGIC RESULT (From coherence_manager.py)
        is_new_thread_current_code = not is_same_thread

        # Check flags in the output string
        if "USE NAME" in char_status:
            if "ambiguous" in char_status:
                flags_triggered.append("AMBIGUOUS")
            if "memory anchor" in char_status:
                flags_triggered.append("ANCHOR")

        if is_new_thread_current_code:
            flags_triggered.append("NEW_THREAD")

        # Inspect memory bank directly for 'Gap' info
        for cid in current_ids:
            if cid in memory_bank.characters:
                # Note: update_and_get_context already updated last_seen to current 'i'
                # So we look at 'consecutive_shots'
                consec = memory_bank.characters[cid]["consecutive_shots"]
                last_act = memory_bank.characters[cid]["last_action"]
                short_act = (last_act[:10] + "..") if len(last_act) > 10 else last_act
                internal_state.append(f"{cid}:sq={consec},act='{short_act}'")

        state_str = ", ".join(internal_state)
        ids_str = str(current_ids)

        # Print Row
        # Highlight interesting rows
        prefix = "  "
        if flags_triggered:
            prefix = "->"

        # Compare logic
        # If Old Sim says YES (cut) but Current Code says NO (continue), that's an improvement [IMPROVED]
        diff_marker = ""
        if is_new_thread_old_sim and not is_new_thread_current_code:
            diff_marker = " [IMPROVED]"
        elif is_new_thread_old_sim != is_new_thread_current_code:
            diff_marker = " [DIFF]"

        print(
            f"{prefix}{i:<2} | {str(current_thread):<8} | {ids_str[:20]:<20} | {str(is_new_thread_old_sim):<15} | {str(is_new_thread_current_code) + diff_marker:<15} | {flags_triggered}"
        )

        results_rows.append(
            {
                "idx": i,
                "thread": current_thread,
                "pred_ids": ids_str,
                "logic_state": state_str,
                "new_thread_old_sim": is_new_thread_old_sim,
                "new_thread_current": is_new_thread_current_code,
                "flags_triggered": str(flags_triggered),
            }
        )

        # Mock update history
        memory_bank.add_generated_ad(text_pred)

    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        pd.DataFrame(results_rows).to_csv(args.save_path, index=False)
        print(f"\nDry run results saved to: {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", required=True, help="Input CSV (stage1 output)")
    parser.add_argument(
        "--anno_path", default=None, help="Annotation CSV containing thread/pred_ids"
    )
    parser.add_argument(
        "--imdb_chars_path", default=None, help="Path to imdb_chars_ext.json"
    )
    parser.add_argument(
        "--save_path",
        default=None,
        help="Optional path to save the dry run results as CSV",
    )
    args = parser.parse_args()
    main(args)
