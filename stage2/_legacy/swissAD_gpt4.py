import os
import sys
import ast
import json
import torch
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from swiss_promptloader import get_user_prompt
from coherence_manager import GlobalMemoryBank

from openai import OpenAI
# os.environ["OPENAI_API_KEY"] = #TODO


def get_samples(lang):
    if lang == "German":
        samples = ["Sie gehen in ein Restaurant.",
                    "Er schiebt den Schirm wieder in die Tür.",
                    "Sie füllt Pasta in eine Plastikdose.",
                    "Die Rotblonde hängt Mantel und Schal in den Spind. Dann zieht sie die Schuhe aus.",
                    "Sie nimmt ein Fläschchen und geht zur Arbeitsfläche. Dort schaut sie auf einen Zettel.",
                    "Die Frau schaut dem Mann ins Gesicht. Dieser hat die Augen geschlossen.\nSchrift: eine Woche davor.",
                    "Tobias geht auf einer teils von Lava verschütteten Strasse. Der Blick entfernt sich.",
                    "Neben dem Bett an der Tür sitzt eine Braunhaarige ca. Mitte 40 und liest.",
                    "Sie gehen den Gang entlang zu Zimmer 6.", "Vor einem Fenster hängt Wäsche an einer Leine."]

    elif lang == "Italian":
        samples = ["In una camionetta della polizia.\nIl portellone posteriore è chiuso.",
                    "Sullo schermo si legge: Zurigo. Millenovecento ottanta.",
                    "Lei arriccia il naso e corruga la fronte.\nDue poliziotti la afferrano.",
                    "In uno studio medico.\nSulla parete, un paesaggio collinare\nsu cui si ergono degli alberi verdi e rigogliosi.",
                    "Christina spinge il passeggino con Lena.\nIndossa un cappotto beige e una collana di perle.",
                    "Su sfondo blu si legge, in bianco: Cinema.\nIn un quadrato bianco si legge, in blu: Rai.\nRai Cinema.",
                    "Due ragazzi in sella a uno scooter\nsi fermano davanti a loro.",
                    "Al rallentatore.\nSegue con lo sguardo il ragazzino che scappa.\nSopraggiunge una volante.",
                    "Lui si allontana."]
    else:
        print(f"Unknown language: {lang}")
        exit(0)
    return samples


def summary_each(client, user_prompt, dataset, idx, mode):
    dataset_text = "movie"

    sys_prompt = (
            f"You are an intelligent chatbot designed for summarizing {dataset_text} audio descriptions. "
            "Here's how you can accomplish the task: INSTRUCTIONS: you should convert the predicted descriptions into one sentence. "
            "You should directly start the answer with the converted results WITHOUT providing ANY more sentences at the beginning or at the end."
    )

    messages = [
        {
            "role": "system",
            "content": sys_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    repeat = True
    iters = 0
    while repeat and iters < 5:
        if iters >=1:
            print("Redo due to error")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
        )
        output_text = response.choices[0].message.content
        if mode == "single" and "{\"summarized_AD\": \"" in output_text:
            repeat = False
        else: # assistant mode
            for ad_idx in range(1, 6):
                if f"\"summarized_AD_{ad_idx}\":" not in output_text:
                    repeat = True
                    continue
                repeat = False
    return output_text


def main(args):
    # Initialise the openai client
    client = OpenAI(
        api_key=args.access_token,
    )

    # Read predicted output from Stage I
    pred_df = pd.read_csv(args.pred_path)

    # Detect optional columns for inter-shot coherence
    has_thread_col = 'thread' in pred_df.columns
    has_pred_ids_col = 'pred_ids' in pred_df.columns

    # Extract GT AD list (w & wo character information)
    gt_df = pd.read_csv(args.few_shot_samples_csv)
    all_gts = gt_df["text_gt"].tolist()
    all_gts_wo_char = gt_df["text_gt_wo_char"].tolist()
    all_gts_num_words = [len(e.strip().split(" ")) for e in all_gts_wo_char]

    text_gen_list = []
    text_gt_list = []
    start_sec_list = []
    end_sec_list = []
    imdbid_list = []
    anno_indices = []
    preceding_ad = ""
    memory_bank = GlobalMemoryBank()

    for shot_idx, (row_idx, row) in enumerate(tqdm(pred_df.iterrows(), total=len(pred_df))):
        # Estimate the number of words based on training split statistics
        duration = round(row['end'] - row['start'], 2)
        # use slope + intercept (from GT data) to predict number of words in output
        rough_num_words = args.LR_intercept + args.LR_slope * duration

        text_gt = row['text_gt']
        text_pred = str(row['text_gen'])

        # --- Inter-shot coherence: parse thread ID and character IDs ---
        current_thread_id = row['thread'] if has_thread_col else None
        if current_thread_id is not None:
            try:
                if pd.isna(current_thread_id):
                    current_thread_id = None
            except (TypeError, ValueError):
                pass  # non-scalar value; keep as-is

        current_character_ids = []
        if has_pred_ids_col:
            pred_ids_raw = row['pred_ids']
            try:
                is_nan = isinstance(pred_ids_raw, float) and pd.isna(pred_ids_raw)
                if not is_nan and str(pred_ids_raw).strip() not in ('', '{}'):
                    pred_ids_dict = ast.literal_eval(str(pred_ids_raw))
                    if isinstance(pred_ids_dict, dict):
                        current_character_ids = list(pred_ids_dict.keys())
            except (ValueError, SyntaxError):
                pass  # malformed pred_ids: skip silently

        char_status, recent_history_str, is_same_thread = memory_bank.update_and_get_context(
            shot_idx, text_pred, current_thread_id, current_character_ids
        )
        # ---------------------------------------------------------------

        # Sample GT ADs with roughly the same length as examples (+-1 word)
        candid_indices = [i for i, s in enumerate(all_gts_num_words) if rough_num_words - 1 <= s <= rough_num_words + 1]
        if len(candid_indices) < args.num_examples:
            candid_indices = list(range(len(all_gts_num_words)))
        sampled_indices = random.choices(candid_indices, k=args.num_examples)
        sampled_examples = [all_gts_wo_char[index] for index in sampled_indices]

        # Formulate the user prompt
        user_prompt = get_user_prompt(
            mode=args.mode, prompt_idx=args.prompt_idx, verb_list=[], text_pred=text_pred,
            word_limit=int(rough_num_words) + 1, examples=sampled_examples,
            language=args.language, preceding_ad=preceding_ad,
            char_status=char_status, recent_history=recent_history_str,
            is_same_thread=is_same_thread,
        )

        # Output AD
        text_summary = summary_each(client, user_prompt, args.dataset, row_idx, args.mode)
        try:
            if args.mode == "single": # default single AD mode
                text_summary = text_summary.replace("{\"summarized_AD\": \"", "").replace("\"}", "").strip()
                if "." != text_summary[-1]:  # Add comma if not existing
                    text_summary = text_summary + "."
                output_ads = text_summary
            else: # assistant mode (predict 5 AD candidates)
                output_ad_list = []
                for ad_idx in range(1, 6):
                    if "\n" not in text_summary:
                        text_summary_tmp = text_summary.split(f"\"summarized_AD_{ad_idx}\":")[-1].split(", \"")[0].split(",\"")[0].replace('\"', "").replace('{', "").replace('}', "").strip()
                    else:
                        text_summary_tmp = text_summary.split(f"\"summarized_AD_{ad_idx}\":")[-1].split(",\n")[0].split("\n")[0].replace('\"', "").replace('{', "").replace('}', "").strip()
                    if "." != text_summary_tmp[-1]:
                        text_summary_tmp = text_summary_tmp + "."
                    output_ad_list.append(text_summary_tmp)
                output_ads = str(output_ad_list)
        except:
            output_ads = ""

        print(output_ads)
        preceding_ad = output_ads
        memory_bank.add_generated_ad(output_ads)
        text_gen_list.append(output_ads)
        text_gt_list.append(text_gt)
        start_sec_list.append(row['start'])
        end_sec_list.append(row['end'])
        imdbid_list.append(row['imdbid'])
        anno_indices.append(row['anno_idx'])


    output_df = pd.DataFrame.from_records({'imdbid': imdbid_list, 'start': start_sec_list, 'end': end_sec_list, 'text_gt': text_gt_list, 'text_gen': text_gen_list, 'anno_idx': anno_indices})
    save_path = os.path.join(args.save_dir, f"stage2_gpt4o_{args.mode}.csv")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    output_df.to_csv(save_path, index=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path', default=None, type=str, help='input directory')
    parser.add_argument('--save_dir', default=None, type=str, help='output directory')
    parser.add_argument('--dataset', default=None, type=str)
    parser.add_argument('--mode', default="single", type=str)
    parser.add_argument('--access_token', default=None, type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_examples', default=10, type=int, help='number of GT ADs')
    parser.add_argument('--prompt_idx', default=0, type=int, help='optional, use to indicate you own prompt')
    parser.add_argument('--LR_intercept', default=0.0, type=float, help="Linear regression to predict AD length based on duration, intercept value.")
    parser.add_argument('--LR_slope', default=0.0, type=float, help="Linear regression to predict AD length based on duration, slope value.")
    parser.add_argument('--language', default="English", type=str, help="Target language.")
    parser.add_argument('--few_shot_samples_csv', type=str, help="csv with ground truth ADs (removed character names).")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    main(args)
