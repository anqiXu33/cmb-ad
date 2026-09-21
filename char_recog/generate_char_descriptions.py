"""
Generate physical descriptions for characters using in-movie representative crops.

Uses Qwen2.5-VL in a single per-movie call so descriptions are mutually distinctive:
the VLM sees all characters from the same film at once and is asked to write descriptions
that differentiate each person from the others.

Falls back to per-character calls if the multi-character JSON output cannot be parsed.

Reads from --imdb_chars_path (e.g. imdb_chars_enriched.json with gender already filled),
writes descriptions back to --output_path.

Usage:
  python char_recog/generate_char_descriptions.py \\
    --imdb_chars_path /path/to/imdb_chars_enriched.json \\
    --rep_crops_dir   /path/to/rep_crops/ \\
    --output_path     /path/to/imdb_chars_enriched.json \\
    --language        German
"""

import os
import re
import json
import glob
import torch
import argparse

from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# ─── prompts ──────────────────────────────────────────────────────────────────

# Multi-character prompt (one call per movie, asks for JSON with distinctive descs)
_MULTI = {
    "German": (
        "Die folgenden Bilder zeigen {n} verschiedene Charaktere aus demselben Film.\n"
        "Schreibe für jeden Charakter eine KURZE Erscheinungsbeschreibung "
        "(max. 8 Wörter, mit bestimmtem Artikel), die ihn/sie von den anderen UNTERSCHEIDET.\n"
        "Falls zwei Personen ähnliche Haare haben, betone andere Unterschiede: "
        "Alter, Barttyp (Vollbart/Dreitagebart/Schnurrbart), Brille, Glatze, Gesichtszüge.\n\n"
        "Antworte NUR mit gültigem JSON mit den Schlüsseln A, B, C … in der abgebildeten Reihenfolge.\n"
        'Beispiel: {{"A": "der ältere Mann mit Vollbart", '
        '"B": "die blonde Frau mit Brille", '
        '"C": "der junge Mann mit kurzem braunem Haar"}}'
    ),
    "Italian": (
        "Le seguenti immagini mostrano {n} personaggi diversi dello stesso film.\n"
        "Scrivi per ogni personaggio una BREVE descrizione dell'aspetto "
        "(max. 8 parole, con articolo determinativo) che lo/la DISTINGUA dagli altri.\n"
        "Se due persone hanno capelli simili, enfatizza: "
        "età, tipo di barba (barba folta/pizzetto/baffi), occhiali, calvizie, tratti del viso.\n\n"
        "Rispondi SOLO con JSON valido con le chiavi A, B, C … nello stesso ordine.\n"
        'Esempio: {{"A": "l\'uomo anziano con la barba folta", '
        '"B": "la donna bionda con gli occhiali", '
        '"C": "il giovane con i capelli castani corti"}}'
    ),
    "French": (
        "Les images suivantes montrent {n} personnages différents du même film.\n"
        "Écris pour chaque personnage une COURTE description (max. 8 mots, article défini) "
        "qui le/la DISTINGUE des autres.\n"
        "Si deux personnes ont des cheveux similaires, mets en avant : "
        "âge, type de barbe (barbe fournie/bouc/moustache), lunettes, calvitie, traits du visage.\n\n"
        "Réponds UNIQUEMENT avec un JSON valide avec les clés A, B, C … dans le même ordre.\n"
        'Exemple: {{"A": "l\'homme âgé avec une barbe fournie", '
        '"B": "la femme blonde avec des lunettes", '
        '"C": "le jeune homme aux cheveux bruns courts"}}'
    ),
}

# Single-character fallback prompt (used when multi-call JSON parsing fails)
_SINGLE = {
    "German": (
        "Describe this person's face in ONE short German phrase "
        "(max 8 words) suitable for audio description. Be as specific as "
        "possible — mention hair color AND style (e.g. kurz, lockig, glatt), "
        "facial hair type (Vollbart, Dreitagebart, Schnurrbart), approximate "
        "age (jung, mittleren Alters, älter), or glasses. "
        "Use a definite article. "
        "Examples: 'der junge Mann mit lockigem braunem Haar', "
        "'die ältere Frau mit kurzem grauem Haar', "
        "'der kahlköpfige Mann mit Dreitagebart'. "
        "Output ONLY the German phrase, nothing else."
    ),
    "Italian": (
        "Describe this person's face in ONE short Italian phrase "
        "(max 8 words) suitable for audio description. Be as specific as "
        "possible — mention hair color AND style (e.g. corti, ricci, lisci), "
        "facial hair type (barba folta, pizzetto, baffi), approximate "
        "age (giovane, di mezza età, anziano), or glasses. "
        "Use a definite article. "
        "Examples: 'il giovane con i capelli ricci castani', "
        "'la donna anziana con i capelli corti grigi', "
        "'l'uomo calvo con il pizzetto'. "
        "Output ONLY the Italian phrase, nothing else."
    ),
    "French": (
        "Describe this person's face in ONE short French phrase "
        "(max 8 words) suitable for audio description. Be as specific as "
        "possible — mention hair color AND style (e.g. courts, bouclés, lisses), "
        "facial hair type (barbe fournie, bouc, moustache), approximate "
        "age (jeune, d'âge moyen, âgé), or glasses. "
        "Use a definite article. "
        "Examples: 'le jeune homme aux cheveux bouclés bruns', "
        "'la femme âgée aux cheveux courts gris', "
        "'l'homme chauve avec un bouc'. "
        "Output ONLY the French phrase, nothing else."
    ),
}

_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# ─── model loading ────────────────────────────────────────────────────────────

def load_model():
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    return processor, model


# ─── helpers ──────────────────────────────────────────────────────────────────

def _generate(processor, model, messages, images, max_new_tokens=512):
    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text_input],
        images=images,
        return_tensors="pt",
        padding=True,
    ).to(model.device)
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    trimmed = out_ids[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def _parse_json_descriptions(text, label_to_id):
    """Extract {label: description} JSON from model output. Returns {} on failure."""
    # Try bare JSON first, then extract first {...} block
    for candidate in [text, re.search(r'\{[^{}]+\}', text, re.DOTALL)]:
        if candidate is None:
            continue
        raw = candidate if isinstance(candidate, str) else candidate.group()
        try:
            parsed = json.loads(raw)
            result = {}
            for label, desc in parsed.items():
                label = label.strip().upper()
                if label in label_to_id:
                    result[label_to_id[label]] = str(desc).strip('"\'').strip()
            if result:
                return result
        except (json.JSONDecodeError, AttributeError):
            continue
    return {}


# ─── core description functions ───────────────────────────────────────────────

def describe_movie_characters(processor, model, char_entries, movie_crops_dir, language):
    """
    One VLM call for all characters in a movie.  Returns {actor_id: description}.
    Falls back to per-character calls if the JSON output cannot be parsed.
    """
    prompt_multi = _MULTI.get(language)
    if prompt_multi is None:
        raise ValueError(f"Unsupported language: {language}")

    # Collect best crop (_rep0) per character
    char_data = []  # [(actor_id, role, Image), ...]
    for entry in char_entries:
        actor_id = str(entry.get("id", ""))
        if not actor_id:
            continue
        crop_pattern = os.path.join(movie_crops_dir, f"{actor_id}_rep*.jpg")
        crop_paths = sorted(glob.glob(crop_pattern))
        if not crop_paths:
            continue
        img = Image.open(crop_paths[0]).convert("RGB")
        char_data.append((actor_id, entry.get("role", "?"), img))

    if not char_data:
        return {}

    # Build message: "Character A:" + image, "Character B:" + image, …, then prompt
    content = []
    label_to_id = {}
    images = []
    for i, (actor_id, role, img) in enumerate(char_data):
        label = _LABELS[i] if i < len(_LABELS) else str(i)
        label_to_id[label] = actor_id
        content.append({"type": "text", "text": f"Character {label}:"})
        content.append({"type": "image", "image": img})
        images.append(img)

    content.append({"type": "text", "text": prompt_multi.format(n=len(char_data))})
    messages = [{"role": "user", "content": content}]

    raw_output = _generate(processor, model, messages, images, max_new_tokens=512)
    print(f"  [multi-call raw] {raw_output[:300]}")

    descriptions = _parse_json_descriptions(raw_output, label_to_id)

    if len(descriptions) < len(char_data):
        missing = [aid for aid, _, _ in char_data if aid not in descriptions]
        print(f"  [WARN] {len(missing)} characters missing from JSON output; "
              f"running per-character fallback for: "
              f"{[role for aid, role, _ in char_data if aid in missing]}")
        for actor_id, role, img in char_data:
            if actor_id not in descriptions:
                desc = _describe_single(processor, model, [img], language)
                descriptions[actor_id] = desc
                print(f"    fallback {role}: {desc}")

    return descriptions


def _describe_single(processor, model, images, language):
    """Single-character fallback."""
    prompt = _SINGLE.get(language, "")
    content = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    desc = _generate(processor, model, messages, images, max_new_tokens=64)
    return desc.strip('"\'').strip()


# ─── main ─────────────────────────────────────────────────────────────────────

def main(args):
    processor, model = load_model()

    with open(args.imdb_chars_path, encoding="utf-8") as f:
        imdb_chars = json.load(f)

    movie_ids = list(imdb_chars.keys())
    if args.movie_ids:
        requested = set(args.movie_ids)
        movie_ids = [mid for mid in movie_ids if mid in requested]
        skipped = requested - set(movie_ids)
        if skipped:
            print(f"[WARN] Movie IDs not found in JSON: {skipped}")
    print(f"Processing {len(movie_ids)} movies  language={args.language}")

    updated_total = 0

    for movie_id in movie_ids:
        chars = imdb_chars[movie_id]
        movie_crops_dir = os.path.join(args.rep_crops_dir, movie_id)
        if not os.path.isdir(movie_crops_dir):
            print(f"[SKIP] No crops directory for {movie_id}")
            continue

        # Skip movie if all characters already have descriptions (checkpoint resume)
        chars_needing_desc = [e for e in chars if not e.get("description", "").strip()]
        if not chars_needing_desc:
            print(f"[SKIP] {movie_id}: all characters already described")
            continue

        print(f"\n{'='*50}\nMovie: {movie_id}  ({len(chars_needing_desc)} characters to describe)")

        descriptions = describe_movie_characters(
            processor, model, chars_needing_desc, movie_crops_dir, args.language
        )

        updated = 0
        for entry in chars:
            actor_id = str(entry.get("id", ""))
            if actor_id in descriptions and descriptions[actor_id]:
                entry["description"] = descriptions[actor_id]
                updated += 1
                print(f"  {entry.get('role', actor_id):30s}  {descriptions[actor_id]}")

        updated_total += updated
        print(f"  Updated {updated}/{len(chars_needing_desc)} descriptions")

        # Checkpoint after each movie
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(imdb_chars, f, indent=2, ensure_ascii=False)
        print(f"  [CHECKPOINT] Saved to {args.output_path}")

    print(f"\nDone. Total updated: {updated_total}")
    print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate mutually-distinctive character descriptions from representative crops"
    )
    parser.add_argument("--imdb_chars_path", required=True,
                        help="Input JSON (e.g. imdb_chars_enriched.json with gender already set)")
    parser.add_argument("--rep_crops_dir", required=True,
                        help="Directory with representative crops (from build_query_bank.py)")
    parser.add_argument("--output_path", required=True,
                        help="Output path (can be same as --imdb_chars_path for in-place update)")
    parser.add_argument("--language", default="German",
                        choices=["German", "Italian", "French"],
                        help="Target language for descriptions")
    parser.add_argument("--movie_ids", nargs="+", default=None,
                        help="Only process these movie IDs. Omit to process all.")
    args = parser.parse_args()
    main(args)
