"""
Auto-correct gender errors in imdb_chars_enriched.json descriptions.

Rules applied:
  German — flip der↔die and Mann↔Frau based on gender field;
           remove beard phrases from female descriptions.
  Italian — flip l'uomo↔la donna (and related nouns) based on gender field;
            remove beard phrases from female descriptions;
            fix adjective agreement (anziano↔anziana, calvo↔calva).

After correction, prints a list of remaining duplicates within each movie
for manual review.

Usage:
  python char_recog/fix_descriptions.py \\
    --input  /path/to/imdb_chars_enriched.json \\
    --output /path/to/imdb_chars_enriched.json
"""

import re
import json
import argparse
from collections import Counter


# ─── German ───────────────────────────────────────────────────────────────────

_DE_BEARD = re.compile(
    r',?\s*(?:und\s+)?(?:mit\s+)?(?:einem?\s+)?'
    r'(?:Vollbart|Dreitagebart|Schnurrbart|Kinnbart|Spitzbart|kurzem\s+Bart|langem\s+Bart|Bart)\b',
    re.IGNORECASE,
)
_DE_BAERTIG = re.compile(r'\bbärtige[rn]?\s+', re.IGNORECASE)


def _fix_german(desc: str, gender: int) -> str:
    if not desc or gender == 0:
        return desc

    if gender == 1:  # female — fix masculine forms
        desc = re.sub(r'^[Dd]er\b', 'die', desc)
        desc = re.sub(r'\bMann\b', 'Frau', desc)
        desc = re.sub(r'\bjunger\s+Mann\b', 'junge Frau', desc, flags=re.IGNORECASE)
        desc = re.sub(r'\bälterer\s+Mann\b', 'ältere Frau', desc, flags=re.IGNORECASE)
        desc = re.sub(r'\bkahlköpfiger\s+Mann\b', 'kahlköpfige Frau', desc, flags=re.IGNORECASE)
        desc = _DE_BEARD.sub('', desc)
        desc = _DE_BAERTIG.sub('', desc)

    elif gender == 2:  # male — fix feminine forms
        desc = re.sub(r'^[Dd]ie\b', 'der', desc)
        desc = re.sub(r'\bFrau\b', 'Mann', desc)

    # Clean up double spaces / trailing commas
    desc = re.sub(r'\s{2,}', ' ', desc).strip().rstrip(',').strip()
    return desc


# ─── Italian ──────────────────────────────────────────────────────────────────

def _remove_it_beard(desc: str) -> str:
    """Remove beard phrases from Italian description, preserving other features."""
    # "con la barba [adj] e [next feature]" → "con [next feature]"
    desc = re.sub(r'\bcon\s+la\s+barba(?:\s+\w+)?\s+e\s+', 'con ', desc, flags=re.IGNORECASE)
    # "e la barba [adj]" at end or before punctuation
    desc = re.sub(r',?\s*e\s+la\s+barba(?:\s+\w+)?\b', '', desc, flags=re.IGNORECASE)
    # standalone "con la barba [adj]"
    desc = re.sub(r',?\s*con\s+la\s+barba(?:\s+\w+)?\b', '', desc, flags=re.IGNORECASE)
    return desc

_IT_MALE_STARTS = [
    (r"^[Ll]'uomo\b",   "la donna"),
    (r'^[Uu]n\s+uomo\b', 'una donna'),
    (r'^[Ii]l\s+giovane\b', 'la giovane'),
    (r'^[Ii]l\s+vecchio\b', 'la vecchia'),
    (r'^[Ii]l\s+bambino\b', 'la donna'),   # bambino → donna (adult woman, not bambina)
    (r'^[Ii]l\s+ragazzo\b', 'la donna'),
]

_IT_FEMALE_STARTS = [
    (r'^[Ll]a\s+donna\b',   "l'uomo"),
    (r'^[Uu]na\s+donna\b',  'un uomo'),
    (r'^[Ll]a\s+giovane\b', 'il giovane'),
    (r'^[Ll]a\s+vecchia\b', 'il vecchio'),
    (r'^[Ll]a\s+bambina\b', "l'uomo"),
    (r'^[Ll]a\s+ragazza\b', 'il giovane'),
]


def _fix_italian(desc: str, gender: int) -> str:
    if not desc or gender == 0:
        return desc

    if gender == 1:  # female
        for pat, repl in _IT_MALE_STARTS:
            if re.match(pat, desc):
                desc = re.sub(pat, repl, desc, count=1)
                break
        desc = _remove_it_beard(desc)
        # Adjective agreement
        desc = re.sub(r'\banziano\b', 'anziana', desc)
        desc = re.sub(r'\bcalvo\b', 'calva', desc)
        desc = re.sub(r'\bvecchio\b', 'vecchia', desc)

    elif gender == 2:  # male
        for pat, repl in _IT_FEMALE_STARTS:
            if re.match(pat, desc):
                desc = re.sub(pat, repl, desc, count=1)
                break
        # Adjective agreement
        desc = re.sub(r'\banziana\b', 'anziano', desc)
        desc = re.sub(r'\bcalva\b', 'calvo', desc)
        desc = re.sub(r'\bvecchia\b', 'vecchio', desc)

    desc = re.sub(r'\s{2,}', ' ', desc).strip().rstrip(',').strip()
    return desc


# ─── Dispatcher ───────────────────────────────────────────────────────────────

def _detect_language(desc: str) -> str:
    if re.match(r'^(der|die|das|ein|eine)\b', desc, re.IGNORECASE):
        return 'de'
    if re.match(r"^(l'|la |il |un |una |lo )", desc, re.IGNORECASE):
        return 'it'
    return 'unknown'


def fix_description(desc: str, gender: int) -> str:
    lang = _detect_language(desc)
    if lang == 'de':
        return _fix_german(desc, gender)
    if lang == 'it':
        return _fix_italian(desc, gender)
    return desc


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    with open(args.input, encoding='utf-8') as f:
        data = json.load(f)

    total_fixed = 0

    for movie_id, entries in data.items():
        for entry in entries:
            desc = entry.get('description', '').strip()
            gender = entry.get('gender', 0)
            if not desc:
                continue
            fixed = fix_description(desc, gender)
            if fixed != desc:
                print(f"  [{movie_id}] {entry.get('role','?'):25s}")
                print(f"    before: {desc}")
                print(f"    after:  {fixed}")
                entry['description'] = fixed
                total_fixed += 1

    print(f"\nFixed {total_fixed} descriptions.")

    # Report remaining duplicates within each movie for manual review
    print("\n=== Remaining duplicates (manual review needed) ===")
    any_dup = False
    for movie_id, entries in data.items():
        descs = [e.get('description', '').strip() for e in entries if e.get('description', '').strip()]
        counts = Counter(descs)
        dups = {d: c for d, c in counts.items() if c > 1}
        if dups:
            any_dup = True
            print(f"\n{movie_id}:")
            for desc, count in dups.items():
                roles = [e.get('role', '?') for e in entries if e.get('description', '').strip() == desc]
                print(f"  [{count}x] \"{desc}\"")
                print(f"       → {roles}")
    if not any_dup:
        print("  None!")

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  required=True, help='Path to imdb_chars_enriched.json')
    parser.add_argument('--output', required=True, help='Output path (can be same as input)')
    main(parser.parse_args())
