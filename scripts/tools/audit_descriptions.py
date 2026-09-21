"""
Audit VLM-generated character descriptions:
  - COLLISION: two+ characters in the same movie share identical description
  - LOW_SIGNAL_MAIN: description is too generic AND character has ≥30 shots
  - ERROR: known grammatical/semantic errors in Italian/German

Outputs manual_descriptions_todo.csv sorted by (COLLISION first, shot_count desc).

Usage (on cluster):
  python scripts/audit_descriptions.py \
    --json_path  /path/to/imdb_chars_enriched.json \
    --anno_dir   /path/to/resources/annotations \
    --output     manual_descriptions_todo.csv \
    [--min_shots 30]
"""

import argparse
import ast
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


# ── German / Italian generic patterns ────────────────────────────────────────
GENERIC_PATTERNS = [
    # German – hair only, no face/age anchor
    "mit kurzen Haaren", "mit langen Haaren", "mit dunklen Haaren",
    "mit hellen Haaren", "mit braunen Haaren", "mit schwarzen Haaren",
    "die blonde Frau", "der blonde Mann",
    "der bärtige Mann", "der Mann mit Bart",
    # Italian – hair only
    "con i capelli corti", "con i capelli lunghi", "con i capelli scuri",
    "con i capelli castani", "con i capelli neri",
    "la donna bionda", "l'uomo con la barba",
]

# Known hard errors (substring match)
ERROR_PATTERNS = [
    "il uomo",        # should be l'uomo
    "corti e calva",  # contradictory: kurz + kahl
    "il ragazzo con i capelli corti e calva",  # same
]


def build_shot_counts(anno_dir: Path, film_ids: list[str]) -> dict[str, Counter]:
    """
    For each film_id, scan all annotation CSVs to find the matching one
    (by imdbid column) and count how many shots each actor_id appears in.

    Returns {film_id: Counter({actor_id: shot_count})}
    """
    # Index all CSVs by imdbid (read only first row to get imdbid)
    csv_by_imdbid: dict[str, Path] = {}
    for csv_path in anno_dir.glob("*_context_faces_threads_shot_scales.csv"):
        try:
            row = pd.read_csv(csv_path, nrows=1)
            iid = str(row["imdbid"].iloc[0])
            csv_by_imdbid[iid] = csv_path
        except Exception:
            continue

    counts: dict[str, Counter] = {}
    for film_id in film_ids:
        csv_path = csv_by_imdbid.get(film_id)
        if csv_path is None:
            print(f"  [WARN] No annotation CSV for {film_id}")
            counts[film_id] = Counter()
            continue

        df = pd.read_csv(csv_path, usecols=["pred_ids"])
        counter: Counter = Counter()
        for val in df["pred_ids"].dropna():
            try:
                d = ast.literal_eval(str(val))
                if isinstance(d, dict):
                    for cid in d:
                        counter[str(cid)] += 1
            except Exception:
                continue
        counts[film_id] = counter
        print(f"  {film_id}: {len(counter)} characters, "
              f"{sum(counter.values())} total shot-appearances")
    return counts


def audit(json_path: str, anno_dir: str, output: str, min_shots: int):
    with open(json_path, encoding="utf-8") as f:
        char_data: dict = json.load(f)

    anno_dir_path = Path(anno_dir)
    film_ids = list(char_data.keys())

    print(f"Computing shot counts for {len(film_ids)} films …")
    shot_counts = build_shot_counts(anno_dir_path, film_ids)

    rows = []  # (film_id, char_id, role, shots, desc, reasons)

    for film_id, chars in char_data.items():
        film_counter = shot_counts.get(film_id, Counter())

        # Build collision map: description → list of chars that share it
        desc_to_chars: dict[str, list] = defaultdict(list)
        for c in chars:
            d = c.get("description", "").strip()
            if d:
                desc_to_chars[d].append(c)

        for c in chars:
            desc = c.get("description", "").strip()
            if not desc:
                continue

            shots = film_counter.get(str(c.get("id", "")), 0)
            reasons = []

            # COLLISION: same description shared by ≥2 characters in this film
            colliders = desc_to_chars[desc]
            if len(colliders) > 1:
                other_roles = [x["role"] for x in colliders if x is not c]
                reasons.append(f"COLLISION_x{len(colliders)}({','.join(other_roles)})")

            # ERROR: known bad substrings
            for pat in ERROR_PATTERNS:
                if pat in desc:
                    reasons.append("ERROR")
                    break

            # LOW_SIGNAL_MAIN: generic description on a character with ≥min_shots shots
            if shots >= min_shots:
                words = desc.split()
                generic_word_count = sum(
                    len(p.split()) for p in GENERIC_PATTERNS if p in desc
                )
                signal = len(words) - generic_word_count
                if signal < 3:
                    reasons.append("LOW_SIGNAL_MAIN")

            if reasons:
                rows.append((film_id, str(c.get("id", "")), c.get("role", ""),
                             shots, desc, ",".join(reasons)))

    # Sort: COLLISION first, then by shot_count descending
    rows.sort(key=lambda x: ("COLLISION" not in x[5], -x[3]))

    with open(output, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["film_id", "char_id", "role", "shot_count",
                    "old_desc", "reason", "new_desc", "reviewed"])
        for row in rows:
            w.writerow(list(row) + ["", "FALSE"])

    print(f"\nTotal flagged: {len(rows)}")
    print(f"Saved → {output}")

    print("\nTop 15 priority:")
    header = f"{'film_id':15} {'role':25} {'shots':>6}  {'reason':35}  desc"
    print(header)
    print("-" * len(header))
    for r in rows[:15]:
        print(f"{r[0]:15} {r[2]:25} {r[3]:>6}  {r[5]:35}  {r[4]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", required=True,
                        help="Path to imdb_chars_enriched.json")
    parser.add_argument("--anno_dir", required=True,
                        help="Directory with *_context_faces_threads_shot_scales.csv files")
    parser.add_argument("--output", default="manual_descriptions_todo.csv")
    parser.add_argument("--min_shots", type=int, default=30,
                        help="Shot threshold below which LOW_SIGNAL is not flagged")
    args = parser.parse_args()
    audit(args.json_path, args.anno_dir, args.output, args.min_shots)


if __name__ == "__main__":
    main()
