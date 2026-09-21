"""
Apply manually-written descriptions from manual_descriptions_todo.csv
back into imdb_chars_enriched.json.

Only rows where reviewed=TRUE and new_desc is non-empty are applied.

Usage:
  python scripts/apply_manual_fixes.py \
    --csv    manual_descriptions_todo.csv \
    --json   /path/to/imdb_chars_enriched.json \
    --output /path/to/imdb_chars_enriched.json
"""

import argparse
import json
import csv
import sys


def main(args):
    with open(args.json, encoding="utf-8") as f:
        data = json.load(f)

    # Build lookup: (film_id, char_id) → entry
    lookup: dict[tuple, dict] = {}
    for film_id, chars in data.items():
        for entry in chars:
            lookup[(film_id, str(entry.get("id", "")))] = entry

    applied, skipped = 0, 0
    with open(args.csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            reviewed = row.get("reviewed", "").strip().upper()
            new_desc = row.get("new_desc", "").strip()

            if reviewed != "TRUE" or not new_desc:
                skipped += 1
                continue

            key = (row["film_id"], row["char_id"])
            entry = lookup.get(key)
            if entry is None:
                print(f"  [WARN] Not found in JSON: {key}")
                skipped += 1
                continue

            old = entry.get("description", "")
            entry["description"] = new_desc
            print(f"  {row['film_id']} | {row['role']:25} | {old!r} → {new_desc!r}")
            applied += 1

    print(f"\nApplied: {applied}  Skipped: {skipped}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",    required=True, help="manual_descriptions_todo.csv")
    parser.add_argument("--json",   required=True, help="imdb_chars_enriched.json (input)")
    parser.add_argument("--output", required=True, help="output path (can equal --json)")
    main(parser.parse_args())
