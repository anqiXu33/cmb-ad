"""
Character metadata utility for CharacterMemoryBank.

Reads imdb_chars_enriched.json to extract:
  - char_gender:       {actor_id: 'male' | 'female' | None}  — pronoun-ambiguity detection
  - char_names:        {actor_id: role_name}                  — display names in char_status
  - char_descriptions: {actor_id: physical_description}       — for REINTRODUCE prompts

Gender is filled manually in imdb_chars_enriched.json (TMDB / gender-guesser no longer used).
Run generate_char_descriptions.py to populate the description fields.

Usage:
    from char_utils import build_char_maps

    char_gender, char_names, char_descriptions = build_char_maps(
        imdb_chars_path="/path/to/imdb_chars_enriched.json",
        movie_imdb_id="tt10949014",
    )
    memory_bank = CharacterMemoryBank(
        char_gender=char_gender, char_names=char_names,
        char_descriptions=char_descriptions,
    )
"""

import json


def build_char_maps(imdb_chars_path: str, movie_imdb_id: str) -> tuple:
    """
    Build gender, name, and description mappings for one movie from imdb_chars_ext.json.

    Args:
        imdb_chars_path:  Path to imdb_chars_ext.json.
        movie_imdb_id:    IMDB title ID, e.g. 'tt10949014'.

    Returns:
        (char_gender, char_names, char_descriptions)
        char_gender:       {actor_id (str): 'male' | 'female' | None}
        char_names:        {actor_id (str): role_name (str)}
        char_descriptions: {actor_id (str): physical_description (str)}
    """
    with open(imdb_chars_path, encoding="utf-8") as f:
        imdb_chars = json.load(f)

    entries = imdb_chars.get(movie_imdb_id, [])
    if not entries:
        print(f"  [WARN] char_utils: no entries for {movie_imdb_id} in {imdb_chars_path}")
        return {}, {}, {}

    char_gender = {}
    char_names = {}
    char_descriptions = {}
    resolved = 0
    n_descriptions = 0

    for entry in entries:
        actor_id = entry.get("id")
        if not actor_id:
            continue
        aid = str(actor_id)

        # Gender
        g = entry.get("gender", 0)  # TMDB int: 0=unknown, 1=female, 2=male
        if g == 1:
            char_gender[aid] = "female"
            resolved += 1
        elif g == 2:
            char_gender[aid] = "male"
            resolved += 1
        else:
            char_gender[aid] = None

        # Role name
        role = entry.get("role", "")
        if role:
            char_names[aid] = role

        # Physical description (from generate_char_descriptions.py)
        desc = entry.get("description", "")
        if desc:
            char_descriptions[aid] = desc
            n_descriptions += 1

    print(f"  Char maps for {movie_imdb_id}: {len(char_names)} names, "
          f"{resolved}/{len(char_gender)} genders resolved, "
          f"{n_descriptions} descriptions")
    return char_gender, char_names, char_descriptions


# Backward-compatible wrapper
def build_char_gender_map(imdb_chars_path: str, movie_imdb_id: str) -> dict:
    """Legacy wrapper — returns only the gender map."""
    char_gender, _, _ = build_char_maps(imdb_chars_path, movie_imdb_id)
    return char_gender
