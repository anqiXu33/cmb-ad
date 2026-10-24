"""Masked text inputs for reference-form selection.

For each (AD, character) row the model sees:

    context : the previous K AD sentences of the same sequence, with every
              name token of the TARGET character replaced by TGTNAME (so the
              model learns "the target was named recently" generically,
              instead of memorising character names)
    current : the current AD with the target's mention replaced by TARGET
              (the answer is hidden; other characters' mentions stay)

Example (label PRN):
    context: "TGTNAME walks into the kitchen. ||| TGTNAME opens the fridge."
    current: "TARGET pours a glass of milk."

This mirrors the referential-form-selection setup in Chen, Same & van Deemter
(INLG 2021), where the referent slot is masked and the model predicts its form.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from followup.mentions_en import _name_tokens

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "followup"

K_CONTEXT = 3
SEP = " ||| "
TARGET = "TARGET"
TGTNAME = "TGTNAME"


def _name_regex(role: str):
    toks = sorted({t for t in _name_tokens(role or "")}, key=len, reverse=True)
    if not toks:
        return None
    return re.compile(r"\b(" + "|".join(re.escape(t) for t in toks) + r")\b(?:['’]s)?")


def build_inputs(df: pd.DataFrame, ads: pd.DataFrame, k: int = K_CONTEXT,
                 mask_name: bool = True, sep: str = SEP) -> pd.DataFrame:
    """Add `context` and `current` columns to df (rows of text_features.csv).

    mask_name=False keeps the target's name in the context (for prompting an
    untrained LLM, which needs to see who was mentioned); the TARGET slot in
    the current AD is always masked."""
    ads = ads.sort_values(["dataset", "seq_id", "start", "anno_idx"]).reset_index(drop=True)
    pos, texts = {}, {}
    for (ds, seq), g in ads.groupby(["dataset", "seq_id"], sort=False):
        t = g["text"].astype(str).tolist()
        texts[(ds, seq)] = t
        for i, a in enumerate(g["anno_idx"]):
            pos[(ds, seq, a)] = i

    ctx_out, cur_out = [], []
    rx_cache: dict[str, re.Pattern | None] = {}
    for r in df.itertuples(index=False):
        role = r.char_role if isinstance(r.char_role, str) else ""
        if role not in rx_cache:
            rx_cache[role] = _name_regex(role)
        rx = rx_cache[role]
        seq_texts = texts[(r.dataset, r.seq_id)]
        i = pos[(r.dataset, r.seq_id, r.anno_idx)]
        prev = seq_texts[max(0, i - k):i]
        if rx is not None and mask_name:
            prev = [rx.sub(TGTNAME, s) for s in prev]
        ctx_out.append(sep.join(prev) if prev else "NOCONTEXT")

        cur = str(r.text)
        a, b = int(r.m_start), int(r.m_start) + len(str(r.m_text))
        if cur[a:b] == str(r.m_text):
            cur = cur[:a] + TARGET + cur[b:]
        else:                               # offset drift (should not happen)
            cur = cur.replace(str(r.m_text), TARGET, 1)
        # a possessive marker left after a masked NAME / DESC ("TARGET's chest")
        # would give the answer away vs. a masked "Her chest" -> drop it
        cur = re.sub(TARGET + r"['’]s\b", TARGET, cur)
        if rx is not None and mask_name:
            cur = rx.sub(TGTNAME, cur)
        cur_out.append(cur)
    df = df.copy()
    df["context"] = ctx_out
    df["current"] = cur_out
    return df


if __name__ == "__main__":
    df = pd.read_csv(OUT / "text_features.csv")
    ads = pd.read_csv(OUT / "text_ads.csv")
    df = build_inputs(df, ads)
    df[["dataset", "seq_id", "anno_idx", "char_id", "label", "context", "current"]].to_csv(
        OUT / "text_inputs.csv", index=False)
    for _, r in df.sample(5, random_state=3).iterrows():
        print(f"[{r.label}]  ctx: {r.context[:160]}\n        cur: {r.current}\n")
