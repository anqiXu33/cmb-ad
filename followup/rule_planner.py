"""Thesis rule planner (CharacterMemoryBank decision logic) replayed on ads.csv.

Reproduces the flag logic of stage2/coherence_manager.py::get_context on the
same per-AD character stream the learned selector sees, so both can be scored
against the same silver labels.  Output flags are collapsed to the label set:

    CONTINUING                      -> PRN
    REINTRODUCE / USE NAME (either) -> NAME   (the rule does not distinguish
                                              name from description; scoring
                                              treats NAME and DESC as one class
                                              when the rule is evaluated)

Gender map comes from silver_labels.csv (same one the features use).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "followup"

HISTORY_WINDOW = 3
ANCHOR_INTERVAL = 4


def _split(s) -> list[str]:
    return [] if (pd.isna(s) or s == "") else str(s).split("|")


def _ambiguous(c: str, cur: set[str], prev: set[str], gender: dict[str, str]) -> bool:
    comp = (cur | prev) - {c}
    if not comp:
        return False
    g = gender.get(c, "")
    if not g:
        return True
    return any((not gender.get(o, "")) or gender.get(o) == g for o in comp)


def replay(ads: pd.DataFrame, gender: dict[str, str],
           anchor_interval: int = ANCHOR_INTERVAL) -> pd.DataFrame:
    ads = ads.sort_values(["dataset", "seq_id", "start", "anno_idx"]).reset_index(drop=True)
    out = []
    for (ds, seq), g in ads.groupby(["dataset", "seq_id"], sort=False):
        chars: dict[str, dict] = {}
        prev_ids: set[str] = set()
        for i, ad in enumerate(g.itertuples(index=False)):
            pids = set(_split(ad.pred_ids))
            for c in sorted(pids):
                if c in chars:
                    gap = i - chars[c]["last"]
                    consecutive = chars[c]["consec"] + 1 if gap == 1 else 1
                    if _ambiguous(c, pids, prev_ids, gender):
                        flag, consec_out = "USE_NAME_AMBIG", 0
                    elif consecutive >= anchor_interval:
                        flag, consec_out = "USE_NAME_ANCHOR", 0
                    elif gap == 1:
                        flag, consec_out = "CONTINUING", consecutive
                    else:
                        flag, consec_out = "REINTRODUCE", 0
                else:
                    flag, consec_out = "FIRST", 0
                chars[c] = {"last": i, "consec": consec_out}
                out.append({"dataset": ds, "seq_id": seq, "anno_idx": ad.anno_idx, "char_id": c,
                            "rule_flag": flag,
                            "rule_pred": "PRN" if flag == "CONTINUING" else "NAME"})
            prev_ids = {str(x) for x in pids}   # empty AD resets the carry, as in the thesis
    return pd.DataFrame(out)


if __name__ == "__main__":
    ads = pd.read_csv(OUT / "ads.csv")
    rows = pd.read_csv(OUT / "silver_labels.csv")
    gender = dict(zip(rows["char_id"], rows["char_gender"].fillna("")))
    r = replay(ads, gender)
    r.to_csv(OUT / "rule_preds.csv", index=False)
    m = rows.merge(r, on=["dataset", "seq_id", "anno_idx", "char_id"])
    print(pd.crosstab(m.label, m.rule_flag))
