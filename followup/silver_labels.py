"""Build per-(AD, character) reference-form silver labels from ground-truth AD.

Datasets (all shipped in resources/):
    madeval  10 full films, movie-level sequences            (thread + shot_scale)
    tvad     100 TV episodes, episode-level sequences        (clips minutes apart)
    cmdad    591 movie clips, clip-level sequences           (thread + shot_scale)

One output row per (AD, character detected in the AD's m-* shots):

    dataset, seq_id, anno_idx, start, end, text, n_chars, pred_ids,
    char_id, label, label_source, thread_switch, shot_scale

label:
    NAME / DESC / PRN   first GT mention bound to the character
    OMIT                character on screen but not referred to
    AMBIG               only unbound PRN/DESC mentions and >1 on-screen chars
Rows with label AMBIG are kept but flagged; training code decides.

Run:
    python -m followup.silver_labels --out results/followup/silver_labels.csv
"""
from __future__ import annotations

import argparse
import ast
import json
import pickle
import sys
import types
from pathlib import Path

import pandas as pd

from followup.mentions_en import (EnglishMentionExtractor, first_mention_form,
                                  guess_gender_from_role, mention_gender, _compatible)

ROOT = Path(__file__).resolve().parent.parent
ANNO = ROOT / "resources" / "annotations"
CB = ROOT / "resources" / "charbanks"


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def parse_pred_ids(val) -> set[str]:
    if pd.isna(val):
        return set()
    s = str(val).strip()
    if s in ("", "{}", "nan"):
        return set()
    try:
        d = ast.literal_eval(s)
        return {str(k) for k in d} if isinstance(d, dict) else set()
    except (ValueError, SyntaxError):
        return set()


def load_madeval_charbank() -> dict[str, list[dict]]:
    shim = types.ModuleType("pandas.core.indexes.numeric")
    shim.Int64Index = pd.Index
    shim.Float64Index = pd.Index
    sys.modules.setdefault("pandas.core.indexes.numeric", shim)
    with open(CB / "madeval_charbank.pkl", "rb") as f:
        d = pickle.load(f)
    out = {}
    for vid, df in d.items():
        r = df.iloc[0]
        out[vid] = [
            {"id": i, "role": ro, "name": na}
            for i, ro, na in zip(r["charbank_nmids"], r["charbank_roles"], r["charbank_names"])
        ]
    return out


def _thread_switch(prev_thread: str | None, cur_thread: str | None) -> int | None:
    """1 if the local thread topology changed between consecutive ADs, else 0.

    The thread column is per-AD local cluster topology, so equality is only a
    weak proxy (see evaluate_coherence.py docstring); kept as a feature, not a
    label.
    """
    if prev_thread is None or cur_thread is None:
        return None
    return int(str(prev_thread) != str(cur_thread))


# ----------------------------------------------------------------------------
# dataset loaders -> list of (seq_id, charbank, per-AD frame)
# ----------------------------------------------------------------------------
def _collapse(df: pd.DataFrame, seq_col: str, text_col: str) -> pd.DataFrame:
    """One row per (seq, anno_idx) with m-* pred_ids union and the first row's meta."""
    m = df[df["shot_label"].astype(str).str.startswith("m-")].copy()
    m["pids"] = m["pred_ids"].apply(parse_pred_ids)
    agg = {
        "pids": lambda s: set().union(*s) if len(s) else set(),
        text_col: "first",
        "start": "first",
        "end": "first",
    }
    for extra in ("thread", "shot_scale"):
        if extra in m.columns:
            agg[extra] = "first"
    g = m.groupby([seq_col, "anno_idx"], sort=False).agg(agg).reset_index()
    g = g.rename(columns={seq_col: "seq_id", text_col: "text"})
    return g.sort_values(["seq_id", "start", "anno_idx"]).reset_index(drop=True)


def load_madeval():
    df = pd.read_csv(ANNO / "madeval_anno_context-3.0-8.0_face-0.2-0.4_scale_thread.csv")
    g = _collapse(df, "movie", "sentence")
    g["dataset"] = "madeval"
    cb = load_madeval_charbank()
    return g, {vid: cb.get(vid, []) for vid in g["seq_id"].unique()}


def load_tvad():
    df = pd.read_csv(ANNO / "tvad_anno_context-9-24_face-0.2-0.4.csv")
    g = _collapse(df, "imdbid_pe", "text")
    g["dataset"] = "tvad"
    cb = json.load(open(CB / "tvad_charbank.json"))
    return g, {pe: cb.get(pe, []) for pe in g["seq_id"].unique()}


def load_cmdad():
    df = pd.read_csv(ANNO / "cmdad_anno_context-3.0-8.0_face-0.2-0.4_scale_thread.csv")
    g = _collapse(df, "cmd_filename", "text")
    g["dataset"] = "cmdad"
    imdb_of_clip = df.drop_duplicates("cmd_filename").set_index("cmd_filename")["imdbid"].to_dict()
    cb = json.load(open(CB / "cmdad_charbank.json"))
    return g, {clip: cb.get(imdb_of_clip.get(clip, ""), []) for clip in g["seq_id"].unique()}


LOADERS = {"madeval": load_madeval, "tvad": load_tvad, "cmdad": load_cmdad}


# ----------------------------------------------------------------------------
# labelling
# ----------------------------------------------------------------------------
def infer_gender(charbank: list[dict], ads: list[tuple[set[str], list]]) -> dict[str, str | None]:
    """Per-character gender: first-name prior, overridden by corpus pronoun votes.

    Votes: in every AD where a character is the ONLY detected face and the AD
    names nobody else, each 3sg pronoun votes for that character's gender.
    """
    prior = {str(c["id"]): guess_gender_from_role(c.get("role", "")) for c in charbank}
    votes: dict[str, dict[str, int]] = {}
    for pids, mentions in ads:
        if len(pids) != 1:
            continue
        if any(m.label == "NAME" and set(m.char_ids) - pids for m in mentions):
            continue
        cid = next(iter(pids))
        for m in mentions:
            g = mention_gender(m) if m.label == "PRN" else None
            if g:
                votes.setdefault(cid, {"M": 0, "F": 0})[g] += 1
    out = dict(prior)
    for cid, v in votes.items():
        tot = v["M"] + v["F"]
        if tot >= 3 and max(v.values()) / tot >= 0.8:
            out[cid] = "M" if v["M"] > v["F"] else "F"
    return out


def label_sequence(seq: pd.DataFrame, charbank: list[dict],
                   gender: dict[str, str | None] | None = None,
                   ad_rows: list[dict] | None = None) -> list[dict]:
    ext = EnglishMentionExtractor(charbank)
    ads = [((ad["pids"] or set()), ext.extract(ad["text"])) for _, ad in seq.iterrows()]
    if gender is None:
        gender = infer_gender(charbank, ads)
    rows = []
    prev_thread = None
    for (_, ad), (pids, mentions) in zip(seq.iterrows(), ads):
        cur_thread = ad.get("thread") if "thread" in ad else None
        tswitch = _thread_switch(prev_thread, cur_thread)
        prev_thread = cur_thread
        named_ids = sorted({c for m in mentions if m.label == "NAME"
                            for c in (set(m.char_ids) & pids)} if pids else set())
        if ad_rows is not None:
            ad_rows.append({
                "dataset": ad["dataset"], "seq_id": ad["seq_id"], "anno_idx": int(ad["anno_idx"]),
                "start": float(ad["start"]), "end": float(ad["end"]), "text": ad["text"],
                "n_chars": len(pids), "pred_ids": "|".join(sorted(pids)),
                "named_ids": "|".join(named_ids),
                "n_name": sum(m.label == "NAME" for m in mentions),
                "n_prn": sum(m.label == "PRN" for m in mentions),
                "n_desc": sum(m.label == "DESC" for m in mentions),
                "thread_switch": tswitch,
                "shot_scale": ad.get("shot_scale") if "shot_scale" in ad else None,
            })
        if not pids:
            continue
        for cid in sorted(pids):
            form = first_mention_form(mentions, cid, pids, gender)
            if form is not None:
                label, src = form, "bound"
            else:
                g_c = gender.get(cid)
                has_unbound = any(
                    m.label in ("PRN", "DESC") and _compatible(mention_gender(m), g_c)
                    for m in mentions
                )
                label = "AMBIG" if (has_unbound and len(pids) > 1) else "OMIT"
                src = "unbound" if has_unbound else "none"
            rows.append({
                "dataset": ad["dataset"],
                "seq_id": ad["seq_id"],
                "anno_idx": int(ad["anno_idx"]),
                "start": float(ad["start"]),
                "end": float(ad["end"]),
                "text": ad["text"],
                "n_chars": len(pids),
                "pred_ids": "|".join(sorted(pids)),
                "char_id": cid,
                "char_gender": gender.get(cid) or "",
                "label": label,
                "label_source": src,
                "n_name": sum(m.label == "NAME" for m in mentions),
                "n_prn": sum(m.label == "PRN" for m in mentions),
                "n_desc": sum(m.label == "DESC" for m in mentions),
                "thread_switch": tswitch,
                "shot_scale": ad.get("shot_scale") if "shot_scale" in ad else None,
            })
    return rows


def build(datasets: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_rows, ad_rows = [], []
    for name in datasets:
        g, cbs = LOADERS[name]()
        for seq_id, seq in g.groupby("seq_id", sort=False):
            all_rows.extend(label_sequence(seq, cbs.get(seq_id, []), ad_rows=ad_rows))
        print(f"[{name}] {g['seq_id'].nunique()} sequences, {len(g)} ADs -> {len(all_rows)} rows so far",
              file=sys.stderr)
    return pd.DataFrame(all_rows), pd.DataFrame(ad_rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["madeval", "tvad", "cmdad"])
    ap.add_argument("--out", default=str(ROOT / "results" / "followup" / "silver_labels.csv"))
    a = ap.parse_args()
    df, ads = build(a.datasets)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    ads.to_csv(Path(a.out).with_name("ads.csv"), index=False)
    print(df.groupby(["dataset", "label"]).size().unstack(fill_value=0).to_string())
