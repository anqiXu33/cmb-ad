"""Load REFRAMED (Sterner et al. 2026; CC BY-NC 4.0) into the text-chain format.

Dataset: https://huggingface.co/datasets/igorsterner/reframed
Each row is one movie clip with SRT strings for the US / UK audio description,
the subtitles, the SDH subtitles, and (for 85 movies) an aligned screenplay.
The test split is encrypted and is not used here.

Produces, per movie (seq_id = imdb_id):
  * an AD table in the same shape followup.text_chains expects
    (dataset, seq_id, anno_idx, start, end, text, clip),
  * a dialogue table (seq_id, clip, start, end, text) from the subtitles,
  * a cast list built WITHOUT IMDb: screenplay <character> names, plus person
    names that spaCy NER finds at least twice in the movie's AD or subtitles.

Time: SRT times are clip-relative. If `video_start` parses as seconds, clips
are placed on one movie timeline (video_start + t); otherwise clips are laid
out in their sort order with a large gap between them. `load()` prints which
case applied, because downstream gap features depend on it.

Export from Hugging Face (run where the hub is reachable, e.g. Kaggle):
    python -m followup.reframed --export results/followup/reframed.parquet
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "followup"

_TS = r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})"
_SRT_TIME = re.compile(_TS + r"\s*-->\s*" + _TS)
_TAG = re.compile(r"<[^>]+>|\{[^}]+\}")
_CLIP_GAP = 100_000.0     # seconds between clips when no movie offset is known


def _sec(h, m, s, ms) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000


def parse_srt(srt) -> list[tuple[float, float, str]]:
    """SRT string -> [(start_sec, end_sec, text)], tolerant to missing indices."""
    if not isinstance(srt, str) or not srt.strip() or srt.strip() == "null":
        return []
    out, cur = [], None
    for line in srt.replace("\r", "").split("\n"):
        mt = _SRT_TIME.search(line)
        if mt:
            if cur and cur[2]:
                out.append((cur[0], cur[1], " ".join(cur[2]).strip()))
            g = mt.groups()
            cur = [_sec(*g[:4]), _sec(*g[4:]), []]
            continue
        if cur is None:
            continue
        s = line.strip()
        if not s:
            if cur[2]:
                out.append((cur[0], cur[1], " ".join(cur[2]).strip()))
            cur = None
            continue
        if s.isdigit() and not cur[2]:
            continue
        cur[2].append(_TAG.sub("", s))
    if cur and cur[2]:
        out.append((cur[0], cur[1], " ".join(cur[2]).strip()))
    return [(a, b, t) for a, b, t in out if t]


# ----------------------------------------------------------------------------
# cast list without IMDb
# ----------------------------------------------------------------------------
_CHAR_TAG = re.compile(r"<character>\s*(.*?)\s*</character>", re.I | re.S)
_PAREN = re.compile(r"\(.*?\)")
_NOT_NAMES = {"man", "woman", "boy", "girl", "voice", "narrator", "both", "all", "everyone",
              "crowd", "guard", "officer", "doctor", "nurse", "waiter", "waitress", "driver",
              "cop", "kid", "mom", "dad", "mother", "father", "the", "a", "an", "and"}


def _norm_name(raw: str) -> str:
    s = _PAREN.sub(" ", raw)
    s = re.sub(r"\b(V\.?O\.?|O\.?S\.?|CONT'D|CONT’D|CONTD)\b", " ", s, flags=re.I)
    s = re.sub(r"[^A-Za-z' \-.]", " ", s)
    s = " ".join(w.capitalize() if w.isupper() or w.islower() else w for w in s.split())
    return s.strip(" .-'")


def screenplay_names(screenplay) -> Counter:
    c = Counter()
    if not isinstance(screenplay, str) or screenplay.strip() in ("", "null"):
        return c
    try:
        scenes = json.loads(screenplay)
        blob = " ".join(str(v) for v in (scenes.values() if isinstance(scenes, dict) else scenes))
    except (ValueError, TypeError):
        blob = screenplay
    for raw in _CHAR_TAG.findall(blob):
        n = _norm_name(raw)
        if n and n.split()[0].lower() not in _NOT_NAMES and len(n) > 1:
            c[n] += 1
    return c


def ner_names(texts: list[str], min_count: int = 2) -> Counter:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])
    except OSError:
        return Counter()
    c = Counter()
    for doc in nlp.pipe(texts, batch_size=256):
        for e in doc.ents:
            if e.label_ == "PERSON":
                n = _norm_name(e.text.replace("’s", "").replace("'s", ""))
                if n and n.split()[0].lower() not in _NOT_NAMES:
                    c[n] += 1
    return Counter({k: v for k, v in c.items() if v >= min_count})


def build_cast(names: Counter) -> list[dict]:
    """Merge 'John' into 'John Smith' when unambiguous; return charbank dicts."""
    full = sorted(names, key=lambda n: (-len(n.split()), -names[n]))
    keep: dict[str, str] = {}                   # name -> canonical
    for n in full:
        toks = n.split()
        if len(toks) == 1:
            owners = [k for k in keep if keep[k] == k and toks[0] in k.split()]
            if len(owners) == 1:
                keep[n] = owners[0]
                continue
        keep[n] = n
    canon = sorted({v for v in keep.values()})
    return [{"id": "c:" + c.lower().replace(" ", "_"), "role": c} for c in canon]


# ----------------------------------------------------------------------------
# main loader
# ----------------------------------------------------------------------------
def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load(path, ad_column: str = "audio_description_us", use_ner: bool = True, verbose=True):
    """Return (ads, dialogue, charbanks) built from an exported parquet / csv."""
    df = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)
    df["seq_id"] = df["imdb_id"].astype(str)
    df["vstart_sec"] = df["video_start"].map(_f)
    has_offset = df["vstart_sec"].notna().mean() > 0.9
    if verbose:
        print(f"[reframed] {len(df)} clips, {df.seq_id.nunique()} movies, "
              f"movie-timeline offsets: {'video_start' if has_offset else 'NONE, clips laid out by order'}",
              file=sys.stderr)
    df = df.sort_values(["seq_id", "vstart_sec" if has_offset else "video_id"]).reset_index(drop=True)

    ad_rows, dl_rows, charbanks = [], [], {}
    for seq, g in df.groupby("seq_id", sort=False):
        names = Counter()
        for sp in g.get("screenplay", pd.Series(dtype=str)):
            names.update(screenplay_names(sp))
        texts = []
        for k, r in enumerate(g.itertuples(index=False)):
            off = r.vstart_sec if has_offset and r.vstart_sec is not None else k * _CLIP_GAP
            for a, b, t in parse_srt(getattr(r, ad_column)):
                ad_rows.append({"dataset": "reframed", "seq_id": seq, "clip": r.video_id,
                                "start": off + a, "end": off + b, "text": t})
                texts.append(t)
            for a, b, t in parse_srt(r.subtitles):
                dl_rows.append({"seq_id": seq, "clip": r.video_id, "start": off + a, "end": off + b, "text": t})
                texts.append(t)
        if use_ner:
            names.update(ner_names(texts))
        charbanks[seq] = build_cast(names)

    ads = pd.DataFrame(ad_rows).sort_values(["seq_id", "start"]).reset_index(drop=True)
    ads["anno_idx"] = ads.groupby("seq_id").cumcount()
    dialogue = pd.DataFrame(dl_rows).sort_values(["seq_id", "start"]).reset_index(drop=True)
    if verbose:
        n_cast = pd.Series({k: len(v) for k, v in charbanks.items()})
        print(f"[reframed] {len(ads)} AD lines, {len(dialogue)} subtitle lines, "
              f"cast size median {n_cast.median():.0f} (min {n_cast.min()}, max {n_cast.max()})",
              file=sys.stderr)
    return ads, dialogue, charbanks


def export(out_path: str, splits=("train", "validation")):
    from datasets import load_dataset
    parts = [load_dataset("igorsterner/reframed", split=s).to_pandas() for s in splits]
    df = pd.concat(parts, ignore_index=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"exported {len(df)} rows -> {out_path}; columns: {list(df.columns)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", help="download train+validation from the hub to this parquet path")
    a = ap.parse_args()
    if a.export:
        export(a.export)
