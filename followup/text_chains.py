"""Text-only reference chains: silver labels and features WITHOUT face recognition.

The human AD transcript is treated as one continuous document per sequence.
Mentions are resolved to characters from the text alone:

    NAME  lexicon match against the cast list (ties broken by recency)
    PRN   most recent mention, within the last MAX_LOOKBACK ADs, of a
          character with compatible gender; a NAME earlier in the same AD wins
    DESC  bound ONLY when its content words occur in exactly one cast member's
          role string ("the witch" -> "Horned Witch"). Recency binding was
          tried and dropped: on a 30-item check about two thirds were wrong,
          because describers use "the driver" precisely for someone OTHER than
          the protagonist just mentioned. Indefinite ("a man"), relational
          ("her brother"), non-individual ("the couple") descriptions and
          "a X ... the X" anaphors of a new person are never bound.
          Consequence: the surviving DESC labels are almost all role-named
          minor characters (Farmer, Co-Pilot), i.e. NOT a genuine name-vs-
          description choice. Treat DESC results with that in mind.

Unresolved PRN / DESC mentions are dropped (counted in the report). The
resolver is the classic recency + gender-agreement heuristic (Hobbs-style),
which is also the listener's working-memory model: the competing antecedent
is whoever was *heard* most recently, not whoever is on screen.

Output rows: one per (AD, character mentioned in it), label = form of the
character's first mention in that AD, plus text-only discourse features.
Face-based pred_ids are deliberately NOT used anywhere here, so this file can
later run on the 214 Swiss scripts, REFRAMED, or any AD transcript.

Run:  python -m followup.text_chains
Out:  results/followup/text_features.csv, text_rule_preds.csv
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from followup.mentions_en import (EnglishMentionExtractor, Mention, guess_gender_from_role,
                                  mention_gender, _compatible)
from followup.silver_labels import LOADERS, _thread_switch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "followup"

MAX_LOOKBACK = 4      # ADs a pronoun may reach back
DESC_LOOKBACK = 2     # ADs a description may reach back
INDEF_BLOCK = 3       # "a man" ... "the man" within this many ADs = the new person

# --- description filters (precision over recall) ---------------------------
# A description only becomes a silver label when it re-describes a character
# that is already in the listener's mind. Everything below is dropped:
_INDEF = {"a", "an", "another", "some", "one", "two", "three", "several"}
_NON_INDIVIDUAL = {"couple", "twins", "siblings", "crowd", "family", "group", "people",
                   "kids", "children", "baby", "infant", "toddler", "pair", "trio", "team"}
_KINSHIP = {"mother", "mom", "father", "dad", "son", "daughter", "brother", "sister",
            "husband", "wife", "boyfriend", "girlfriend", "grandmother", "grandma",
            "grandfather", "grandpa", "stepfather", "stepmother", "uncle", "aunt",
            "cousin", "nephew", "niece", "fiance", "fiancee", "parents"}


_DESC_STOP = {"the", "a", "an", "his", "her", "their", "its", "young", "old", "man", "woman",
              "guy", "boy", "girl", "lady", "person", "figure", "mr", "mrs", "ms", "dr", "of",
              "and", "s", "little", "big", "tall", "short", "other", "first", "second", ""}


def _desc_tokens(m: Mention) -> list[str]:
    return [t.strip(".,'’\"").lower() for t in m.text.split()]


def desc_filter_reason(m: Mention, i: int, indef_heads: list[tuple[int, str]]) -> str | None:
    """Why a DESC mention must NOT be bound to a known character (None = may bind)."""
    toks = _desc_tokens(m)
    head = m.head_lemma or (toks[-1] if toks else "")
    if toks and toks[0] in _INDEF:
        return "indefinite"                 # introduces a new person
    if any(t in _NON_INDIVIDUAL for t in toks):
        return "non_individual"
    if head in _KINSHIP or any(t in _KINSHIP for t in toks):
        return "relational"                 # "her brother", "Pete's mom"
    if any(i - j <= INDEF_BLOCK and h == head for j, h in indef_heads):
        return "anaphor_of_new"             # "a man ..." -> "the man ..."
    return None
FORMS = ["NAME", "DESC", "PRN", "none"]
ANCHOR_INTERVAL = 4


# ----------------------------------------------------------------------------
# gender (text only): first-name prior + votes from "NAME ... PRN" in one AD
# ----------------------------------------------------------------------------
def text_gender(charbank: list[dict], ad_mentions: list[list[Mention]]) -> dict[str, str | None]:
    prior = {str(c["id"]): guess_gender_from_role(c.get("role", "")) for c in charbank}
    votes: dict[str, Counter] = {}
    for ms in ad_mentions:
        names = [m for m in ms if m.label == "NAME" and len(m.char_ids) == 1]
        if len(names) != 1:
            continue
        cid = names[0].char_ids[0]
        for m in ms:
            if m.label == "PRN" and m.start > names[0].start:
                g = mention_gender(m)
                if g:
                    votes.setdefault(cid, Counter())[g] += 1
    out = dict(prior)
    for cid, v in votes.items():
        tot = sum(v.values())
        if tot >= 3 and max(v.values()) / tot >= 0.8:
            out[cid] = v.most_common(1)[0][0]
    return out


# ----------------------------------------------------------------------------
# resolution
# ----------------------------------------------------------------------------
def resolve_sequence(seq: pd.DataFrame, charbank: list[dict]):
    ext = EnglishMentionExtractor(charbank)
    role_of = {str(c["id"]): c.get("role") or "" for c in charbank}
    role_toks = {cid: {w.strip(".,'’\"()").lower() for w in r.replace("-", " ").split()} - _DESC_STOP
                 for cid, r in role_of.items()}
    ad_mentions = [ext.extract(t) for t in seq["text"]]
    gender = text_gender(charbank, ad_mentions)

    last_mention_ad: dict[str, int] = {}     # char -> AD index of last mention
    rows, stats = [], Counter()
    prev_thread = None
    history: list[tuple[int, str]] = []       # (ad_index, char) mention events in order
    indef_heads: list[tuple[int, str]] = []   # (ad_index, head noun) of "a X" introductions

    for i, (ad, ms) in enumerate(zip(seq.itertuples(index=False), ad_mentions)):
        cur_thread = getattr(ad, "thread", None)
        tswitch = _thread_switch(prev_thread, cur_thread)
        prev_thread = cur_thread
        bound: dict[str, str] = {}           # char -> first form in this AD
        span: dict[str, tuple[int, str]] = {} # char -> (start, text) of that mention
        resolved_in_ad: list[str] = []

        for m in ms:
            cid = None
            if m.label == "NAME":
                cands = m.char_ids
                if len(cands) == 1:
                    cid = cands[0]
                elif cands:
                    cid = max(cands, key=lambda c: last_mention_ad.get(c, -1))
                stats["NAME"] += 1
            elif m.label == "DESC":
                reason = desc_filter_reason(m, i, indef_heads)
                toks = _desc_tokens(m)
                if toks and toks[0] in _INDEF:
                    indef_heads.append((i, m.head_lemma or toks[-1]))
                if reason:
                    stats[f"DESC_filtered_{reason}"] += 1
                    continue
                # Recency does NOT work for descriptions: describers use "the
                # driver" precisely for someone other than the protagonist just
                # mentioned. Bind only when the description's content words
                # occur in exactly one cast member's role string ("the witch"
                # -> "Horned Witch", "the staff sergeant" -> "SSgt. ... ").
                content = {t for t in toks if t not in _DESC_STOP}
                hits = [c for c, rt in role_toks.items()
                        if content and (content & rt) and c not in resolved_in_ad]
                if len(hits) == 1:
                    cid = hits[0]
                stats["DESC_resolved" if cid else "DESC_unresolved"] += 1
            else:
                g = mention_gender(m)
                # 1) a compatible character already resolved earlier in this AD
                same_ad = [c for c in resolved_in_ad if _compatible(g, gender.get(c))]
                if same_ad:
                    cid = same_ad[0]
                else:
                    # 2) most recent compatible character in the last MAX_LOOKBACK ADs
                    for j, c in reversed(history):
                        if i - j > MAX_LOOKBACK:
                            break
                        if _compatible(g, gender.get(c)):
                            cid = c
                            break
                stats[f"{m.label}_resolved" if cid else f"{m.label}_unresolved"] += 1
            if cid is None:
                continue
            if cid not in bound:
                bound[cid] = m.label
                span[cid] = (m.start, m.text)
            resolved_in_ad.append(cid)

        for cid, form in bound.items():
            rows.append({"dataset": ad.dataset, "seq_id": ad.seq_id, "anno_idx": int(ad.anno_idx),
                         "start": float(ad.start), "end": float(ad.end), "text": ad.text,
                         "char_id": cid, "char_gender": gender.get(cid) or "", "label": form,
                         "m_start": span[cid][0], "m_text": span[cid][1],
                         "char_role": role_of.get(cid, ""),
                         "n_mentioned_cur": len(bound), "thread_switch": tswitch,
                         "shot_scale": getattr(ad, "shot_scale", None)})
        for cid in bound:
            history.append((i, cid))
            last_mention_ad[cid] = i
    return rows, stats


# ----------------------------------------------------------------------------
# features + rule replay on the text stream
# ----------------------------------------------------------------------------
def build_features(rows: pd.DataFrame, ads: pd.DataFrame) -> pd.DataFrame:
    """Text-only discourse features, computed sequentially per sequence."""
    feats = []
    rule = []
    for (ds, seq), g in ads.groupby(["dataset", "seq_id"], sort=False):
        r = rows[(rows.dataset == ds) & (rows.seq_id == seq)]
        by_ad = {a: grp for a, grp in r.groupby("anno_idx")}
        gender = dict(zip(r.char_id, r.char_gender))
        state: dict[str, dict] = {}
        prev_chars: set[str] = set()
        rule_state: dict[str, dict] = {}
        recent: list[set[str]] = []           # chars mentioned in each of the last ADs
        for i, ad in enumerate(g.itertuples(index=False)):
            grp = by_ad.get(ad.anno_idx)
            cur = set(grp.char_id) if grp is not None else set()
            for c in sorted(cur):
                st = state.get(c)
                gc = gender.get(c, "")
                last3 = set().union(*recent[-3:]) if recent else set()
                between = st["others_since"] if st else 0
                f = {
                    "dataset": ds, "seq_id": seq, "anno_idx": ad.anno_idx, "char_id": c,
                    "gap_ads": (i - st["last_idx"]) if st else -1,
                    "gap_sec": (ad.start - st["last_end"]) if st else -1.0,
                    "first_in_seq": int(st is None),
                    "consecutive": (st["consecutive"] if st and st["last_idx"] == i - 1 else 0),
                    "ad_dur": ad.end - ad.start,
                    "n_mentioned_cur": len(cur),
                    "n_mentioned_prev": len(prev_chars),
                    "n_same_gender_cur": sum(1 for o in cur if o != c and gc and gender.get(o, "") == gc),
                    "n_same_gender_last3": sum(1 for o in last3 if o != c and gc and gender.get(o, "") == gc),
                    "n_other_mentioned_between": between,
                    "competitor_in_prev": int(bool(prev_chars - {c})),
                    "self_in_prev": int(c in prev_chars),
                    "gender_known": int(bool(gc)),
                    "n_prior_mentions": st["n"] if st else 0,
                    "n_prior_name": st["n_name"] if st else 0,
                    "n_prior_prn": st["n_prn"] if st else 0,
                    "thread_switch": ad.thread_switch if hasattr(ad, "thread_switch") else None,
                    "shot_scale": getattr(ad, "shot_scale", None),
                }
                f["log_gap_sec"] = np.log1p(f["gap_sec"]) if f["gap_sec"] >= 0 else -1.0
                f["name_ratio_prior"] = f["n_prior_name"] / f["n_prior_mentions"] if f["n_prior_mentions"] else -1.0
                pf = st["last_form"] if st else "none"
                for form in FORMS:
                    f[f"prev_form_{form}"] = int(pf == form)
                feats.append(f)

                # thesis rule replayed with "present" := "mentioned"
                rs = rule_state.get(c)
                comp = (cur | prev_chars) - {c}
                amb = bool(comp) and (not gc or any((not gender.get(o, "")) or gender.get(o) == gc for o in comp))
                if rs is None:
                    flag, cons_out = "FIRST", 0
                else:
                    gap = i - rs["last"]
                    cons = rs["consec"] + 1 if gap == 1 else 1
                    if amb:
                        flag, cons_out = "USE_NAME_AMBIG", 0
                    elif cons >= ANCHOR_INTERVAL:
                        flag, cons_out = "USE_NAME_ANCHOR", 0
                    elif gap == 1:
                        flag, cons_out = "CONTINUING", cons
                    else:
                        flag, cons_out = "REINTRODUCE", 0
                rule_state[c] = {"last": i, "consec": cons_out}
                rule.append({"dataset": ds, "seq_id": seq, "anno_idx": ad.anno_idx, "char_id": c,
                             "rule_flag": flag, "rule_pred": "PRN" if flag == "CONTINUING" else "NAME"})

            # update state
            if grp is not None:
                forms = dict(zip(grp.char_id, grp.label))
                for c in cur:
                    st = state.setdefault(c, {"last_idx": -99, "last_end": 0.0, "consecutive": 0,
                                               "n": 0, "n_name": 0, "n_prn": 0, "last_form": "none",
                                               "others_since": 0})
                    st["consecutive"] = st["consecutive"] + 1 if st["last_idx"] == i - 1 else 1
                    st["last_idx"], st["last_end"] = i, ad.end
                    st["n"] += 1
                    st["n_name"] += int(forms[c] == "NAME")
                    st["n_prn"] += int(forms[c] == "PRN")
                    st["last_form"] = forms[c]
                    st["others_since"] = 0
                for c, st in state.items():
                    if c not in cur:
                        st["others_since"] += len(cur)
            prev_chars = cur
            recent.append(cur)
    F = pd.DataFrame(feats)
    R = pd.DataFrame(rule)
    keep = [c for c in rows.columns if c not in ("thread_switch", "shot_scale", "n_mentioned_cur")]
    df = rows[keep].merge(F, on=["dataset", "seq_id", "anno_idx", "char_id"], how="left")
    return df, R


TEXT_FEATURE_COLS = [
    "gap_ads", "gap_sec", "log_gap_sec", "first_in_seq", "consecutive", "ad_dur",
    "n_mentioned_cur", "n_mentioned_prev", "n_same_gender_cur", "n_same_gender_last3",
    "n_other_mentioned_between", "competitor_in_prev", "self_in_prev", "gender_known",
    "n_prior_mentions", "n_prior_name", "n_prior_prn", "name_ratio_prior",
    "thread_switch", "shot_scale",
] + [f"prev_form_{f}" for f in FORMS]

# Syntactic position of the labelled mention in its AD sentence. Known when the
# sentence exists (training on human AD, the classic referential-form-selection
# setting); a planner that runs BEFORE generation does not know it, and should
# instead output one recommendation per position (see followup/role_eval.py).
ROLES = ["subj", "obj", "poss", "pobj", "other"]
ROLE_COLS = [f"role_{r}" for r in ROLES]
TEXT_FEATURE_COLS = TEXT_FEATURE_COLS + ROLE_COLS


def mention_role(doc, start: int, length: int) -> str:
    span = doc.char_span(start, start + length, alignment_mode="expand")
    if span is None:
        return "other"
    t = span.root
    if t.dep_ == "conj":                     # "Raj and Penny walk": Penny shares Raj's role
        while t.dep_ == "conj" and t.head is not t:
            t = t.head
    dep = t.dep_
    if dep in ("nsubj", "nsubjpass"):
        return "subj"
    if dep in ("dobj", "dative", "iobj"):
        return "obj"
    if dep == "poss" or span.root.tag_ == "PRP$":
        return "poss"
    if dep == "pobj":
        return "pobj"
    return "other"


def add_syntactic_role(df: pd.DataFrame) -> pd.DataFrame:
    from followup.mentions_en import _nlp
    nlp = _nlp()
    roles = []
    for doc, a, m in zip(nlp.pipe(df["text"].astype(str), batch_size=512), df["m_start"], df["m_text"]):
        roles.append(mention_role(doc, int(a), len(str(m))))
    df = df.copy()
    df["role"] = roles
    for r in ROLES:
        df[f"role_{r}"] = (df["role"] == r).astype(int)
    return df


def main(datasets=("madeval", "tvad", "cmdad")):
    all_rows, all_ads, stats = [], [], Counter()
    for name in datasets:
        g, cbs = LOADERS[name]()
        for seq_id, seq in g.groupby("seq_id", sort=False):
            rows, st = resolve_sequence(seq, cbs.get(seq_id, []))
            all_rows.extend(rows)
            stats.update(st)
        ads = g[["dataset", "seq_id", "anno_idx", "start", "end", "text"] +
                [c for c in ("thread", "shot_scale") if c in g.columns]].copy()
        ads["thread_switch"] = ads.groupby("seq_id")["thread"].transform(
            lambda s: (s.astype(str) != s.astype(str).shift()).astype(float).where(s.shift().notna())
        ) if "thread" in ads.columns else None
        all_ads.append(ads)
        print(f"[{name}] rows so far {len(all_rows)}", file=sys.stderr)
    rows = pd.DataFrame(all_rows)
    ads = pd.concat(all_ads, ignore_index=True)
    df, R = build_features(rows, ads)
    df = add_syntactic_role(df)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "text_features.csv", index=False)
    ads.to_csv(OUT / "text_ads.csv", index=False)
    R.to_csv(OUT / "text_rule_preds.csv", index=False)
    print("resolution:", dict(stats))
    print(df.groupby(["dataset", "label"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
