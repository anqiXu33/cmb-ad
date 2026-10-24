"""Italian null-subject (pro-drop) detector.

Why: the thesis prompt tells the model to drop 'Lui/Lei' in Italian, so a
correct pronominal reference surfaces as a finite verb with no overt subject.
The mention extractor only counts overt tokens, so Italian PRN rates come out
at 5–9 % while human AD sits at 24–39 %. This module makes the implicit
pronoun countable.

Rule (spaCy it_core_news_sm dependency parse):
    a finite verb (or copular AUX head) is a ZERO-SUBJECT mention if
      * it is 3rd person singular (Person=3, Number=Sing), and
      * it has no nsubj / nsubj:pass / csubj child, and
      * it is not impersonal / weather-like / existential ("piove", "c'è"),
        and not inside a relative clause whose head noun is its subject, and
      * the sentence has no other finite verb that already carries the
        subject in a coordination ("Maria entra e si siede" counts once:
        the second verb shares Maria's subject via conj).

Each hit is returned as Mention(label="PRN", text="∅<verb>", ...), so it can
be merged with overt mentions and bound to a character by the same gender
logic (gender is unknown for a zero subject, i.e. compatible with anyone;
agreement on a predicate adjective / participle, when present, gives M/F).

Run self-test:   python -m followup.zero_subject_it
"""
from __future__ import annotations

from dataclasses import dataclass, field

import spacy

_NLP = None


def _nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("it_core_news_sm", disable=["ner"])
    return _NLP


_IMPERSONAL = {"piovere", "nevicare", "bisognare", "sembrare", "parere", "bastare",
               "esserci", "esistere", "accadere", "succedere", "capitare", "occorrere"}
_SUBJ_DEPS = {"nsubj", "nsubj:pass", "csubj", "csubj:pass"}


@dataclass
class ZeroMention:
    label: str
    text: str
    start: int
    gender: str | None = None
    verb: str = ""
    char_ids: list = field(default_factory=list)
    head_lemma: str = ""


def _finite_3sg(m) -> bool:
    return "Person=3" in m and "Number=Sing" in m and "VerbForm=Fin" in m


def _is_3sg(tok) -> bool:
    if _finite_3sg(tok.morph):
        return True
    # participle / adjective predicate carried by a finite 3sg auxiliary or copula
    if "VerbForm=Part" in tok.morph or tok.pos_ == "ADJ":
        return any(ch.dep_ in {"aux", "aux:pass", "cop"} and _finite_3sg(ch.morph)
                   for ch in tok.children)
    return False


def _has_overt_subject(tok) -> bool:
    for ch in tok.children:
        if ch.dep_ in _SUBJ_DEPS:
            return True
    # relative clause: "l'uomo che entra" -> subject is the antecedent
    if tok.dep_ in {"acl:relcl", "acl"}:
        return True
    # shared subject via coordination: look at the conj head chain
    head = tok
    while head.dep_ == "conj" and head.head is not head:
        head = head.head
        for ch in head.children:
            if ch.dep_ in _SUBJ_DEPS:
                return True
    return False


def _predicate_gender(tok) -> str | None:
    """Gender from an agreeing participle / predicate adjective, if any."""
    cands = [tok] + [c for c in tok.children if c.dep_ in {"xcomp", "cop", "aux", "aux:pass"}]
    cands += [c for c in tok.children if c.pos_ == "ADJ" and c.dep_ in {"xcomp", "advcl", "amod"}]
    for c in cands:
        m = c.morph
        if ("VerbForm=Part" in m or c.pos_ == "ADJ") and "Number=Sing" in m:
            if "Gender=Masc" in m:
                return "M"
            if "Gender=Fem" in m:
                return "F"
    return None


def detect_zero_subjects(text: str) -> list[ZeroMention]:
    if not text or not isinstance(text, str):
        return []
    doc = _nlp()(text)
    # One-word AD sentences ("Sorride.") are often mis-tagged PROPN because of
    # the capital; re-parse lower-cased when the root is a lone PROPN.
    if len(doc) <= 2 and doc[0].pos_ == "PROPN":
        doc2 = _nlp()(text[0].lower() + text[1:])
        if doc2[0].pos_ in {"VERB", "AUX"}:
            doc = doc2
    out: list[ZeroMention] = []
    for sent in doc.sents:
        for tok in sent:
            if tok.pos_ not in {"VERB", "AUX"}:
                continue
            if tok.dep_ in {"aux", "aux:pass", "cop"}:
                continue
            if not _is_3sg(tok):
                continue
            if tok.lemma_.lower() in _IMPERSONAL:
                continue
            # impersonal / passive "si" ("si vede un treno"): skip. A reflexive
            # "si" ("si alza") is kept: it still has a dropped human subject.
            if any(ch.lower_ == "si" and ch.dep_ in {"expl:impers", "expl:pass"} for ch in tok.children):
                continue
            if any(ch.lower_ == "si" and ch.dep_ == "expl" for ch in tok.children) and \
               any(ch.dep_ in {"nsubj", "nsubj:pass"} for ch in tok.children):
                continue
            if _has_overt_subject(tok):
                continue
            # second verb of a coordination with a dropped subject shares it:
            # "Prende il telefono e compone un numero" is one zero subject.
            if tok.dep_ == "conj" and tok.head.pos_ in {"VERB", "AUX"} and _is_3sg(tok.head):
                continue
            out.append(ZeroMention("PRN", f"∅{tok.text}", tok.idx,
                                   gender=_predicate_gender(tok), verb=tok.lemma_))
    return out


# ---------------------------------------------------------------------------
# self-test: hand-built cases in the register of Italian AD
# ---------------------------------------------------------------------------
_CASES = [
    # (sentence, expected number of zero subjects)
    ("Inizia a cucinare.", 1),
    ("Si alza e esce dalla stanza.", 1),           # coordination shares one dropped subject
    ("Apre la porta.", 1),
    ("Guarda fuori dalla finestra, poi si volta.", 1),
    ("È seduta al tavolo.", 1),                    # copula + agreeing participle (F)
    ("Marco afferra il coltello mentre Luca si alza.", 0),
    ("Laura inizia a cucinare.", 0),
    ("La donna dai capelli scuri entra in cucina.", 0),
    ("L'uomo che entra in cucina sorride.", 0),    # relative clause, antecedent is subject
    ("Piove sulla città.", 0),                     # weather verb
    ("C'è una lettera sul tavolo.", 0),            # existential
    ("Si vede un treno in lontananza.", 0),        # si-passive
    ("Le due donne camminano lungo il fiume.", 0), # plural
    ("Un treno attraversa il paesaggio innevato.", 0),
    ("Prende il telefono e compone un numero.", 1),
    ("Sorride.", 1),
    ("Si guardano negli occhi.", 0),               # plural reciprocal
    ("Il sole tramonta dietro le montagne.", 0),
    ("Corre verso la macchina, sale e parte.", 1),
    ("Lei inizia a cucinare.", 0),                 # overt pronoun -> not a zero subject
]


def self_test(verbose: bool = True) -> tuple[int, int]:
    ok = 0
    for sent, exp in _CASES:
        got = detect_zero_subjects(sent)
        hit = len(got) == exp
        ok += hit
        if verbose:
            mark = "ok " if hit else "XX "
            print(f"{mark} exp={exp} got={len(got):d}  {sent}   {[ (m.text, m.gender) for m in got]}")
    if verbose:
        print(f"\n{ok}/{len(_CASES)} cases correct")
    return ok, len(_CASES)


if __name__ == "__main__":
    self_test()
