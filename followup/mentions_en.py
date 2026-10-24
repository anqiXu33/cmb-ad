"""English NAME / DESC / PRN mention extractor (deterministic, spaCy-based).

Mirrors the label set used by the thesis mention-level suite
(scripts/eval/mention_level_eval.py) so English silver labels are comparable
with the German / Italian ones:

    NAME  a proper-name reference to a cast member (first name, surname,
          nickname, or the full role string); possessives count ("Rachel's").
    DESC  a definite / indefinite noun phrase whose head is a person noun
          ("the tall man", "a woman in red", "the officer").
    PRN   a 3rd-person singular personal / possessive / reflexive pronoun
          referring to a person (he, she, him, her, his, hers, himself,
          herself).  Plural "they/them" is excluded: it never maps to one
          tracked character.

Each mention carries the character id(s) it can be aligned to, where
alignment is lexical for NAME (via the per-title name lexicon built from the
charbank) and left to the caller for DESC / PRN (which need pred_ids).

Usage
-----
    ext = EnglishMentionExtractor(charbank)      # list of {id, role, name}
    ext.extract("Rachel sits next to him.")
    -> [Mention(label='NAME', text='Rachel', start=0, char_ids=['nm0000098']),
        Mention(label='PRN',  text='him',    start=20, char_ids=[])]
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import spacy

_NLP = None


def _nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    return _NLP


# Titles / ranks that precede a name in cast lists but are not names.
_TITLE_WORDS = {
    "mr", "mrs", "ms", "miss", "dr", "doctor", "sir", "lady", "lord", "captain",
    "capt", "lieutenant", "lt", "commander", "cmdr", "sergeant", "sgt", "officer",
    "coach", "counselor", "counsellor", "professor", "prof", "father", "mother",
    "uncle", "aunt", "king", "queen", "prince", "princess", "general", "colonel",
    "major", "private", "detective", "agent", "judge", "nurse", "reverend",
    "senator", "governor", "president", "mayor", "chief", "young", "old", "the",
    "of", "de", "van", "von", "jr", "sr", "ii", "iii", "self", "himself", "herself",
    "voice", "narrator", "boy", "girl", "man", "woman", "kid", "baby", "#1", "#2",
}

# Person nouns that head a DESC phrase. Deliberately broad; false positives
# are filtered later by requiring that the AD has an on-screen character.
_PERSON_NOUNS = {
    "man", "woman", "boy", "girl", "guy", "lady", "gentleman", "kid", "child",
    "teenager", "teen", "youth", "stranger", "figure", "person", "soldier",
    "officer", "cop", "policeman", "policewoman", "detective", "guard", "nurse",
    "doctor", "waiter", "waitress", "driver", "pilot", "priest", "monk", "nun",
    "teacher", "student", "coach", "player", "captain", "sergeant", "lieutenant",
    "general", "king", "queen", "prince", "princess", "witch", "wizard", "knight",
    "warrior", "hunter", "farmer", "worker", "clerk", "bartender", "customer",
    "passenger", "mother", "father", "mom", "dad", "son", "daughter", "brother",
    "sister", "husband", "wife", "boyfriend", "girlfriend", "friend", "neighbor",
    "neighbour", "couple", "blonde", "brunette", "redhead", "newcomer", "intruder",
    "attacker", "victim", "patient", "survivor", "marine", "corporal", "private",
    "agent", "senator", "governor", "candidate", "reporter", "journalist",
    "photographer", "cameraman", "assistant", "secretary", "boss", "manager",
    "roommate", "classmate", "stepfather", "stepmother", "grandfather",
    "grandmother", "grandma", "grandpa", "toddler", "infant", "baby", "twins",
}

_PRN_SG = {"he", "she", "him", "her", "his", "hers", "himself", "herself"}

_QUOTED_NICK = re.compile(r"[\"'“”‘’]([^\"'“”‘’]+)[\"'“”‘’]")


@dataclass
class Mention:
    label: str            # NAME | DESC | PRN
    text: str
    start: int            # character offset in the AD
    char_ids: list[str] = field(default_factory=list)  # NAME only; [] if unresolved
    head_lemma: str = ""  # DESC only


def _name_tokens(role: str) -> list[str]:
    """Split a cast-list role string into candidate name tokens."""
    if not role:
        return []
    toks: list[str] = []
    for nick in _QUOTED_NICK.findall(role):
        toks.extend(nick.split())
    cleaned = _QUOTED_NICK.sub(" ", role)
    cleaned = re.sub(r"\(.*?\)", " ", cleaned)
    for t in re.split(r"[\s\-/]+", cleaned):
        t = t.strip(".,'’\"")
        if not t or t.lower() in _TITLE_WORDS or len(t) < 2:
            continue
        if t.lower() in _PERSON_NOUNS:          # "The Waitress", "Guy in the Coffee House"
            continue
        if not t[0].isupper():
            continue
        toks.append(t)
    return toks


def build_name_lexicon(charbank: list[dict]) -> dict[str, set[str]]:
    """token(lower) -> set of char ids that own it (first names, surnames, nicks)."""
    lex: dict[str, set[str]] = {}
    for c in charbank:
        cid = str(c.get("id") or c.get("nmid") or "")
        if not cid:
            continue
        for t in _name_tokens(c.get("role") or ""):
            lex.setdefault(t.lower(), set()).add(cid)
    return lex


class EnglishMentionExtractor:
    def __init__(self, charbank: list[dict]):
        self.lex = build_name_lexicon(charbank)

    # ------------------------------------------------------------------
    def extract(self, text: str) -> list[Mention]:
        if not text or not isinstance(text, str):
            return []
        doc = _nlp()(text)
        mentions: list[Mention] = []
        used: set[int] = set()  # token indices consumed by a NAME

        # 1) NAME: maximal runs of consecutive tokens found in the lexicon
        i = 0
        while i < len(doc):
            tok = doc[i]
            key = tok.text.lower().rstrip("’'s") if tok.text.lower().endswith(("'s", "’s")) else tok.text.lower()
            if key in self.lex and tok.text[0].isupper() and (tok.i > 0 or tok.pos_ in {"PROPN", "NOUN", "X"}):
                j = i + 1
                ids = set(self.lex[key])
                while j < len(doc):
                    k2 = doc[j].text.lower()
                    if k2 in self.lex and (ids & self.lex[k2]):
                        ids &= self.lex[k2]
                        j += 1
                    else:
                        break
                span = doc[i:j]
                mentions.append(Mention("NAME", span.text, span.start_char, sorted(ids)))
                used.update(range(i, j))
                i = j
                continue
            i += 1

        # 2) PRN
        for tok in doc:
            if tok.i in used:
                continue
            low = tok.text.lower()
            if low in _PRN_SG and tok.pos_ in {"PRON", "DET"}:
                mentions.append(Mention("PRN", tok.text, tok.idx))

        # 3) DESC: noun chunks headed by a person noun, not containing a NAME
        for chunk in doc.noun_chunks:
            head = chunk.root
            if head.i in used or any(t.i in used for t in chunk):
                continue
            if head.tag_ == "NNS":          # plural ("the students") never maps to one character
                continue
            lemma = head.text.lower().rstrip("s") if head.text.lower() not in _PERSON_NOUNS else head.text.lower()
            if lemma in _PERSON_NOUNS or head.text.lower() in _PERSON_NOUNS:
                mentions.append(Mention("DESC", chunk.text, chunk.start_char, head_lemma=lemma))

        # 3b) fallback for person nouns the chunker mis-attached (small-model
        #     parse errors such as "the waitress watches" as one NP)
        covered = {(m.start, m.start + len(m.text)) for m in mentions if m.label == "DESC"}
        for tok in doc:
            if tok.i in used or tok.pos_ != "NOUN" or tok.tag_ != "NN":
                continue
            if tok.text.lower() not in _PERSON_NOUNS:
                continue
            if any(a <= tok.idx < b for a, b in covered):
                continue
            left = tok.i
            while left > 0 and doc[left - 1].pos_ in {"DET", "ADJ", "NUM"} and (left - 1) not in used:
                left -= 1
            span = doc[left:tok.i + 1]
            mentions.append(Mention("DESC", span.text, span.start_char, head_lemma=tok.text.lower()))

        mentions.sort(key=lambda m: m.start)
        return mentions


# ----------------------------------------------------------------------------
# gender helpers (M / F / None)
# ----------------------------------------------------------------------------
_PRN_GENDER = {"he": "M", "him": "M", "his": "M", "himself": "M",
               "she": "F", "her": "F", "hers": "F", "herself": "F"}

_NOUN_GENDER = {
    "man": "M", "boy": "M", "guy": "M", "gentleman": "M", "father": "M", "dad": "M",
    "son": "M", "brother": "M", "husband": "M", "boyfriend": "M", "king": "M",
    "prince": "M", "waiter": "M", "policeman": "M", "monk": "M", "priest": "M",
    "grandfather": "M", "grandpa": "M", "stepfather": "M", "wizard": "M",
    "woman": "F", "girl": "F", "lady": "F", "mother": "F", "mom": "F", "daughter": "F",
    "sister": "F", "wife": "F", "girlfriend": "F", "queen": "F", "princess": "F",
    "waitress": "F", "policewoman": "F", "nun": "F", "witch": "F", "grandmother": "F",
    "grandma": "F", "stepmother": "F", "blonde": "F", "brunette": "F",
}

_GG = None


def guess_gender_from_role(role: str) -> str | None:
    """M / F / None from the first name token of a cast-list role string."""
    global _GG
    toks = _name_tokens(role or "")
    if not toks:
        return None
    if _GG is None:
        try:
            import gender_guesser.detector as gd
            _GG = gd.Detector(case_sensitive=False)
        except ImportError:
            return None
    g = _GG.get_gender(toks[0])
    if g in ("male", "mostly_male"):
        return "M"
    if g in ("female", "mostly_female"):
        return "F"
    return None


def mention_gender(m: Mention) -> str | None:
    if m.label == "PRN":
        return _PRN_GENDER.get(m.text.lower())
    if m.label == "DESC":
        return _NOUN_GENDER.get(m.head_lemma)
    return None


def _compatible(g_mention: str | None, g_char: str | None) -> bool:
    return g_mention is None or g_char is None or g_mention == g_char


def first_mention_form(
    mentions: list[Mention],
    char_id: str,
    pred_ids: set[str],
    gender: dict[str, str | None] | None = None,
) -> str | None:
    """Reference form of the mention bound to `char_id` in this AD.

    Binding rules:
      NAME        bound if char_id is among its resolved ids and no other
                  on-screen char shares them. A bound NAME always wins.
      PRN / DESC  bound if, among on-screen characters that are NOT named
                  elsewhere in the AD, char_id is the only one whose gender
                  is compatible with the mention (unknown gender = compatible).
    Returns NAME | DESC | PRN, or None when nothing binds.
    """
    gender = gender or {}
    named: set[str] = set()
    for m in mentions:
        if m.label == "NAME":
            cands = set(m.char_ids) & pred_ids
            if len(cands) == 1:
                named |= cands
    if char_id in named:
        return "NAME"
    unnamed = pred_ids - named
    if char_id not in unnamed:
        return None
    for m in mentions:
        if m.label == "NAME":
            continue
        gm = mention_gender(m)
        compatible = {c for c in unnamed if _compatible(gm, gender.get(c))}
        if compatible == {char_id}:
            return m.label
    return None
