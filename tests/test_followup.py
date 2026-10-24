"""Smoke tests for the follow-up modules (English extractor, Italian zero subjects)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from followup.mentions_en import EnglishMentionExtractor, first_mention_form  # noqa: E402
from followup.zero_subject_it import self_test                                 # noqa: E402

CB = [{"id": "r", "role": "Rachel Green"}, {"id": "j", "role": "Joey Tribbiani"},
      {"id": "w", "role": "The Waitress"}]


def test_english_labels():
    ext = EnglishMentionExtractor(CB)
    m = ext.extract("Rachel sits next to him and the waitress smiles.")
    assert [(x.label, x.text) for x in m] == [("NAME", "Rachel"), ("PRN", "him"), ("DESC", "the waitress")]


def test_name_beats_pronoun_and_gender_blocks_binding():
    ext = EnglishMentionExtractor(CB)
    m = ext.extract("She opens the door and faces Joey.")
    gender = {"j": "M", "r": "F"}
    assert first_mention_form(m, "j", {"j"}, gender) == "NAME"
    assert first_mention_form(m, "r", {"r"}, gender) == "PRN"
    assert first_mention_form(m, "j", {"j", "r"}, gender) == "NAME"
    assert first_mention_form(m, "r", {"j", "r"}, gender) == "PRN"   # only F candidate


def test_italian_zero_subject_selftest():
    ok, n = self_test(verbose=False)
    assert ok == n


def test_parse_srt_and_cast():
    from followup.reframed import build_cast, parse_srt, screenplay_names
    srt = "1\n00:00:01,500 --> 00:00:03,000\n<i>Anna</i> sits.\n\n2\n00:01:00,000 --> 00:01:02,250\nShe\nsmiles.\n"
    assert parse_srt(srt) == [(1.5, 3.0, "Anna sits."), (60.0, 62.25, "She smiles.")]
    assert parse_srt("null") == []
    names = screenplay_names('{"1": "<character>ANNA (V.O.)</character><character>ANNA SMITH</character>"}')
    cast = build_cast(names)
    assert [c["role"] for c in cast] == ["Anna Smith"]
