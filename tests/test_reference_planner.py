"""Tests for the ReferencePlanner refactor.

1. Unit tests of RulePlanner flags (the thesis rules, one case each).
2. Regression: the refactored CharacterMemoryBank must produce byte-identical
   char_status / recent_history output to the frozen thesis version
   (tests/_legacy_coherence_manager.py) on 300 random character streams.
3. A toy learned planner can be swapped in without touching the bank.

Run:  pytest tests/ -q
"""
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "stage2"))
sys.path.insert(0, str(ROOT / "tests"))

from coherence_manager import CharacterMemoryBank              # noqa: E402
from reference_planner import CharState, Decision, ReferencePlanner, RulePlanner  # noqa: E402
import _legacy_coherence_manager as legacy                      # noqa: E402

GENDER = {"a": "male", "b": "male", "c": "female", "d": None}
NAMES = {"a": "Hans", "b": "Karl", "c": "Maria", "d": "Unknown"}
DESCS = {"a": "der Dunkelhaarige", "c": "die Frau im roten Mantel"}


# ---------------------------------------------------------------------------
# 1. RulePlanner unit tests
# ---------------------------------------------------------------------------
class TestRulePlanner:
    def setup_method(self):
        self.p = RulePlanner(GENDER, anchor_interval=4)

    def test_first_appearance(self):
        d = self.p.decide("a", 0, None, ["a"], [])
        assert d.is_first and "use name if known" in d.flag and d.consecutive_out == 0

    def test_continuing_builds_chain(self):
        d = self.p.decide("a", 5, CharState(4, 2), ["a"], ["a"])
        assert d.flag.startswith("CONTINUING") and d.consecutive_out == 3 and d.gap == 1

    def test_anchor_fires_and_resets(self):
        d = self.p.decide("a", 5, CharState(4, 3), ["a"], ["a"])
        assert "memory anchor after 4" in d.flag and d.consecutive_out == 0

    def test_gap_reintroduces(self):
        d = self.p.decide("a", 7, CharState(4, 3), ["a"], [])
        assert d.flag.startswith("REINTRODUCE") and d.gap == 3 and d.consecutive_out == 0

    def test_same_gender_competitor_in_current_ad(self):
        d = self.p.decide("a", 5, CharState(4, 1), ["a", "b"], ["a"])
        assert d.flag.startswith("USE NAME — same-gender")

    def test_same_gender_competitor_in_previous_ad(self):
        d = self.p.decide("a", 5, CharState(4, 1), ["a"], ["b"])
        assert d.flag.startswith("USE NAME — same-gender")

    def test_other_gender_competitor_is_fine(self):
        d = self.p.decide("a", 5, CharState(4, 1), ["a", "c"], ["a"])
        assert d.flag.startswith("CONTINUING")

    def test_unknown_gender_is_conservative(self):
        d = self.p.decide("a", 5, CharState(4, 1), ["a", "d"], [])
        assert d.flag.startswith("USE NAME")

    def test_no_gender_map_is_conservative(self):
        p = RulePlanner({}, 4)
        d = p.decide("a", 5, CharState(4, 1), ["a", "c"], [])
        assert d.flag.startswith("USE NAME")

    def test_ambiguity_beats_anchor(self):
        d = self.p.decide("a", 5, CharState(4, 3), ["a", "b"], [])
        assert "same-gender" in d.flag


# ---------------------------------------------------------------------------
# 2. Regression against the frozen thesis implementation
# ---------------------------------------------------------------------------
def _random_stream(rng, n_ads=40):
    ids = list(GENDER)
    stream = []
    for _ in range(n_ads):
        k = rng.choice([0, 1, 1, 1, 2, 2, 3])
        stream.append(rng.sample(ids, k))
    return stream


def _run(bank_cls, stream, long_text=False):
    bank = bank_cls(char_gender=GENDER, char_names=NAMES, char_descriptions=DESCS)
    outs = []
    for i, ids in enumerate(stream):
        text = (f"Stage one description number {i}. " * (8 if long_text else 1)).strip()
        status, hist = bank.get_context(i, text, ids)
        outs.append((status, hist))
        bank.add_ad(f"AD {i}.")
        if i % 7 == 6:
            bank.sync_history([f"real AD {j}." for j in range(i + 1)])
            bank.sync_last_actions({i: ids}, {i: f"real AD {i} about {ids}."})
    return outs, bank.characters


@pytest.mark.parametrize("seed", range(300))
def test_refactor_matches_legacy(seed):
    rng = random.Random(seed)
    stream = _random_stream(rng)
    new, new_state = _run(CharacterMemoryBank, stream, long_text=(seed % 3 == 0))
    old, old_state = _run(legacy.CharacterMemoryBank, stream, long_text=(seed % 3 == 0))
    assert new == old
    assert new_state == old_state


def test_legacy_ambiguity_helper_still_available():
    bank = CharacterMemoryBank(char_gender=GENDER)
    assert bank._is_pronoun_ambiguous_for("a", ["a", "b"], []) is True
    assert bank._is_pronoun_ambiguous_for("a", ["a", "c"], []) is False


# ---------------------------------------------------------------------------
# 3. Swapping in another planner
# ---------------------------------------------------------------------------
class AlwaysNamePlanner(ReferencePlanner):
    """Baseline-2 behaviour expressed as a planner."""

    def decide(self, char_id, ad_idx, state, current_ids, prev_ids) -> Decision:
        if state is None:
            return Decision("REINTRODUCE — use name if known", 0, None)
        return Decision("USE NAME — always", 0, ad_idx - state.last_seen_ad_idx)


def test_custom_planner_is_used():
    bank = CharacterMemoryBank(char_gender=GENDER, char_names=NAMES, planner=AlwaysNamePlanner())
    bank.get_context(0, "t", ["a"])
    status, _ = bank.get_context(1, "t", ["a"])
    assert "[USE NAME — always]" in status
    assert bank.characters["a"]["consecutive_ads"] == 0
