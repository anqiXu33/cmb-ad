"""Reference-form planners: the decision layer of the Character Memory Bank.

The CMB separates two concerns that the thesis version kept in one method:

  * *state*  (who was seen when, running pronoun chains, last actions):
    owned by CharacterMemoryBank in coherence_manager.py
  * *decision* (given that state, which reference form should this
    character get in this AD): owned by a ReferencePlanner.

Any planner that implements `decide()` can be dropped into the bank, so the
rule planner from the thesis and future learned planners share one interface
and one evaluation path.

Flags (unchanged from the thesis; the prompt builder keys on these strings):
    "CONTINUING — pronoun ok"
    "REINTRODUCE — use name or description"
    "REINTRODUCE — use name if known"            (first appearance)
    "USE NAME — same-gender competitor present, pronoun is ambiguous"
    "USE NAME — memory anchor after {n} consecutive ADs"
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class CharState:
    """What the bank knows about one character before the current AD."""
    last_seen_ad_idx: int
    consecutive_ads: int


@dataclass(frozen=True)
class Decision:
    flag: str
    consecutive_out: int   # value the bank stores back as consecutive_ads
    gap: int | None        # ADs since last appearance; None on first appearance

    @property
    def is_first(self) -> bool:
        return self.gap is None

    @property
    def needs_description_hint(self) -> bool:
        return "REINTRODUCE" in self.flag or "USE NAME" in self.flag


class ReferencePlanner(ABC):
    """Decide the reference form for one character in one AD."""

    @abstractmethod
    def decide(
        self,
        char_id: str,
        ad_idx: int,
        state: CharState | None,
        current_ids: list[str],
        prev_ids: list[str],
    ) -> Decision:
        """`state` is None on the character's first appearance."""


class RulePlanner(ReferencePlanner):
    """The thesis rule planner (Ch. 3 / Ch. 4), verbatim logic.

    Priority: ambiguity > anchor > continuing > reintroduce.
    """

    def __init__(self, char_gender: dict | None = None, anchor_interval: int = 4):
        self.char_gender = char_gender or {}
        self.anchor_interval = anchor_interval

    # -- ambiguity -------------------------------------------------------
    def is_pronoun_ambiguous_for(self, char_id, current_ids: list, prev_ids: list) -> bool:
        """True if a pronoun for `char_id` would be ambiguous.

        Competitor set = (current ∪ previous) − {char_id}. Conservative when
        gender is missing anywhere (see thesis §4.2).
        """
        competitors = set(str(c) for c in current_ids) | set(str(c) for c in prev_ids)
        competitors.discard(str(char_id))
        if not competitors:
            return False
        if not self.char_gender:
            return True
        my_gender = self.char_gender.get(str(char_id))
        if my_gender is None:
            return True
        for comp in competitors:
            comp_gender = self.char_gender.get(comp)
            if comp_gender is None or comp_gender == my_gender:
                return True
        return False

    # -- decision --------------------------------------------------------
    def decide(self, char_id, ad_idx, state, current_ids, prev_ids) -> Decision:
        if state is None:
            return Decision("REINTRODUCE — use name if known", 0, None)
        gap = ad_idx - state.last_seen_ad_idx
        consecutive = state.consecutive_ads + 1 if gap == 1 else 1
        if self.is_pronoun_ambiguous_for(char_id, current_ids, prev_ids):
            return Decision("USE NAME — same-gender competitor present, pronoun is ambiguous", 0, gap)
        if consecutive >= self.anchor_interval:
            return Decision(f"USE NAME — memory anchor after {consecutive} consecutive ADs", 0, gap)
        if gap == 1:
            return Decision("CONTINUING — pronoun ok", consecutive, gap)
        return Decision("REINTRODUCE — use name or description", 0, gap)
