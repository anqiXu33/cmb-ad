import os
import nltk
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize


class CharacterMemoryBank:
    """
    Tracks character state and AD history across ADs for inter-shot coherence.
    One instance per movie inference run.

    Usage:
        bank = CharacterMemoryBank(char_gender=char_gender)

        # Before each AD generation:
        char_status, recent_history_str = bank.get_context(
            ad_idx, text_pred, pred_ids={"nm001": "Hans", "nm002": "Maria"}
        )

        # During chunk: proxy update with Stage-1 text
        bank.add_ad(text_pred)

        # After each chunk: replace proxy history with real LLM outputs
        bank.sync_history(real_ads)

    Args:
        char_gender: Optional dict {actor_imdb_id: 'male' | 'female' | None}
                     produced by char_utils.build_char_maps().
                     When provided, pronoun-ambiguity detection is gender-aware:
                     a character is flagged only if a same-gender competitor is
                     present in the current AD or the immediately preceding AD.
                     When absent, the bank is conservative — any character with
                     at least one competitor in the same or previous AD is
                     flagged regardless of gender.
        char_names:  Optional dict {actor_imdb_id: role_name}
                     produced by char_utils.build_char_maps().
                     When provided, char_status output shows role names
                     (e.g. "Hans [nm001]") instead of bare IDs.
        char_descriptions: Optional dict {actor_imdb_id: physical_description}
                     produced by generate_char_descriptions.py via VLM from
                     in-movie representative crops. When provided, any
                     "must-name" flag (REINTRODUCE / USE NAME — ambiguous /
                     USE NAME — anchor) is appended with the description so
                     the LLM can use it as an alternative to the character name.

    Flags emitted per character in char_status:
        [CONTINUING — pronoun ok]
            gap == 1 and no override flag. Pronoun is unambiguous.
        [REINTRODUCE — use name or description]
            gap > 1, or first appearance. Character needs re-introduction.
        [USE NAME — same-gender competitor present, pronoun is ambiguous]
            Another character of the same (or unknown) grammatical gender is
            present in the current AD or was present in the immediately
            preceding AD. Pronoun would be unresolvable for visually impaired
            listeners — either because the listener can't tell which on-screen
            character is meant, or because a recently-named off-screen
            same-gender antecedent is still active in their mental model.
            consecutive_ads counter resets to 0 (name-use period does not
            build a pronoun chain, so anchor should not fire immediately after).
        [USE NAME — memory anchor after N consecutive ADs]
            Character has appeared in N or more consecutive ADs with pronouns.
            Prevents indefinite pronoun chaining. Counter resets after firing.

    `Last known action: "..."` is only appended when gap > 1. For gap == 1
    the action is the same truncated text as `preceding_ad` (shown separately
    to the LLM), so duplicating it per-character would waste prompt tokens.
    """

    HISTORY_WINDOW = 3   # number of past ADs kept in recent_history
    ANCHOR_INTERVAL = 4  # re-anchor character name after this many consecutive ADs

    def __init__(self, char_gender: dict = None, char_names: dict = None,
                 char_descriptions: dict = None):
        # {char_id: {'last_seen_ad_idx': int, 'last_action': str, 'consecutive_ads': int}}
        self.characters = {}
        self.recent_history = []           # rolling list of last N ADs
        # Char IDs from the immediately preceding AD — used for cross-AD pronoun
        # ambiguity detection (a same-gender character recently named off-screen
        # remains a competing antecedent for "er" / "sie" in the current AD).
        self._prev_ad_char_ids: list = []
        self.char_gender = char_gender or {}
        self.char_names = char_names or {}  # {actor_id: role_name}
        self.char_descriptions = char_descriptions or {}  # {actor_id: physical_description}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _truncate_to_sentence(text: str, max_chars: int = 150) -> str:
        """Return the longest prefix of complete sentences within max_chars."""
        if len(text) <= max_chars:
            return text
        sentences = sent_tokenize(text)
        result = ""
        for sent in sentences:
            candidate = (result + " " + sent).strip() if result else sent
            if len(candidate) <= max_chars:
                result = candidate
            else:
                break
        return result if result else text[:max_chars]

    def _display_name(self, char_id: str) -> str:
        """Return 'RoleName [char_id]' if known, else just '[char_id]'."""
        name = self.char_names.get(str(char_id))
        return f"{name} [{char_id}]" if name else f"[{char_id}]"

    def _description_hint(self, char_id: str) -> str:
        """Return description hint for REINTRODUCE, or empty string.

        Set CMB_DISABLE_DESC_HINT=1 to force-disable for ablation (test B).
        """
        if os.getenv("CMB_DISABLE_DESC_HINT"):
            return ""
        desc = self.char_descriptions.get(str(char_id), "")
        return f' (appearance: "{desc}")' if desc else ""

    def _is_pronoun_ambiguous_for(self, char_id, current_ids: list,
                                  prev_ids: list) -> bool:
        """Return True if a pronoun for `char_id` would be ambiguous.

        Competitor set = (current_ids ∪ prev_ids) − {char_id}.

        Same-AD competitors model the within-shot case ("Hans and Karl
        argue — er …" → which one?).
        Previous-AD competitors model the cross-AD case ("AD N: Hans
        sits down. AD N+1: Karl reads — er …" → listener still has
        Hans as the active male antecedent).

        Conservative fallbacks (return True when uncertain):
          * `char_gender` map is empty (no gender info at all).
          * `char_id`'s own gender is unknown.
          * any competitor has unknown gender.
        Otherwise → True iff any competitor shares `char_id`'s gender.
        """
        competitors = set(str(c) for c in current_ids) | set(str(c) for c in prev_ids)
        competitors.discard(str(char_id))
        if not competitors:
            return False
        if not self.char_gender:
            return True  # no gender info → conservative
        my_gender = self.char_gender.get(str(char_id))
        if my_gender is None:
            return True  # own gender unknown → conservative
        for comp in competitors:
            comp_gender = self.char_gender.get(comp)
            if comp_gender is None or comp_gender == my_gender:
                return True
        return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_context(self, ad_idx: int, text_pred: str, char_ids: list):
        """
        Compute narrative context for the current AD, then update internal state.
        Call once per AD BEFORE generation.

        Args:
            ad_idx:    Monotonically increasing index for the current AD.
            text_pred: Stage-1 raw description for this AD.
            char_ids:  List of character ID strings from pred_ids keys.

        Returns:
            (char_status, recent_history_str)
            Each character in char_status carries its own continuity flag:
              [CONTINUING — pronoun ok]
              [REINTRODUCE — use name or description]
              [USE NAME — same-gender competitor present, pronoun is ambiguous]
              [USE NAME — memory anchor after N consecutive ADs]
        """
        prev_char_ids = self._prev_ad_char_ids
        status_parts = []

        if not char_ids:
            status_parts.append(
                "  - [EMPTY SHOT] No tracked characters — describe environment or objects."
            )

        for char_id in char_ids:
            display = self._display_name(char_id)

            if char_id in self.characters:
                gap = ad_idx - self.characters[char_id]["last_seen_ad_idx"]
                last_action = self.characters[char_id]["last_action"]
                prev_consecutive = self.characters[char_id]["consecutive_ads"]
                consecutive = prev_consecutive + 1 if gap == 1 else 1

                if self._is_pronoun_ambiguous_for(char_id, char_ids, prev_char_ids):
                    flag = "USE NAME — same-gender competitor present, pronoun is ambiguous"
                    consecutive_out = 0  # name-use period: no pronoun chain being built
                elif consecutive >= self.ANCHOR_INTERVAL:
                    flag = f"USE NAME — memory anchor after {consecutive} consecutive ADs"
                    consecutive_out = 0  # reset so next anchor fires after ANCHOR_INTERVAL more ADs
                elif gap == 1:
                    flag = "CONTINUING — pronoun ok"
                    consecutive_out = consecutive
                else:
                    flag = "REINTRODUCE — use name or description"
                    consecutive_out = 0  # name event: same as ambiguous/anchor — don't immediately re-anchor

                desc_hint = (
                    self._description_hint(char_id)
                    if ("REINTRODUCE" in flag or "USE NAME" in flag)
                    else ""
                )
                # Only emit `Last known action` when it adds info beyond the
                # separately-shown `preceding_ad`. When gap == 1 the two are
                # the same truncated text, so we drop the per-character line
                # to avoid duplicating tokens in the prompt.
                if gap > 1:
                    status_parts.append(
                        f"  - {display}: last seen {gap} AD(s) ago. [{flag}]{desc_hint}\n"
                        f'    Last known action: "{last_action}"'
                    )
                else:
                    status_parts.append(
                        f"  - {display}: last seen {gap} AD(s) ago. [{flag}]{desc_hint}"
                    )
                self.characters[char_id] = {
                    "last_seen_ad_idx": ad_idx,
                    "last_action": self._truncate_to_sentence(text_pred.strip()),
                    "consecutive_ads": consecutive_out,
                }
            else:
                desc_hint = self._description_hint(char_id)
                status_parts.append(
                    f"  - {display}: first appearance. [REINTRODUCE — use name if known]{desc_hint}"
                )
                self.characters[char_id] = {
                    "last_seen_ad_idx": ad_idx,
                    "last_action": self._truncate_to_sentence(text_pred.strip()),
                    # Name event: align with REINTRODUCE / ambiguous / anchor —
                    # the next CONTINUING AD starts the pronoun chain at 1.
                    "consecutive_ads": 0,
                }

        char_status = (
            "\n".join(status_parts)
            if status_parts
            else "  No tracked characters in this shot."
        )
        # Exclude the most recent AD: it is already shown to the LLM as `preceding_ad`.
        # Showing it again here would be redundant and inflate token count.
        older_history = self.recent_history[:-1]
        recent_history_str = (
            "\n".join(f"  [{i + 1}] {ad}" for i, ad in enumerate(older_history))
            if older_history
            else "  (None yet)"
        )
        # Carry over current char IDs for the next AD's cross-AD ambiguity check.
        # Empty AD ([EMPTY SHOT]) intentionally resets the carry — the listener's
        # pronoun antecedent fades when the narration switches to environment.
        self._prev_ad_char_ids = [str(c) for c in char_ids]
        return char_status, recent_history_str

    def add_ad(self, ad_text: str):
        """Push an AD into the rolling history window (proxy or real)."""
        if not ad_text:
            return
        self.recent_history.append(ad_text.strip())
        if len(self.recent_history) > self.HISTORY_WINDOW:
            self.recent_history = self.recent_history[-self.HISTORY_WINDOW:]

    def sync_history(self, real_ads: list):
        """Replace proxy history with real LLM outputs after a chunk."""
        real = [ad for ad in real_ads if ad]
        self.recent_history = real[-self.HISTORY_WINDOW:]

    def sync_last_actions(self, chunk_char_map: dict, chunk_ad_texts: dict):
        """Overwrite last_action with real LLM output for each character in the chunk.

        Args:
            chunk_char_map: {ad_idx: [char_id, ...]} built during Step 1.
            chunk_ad_texts: {ad_idx: str} real generated AD per AD index.
        """
        for ad_idx, char_ids in chunk_char_map.items():
            real_ad = chunk_ad_texts.get(ad_idx, "")
            if not real_ad:
                continue
            truncated = self._truncate_to_sentence(real_ad.strip())
            for char_id in char_ids:
                entry = self.characters.get(char_id)
                if entry and entry["last_seen_ad_idx"] == ad_idx:
                    entry["last_action"] = truncated
