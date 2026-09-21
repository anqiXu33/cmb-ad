"""
Swiss-specific Stage-2 prompt builder.

Prompt structure (for German/Italian single mode):
  [1. TASK]       — one-line task instruction
  [2. INPUT]      — Description + Narrative Context (history, preceding AD, character status)
  [3. COHERENCE]  — language-specific character referencing rules
  [4. STYLE]      — AD writing constraints (grouped)
  [5. OUTPUT]     — JSON template, word limit, few-shot examples

French and assistant-mode prompts are kept simpler (no CMB integration).
"""


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

_STYLE_RULES = (
    "STYLE RULES:\n"
    "- Focus on characters and their actions, supplemented by key objects.\n"
    "- For actions: avoid mentioning the camera; do not focus on 'talking'.\n"
    "- For objects (especially without characters): describe concrete, specific ones.\n"
    "- Do not mention characters' mood.\n"
    "- Narrator perspective. Stay neutral; do not patronize the audience.\n"
    "- Never use words that indicate seeing ('We see..', '..is visible').\n"
    "- Do not add information not present in the input.\n"
    "- Use present tense. Use short, simple sentences with standard syntax.\n"
    "- If on-screen text is plot-relevant (or credits), include its content.\n"
    "- Do not mention 'Shot number'.\n"
)

_STYLE_RULES_DE = (
    _STYLE_RULES
    + "- Avoid long composite words.\n"
    + "- Write in German, use Swiss spelling.\n"
)

_STYLE_RULES_IT = (
    _STYLE_RULES
    + "- Write in Italian, use Swiss spelling.\n"
)


# ---------------------------------------------------------------------------
# Simplified coherence blocks for the --no_cmb baseline (B1)
# ---------------------------------------------------------------------------
# Used when CharacterMemoryBank is disabled. The model has no char_status,
# no recent_history, and no PRIORITY rules. It only sees the Description and
# the preceding AD, so the coherence block can only give general guidance on
# how to refer to characters mentioned in the Description.

_COHERENCE_DE_NOCMB = (
    "CHARACTER REFERENCING:\n"
    "  - If a character's name is mentioned in the Description, use that name (drop titles).\n"
    "  - Otherwise describe with a vivid visual feature (hair, clothing, age).\n"
    "    Avoid generic terms like 'ein Mann' or 'eine Frau'.\n"
    "  - Use pronouns ('Er' / 'Sie') only when the reference is unambiguous from the preceding AD.\n"
)

_COHERENCE_IT_NOCMB = (
    "CHARACTER REFERENCING:\n"
    "  - If a character's name is mentioned in the Description, use that name (drop titles).\n"
    "  - Otherwise describe with a vivid visual feature (hair, clothing, age).\n"
    "    Avoid generic terms like 'un uomo' or 'una donna'.\n"
    "  - Italian is pro-drop: do NOT write 'Lui' / 'Lei'. Start directly with the conjugated verb when continuing.\n"
)


def _build_input_block(text_pred, preceding_ad, recent_history, char_status):
    """Build the unified INPUT section with description and narrative context."""
    parts = [
        f"INPUT:\n"
        f"  Description: {text_pred}\n"
    ]
    # Narrative context — only shown when CMB is active
    if char_status or recent_history:
        parts.append(
            "\n  --- NARRATIVE CONTEXT ---\n"
            f"  [Preceding AD (avoid repeating)]: '{preceding_ad}'\n"
            f"  [Recent AD History (last 3)]:\n{recent_history}\n"
            f"  [Character Tracking Status]:\n{char_status}\n"
            "  --------------------------\n"
        )
    elif preceding_ad:
        parts.append(f"\n  Preceding AD (avoid repeating): '{preceding_ad}'\n")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Language-specific coherence rules
# ---------------------------------------------------------------------------

_COHERENCE_DE = (
    "PRIORITY — Character Tracking Status is AUTHORITATIVE for character identity.\n"
    "  The Description is free-text from a VLM and may disagree with the Status\n"
    "  block. When they conflict, follow Status:\n"
    "    (a) Character listed in Status but NOT mentioned in the Description:\n"
    "        Stage-1 missed them. Do NOT invent actions — omit them from the AD.\n"
    "    (b) Character mentioned in the Description but NOT listed in Status:\n"
    "        Face detection missed them. Describe with a visual feature (hair,\n"
    "        clothing, age) — never invent a name. Applies even if Status is\n"
    "        [EMPTY SHOT].\n"
    "    (c) The Description names a character differently from Status\n"
    "        (e.g. wrong person, wrong color attribution):\n"
    "        Prefer the name/role from the Status block.\n"
    "    (d) Character appears in both with matching reference:\n"
    "        Apply the flag-specific rule 1–4 below.\n"
    "\n"
    "CHARACTER REFERENCING RULES (apply per character based on their flag above):\n"
    "  1. [CONTINUING — pronoun ok]:\n"
    "     Use the correct German pronoun ('Er' / 'Sie').\n"
    "     Example: Anna [CONTINUING] -> 'Sie beginnt zu kochen.'\n"
    "\n"
    "  2. [REINTRODUCE] or [first appearance]:\n"
    "     Re-introduce with their name OR a vivid physical description.\n"
    "     If an (appearance: \"...\") hint is shown, prefer that phrase.\n"
    "     Avoid generic terms like 'ein Mann' or 'eine Frau'.\n"
    "     Example: Paul [REINTRODUCE] (appearance: \"der Dunkelhaarige\") -> 'Der Dunkelhaarige steht auf.' or 'Paul steht auf.'\n"
    "\n"
    "  3. [USE NAME] (highest priority — overrides Rule 1):\n"
    "     3a. Pronoun ambiguity (2+ characters, same gender):\n"
    "         Use each character's name for every action.\n"
    "         Example: Paul, Thomas [USE NAME] -> 'Paul greift nach dem Messer, während Thomas aufsteht.'\n"
    "     3b. Memory anchor (after N consecutive ADs):\n"
    "         Re-state the name ONCE, then resume pronouns.\n"
    "         Example: Laura [USE NAME — anchor] -> 'Laura beginnt zu kochen.'\n"
    "\n"
    "  4. [EMPTY SHOT]:\n"
    "     No characters tracked in Status. If the Description also has no person,\n"
    "     describe environment, objects, or on-screen text; do NOT use pronouns\n"
    "     or invent named characters. If the Description clearly shows a person,\n"
    "     fall back to PRIORITY rule (b): visual feature, no invented name.\n"
    "\n"
    "  General: use first names (drop titles like 'Mr.', 'Dr.').\n"
)

_COHERENCE_IT = (
    "PRIORITY — Character Tracking Status is AUTHORITATIVE for character identity.\n"
    "  The Description is free-text from a VLM and may disagree with the Status\n"
    "  block. When they conflict, follow Status:\n"
    "    (a) Character listed in Status but NOT mentioned in the Description:\n"
    "        Stage-1 missed them. Do NOT invent actions — omit them from the AD.\n"
    "    (b) Character mentioned in the Description but NOT listed in Status:\n"
    "        Face detection missed them. Describe with a visual feature (hair,\n"
    "        clothing, age) — never invent a name. Applies even if Status is\n"
    "        [EMPTY SHOT].\n"
    "    (c) The Description names a character differently from Status\n"
    "        (e.g. wrong person, wrong color attribution):\n"
    "        Prefer the name/role from the Status block.\n"
    "    (d) Character appears in both with matching reference:\n"
    "        Apply the flag-specific rule 1–4 below.\n"
    "\n"
    "CHARACTER REFERENCING RULES (apply per character based on their flag above):\n"
    "  1. [CONTINUING — pronoun ok]:\n"
    "     Italian is pro-drop: DO NOT write 'Lui' or 'Lei'.\n"
    "     Start directly with the conjugated verb.\n"
    "     Example: Laura [CONTINUING] -> 'Inizia a cucinare.' (NOT 'Lei inizia a cucinare.')\n"
    "\n"
    "  2. [REINTRODUCE] or [first appearance]:\n"
    "     Re-introduce with name or a descriptive feature.\n"
    "     If an (appearance: \"...\") hint is shown, prefer that phrase.\n"
    "     Avoid generic terms like 'un uomo' or 'una donna'. Do not invent names.\n"
    "     Example: Paul [REINTRODUCE] (appearance: \"l'uomo dai capelli scuri\") -> 'L\\'uomo dai capelli scuri entra in cucina.' or 'Paul entra in cucina.'\n"
    "\n"
    "  3. [USE NAME] (highest priority — overrides Rule 1):\n"
    "     3a. Pronoun ambiguity (2+ characters, same gender):\n"
    "         Use each character's name for every action.\n"
    "         Example: Marco, Luca [USE NAME] -> 'Marco afferra il coltello mentre Luca si alza.'\n"
    "     3b. Memory anchor (after N consecutive ADs):\n"
    "         Re-state the name ONCE, then resume pro-drop.\n"
    "         Example: Laura [USE NAME — anchor] -> 'Laura inizia a cucinare.'\n"
    "\n"
    "  4. [EMPTY SHOT]:\n"
    "     No characters tracked in Status. If the Description also has no person,\n"
    "     describe environment, objects, or on-screen text; do NOT use pronouns\n"
    "     or invent named characters. If the Description clearly shows a person,\n"
    "     fall back to PRIORITY rule (b): visual feature, no invented name.\n"
    "\n"
    "  General: use first names (drop titles like 'Mr.', 'Dr.').\n"
)


def _build_output_block(template, word_limit, example_sentence):
    """Build the OUTPUT section with template, word limit, and examples."""
    return (
        f"OUTPUT:\n"
        f"  JSON format: {template}\n"
        f"  Maximum length: {word_limit} words.\n"
        f"  Example outputs:\n{example_sentence}"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_user_prompt(
    mode,
    prompt_idx,
    verb_list,
    text_pred,
    word_limit,
    examples,
    language="German",
    preceding_ad="",
    char_status="",
    recent_history="",
    enable_cmb=True,
):
    """
    Build a user prompt for Swiss AD generation.

    Args:
        mode:           "single" or "assistant".
        prompt_idx:     Prompt variant index (0 = default).
        verb_list:      List of target motion verbs (may be None).
        text_pred:      Stage-1 raw description string.
        word_limit:     Approximate target word count.
        examples:       List of few-shot GT AD strings.
        language:       "German", "Italian", or "French".
        preceding_ad:   The immediately preceding generated AD sentence.
        char_status:    Character tracking status string from CharacterMemoryBank.
        recent_history: Recent history string from CharacterMemoryBank.
        enable_cmb:     When False (Baseline 1 ablation), use the simplified
                        coherence block instead of the full CMB-aware rules.
                        char_status and recent_history are ignored.
    """
    text_pred = f'"{text_pred.strip()}"'

    # Format few-shot examples
    example_sentence = ""
    for selected_example in examples:
        example_sentence += '  {"summarized_AD": "' + f"{selected_example}" + '"}\n'

    if mode == "single":
        template = '{"summarized_AD": ""}'

        # [1. TASK]
        task = f"Summarize the following movie clip description into ONE succinct audio description (AD) sentence in {language}.\n\n"

        # [2. INPUT]
        # When CMB is disabled, drop char_status and recent_history so the
        # input block degrades to "Description + Preceding AD" only.
        if enable_cmb:
            input_block = _build_input_block(text_pred, preceding_ad, recent_history, char_status)
        else:
            input_block = _build_input_block(text_pred, preceding_ad, "", "")

        if language == "German":
            # [3. COHERENCE]  [4. STYLE]  [5. OUTPUT]
            coherence_block = _COHERENCE_DE if enable_cmb else _COHERENCE_DE_NOCMB
            user_prompt = (
                task
                + input_block + "\n"
                + coherence_block + "\n"
                + _STYLE_RULES_DE + "\n"
                + _build_output_block(template, word_limit, example_sentence)
            )

        elif language == "Italian":
            coherence_block = _COHERENCE_IT if enable_cmb else _COHERENCE_IT_NOCMB
            user_prompt = (
                task
                + input_block + "\n"
                + coherence_block + "\n"
                + _STYLE_RULES_IT + "\n"
                + _build_output_block(template, word_limit, example_sentence)
            )

        elif language == "French":
            # French: no CMB integration, simpler coherence rules
            french_coherence = (
                "CHARACTER RULES:\n"
                "  Use first names (drop titles like 'Mr.', 'Dr.').\n"
                "  If a name is not available, use a vivid description "
                "(e.g., 'la fille aux cheveux bruns', 'l\\'homme à la barbe'). "
                "Avoid generic 'un homme' or 'une femme'.\n"
                "  Use a character's name if NOT already in the preceding AD; "
                "otherwise use a pronoun. Examples:\n"
                "    'Laura enters and cooks.' + preceding: 'Laura entre dans la cuisine.' "
                "-> 'Elle commence à cuisiner.'\n"
                "    'Laura enters and cooks.' + preceding: 'Paul entre dans la cuisine.' "
                "-> 'Laura entre dans la cuisine et commence à cuisiner.'\n"
            )
            french_style = (
                _STYLE_RULES
                + "- Write in French, use Swiss spelling.\n"
            )
            input_block_fr = (
                f"INPUT:\n"
                f"  Description: {text_pred}\n"
                f"\n  Preceding AD (avoid repeating): '{preceding_ad}'\n"
            )
            user_prompt = (
                task
                + input_block_fr + "\n"
                + french_coherence + "\n"
                + french_style + "\n"
                + _build_output_block(template, word_limit, example_sentence)
            )

        else:
            raise ValueError(
                f"swiss_promptloader: unsupported language '{language}'. "
                "Use 'German', 'Italian', or 'French'."
            )

    else:  # assistant mode — no CMB, 5 candidates
        template = (
            '{"summarized_AD_1": "",\n"summarized_AD_2": "",\n'
            '"summarized_AD_3": "",\n"summarized_AD_4": "",\n"summarized_AD_5": ""}'
        )
        user_prompt = (
            "Summarize the following movie clip description into ONE succinct AD sentence.\n"
            f"Description: {text_pred}\n\n"
            "Focus on characters, their actions, and key objects.\n"
            "Use first names (drop 'Mr.', 'Dr.'). If unavailable, use pronouns ('He', 'her'); "
            "avoid 'a man' or 'a woman'.\n"
            "Avoid mentioning the camera or focusing on 'talking'.\n"
            "Do not mention mood. Do not hallucinate.\n"
            f"Identify these motions (decreasing priority): {verb_list}, and use them.\n"
            "Provide 5 distinct ADs from a narrator perspective, each emphasizing "
            "different characters, actions, or movements.\n\n"
            f"Output template (JSON): {template}\n"
            f"Max {word_limit} words per AD.\n"
            "Examples:\n"
            f"{example_sentence}"
        )

    return user_prompt
