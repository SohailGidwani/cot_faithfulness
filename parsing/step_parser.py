"""Parse Chain-of-Thought text into discrete reasoning steps."""

from __future__ import annotations

import re


def parse_into_steps(cot_text: str) -> list[str]:
    """Parse CoT into discrete reasoning steps.

    Uses a hierarchy of strategies:
    1. Numbered markers (1., Step 1:, (1))
    2. Transition words (First, Then, Next, Therefore, Finally, ...)
    3. Fallback: sentence boundaries
    """
    if not cot_text or not cot_text.strip():
        return []

    # Strategy 1: Numbered patterns
    numbered = re.split(r"(?:^|\n)\s*(?:\d+[\.\):]|Step\s+\d+:?)", cot_text)
    numbered = [s.strip() for s in numbered if s.strip()]
    if len(numbered) >= 2:
        return numbered

    # Strategy 2: Transition words
    transitions = re.split(
        r"(?:^|\.\s+)(First(?:ly)?|Second(?:ly)?|Third(?:ly)?|"
        r"Then|Next|After that|Finally|Therefore|Thus|Hence|So|"
        r"This means|As a result|Consequently)[,\s]",
        cot_text,
        flags=re.IGNORECASE,
    )
    transitions = [s.strip() for s in transitions if s.strip() and len(s) > 20]
    if len(transitions) >= 2:
        return transitions

    # Strategy 3: Sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", cot_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences
