from __future__ import annotations
import re


def parse_into_steps(cot_text):
    """Split a CoT response into discrete reasoning steps."""
    if not cot_text or not cot_text.strip():
        return []

    # try numbered patterns first  (1. / Step 1: / (1) etc)
    parts = re.split(r"(?:^|\n)\s*(?:\d+[\.\):]|Step\s+\d+:?)", cot_text)
    parts = [s.strip() for s in parts if s.strip()]
    if len(parts) >= 2:
        return parts

    # transition words
    parts = re.split(
        r"(?:^|\.\s+)(First(?:ly)?|Second(?:ly)?|Third(?:ly)?|"
        r"Then|Next|After that|Finally|Therefore|Thus|Hence|So|"
        r"This means|As a result|Consequently)[,\s]",
        cot_text, flags=re.IGNORECASE
    )
    parts = [s.strip() for s in parts if s.strip() and len(s) > 20]
    if len(parts) >= 2:
        return parts

    # fall back to sentence splitting
    sents = re.split(r"(?<=[.!?])\s+", cot_text)
    return [s.strip() for s in sents if s.strip()]
