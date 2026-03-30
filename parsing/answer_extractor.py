"""Extract final answers from model responses."""

from __future__ import annotations

import re


def extract_answer_gsm8k(response: str) -> str:
    """Extract numeric answer from a GSM8K response.

    Tries in order:
    1. #### {number} pattern
    2. 'answer is {number}' patterns
    3. Last number in response
    """
    if not response:
        return ""

    # Pattern 1: #### marker
    match = re.search(r"####\s*\$?([\d,]+(?:\.\d+)?)", response)
    if match:
        return match.group(1).replace(",", "")

    # Pattern 2: "answer/result/total is {number}"
    match = re.search(
        r"(?:answer|result|total|sum|value)\s*(?:is|=|:)\s*\$?([\d,]+(?:\.\d+)?)",
        response,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).replace(",", "")

    # Pattern 3: "= {number}" near end
    match = re.search(
        r"=\s*\$?([\d,]+(?:\.\d+)?)\s*$",
        response,
        re.MULTILINE,
    )
    if match:
        return match.group(1).replace(",", "")

    # Fallback: last number in response
    numbers = re.findall(r"\$?([\d,]+(?:\.\d+)?)", response)
    if numbers:
        return numbers[-1].replace(",", "")

    return ""


def extract_answer_arc(response: str, choice_labels: list[str] | None = None) -> str:
    """Extract letter choice from an ARC response.

    Tries in order:
    1. 'answer is (X)' patterns
    2. Standalone letter at end of response
    3. First mentioned valid choice label
    """
    if not response:
        return ""

    valid = set(choice_labels) if choice_labels else {"A", "B", "C", "D"}

    # Pattern 1: "answer/choice is X"
    match = re.search(
        r"(?:answer|choice|option)\s*(?:is|:)?\s*\(?([A-Z])\)?",
        response,
        re.IGNORECASE,
    )
    if match and match.group(1).upper() in valid:
        return match.group(1).upper()

    # Pattern 2: Standalone letter at end
    match = re.search(r"\b([A-Z])\b\s*[.!]?\s*$", response)
    if match and match.group(1).upper() in valid:
        return match.group(1).upper()

    # Pattern 3: First mentioned valid label
    for m in re.finditer(r"\b([A-Z])\b", response):
        if m.group(1).upper() in valid:
            return m.group(1).upper()

    return ""


def answers_match(answer1: str, answer2: str, dataset: str) -> bool:
    """Check if two extracted answers are equivalent."""
    if not answer1 or not answer2:
        return False
    if dataset == "gsm8k":
        try:
            return float(answer1) == float(answer2)
        except ValueError:
            return answer1.strip() == answer2.strip()
    else:
        return answer1.strip().upper() == answer2.strip().upper()
