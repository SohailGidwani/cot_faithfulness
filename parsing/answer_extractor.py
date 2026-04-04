from __future__ import annotations
import re


def extract_answer_gsm8k(text):
    if not text:
        return ""

    # #### marker
    m = re.search(r"####\s*\$?([\d,]+(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "")

    # "answer is 42" style
    m = re.search(
        r"(?:answer|result|total|sum|value)\s*(?:is|=|:)\s*\$?([\d,]+(?:\.\d+)?)",
        text, re.IGNORECASE)
    if m:
        return m.group(1).replace(",", "")

    # equals sign near end of a line
    m = re.search(r"=\s*\$?([\d,]+(?:\.\d+)?)\s*$", text, re.MULTILINE)
    if m:
        return m.group(1).replace(",", "")

    # last number in the whole response
    nums = re.findall(r"\$?([\d,]+(?:\.\d+)?)", text)
    if nums:
        return nums[-1].replace(",", "")
    return ""


def extract_answer_arc(text, choice_labels=None):
    if not text:
        return ""

    valid = set(choice_labels) if choice_labels else {"A", "B", "C", "D"}

    m = re.search(r"(?:answer|choice|option)\s*(?:is|:)?\s*\(?([A-Z])\)?",
                  text, re.IGNORECASE)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()

    # standalone letter at the very end
    m = re.search(r"\b([A-Z])\b\s*[.!]?\s*$", text)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()

    for hit in re.finditer(r"\b([A-Z])\b", text):
        if hit.group(1).upper() in valid:
            return hit.group(1).upper()

    return ""


def answers_match(a, b, dataset):
    if not a or not b:
        return False
    if dataset == "gsm8k":
        try:
            return float(a) == float(b)
        except ValueError:
            return a.strip() == b.strip()
    return a.strip().upper() == b.strip().upper()
