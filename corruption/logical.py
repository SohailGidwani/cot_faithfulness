"""Rule-based logical/factual corruption for ARC-Challenge reasoning steps."""

from __future__ import annotations

import re
import random


# Antonym pairs for negation
_ANTONYM_PAIRS = [
    ("increases", "decreases"),
    ("increase", "decrease"),
    ("expands", "contracts"),
    ("expand", "contract"),
    ("heats", "cools"),
    ("heat", "cool"),
    ("rises", "falls"),
    ("rise", "fall"),
    ("absorbs", "reflects"),
    ("absorb", "reflect"),
    ("attracts", "repels"),
    ("attract", "repel"),
    ("accelerates", "decelerates"),
    ("faster", "slower"),
    ("larger", "smaller"),
    ("more", "less"),
    ("higher", "lower"),
    ("hotter", "colder"),
    ("greater", "lesser"),
    ("positive", "negative"),
    ("solid", "liquid"),
    ("liquid", "gas"),
    ("heavy", "light"),
    ("strong", "weak"),
    ("bright", "dim"),
    ("thick", "thin"),
    ("long", "short"),
    ("above", "below"),
    ("before", "after"),
    ("true", "false"),
]

# Build bidirectional lookup
_ANTONYM_MAP = {}
for a, b in _ANTONYM_PAIRS:
    _ANTONYM_MAP[a.lower()] = b
    _ANTONYM_MAP[b.lower()] = a


def negate_conclusion(step: str, rng: random.Random | None = None) -> str:
    """Negate a conclusion by adding/removing 'not' or swapping antonyms."""
    if rng is None:
        rng = random.Random()

    # Strategy 1: Swap antonyms
    words = step.split()
    swappable = [(i, w) for i, w in enumerate(words) if w.lower().rstrip(".,;:!?") in _ANTONYM_MAP]
    if swappable:
        idx, word = rng.choice(swappable)
        clean = word.lower().rstrip(".,;:!?")
        suffix = word[len(clean):]
        replacement = _ANTONYM_MAP[clean]
        # Preserve capitalization
        if word[0].isupper():
            replacement = replacement.capitalize()
        words[idx] = replacement + suffix
        return " ".join(words)

    # Strategy 2: Insert or remove 'not'
    # Remove existing 'not'
    if " not " in step:
        return step.replace(" not ", " ", 1)

    # Insert 'not' after a verb-like word (is, are, was, were, does, do, can, will)
    verb_pattern = r"\b(is|are|was|were|does|do|can|will|would|could|should)\b"
    match = re.search(verb_pattern, step, re.IGNORECASE)
    if match:
        insert_pos = match.end()
        return step[:insert_pos] + " not" + step[insert_pos:]

    return step


def introduce_factual_error(step: str, rng: random.Random | None = None) -> str:
    """Change numbers or physical properties to introduce a factual error."""
    if rng is None:
        rng = random.Random()

    # Strategy 1: Perturb numbers (temperatures, distances, percentages)
    pattern = r"(\d+(?:\.\d+)?)\s*(°[CF]|degrees|percent|%|km|m|cm|mm|kg|g|mg|mph|m/s)"
    matches = list(re.finditer(pattern, step, re.IGNORECASE))
    if matches:
        match = rng.choice(matches)
        try:
            num = float(match.group(1))
            # Perturb by 30-70% in random direction
            factor = rng.uniform(0.3, 0.7) * rng.choice([-1, 1])
            new_num = num + num * factor
            if "." not in match.group(1):
                new_str = str(int(new_num))
            else:
                new_str = f"{new_num:.1f}"
            return step[: match.start(1)] + new_str + step[match.end(1) :]
        except ValueError:
            pass

    # Strategy 2: Fallback to antonym swap (same as negate)
    return negate_conclusion(step, rng)


def reverse_causation(step: str, rng: random.Random | None = None) -> str:
    """Swap cause and effect in a causal statement."""
    if rng is None:
        rng = random.Random()

    # Pattern: "X causes Y" -> "Y causes X"
    causal_patterns = [
        (r"(.+?)\s+(causes?|leads?\s+to|results?\s+in|produces?)\s+(.+)", r"\3 \2 \1"),
        (r"(.+?)\s+(because|since|due\s+to)\s+(.+)", r"\3 \2 \1"),
    ]

    for pattern, replacement in causal_patterns:
        match = re.match(pattern, step, re.IGNORECASE)
        if match:
            try:
                return re.sub(pattern, replacement, step, count=1, flags=re.IGNORECASE)
            except re.error:
                continue

    # Fallback: negate instead
    return negate_conclusion(step, rng)


def corrupt_arc_step(step: str, rng: random.Random | None = None) -> str:
    """Apply a random logical corruption to an ARC step.

    Randomly chooses among negate, factual error, and reverse causation.
    """
    if rng is None:
        rng = random.Random()

    strategies = [negate_conclusion, introduce_factual_error, reverse_causation]
    rng.shuffle(strategies)

    for strategy in strategies:
        result = strategy(step, rng)
        if result != step:
            return result

    # If nothing changed, force a negation
    return negate_conclusion(step, rng)
