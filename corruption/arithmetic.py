"""Rule-based arithmetic corruption for GSM8K reasoning steps."""

from __future__ import annotations

import re
import random

from config import CORRUPTION_ARITHMETIC_DELTA_RANGE


def corrupt_arithmetic_step(step: str, rng: random.Random | None = None) -> str:
    """Corrupt an arithmetic result in a reasoning step.

    Finds patterns like '= 42' or 'equals 42' and perturbs the number
    by a random delta in CORRUPTION_ARITHMETIC_DELTA_RANGE.
    Preserves the original structure and length as much as possible.
    """
    if rng is None:
        rng = random.Random()

    lo, hi = CORRUPTION_ARITHMETIC_DELTA_RANGE

    # Match "= number", "equals number", "is number" in arithmetic contexts
    pattern = r"(=\s*\$?|equals\s+\$?|is\s+\$?)([\d,]+(?:\.\d+)?)"

    matches = list(re.finditer(pattern, step, re.IGNORECASE))
    if not matches:
        # Fallback: try to corrupt any number in the step
        matches = list(re.finditer(r"(\$?)([\d,]+(?:\.\d+)?)", step))
        if not matches:
            return step

    # Pick a random match to corrupt
    match = rng.choice(matches)
    original_num_str = match.group(2).replace(",", "")

    try:
        original_num = float(original_num_str)
    except ValueError:
        return step

    # Generate delta that changes the number
    delta = rng.randint(lo, hi) * rng.choice([-1, 1])
    if original_num + delta <= 0 and original_num > 0:
        delta = abs(delta)

    new_num = original_num + delta
    # Preserve integer formatting if original was integer
    if "." not in match.group(2):
        new_num_str = str(int(new_num))
    else:
        new_num_str = f"{new_num:.2f}"

    corrupted = step[: match.start(2)] + new_num_str + step[match.end(2) :]
    return corrupted


def swap_operator(step: str, rng: random.Random | None = None) -> str:
    """Swap an arithmetic operator in a reasoning step.

    Replaces + with -, * with /, etc.
    """
    if rng is None:
        rng = random.Random()

    swap_map = {"+": "-", "-": "+", "*": "/", "/": "*", "×": "÷", "÷": "×"}

    # Find operators surrounded by numbers/spaces
    pattern = r"(\d\s*)([\+\-\*\/×÷])(\s*\d)"
    matches = list(re.finditer(pattern, step))
    if not matches:
        return step

    match = rng.choice(matches)
    op = match.group(2)
    new_op = swap_map.get(op, op)
    if new_op == op:
        return step

    corrupted = step[: match.start(2)] + new_op + step[match.end(2) :]
    return corrupted


def corrupt_gsm8k_step(step: str, rng: random.Random | None = None) -> str:
    """Apply a random arithmetic corruption to a GSM8K step.

    Randomly chooses between corrupt_arithmetic_step and swap_operator.
    """
    if rng is None:
        rng = random.Random()

    # Try both strategies, prefer the one that changes the step
    if rng.random() < 0.6:
        result = corrupt_arithmetic_step(step, rng)
        if result != step:
            return result
        return swap_operator(step, rng)
    else:
        result = swap_operator(step, rng)
        if result != step:
            return result
        return corrupt_arithmetic_step(step, rng)
