from __future__ import annotations
import re, random
from config import ARITH_DELTA


def corrupt_arithmetic_step(step, rng=None):
    """Mess with a numeric result in a reasoning step."""
    if rng is None:
        rng = random.Random()
    lo, hi = ARITH_DELTA

    # look for "= number" or "equals number" patterns
    pat = r"(=\s*\$?|equals\s+\$?|is\s+\$?)([\d,]+(?:\.\d+)?)"
    hits = list(re.finditer(pat, step, re.IGNORECASE))
    if not hits:
        hits = list(re.finditer(r"(\$?)([\d,]+(?:\.\d+)?)", step))
        if not hits:
            return step

    pick = rng.choice(hits)
    raw = pick.group(2).replace(",", "")
    try:
        val = float(raw)
    except ValueError:
        return step

    delta = rng.randint(lo, hi) * rng.choice([-1, 1])
    if val + delta <= 0 and val > 0:
        delta = abs(delta)
    new_val = val + delta

    if "." not in pick.group(2):
        replacement = str(int(new_val))
    else:
        replacement = "%.2f" % new_val

    return step[:pick.start(2)] + replacement + step[pick.end(2):]


def swap_operator(step, rng=None):
    if rng is None:
        rng = random.Random()

    swaps = {"+": "-", "-": "+", "*": "/", "/": "*", "×": "÷", "÷": "×"}
    hits = list(re.finditer(r"(\d\s*)([\+\-\*\/×÷])(\s*\d)", step))
    if not hits:
        return step

    pick = rng.choice(hits)
    op = pick.group(2)
    new_op = swaps.get(op, op)
    if new_op == op:
        return step
    return step[:pick.start(2)] + new_op + step[pick.end(2):]


def corrupt_gsm8k_step(step, rng=None):
    if rng is None:
        rng = random.Random()

    # try one strategy, fall back to the other
    if rng.random() < 0.6:
        out = corrupt_arithmetic_step(step, rng)
        return out if out != step else swap_operator(step, rng)
    else:
        out = swap_operator(step, rng)
        return out if out != step else corrupt_arithmetic_step(step, rng)
