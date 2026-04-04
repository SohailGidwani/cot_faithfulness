from __future__ import annotations
import re, random

_PAIRS = [
    ("increases", "decreases"), ("increase", "decrease"),
    ("expands", "contracts"), ("expand", "contract"),
    ("heats", "cools"), ("heat", "cool"),
    ("rises", "falls"), ("rise", "fall"),
    ("absorbs", "reflects"), ("absorb", "reflect"),
    ("attracts", "repels"), ("attract", "repel"),
    ("accelerates", "decelerates"),
    ("faster", "slower"), ("larger", "smaller"),
    ("more", "less"), ("higher", "lower"),
    ("hotter", "colder"), ("greater", "lesser"),
    ("positive", "negative"), ("solid", "liquid"),
    ("liquid", "gas"), ("heavy", "light"),
    ("strong", "weak"), ("bright", "dim"),
    ("thick", "thin"), ("long", "short"),
    ("above", "below"), ("before", "after"),
    ("true", "false"),
]

_ANT = {}
for _a, _b in _PAIRS:
    _ANT[_a.lower()] = _b
    _ANT[_b.lower()] = _a


def negate_conclusion(step, rng=None):
    if rng is None:
        rng = random.Random()

    words = step.split()
    swappable = []
    for i, w in enumerate(words):
        clean = w.lower().rstrip(".,;:!?")
        if clean in _ANT:
            swappable.append((i, w, clean))
    if swappable:
        idx, orig, clean = rng.choice(swappable)
        suffix = orig[len(clean):]
        repl = _ANT[clean]
        if orig[0].isupper():
            repl = repl.capitalize()
        words[idx] = repl + suffix
        return " ".join(words)

    # try removing "not" if present
    if " not " in step:
        return step.replace(" not ", " ", 1)

    # or insert "not" after a common verb
    m = re.search(r"\b(is|are|was|were|does|do|can|will|would|could|should)\b",
                  step, re.IGNORECASE)
    if m:
        return step[:m.end()] + " not" + step[m.end():]
    return step


def introduce_factual_error(step, rng=None):
    if rng is None:
        rng = random.Random()

    pat = r"(\d+(?:\.\d+)?)\s*(°[CF]|degrees|percent|%|km|m|cm|mm|kg|g|mg|mph|m/s)"
    hits = list(re.finditer(pat, step, re.IGNORECASE))
    if hits:
        pick = rng.choice(hits)
        try:
            num = float(pick.group(1))
            factor = rng.uniform(0.3, 0.7) * rng.choice([-1, 1])
            new_num = num + num * factor
            fmt = str(int(new_num)) if "." not in pick.group(1) else "%.1f" % new_num
            return step[:pick.start(1)] + fmt + step[pick.end(1):]
        except ValueError:
            pass

    return negate_conclusion(step, rng)


def reverse_causation(step, rng=None):
    if rng is None:
        rng = random.Random()

    patterns = [
        (r"(.+?)\s+(causes?|leads?\s+to|results?\s+in|produces?)\s+(.+)", r"\3 \2 \1"),
        (r"(.+?)\s+(because|since|due\s+to)\s+(.+)", r"\3 \2 \1"),
    ]
    for pat, repl in patterns:
        if re.match(pat, step, re.IGNORECASE):
            try:
                return re.sub(pat, repl, step, count=1, flags=re.IGNORECASE)
            except re.error:
                continue
    return negate_conclusion(step, rng)


def corrupt_arc_step(step, rng=None):
    """Pick a random corruption strategy for an ARC reasoning step."""
    if rng is None:
        rng = random.Random()

    fns = [negate_conclusion, introduce_factual_error, reverse_causation]
    rng.shuffle(fns)
    for fn in fns:
        out = fn(step, rng)
        if out != step:
            return out
    return negate_conclusion(step, rng)
