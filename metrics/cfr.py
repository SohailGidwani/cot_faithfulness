from __future__ import annotations
from metrics.statistical_tests import confidence_interval


def compute_cfr(orig_answers, corrupt_answers):
    assert len(orig_answers) == len(corrupt_answers)
    changed = [o.strip() != c.strip()
               for o, c in zip(orig_answers, corrupt_answers) if o and c]
    n = len(changed)
    if n == 0:
        return {"cfr": 0.0, "n_changed": 0, "n_total": 0,
                "ci_lower": 0.0, "ci_upper": 0.0}
    k = sum(changed)
    lo, hi = confidence_interval(changed)
    return {"cfr": k / n, "n_changed": k, "n_total": n,
            "ci_lower": lo, "ci_upper": hi}
