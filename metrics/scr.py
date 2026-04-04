from __future__ import annotations
from metrics.statistical_tests import confidence_interval


def compute_scr(trunc_answers, full_answers):
    assert len(trunc_answers) == len(full_answers)
    matches = [t.strip() == f.strip()
               for t, f in zip(trunc_answers, full_answers) if t and f]
    n = len(matches)
    if n == 0:
        return {"scr": 0.0, "n_consistent": 0, "n_total": 0,
                "ci_lower": 0.0, "ci_upper": 0.0}
    hit = sum(matches)
    lo, hi = confidence_interval(matches)
    return {"scr": hit / n, "n_consistent": hit, "n_total": n,
            "ci_lower": lo, "ci_upper": hi}
