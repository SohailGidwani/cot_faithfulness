"""Step Consistency Rate (SCR) computation for Experiment 1."""

from __future__ import annotations

from metrics.statistical_tests import confidence_interval


def compute_scr(truncated_answers: list[str], full_answers: list[str]) -> dict:
    """Compute Step Consistency Rate.

    SCR = (# samples where truncated answer == full answer) / total samples

    Args:
        truncated_answers: Answers from truncated CoT at step k.
        full_answers: Answers from full CoT.

    Returns:
        Dict with scr, n_consistent, n_total, and confidence interval.
    """
    assert len(truncated_answers) == len(full_answers)

    matches = [
        t.strip() == f.strip()
        for t, f in zip(truncated_answers, full_answers)
        if t and f
    ]
    n_total = len(matches)
    if n_total == 0:
        return {"scr": 0.0, "n_consistent": 0, "n_total": 0, "ci_lower": 0.0, "ci_upper": 0.0}

    n_consistent = sum(matches)
    scr = n_consistent / n_total
    ci_lower, ci_upper = confidence_interval(matches)

    return {
        "scr": scr,
        "n_consistent": n_consistent,
        "n_total": n_total,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }
