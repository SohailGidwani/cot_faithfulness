"""Statistical tests: confidence intervals and McNemar's test."""

from __future__ import annotations

import numpy as np
from scipy import stats

from config import CONFIDENCE_LEVEL


def confidence_interval(data: list[bool], confidence: float = CONFIDENCE_LEVEL) -> tuple[float, float]:
    """Compute confidence interval for a proportion.

    Uses the normal approximation (Wald interval).

    Args:
        data: List of boolean outcomes.
        confidence: Confidence level (default 0.95).

    Returns:
        (lower_bound, upper_bound) tuple.
    """
    n = len(data)
    if n == 0:
        return (0.0, 0.0)

    p = sum(data) / n
    z = stats.norm.ppf((1 + confidence) / 2)
    margin = z * np.sqrt(p * (1 - p) / n)
    return (max(0.0, p - margin), min(1.0, p + margin))


def mcnemar_test(table: np.ndarray) -> dict:
    """McNemar's test for paired comparisons between two models.

    Uses the exact binomial test on the discordant cells (b, c).

    Args:
        table: 2x2 contingency table where:
            table[0][0] = both correct
            table[0][1] = model1 correct, model2 wrong  (b)
            table[1][0] = model1 wrong, model2 correct   (c)
            table[1][1] = both wrong

    Returns:
        Dict with statistic and pvalue.
    """
    b = int(table[0][1])
    c = int(table[1][0])
    n = b + c

    if n == 0:
        return {"statistic": 0.0, "pvalue": 1.0}

    # Exact binomial test: under H0 b ~ Binomial(n, 0.5)
    pvalue = float(stats.binomtest(b, n, 0.5).pvalue)
    statistic = float((b - c) ** 2 / n) if n > 0 else 0.0

    return {
        "statistic": statistic,
        "pvalue": pvalue,
    }
