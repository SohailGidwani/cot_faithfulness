from __future__ import annotations
import numpy as np
from scipy import stats
from config import CONF_LEVEL


def confidence_interval(data, confidence=CONF_LEVEL):
    n = len(data)
    if n == 0:
        return (0.0, 0.0)
    p = sum(data) / n
    z = stats.norm.ppf((1 + confidence) / 2)
    margin = z * np.sqrt(p * (1 - p) / n)
    return (max(0.0, p - margin), min(1.0, p + margin))


def mcnemar_test(table):
    """Paired comparison via exact binomial on the off-diagonal cells."""
    b = int(table[0][1])
    c = int(table[1][0])
    n = b + c
    if n == 0:
        return {"statistic": 0.0, "pvalue": 1.0}

    pval = float(stats.binomtest(b, n, 0.5).pvalue)
    chi2 = float((b - c) ** 2 / n)
    return {"statistic": chi2, "pvalue": pval}
