"""Detection evaluation metrics: TPR at a fixed FPR, the paper's main metric."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def tpr_at_fpr(
    watermarked_scores: Sequence[float],
    unwatermarked_scores: Sequence[float],
    fpr: float = 0.01,
) -> tuple[float, float]:
    """Compute the true-positive rate at a fixed false-positive rate.

    Threshold is set at the (1 - fpr) quantile of the unwatermarked scores,
    then TPR is the fraction of watermarked scores above that threshold.
    This mirrors the paper's empirical TPR@FPR methodology exactly.

    Returns:
        (tpr, threshold)
    """
    unwatermarked_scores = np.asarray(unwatermarked_scores, dtype=float)
    watermarked_scores = np.asarray(watermarked_scores, dtype=float)

    threshold = float(np.quantile(unwatermarked_scores, 1 - fpr))
    tpr = float(np.mean(watermarked_scores > threshold))
    return tpr, threshold