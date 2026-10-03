def detect_watermark(
    score: float,
    threshold: float,
) -> bool:
    """
    Detect a watermark using a score threshold.

    Used for scoring methods where a larger score indicates
    stronger watermark evidence.
    """
    return score >= threshold


def detect_frequentist(
    p_value: float,
    alpha: float,
) -> bool:
    """
    Detect a watermark using a frequentist p-value.

    A smaller p-value provides stronger evidence against
    the unwatermarked null hypothesis.
    """
    if not 0 <= p_value <= 1:
        raise ValueError("p_value must be between 0 and 1.")

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")

    return p_value <= alpha