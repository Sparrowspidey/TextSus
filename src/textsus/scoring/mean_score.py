"""Watermark detection scoring: the mean-g-value score from the paper.

Score(x) = (1 / (m*T)) * sum_t sum_l g_l(x_t, r_t)

Requires only the token ids and the watermarking key -- no LLM access.
"""

from __future__ import annotations

from typing import Sequence

from textsus.gvalues.gvalues import g_value
from textsus.seed.random_seed import sliding_window_seed


def mean_score(
    token_ids: Sequence[int],
    key: int,
    m: int = 4,
    H: int = 4,
    distribution: str = "bernoulli",
) -> float:
    """Compute the mean g-value score for a sequence of tokens.

    Args:
        token_ids: the full generated response, as token ids.
        key: the watermarking key used at generation time.
        m: number of tournament layers used at generation time.
        H: sliding-window size used at generation time.
        distribution: g-value distribution used at generation time.

    Returns:
        A float score. For Bernoulli(0.5) g-values, unwatermarked text
        scores near 0.5; watermarked text scores noticeably higher.
    """
    if not token_ids:
        return 0.0

    total = 0.0
    count = 0
    for t in range(len(token_ids)):
        context = token_ids[:t]
        seed = sliding_window_seed(context, key, H=H)
        for layer in range(1, m + 1):
            total += g_value(token_ids[t], layer, seed, distribution)
            count += 1

    return total / count if count else 0.0