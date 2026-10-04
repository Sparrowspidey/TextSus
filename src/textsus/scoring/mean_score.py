<<<<<<< HEAD
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
=======
from textsus.gvalues.gvalues import generate_g_value
from textsus.seed.random_seed import generate_random_seed


def mean_score(
    token_ids: list[int],
    watermarking_key: int,
    num_layers: int,
) -> float:
    """
    Compute the mean watermark score.

    The score is the mean of the g-values across all
    token positions and tournament layers.
    """
    if not token_ids:
        raise ValueError("token_ids must not be empty.")

    if num_layers < 1:
        raise ValueError("num_layers must be at least 1.")

    total_g_value = 0.0

    for t, token_id in enumerate(token_ids):
        # The seed is generated from the preceding token context.
        seed = generate_random_seed(
            token_ids[:t],
            watermarking_key,
        )

        for layer in range(1, num_layers + 1):
            g_value = generate_g_value(
                token_id=token_id,
                seed=seed,
                layer=layer,
            )
            total_g_value += g_value

    return total_g_value / (len(token_ids) * num_layers)
>>>>>>> ac23df183f6bca3a9ff72a1ade999c42642d7a0b
