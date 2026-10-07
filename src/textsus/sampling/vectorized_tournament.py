"""Vectorized tournament reduction for sampled token candidates.

This module selects one token from candidates that have already been sampled
from the language-model distribution. It vectorizes over any leading batch
dimensions and over all matches within each tournament layer.
"""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt


TokenArray = npt.NDArray[np.int64]
ScoreArray = npt.NDArray[np.float64]
GValueFunction = Callable[[TokenArray, int], ScoreArray]


def vectorized_tournament_sample(
    candidate_token_ids: TokenArray,
    *,
    num_layers: int,
    num_competitors: int,
    g_value_fn: GValueFunction,
    rng: np.random.Generator,
) -> TokenArray:
    """Reduce ``N**m`` sampled candidates to one token per batch item.

    Args:
        candidate_token_ids: Integer token IDs. The last axis must contain
            ``num_competitors ** num_layers`` independently sampled
            candidates. Any preceding axes are treated as batch dimensions.
        num_layers: Number of tournament layers (``m``), at least one.
        num_competitors: Number of competitors in each match (``N``), at
            least two.
        g_value_fn: Vectorized function ``g_value_fn(token_ids, layer)``.
            It must return one score per token ID, with the same shape as
            ``token_ids``. Capture the generation seed in this function.
        rng: NumPy random generator used to break ties uniformly at random.

    Returns:
        One selected token ID for each item in the leading batch dimensions.
        If the input is one-dimensional, returns a scalar-shaped array.

    Notes:
        At every layer, candidates are grouped in consecutive blocks of N.
        This is equivalent to random grouping when the initial candidates
        are independent samples and their order is exchangeable. Ties are
        resolved by assigning independent random priorities to tied
        candidates and selecting the highest priority.
    """
    candidates = np.asarray(candidate_token_ids)

    if candidates.ndim < 1:
        raise ValueError("candidate_token_ids must have a candidate axis")
    if not np.issubdtype(candidates.dtype, np.integer):
        raise TypeError("candidate_token_ids must contain integer token IDs")
    if isinstance(num_layers, bool) or not isinstance(num_layers, int):
        raise TypeError("num_layers must be an integer")
    if isinstance(num_competitors, bool) or not isinstance(num_competitors, int):
        raise TypeError("num_competitors must be an integer")
    if num_layers < 1:
        raise ValueError("num_layers must be at least 1")
    if num_competitors < 2:
        raise ValueError("num_competitors must be at least 2")

    expected_candidates = num_competitors**num_layers
    if candidates.shape[-1] != expected_candidates:
        raise ValueError(
            "last candidate axis must have length "
            f"num_competitors ** num_layers ({expected_candidates}); "
            f"got {candidates.shape[-1]}"
        )

    current = candidates
    for layer in range(num_layers):
        scores = np.asarray(g_value_fn(current, layer))
        if scores.shape != current.shape:
            raise ValueError(
                "g_value_fn must return one score per candidate with the "
                f"same shape; got {scores.shape}, expected {current.shape}"
            )

        group_shape = current.shape[:-1] + (
            current.shape[-1] // num_competitors,
            num_competitors,
        )
        grouped_tokens = current.reshape(group_shape)
        grouped_scores = scores.reshape(group_shape)

        maximum = np.max(grouped_scores, axis=-1, keepdims=True)
        tied_for_maximum = grouped_scores == maximum

        # Independent continuous priorities make every maximum-score token
        # equally likely to win a tie.
        tie_priorities = rng.random(group_shape)
        priorities = np.where(tied_for_maximum, tie_priorities, -np.inf)
        winner_offsets = np.argmax(priorities, axis=-1)

        current = np.take_along_axis(
            grouped_tokens,
            winner_offsets[..., np.newaxis],
            axis=-1,
        )[..., 0]

    return np.squeeze(current, axis=-1)
