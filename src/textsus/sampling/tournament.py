<<<<<<< HEAD
"""Tournament sampling (Algorithms 1 and 2 from the SynthID-Text paper).

Reference implementation, not yet vectorized -- correct and easy to read,
good enough for m up to ~8-10 layers on CPU. Member 2 owns optimizing this
(the paper's vectorized version) if we need more layers or more speed.
"""

from __future__ import annotations

import random
from typing import Sequence

import torch

from textsus.gvalues.gvalues import g_value
from textsus.seed.random_seed import sliding_window_seed


def single_layer_tournament(
    candidates: Sequence[int],
    seed: int,
    layer: int = 1,
    distribution: str = "bernoulli",
) -> int:
    """Algorithm 1: from N candidate tokens, return the one with max g-value.

    Ties are broken uniformly at random.
    """
    scores = [g_value(tok, layer, seed, distribution) for tok in candidates]
    best_score = max(scores)
    winners = [tok for tok, s in zip(candidates, scores) if s == best_score]
    return random.choice(winners)


def multilayer_tournament_sample(
    probs: torch.Tensor,
    context_tokens: Sequence[int],
    key: int,
    m: int = 4,
    H: int = 4,
    distribution: str = "bernoulli",
) -> int:
    """Algorithm 2: sample one token using an m-layer Tournament.

    Args:
        probs: 1D tensor of next-token probabilities from the LLM
            (after top-k / top-p / temperature have already been applied).
        context_tokens: token ids generated so far, for the seed hash.
        key: secret watermarking key.
        m: number of tournament layers. N = 2**m candidates are drawn.
           Keep this small (4-8) for CPU-scale experiments; the paper uses
           m=30 with production-scale vectorized infrastructure.
        H: sliding-window size for the seed generator.
        distribution: g-value distribution, "bernoulli" or "uniform".

    Returns:
        The winning token id (int).
    """
    seed = sliding_window_seed(context_tokens, key, H=H)
    n_candidates = 2**m

    # Draw N candidate tokens directly from the (already filtered) LLM distribution.
    candidates = torch.multinomial(probs, n_candidates, replacement=True).tolist()

    for layer in range(1, m + 1):
        random.shuffle(candidates)
        next_round = []
        for i in range(0, len(candidates), 2):
            pair = candidates[i : i + 2]
            winner = single_layer_tournament(pair, seed, layer=layer, distribution=distribution)
            next_round.append(winner)
        candidates = next_round
=======
import random

from textsus.gvalues.gvalues import generate_g_value


def single_layer_tournament_sample(
    token_ids: list[int],
    seed: int,
) -> int:
    """
    Perform single-layer Tournament Sampling.

    Algorithm 1 from the SynthID-Text paper:
    1. Compute g-values for the sampled candidate tokens.
    2. Keep all candidates with the maximum g-value.
    3. Select uniformly among those candidates.

    Parameters
    ----------
    token_ids : list[int]
        Candidate tokens sampled from the LLM distribution.
    seed : int
        Random seed for the g-value function.

    Returns
    -------
    int
        Selected token.
    """

    if len(token_ids) < 2:
        raise ValueError("Tournament sampling requires at least 2 candidates.")

    # Compute the g-value for every sampled candidate.
    g_values = [
        generate_g_value(token_id, seed, layer=1)
        for token_id in token_ids
    ]

    # Find the maximum g-value.
    max_g_value = max(g_values)

    # Keep ALL candidates having the maximum g-value.
    max_candidates = [
        token_id
        for token_id, g_value in zip(token_ids, g_values)
        if g_value == max_g_value
    ]

    # Choose uniformly among the maximum-g candidates.
    return random.choice(max_candidates)
def tournament_sample(
    token_ids: list[int],
    seed: int,
    num_layers: int,
) -> int:
    """
    Perform multilayer Tournament Sampling.

    Algorithm 2 from the SynthID-Text paper:
    1. Start with M = 2^m candidate tokens.
    2. Pair the candidates.
    3. At each layer, compare the g-values of each pair.
    4. The candidate with the higher g-value wins.
    5. If the g-values tie, choose randomly.
    6. Continue with the winners at the next layer.
    7. Return the final remaining token.
    """

    if num_layers < 1:
        raise ValueError("Number of layers must be at least 1.")

    expected_candidates = 2**num_layers

    if len(token_ids) != expected_candidates:
        raise ValueError(
            f"Expected {expected_candidates} candidates for "
            f"{num_layers} layers, got {len(token_ids)}."
        )

    candidates = token_ids.copy()

    for layer in range(1, num_layers + 1):
        winners = []

        for i in range(0, len(candidates), 2):
            token_a = candidates[i]
            token_b = candidates[i + 1]

            g_a = generate_g_value(token_a, seed, layer)
            g_b = generate_g_value(token_b, seed, layer)

            if g_a > g_b:
                winner = token_a
            elif g_b > g_a:
                winner = token_b
            else:
                winner = random.choice([token_a, token_b])

            winners.append(winner)

        candidates = winners
>>>>>>> ac23df183f6bca3a9ff72a1ade999c42642d7a0b

    return candidates[0]