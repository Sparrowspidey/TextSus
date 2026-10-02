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

    return candidates[0]