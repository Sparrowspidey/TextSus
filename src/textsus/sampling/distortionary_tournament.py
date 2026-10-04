"""Distortionary Tournament sampling: a generalization of the team's
tournament_sample() (textsus.sampling.tournament) that allows MORE than
two competitors per match.

The paper distinguishes two configurations:
  - Exactly 2 competitors per match -> single-token non-distortionary
    (on average, the output token distribution equals the original LLM
    distribution). This is textsus.sampling.tournament.tournament_sample.
  - More than 2 competitors per match -> distortionary: selection is
    biased more strongly toward high-g-value tokens each round, which
    strengthens the watermark (higher detectability) at the cost of
    pulling the output distribution further from the original LLM
    distribution (lower text quality/diversity).

With competitors_per_match=2 this function is mathematically identical
to tournament_sample; it exists as a separate module so the team's
original, tested implementation stays untouched.
"""

import random

from textsus.gvalues.gvalues import generate_g_value


def distortionary_tournament_sample(
    token_ids: list[int],
    seed: int,
    num_layers: int,
    competitors_per_match: int = 2,
) -> int:
    """
    Perform multilayer Tournament Sampling with a configurable number of
    competitors per match (Algorithm 2, generalized).

    At each of ``num_layers`` layers, candidates are grouped into blocks
    of ``competitors_per_match``; within each block, the candidate with
    the highest g-value at that layer wins (ties broken uniformly at
    random). Winners proceed to the next layer. The final surviving
    candidate is returned.

    Parameters
    ----------
    token_ids : list[int]
        Initial candidate tokens, sampled from the LLM distribution.
        Must contain exactly competitors_per_match ** num_layers tokens.
    seed : int
        Random seed for this generation step.
    num_layers : int
        Number of tournament layers.
    competitors_per_match : int, default=2
        Number of candidates competing in each match. 2 reproduces the
        paper's non-distortionary configuration; >2 is distortionary
        (stronger watermark, more quality cost).

    Returns
    -------
    int
        Selected token.
    """
    if num_layers < 1:
        raise ValueError("num_layers must be at least 1.")

    if competitors_per_match < 2:
        raise ValueError("competitors_per_match must be at least 2.")

    expected_candidates = competitors_per_match**num_layers
    if len(token_ids) != expected_candidates:
        raise ValueError(
            f"Expected {expected_candidates} candidates for {num_layers} layers "
            f"with {competitors_per_match} competitors/match, got {len(token_ids)}."
        )

    candidates = token_ids.copy()

    for layer in range(1, num_layers + 1):
        winners = []

        for i in range(0, len(candidates), competitors_per_match):
            group = candidates[i : i + competitors_per_match]

            g_values = [generate_g_value(tok, seed, layer) for tok in group]
            max_g_value = max(g_values)

            best_candidates = [
                tok for tok, g in zip(group, g_values) if g == max_g_value
            ]
            winners.append(random.choice(best_candidates))

        candidates = winners

    return candidates[0]