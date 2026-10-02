import random

from src.textsus.gvalues.gvalues import generate_g_value


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