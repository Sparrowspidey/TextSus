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

    return candidates[0]