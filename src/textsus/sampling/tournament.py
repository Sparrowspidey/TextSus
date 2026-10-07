import random

from textsus.gvalues.gvalues import generate_g_value


def single_layer_tournament_sample(
    token_ids: list[int],
    seed: int,
    num_competitors: int = 2,
) -> int:
    """
    Perform single-layer Tournament Sampling.

    Algorithm 1 from the SynthID-Text paper:
    1. Consider N candidate tokens.
    2. Compute the g-value for each candidate.
    3. Keep all candidates with the maximum g-value.
    4. Select uniformly among the maximum-g candidates.

    Parameters
    ----------
    token_ids : list[int]
        Candidate token IDs.
    seed : int
        Random watermarking seed.
    num_competitors : int
        Number of candidates competing in the match.

    Returns
    -------
    int
        Selected token ID.
    """
    if num_competitors < 2:
        raise ValueError("Number of competitors must be at least 2.")

    if len(token_ids) != num_competitors:
        raise ValueError(
            f"Expected {num_competitors} candidates, "
            f"got {len(token_ids)}."
        )

    g_values = [
        generate_g_value(
            token_id=token_id,
            seed=seed,
            layer=1,
        )
        for token_id in token_ids
    ]

    max_g_value = max(g_values)

    max_candidates = [
        token_id
        for token_id, g_value in zip(token_ids, g_values)
        if g_value == max_g_value
    ]

    return random.choice(max_candidates)


def tournament_sample(
    token_ids: list[int],
    seed: int,
    num_layers: int,
    num_competitors: int = 2,
) -> int:
    """
    Perform multilayer Tournament Sampling.

    Algorithm 2 from the SynthID-Text paper.

    For N competitors and m tournament layers:
        initial candidates = N^m

    At each layer:
    1. Divide candidates into groups of N.
    2. Compute the layer-specific g-value for every candidate.
    3. Keep all candidates with the maximum g-value in each group.
    4. Select uniformly among tied maximum candidates.
    5. Use the winners as candidates for the next layer.

    Parameters
    ----------
    token_ids : list[int]
        Initial candidate tokens. Must contain N^m candidates.
    seed : int
        Random watermarking seed.
    num_layers : int
        Number of tournament layers.
    num_competitors : int
        Number of competitors in each match.

    Returns
    -------
    int
        Final tournament winner.
    """
    if num_layers < 1:
        raise ValueError("Number of layers must be at least 1.")

    if num_competitors < 2:
        raise ValueError("Number of competitors must be at least 2.")

    expected_candidates = num_competitors ** num_layers

    if len(token_ids) != expected_candidates:
        raise ValueError(
            f"Expected {expected_candidates} candidates for "
            f"{num_competitors} competitors and {num_layers} layers, "
            f"got {len(token_ids)}."
        )

    candidates = token_ids.copy()

    for layer in range(1, num_layers + 1):
        winners = []

        for i in range(0, len(candidates), num_competitors):
            group = candidates[i : i + num_competitors]

            g_values = [
                generate_g_value(
                    token_id=token_id,
                    seed=seed,
                    layer=layer,
                )
                for token_id in group
            ]

            max_g_value = max(g_values)

            max_candidates = [
                token_id
                for token_id, g_value in zip(group, g_values)
                if g_value == max_g_value
            ]

            winner = random.choice(max_candidates)
            winners.append(winner)

        candidates = winners

    return candidates[0]