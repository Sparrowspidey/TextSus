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