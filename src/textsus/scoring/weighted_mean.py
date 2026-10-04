from textsus.gvalues.gvalues import generate_g_value
from textsus.seed.random_seed import generate_random_seed


def weighted_mean_score(
    token_ids: list[int],
    watermarking_key: int,
    layer_weights: list[float],
) -> float:
    """
    Compute the weighted mean watermark score.

    Evidence from each tournament layer is multiplied by
    its corresponding layer weight.
    """
    if not token_ids:
        raise ValueError("token_ids must not be empty.")

    if not layer_weights:
        raise ValueError("layer_weights must not be empty.")

    if any(weight < 0 for weight in layer_weights):
        raise ValueError("layer_weights must be non-negative.")

    total_weight = sum(layer_weights)

    if total_weight == 0:
        raise ValueError("At least one layer weight must be positive.")

    weighted_g_value_sum = 0.0

    for t, token_id in enumerate(token_ids):
        seed = generate_random_seed(
            token_ids[:t],
            watermarking_key,
        )

        for layer, weight in enumerate(layer_weights, start=1):
            g_value = generate_g_value(
                token_id=token_id,
                seed=seed,
                layer=layer,
            )

            weighted_g_value_sum += weight * g_value

    return weighted_g_value_sum / (
        len(token_ids) * total_weight
    )