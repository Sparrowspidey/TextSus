import math

from textsus.gvalues.gvalues import generate_g_value
from textsus.seed.random_seed import generate_random_seed


def _collect_g_values(
    token_ids: list[int],
    watermarking_key: int,
    num_layers: int,
) -> list[int]:
    """Collect Bernoulli(0.5) g-values for all tokens and layers."""
    if not token_ids:
        raise ValueError("token_ids must not be empty.")

    if num_layers < 1:
        raise ValueError("num_layers must be at least 1.")

    g_values = []

    for t, token_id in enumerate(token_ids):
        seed = generate_random_seed(
            token_ids[:t],
            watermarking_key,
        )

        for layer in range(1, num_layers + 1):
            g_value = generate_g_value(
                token_id=token_id,
                seed=seed,
                layer=layer,
                distribution="bernoulli",
            )

            g_values.append(int(g_value))

    return g_values


def frequentist_mean_score(
    token_ids: list[int],
    watermarking_key: int,
    num_layers: int,
) -> float:
    """
    Compute the frequentist p-value for the mean watermark score.

    Under H0, the Bernoulli(0.5) g-values are used to calculate
    the probability of observing the measured number of successes
    or more.

    Returns
    -------
    float
        One-sided p-value.
    """
    g_values = _collect_g_values(
        token_ids,
        watermarking_key,
        num_layers,
    )

    successes = sum(g_values)
    total = len(g_values)

    # P(X >= successes), X ~ Binomial(total, 0.5)
    p_value = sum(
        math.comb(total, k) * (0.5 ** total)
        for k in range(successes, total + 1)
    )

    return p_value


def frequentist_weighted_mean_score(
    token_ids: list[int],
    watermarking_key: int,
    layer_weights: list[float],
) -> float:
    """
    Compute a frequentist p-value for the weighted mean score.

    Under H0, each Bernoulli(0.5) g-value has:
        E[g] = 0.5
        Var[g] = 0.25

    The weighted mean is standardized using its null mean
    and variance and converted to a one-sided normal p-value.
    """
    if not token_ids:
        raise ValueError("token_ids must not be empty.")

    if not layer_weights:
        raise ValueError("layer_weights must not be empty.")

    if any(weight < 0 for weight in layer_weights):
        raise ValueError("layer_weights must be non-negative.")

    weight_sum = sum(layer_weights)

    if weight_sum == 0:
        raise ValueError("At least one layer weight must be positive.")

    num_layers = len(layer_weights)

    weighted_sum = 0.0

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
                distribution="bernoulli",
            )

            weighted_sum += weight * g_value

    num_tokens = len(token_ids)

    weighted_mean = weighted_sum / (num_tokens * weight_sum)

    # Null expectation:
    # E[weighted_mean] = 0.5
    null_mean = 0.5

    # Null variance:
    #
    # Var(weighted_mean)
    # = 1/(4T) * sum(w_l^2)/(sum(w_l)^2)
    null_variance = (
        sum(weight ** 2 for weight in layer_weights)
        / (4 * num_tokens * weight_sum ** 2)
    )

    standard_error = math.sqrt(null_variance)

    z = (weighted_mean - null_mean) / standard_error

    # One-sided upper-tail probability.
    p_value = 0.5 * math.erfc(z / math.sqrt(2))

    return p_value