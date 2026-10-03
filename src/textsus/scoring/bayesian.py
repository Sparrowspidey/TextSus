import math

from textsus.gvalues.gvalues import generate_g_value
from textsus.seed.random_seed import generate_random_seed


def _sigmoid(value: float) -> float:
    """Numerically stable sigmoid."""
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)

    z = math.exp(value)
    return z / (1.0 + z)


def _get_g_values(
    token_ids: list[int],
    watermarking_key: int,
    num_layers: int,
) -> list[list[int]]:
    """Generate Bernoulli g-values for each token and layer."""
    if not token_ids:
        raise ValueError("token_ids must not be empty.")

    if num_layers < 1:
        raise ValueError("num_layers must be at least 1.")

    all_g_values = []

    for t, token_id in enumerate(token_ids):
        seed = generate_random_seed(
            token_ids[:t],
            watermarking_key,
        )

        token_g_values = []

        for layer in range(1, num_layers + 1):
            g_value = generate_g_value(
                token_id=token_id,
                seed=seed,
                layer=layer,
                distribution="bernoulli",
            )

            token_g_values.append(int(g_value))

        all_g_values.append(token_g_values)

    return all_g_values


def bayesian_score(
    token_ids: list[int],
    watermarking_key: int,
    parameters: dict,
) -> float:
    """
    Compute the parameterized Bayesian watermark score.

    Parameters
    ----------
    token_ids:
        Tokenized text.

    watermarking_key:
        Watermarking key used to reproduce the g-values.

    parameters:
        Learned Bayesian parameters containing:

        {
            "beta": list[float],
            "delta": list[list[float]],
            "prior": float
        }

        beta[l] is the bias for layer l.

        delta[l][j] describes how previous-layer g-values
        influence the latent probability for layer l.

        prior is P(watermarked).

    Returns
    -------
    float
        Posterior probability P(watermarked | g-values).
    """
    beta = parameters["beta"]
    delta = parameters["delta"]
    prior = parameters.get("prior", 0.5)

    num_layers = len(beta)

    if num_layers < 1:
        raise ValueError("Bayesian parameters must contain at least one layer.")

    if len(delta) != num_layers:
        raise ValueError("delta must contain one row per layer.")

    if any(len(row) != num_layers for row in delta):
        raise ValueError(
            "delta must be a square matrix with size num_layers."
        )

    if not 0 < prior < 1:
        raise ValueError("prior must be between 0 and 1.")

    g_values = _get_g_values(
        token_ids,
        watermarking_key,
        num_layers,
    )

    log_likelihood_ratio = 0.0

    for token_g_values in g_values:
        for layer_index in range(num_layers):
            # The Bayesian model predicts the probability that the
            # tournament match had two unique candidates.
            logit = beta[layer_index]

            # Autoregressive dependence on previous layers.
            for previous_layer in range(layer_index):
                logit += (
                    delta[layer_index][previous_layer]
                    * token_g_values[previous_layer]
                )

            p_two_unique_tokens = _sigmoid(logit)
            p_one_unique_token = 1.0 - p_two_unique_tokens

            g_value = token_g_values[layer_index]

            # P(g | watermarked)
            likelihood_watermarked = 0.5 * (
                (g_value + 0.5) * p_two_unique_tokens
                + p_one_unique_token
            )

            # P(g | unwatermarked)
            likelihood_unwatermarked = 0.5

            likelihood_watermarked = max(
                likelihood_watermarked,
                1e-30,
            )

            likelihood_unwatermarked = max(
                likelihood_unwatermarked,
                1e-30,
            )

            log_likelihood_ratio += (
                math.log(likelihood_watermarked)
                - math.log(likelihood_unwatermarked)
            )

    # Prior odds.
    log_prior_odds = math.log(prior) - math.log(1.0 - prior)

    # Posterior log-odds.
    log_posterior_odds = (
        log_prior_odds + log_likelihood_ratio
    )

    return _sigmoid(log_posterior_odds)