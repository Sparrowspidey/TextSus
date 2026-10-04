import hashlib
import math

from textsus.seed.random_seed import generate_random_seed


def _green_list_hash(token_id: int, seed: int) -> float:
    """Hash (token_id, seed) to a value uniform in (0, 1), used to
    pseudorandomly assign the token to the green or red list."""
    data = f"soft_red_list|{token_id}|{seed}".encode("utf-8")
    digest = hashlib.sha256(data).digest()
    hash_value = int.from_bytes(digest, byteorder="big")
    n = 8 * len(digest)
    return hash_value / (2**n)


def is_green(token_id: int, seed: int, gamma: float = 0.5) -> bool:
    """
    Determine whether a token falls in the pseudorandom green list.

    Parameters
    ----------
    token_id : int
        Candidate token id.
    seed : int
        Random seed for this generation step.
    gamma : float, default=0.5
        Fraction of the vocabulary assigned to the green list.
    """
    return _green_list_hash(token_id, seed) < gamma


def apply_soft_red_list_bias(
    logits: dict[int, float],
    seed: int,
    gamma: float = 0.5,
    delta: float = 2.0,
) -> dict[int, float]:
    """
    Apply the Soft Red List logit bias (Kirchenbauer et al., ICML 2023).

    Every candidate token in the pseudorandom "green list" has its logit
    boosted by delta before sampling; "red list" tokens are left
    unchanged. Larger delta = stronger (more distortionary) watermark at
    the cost of text quality -- this is the strength hyperparameter the
    paper trades off against detectability.

    Parameters
    ----------
    logits : dict[int, float]
        Mapping from candidate token id to its raw logit.
    seed : int
        Random seed for this generation step.
    gamma : float, default=0.5
        Fraction of the vocabulary assigned to the green list.
    delta : float, default=2.0
        Logit bias strength added to green-list tokens.

    Returns
    -------
    dict[int, float]
        Logits with the green-list bias applied, same keys as input.
    """
    biased = {}
    for token_id, logit in logits.items():
        biased[token_id] = logit + delta if is_green(token_id, seed, gamma) else logit
    return biased


def green_fraction_score(
    token_ids: list[int],
    watermarking_key: int,
    gamma: float = 0.5,
) -> float:
    """
    Compute the fraction of tokens in a sequence that fall in the
    pseudorandom green list under their own generation-time seed.

    Watermarked text (biased toward the green list during generation)
    should show a green fraction well above gamma; unwatermarked text
    should sit close to gamma.
    """
    if not token_ids:
        raise ValueError("token_ids must not be empty.")

    green_count = 0
    for t, token_id in enumerate(token_ids):
        seed = generate_random_seed(token_ids[:t], watermarking_key)
        if is_green(token_id, seed, gamma):
            green_count += 1

    return green_count / len(token_ids)


def z_score(
    token_ids: list[int],
    watermarking_key: int,
    gamma: float = 0.5,
) -> float:
    """
    Compute the Kirchenbauer et al. z-statistic for watermark detection.

    Under the null hypothesis (unwatermarked text), the green-list
    fraction is Binomial(n, gamma); the z-score measures how many
    standard deviations the observed green count sits above the
    expected count. Higher z = stronger evidence of watermarking.
    """
    if not token_ids:
        raise ValueError("token_ids must not be empty.")

    n = len(token_ids)
    green_count = round(green_fraction_score(token_ids, watermarking_key, gamma) * n)
    expected = gamma * n
    std = math.sqrt(n * gamma * (1 - gamma))

    return (green_count - expected) / std if std > 0 else 0.0