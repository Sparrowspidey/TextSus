import hashlib
import math

from textsus.seed.random_seed import generate_random_seed


def _uniform_hash(token_id: int, seed: int) -> float:
    """Hash (token_id, seed) to a value uniform in (0, 1)."""
    data = f"gumbel|{token_id}|{seed}".encode("utf-8")
    digest = hashlib.sha256(data).digest()
    hash_value = int.from_bytes(digest, byteorder="big")
    n = 8 * len(digest)
    # Keep strictly inside (0, 1) so log() below is always well-defined.
    return min(max(hash_value / (2**n), 1e-12), 1 - 1e-12)


def gumbel_sample(probs: dict[int, float], seed: int) -> int:
    """
    Select a token using Gumbel (exponential-minimum) watermarked sampling.

    Aaronson & Kirchner's method: for every candidate token v with LLM
    probability p_v, draw a pseudorandom r_v ~ Uniform(0, 1) from a keyed
    hash of (v, seed), then pick the token maximizing r_v ** (1 / p_v).
    This reproduces the original LLM distribution exactly on average
    (single-token non-distortionary) while leaving a detectable trace.

    Parameters
    ----------
    probs : dict[int, float]
        Mapping from candidate token id to its LLM probability. Only
        tokens with non-zero probability need to be included (e.g.
        after top-k/top-p filtering) -- looping the full vocabulary is
        wasteful and unnecessary.
    seed : int
        Random seed for this generation step.

    Returns
    -------
    int
        Selected token id.
    """
    if not probs:
        raise ValueError("probs must not be empty.")

    best_token = None
    best_score = -1.0

    for token_id, p in probs.items():
        if p <= 0:
            continue
        u = _uniform_hash(token_id, seed)
        score = u ** (1.0 / p)
        if score > best_score:
            best_score = score
            best_token = token_id

    if best_token is None:
        raise ValueError("No candidate token has non-zero probability.")

    return best_token


def gumbel_token_score(token_id: int, seed: int) -> float:
    """
    Compute the per-token detection score for Gumbel sampling.

    Score(x_t) = -log(1 - r_t), where r_t is the same pseudorandom
    value used to select x_t during generation. Watermarked tokens
    tend to have been chosen precisely because r_t was large, so this
    score runs higher on average for watermarked text than for
    unwatermarked text (where r_t is unrelated to the token actually
    present).
    """
    u = _uniform_hash(token_id, seed)
    return -math.log(max(1.0 - u, 1e-12))


def gumbel_mean_score(token_ids: list[int], watermarking_key: int) -> float:
    """
    Compute the mean Gumbel detection score for a sequence of tokens.

    Requires only the token ids and the watermarking key -- no LLM access.
    """
    if not token_ids:
        raise ValueError("token_ids must not be empty.")

    total = 0.0
    for t, token_id in enumerate(token_ids):
        seed = generate_random_seed(token_ids[:t], watermarking_key)
        total += gumbel_token_score(token_id, seed)

    return total / len(token_ids)