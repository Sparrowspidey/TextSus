import hashlib


def generate_g_value(
    token_id: int,
    seed: int,
    layer: int,
    distribution: str = "bernoulli",
) -> float | int:
    """
    Generate a deterministic pseudorandom g-value for a token.

    The SynthID-Text paper defines the g-value as a function of:
        - token x
        - random seed r
        - tournament layer l

    The hash output is converted to a value in [0, 1], then mapped
    to the requested g-value distribution.

    Parameters
    ----------
    token_id : int
        Candidate token ID.
    seed : int
        Random seed for the current generation step.
    layer : int
        Tournament layer.
    distribution : str, default="bernoulli"
        G-value distribution:
        - "bernoulli": Bernoulli(0.5)
        - "uniform": Uniform[0, 1]

    Returns
    -------
    float | int
        Generated g-value.
    """

    # Hash token, layer, and seed together.
    data = f"{token_id}|{layer}|{seed}".encode("utf-8")
    digest = hashlib.sha256(data).digest()

    # Convert hash to a deterministic value in [0, 1].
    hash_value = int.from_bytes(digest, byteorder="big")
    n = 8 * len(digest)
    uniform_value = hash_value / (2**n)

    if distribution == "bernoulli":
        # Bernoulli(0.5)
        return 1 if uniform_value >= 0.5 else 0

    if distribution == "uniform":
        # Uniform[0, 1]
        return uniform_value

    raise ValueError(
        f"Unsupported g-value distribution: {distribution}"
    )