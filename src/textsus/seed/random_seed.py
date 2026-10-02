import hashlib


H = 4


def generate_random_seed(tokens: list[int], key: int) -> int:
    """
    Generate a deterministic random seed from the last H tokens
    and a watermarking key.

    The SynthID-Text paper uses a sliding-window hash:
        r_t = h(x_{t-H}, ..., x_{t-1}, k)

    Parameters
    ----------
    tokens : list[int]
        Tokens generated so far.
    key : int
        Watermarking key.

    Returns
    -------
    int
        Deterministic integer seed.
    """

    # Use only the most recent H tokens.
    window = tokens[-H:]

    # Serialize tokens and key deterministically.
    data = ",".join(map(str, window)) + f"|{key}"

    # Keyed hash -> deterministic seed.
    digest = hashlib.sha256(data.encode("utf-8")).digest()

    # Convert hash bytes to an integer.
    return int.from_bytes(digest, byteorder="big")