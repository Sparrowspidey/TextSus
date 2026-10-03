from collections import deque


H = 4


def create_context_history(
    k: int,
) -> deque[set[tuple[int, ...]]]:
    """
    Create K-sequence context history.

    K controls how many response histories are retained.
    """

    if k < 1:
        raise ValueError("K must be at least 1.")

    return deque(maxlen=k)


def is_context_repeated(
    context: tuple[int, ...],
    context_history: list[set[tuple[int, ...]]],
) -> bool:
    """
    Check whether a context has already been used for watermarking.

    Parameters
    ----------
    context : tuple[int, ...]
        Current H-token context window.

    context_history : list[set[tuple[int, ...]]]
        Contexts used in previous/current responses.

    Returns
    -------
    bool
        True if the context has already been used.
        False otherwise.
    """

    return any(
        context in response_history
        for response_history in context_history
    )