"""Random seed generator for SynthID-Text style watermarking.

Implements the sliding-window method from the paper: the seed at step t
is a keyed hash of the last H generated tokens.
"""

from __future__ import annotations

import hashlib
from typing import Sequence


def sliding_window_seed(context_tokens: Sequence[int], key: int, H: int = 4) -> int:
    """Compute a random seed from the last ``H`` tokens and the watermarking key.

    Args:
        context_tokens: token ids generated so far (x_1, ..., x_{t-1}).
        key: secret watermarking key (any int).
        H: sliding window size (paper default: 4).

    Returns:
        A 64-bit integer seed, deterministic given the same inputs.
    """
    window = list(context_tokens[-H:]) if len(context_tokens) >= H else list(context_tokens)
    payload = f"{key}|" + ",".join(str(t) for t in window)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big")