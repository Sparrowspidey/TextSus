"""g-value functions used to referee Tournament sampling matches.

g_layer(token, seed) is a pseudorandom function of (token id, layer, seed)
that the detector can recompute exactly given the watermarking key -- no
LLM access required.
"""

from __future__ import annotations

import hashlib


def _uniform_hash(token_id: int, layer: int, seed: int) -> float:
    """Hash (token_id, layer, seed) to a value uniform in [0, 1)."""
    payload = f"{token_id}|{layer}|{seed}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    as_int = int.from_bytes(digest[:8], byteorder="big")
    return as_int / 2**64


def g_value(token_id: int, layer: int, seed: int, distribution: str = "bernoulli") -> float:
    """Compute the layer-``layer`` g-value of ``token_id`` under ``seed``.

    Args:
        token_id: candidate token id.
        layer: tournament layer index (1-indexed).
        seed: random seed for this generation step (from random_seed.py).
        distribution: "bernoulli" (default, paper's main setting) or "uniform".

    Returns:
        0.0 or 1.0 for Bernoulli(0.5); a float in [0, 1) for Uniform.
    """
    u = _uniform_hash(token_id, layer, seed)
    if distribution == "bernoulli":
        return 1.0 if u < 0.5 else 0.0
    if distribution == "uniform":
        return u
    raise ValueError(f"Unknown g-value distribution: {distribution!r}")