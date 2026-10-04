"""Perplexity computation, used to check that watermarking preserves quality."""

from __future__ import annotations

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


@torch.no_grad()
def compute_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    device: str = "cpu",
) -> float:
    """Compute the perplexity of ``text`` under ``model`` (lower is more fluent).

    Used the same way as in the paper: compare mean perplexity of
    watermarked vs. unwatermarked responses -- no significant difference
    should indicate the watermark is not degrading quality.
    """
    if not text.strip():
        return float("nan")

    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    if input_ids.shape[1] < 2:
        return float("nan")

    outputs = model(input_ids, labels=input_ids)
    return float(torch.exp(outputs.loss).item())