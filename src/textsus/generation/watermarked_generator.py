"""Watermarked (and unwatermarked) text generation with a HuggingFace model.

This is Member 3's main integration point: it plugs Tournament sampling
into a normal autoregressive generation loop, one token at a time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from textsus.sampling.tournament import multilayer_tournament_sample


@dataclass
class GenerationResult:
    text: str
    token_ids: list[int]
    watermarked: bool


class WatermarkedGenerator:
    def __init__(
        self,
        model_name: str,
        key: int,
        m: int = 4,
        H: int = 4,
        top_k: int = 100,
        temperature: float = 0.7,
        distribution: str = "bernoulli",
        device: Optional[str] = None,
    ):
        """
        Args:
            model_name: HuggingFace model id, e.g. "google/gemma-2b-it".
            key: secret watermarking key (any int -- keep it fixed within
                an experiment so the detector can use the same key).
            m: number of tournament layers (see tournament.py for guidance).
            H: sliding-window context size for the seed generator.
            top_k: truncate the LLM distribution to the top-k tokens before
                sampling (paper default: 100).
            temperature: softmax temperature applied before top-k.
            distribution: g-value distribution, "bernoulli" or "uniform".
            device: "cuda", "cpu", or None to auto-detect.
        """
        self.key = key
        self.m = m
        self.H = H
        self.top_k = top_k
        self.temperature = temperature
        self.distribution = distribution

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _next_token_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature and top-k filtering, return a probability vector."""
        logits = logits / max(self.temperature, 1e-5)
        if self.top_k > 0:
            top_k = min(self.top_k, logits.shape[-1])
            values, indices = torch.topk(logits, top_k)
            filtered = torch.full_like(logits, float("-inf"))
            filtered.scatter_(0, indices, values)
            logits = filtered
        return torch.softmax(logits, dim=-1)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        watermark: bool = True,
    ) -> GenerationResult:
        """Generate a response to ``prompt``, token by token.

        Set ``watermark=False`` to generate with plain multinomial sampling
        from the same (temperature- and top-k-filtered) distribution -- this
        is your unwatermarked baseline/negative for detection experiments.
        """
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        generated: list[int] = []
        eos_id = self.tokenizer.eos_token_id

        for _ in range(max_new_tokens):
            outputs = self.model(input_ids)
            next_logits = outputs.logits[0, -1, :]
            probs = self._next_token_probs(next_logits)

            if watermark:
                next_token = multilayer_tournament_sample(
                    probs,
                    context_tokens=generated,
                    key=self.key,
                    m=self.m,
                    H=self.H,
                    distribution=self.distribution,
                )
            else:
                next_token = torch.multinomial(probs, 1).item()

            if next_token == eos_id:
                break

            generated.append(next_token)
            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_token]], device=self.device)], dim=1
            )

        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return GenerationResult(text=text, token_ids=generated, watermarked=watermark)