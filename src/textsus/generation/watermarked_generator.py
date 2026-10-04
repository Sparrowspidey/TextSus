"""Watermarked (and unwatermarked) text generation with a HuggingFace model.

Plugs the team's real Tournament sampling implementation
(textsus.sampling.tournament) into a token-by-token generation loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from textsus.sampling.tournament import tournament_sample
from textsus.seed.random_seed import generate_random_seed


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
        top_k: int = 100,
        temperature: float = 0.7,
        device: Optional[str] = None,
    ):
        """
        Args:
            model_name: HuggingFace model id, e.g. "google/gemma-2b-it".
            key: secret watermarking key (any int -- keep it fixed within
                an experiment so the detector can use the same key).
            m: number of tournament layers. N = 2**m candidates are drawn
               per step -- keep this small (4-8) for CPU-scale experiments.
               Note: the sliding-window size H is fixed at 4 inside
               textsus.seed.random_seed (not configurable from here).
            top_k: truncate the LLM distribution to the top-k tokens before
                sampling (paper default: 100).
            temperature: softmax temperature applied before top-k.
            device: "cuda", "cpu", or None to auto-detect.
        """
        self.key = key
        self.m = m
        self.top_k = top_k
        self.temperature = temperature

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
        n_candidates = 2**self.m

        for _ in range(max_new_tokens):
            outputs = self.model(input_ids)
            next_logits = outputs.logits[0, -1, :]
            probs = self._next_token_probs(next_logits)

            if watermark:
                candidates = torch.multinomial(
                    probs, n_candidates, replacement=True
                ).tolist()
                seed = generate_random_seed(generated, self.key)
                next_token = tournament_sample(candidates, seed, num_layers=self.m)
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