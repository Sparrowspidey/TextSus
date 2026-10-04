"""Watermarked (and unwatermarked) text generation with a HuggingFace model.

Supports three sampling methods, selected via ``method``:
  - "tournament"    SynthID-Text's Tournament sampling (the paper's method)
  - "gumbel"        Aaronson & Kirchner's Gumbel sampling (non-distortionary baseline)
  - "soft_red_list" Kirchenbauer et al.'s Soft Red List (distortionary baseline)

All three share the same sliding-window random seed generator
(textsus.seed.random_seed), as in the paper's like-for-like comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from textsus.sampling.gumbel import gumbel_sample
from textsus.sampling.soft_red_list import apply_soft_red_list_bias
from textsus.sampling.tournament import tournament_sample
from textsus.seed.random_seed import generate_random_seed


@dataclass
class GenerationResult:
    text: str
    token_ids: list[int]
    watermarked: bool
    method: str


class WatermarkedGenerator:
    def __init__(
        self,
        model_name: str,
        key: int,
        m: int = 4,
        top_k: int = 100,
        temperature: float = 0.7,
        gamma: float = 0.5,
        delta: float = 2.0,
        device: Optional[str] = None,
    ):
        """
        Args:
            model_name: HuggingFace model id, e.g. "google/gemma-2b-it".
            key: secret watermarking key (any int -- keep it fixed within
                an experiment so the detector can use the same key).
            m: number of Tournament sampling layers (only used when
               method="tournament"). N = 2**m candidates are drawn per
               step -- keep this small (4-8) for CPU-scale experiments.
               Note: the sliding-window size H is fixed at 4 inside
               textsus.seed.random_seed (not configurable from here).
            top_k: truncate the LLM distribution to the top-k tokens before
                sampling (paper default: 100).
            temperature: softmax temperature applied before top-k.
            gamma: green-list fraction (only used when method="soft_red_list").
            delta: green-list logit bias strength (only used when
                method="soft_red_list"; higher = stronger, more distortionary).
            device: "cuda", "cpu", or None to auto-detect.
        """
        self.key = key
        self.m = m
        self.top_k = top_k
        self.temperature = temperature
        self.gamma = gamma
        self.delta = delta

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _top_k_logits(self, logits: torch.Tensor) -> dict[int, float]:
        """Apply temperature, then return a {token_id: logit} dict for the
        top-k tokens only (used by the soft_red_list method, which biases
        logits directly rather than probabilities)."""
        logits = logits / max(self.temperature, 1e-5)
        top_k = min(self.top_k, logits.shape[-1])
        values, indices = torch.topk(logits, top_k)
        return {int(idx): float(val) for idx, val in zip(indices.tolist(), values.tolist())}

    def _next_token_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature and top-k filtering, return a dense probability
        vector (used by the plain/unwatermarked and tournament methods)."""
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
        method: str = "tournament",
        ignore_eos: bool = False,
    ) -> GenerationResult:
        """Generate a response to ``prompt``, token by token.

        Set ``watermark=False`` to generate with plain multinomial sampling
        (ignores ``method``) -- this is your unwatermarked baseline/negative
        for detection experiments.

        ``method`` selects the watermarking scheme when watermark=True:
        "tournament" (default), "gumbel", or "soft_red_list".

        ``ignore_eos``: if True, never stop early on an end-of-sequence
        token -- always generate exactly ``max_new_tokens``. Use this for
        length-sweep experiments where every sample needs the same token
        count; leave it False for normal, natural-length generation.
        """
        if watermark and method not in ("tournament", "gumbel", "soft_red_list"):
            raise ValueError(f"Unknown method: {method!r}")

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        generated: list[int] = []
        eos_id = self.tokenizer.eos_token_id
        n_candidates = 2**self.m

        for _ in range(max_new_tokens):
            outputs = self.model(input_ids)
            next_logits = outputs.logits[0, -1, :]

            if not watermark:
                probs = self._next_token_probs(next_logits)
                next_token = torch.multinomial(probs, 1).item()

            elif method == "tournament":
                probs = self._next_token_probs(next_logits)
                candidates = torch.multinomial(
                    probs, n_candidates, replacement=True
                ).tolist()
                seed = generate_random_seed(generated, self.key)
                next_token = tournament_sample(candidates, seed, num_layers=self.m)

            elif method == "gumbel":
                probs = self._next_token_probs(next_logits)
                nonzero = torch.nonzero(probs, as_tuple=True)[0]
                probs_dict = {int(i): float(probs[i]) for i in nonzero}
                seed = generate_random_seed(generated, self.key)
                next_token = gumbel_sample(probs_dict, seed)

            else:  # soft_red_list
                logits_dict = self._top_k_logits(next_logits)
                seed = generate_random_seed(generated, self.key)
                biased = apply_soft_red_list_bias(
                    logits_dict, seed, gamma=self.gamma, delta=self.delta
                )
                token_ids = list(biased.keys())
                biased_logits = torch.tensor([biased[t] for t in token_ids])
                probs = torch.softmax(biased_logits, dim=-1)
                next_token = token_ids[torch.multinomial(probs, 1).item()]

            if next_token == eos_id and not ignore_eos:
                break

            generated.append(next_token)
            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_token]], device=self.device)], dim=1
            )

        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        used_method = method if watermark else "none"
        return GenerationResult(
            text=text, token_ids=generated, watermarked=watermark, method=used_method
        )