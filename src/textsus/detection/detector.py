"""Watermark detector: text + key -> score -> watermarked/not decision."""

from __future__ import annotations

from typing import Sequence

from textsus.scoring.mean_score import mean_score


class Detector:
    """Detects whether a piece of tokenized text was watermarked with the
    given key and watermarking configuration.

    The threshold should be calibrated empirically on unwatermarked text to
    hit a target false-positive rate -- see evaluation/metrics.tpr_at_fpr,
    which computes both the threshold and the resulting true-positive rate.
    """

    def __init__(
        self,
        key: int,
        m: int = 4,
        H: int = 4,
        distribution: str = "bernoulli",
        threshold: float = 0.6,
    ):
        self.key = key
        self.m = m
        self.H = H
        self.distribution = distribution
        self.threshold = threshold

    def score(self, token_ids: Sequence[int]) -> float:
        return mean_score(
            token_ids,
            key=self.key,
            m=self.m,
            H=self.H,
            distribution=self.distribution,
        )

    def is_watermarked(self, token_ids: Sequence[int]) -> bool:
        return self.score(token_ids) > self.threshold