"""Self-BLEU: measures inter-response diversity, used in the paper's
Extended Data Fig. 4 to show the detectability/diversity trade-off.

For each generated text, compute its BLEU score against all the OTHER
generated texts (treating them as references) and average. High Self-BLEU
means responses look similar to each other (low diversity); low Self-BLEU
means they're more varied.
"""

from __future__ import annotations

from typing import Sequence

import nltk
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


def _ensure_punkt():
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


def self_bleu(texts: Sequence[str]) -> float:
    """Compute the mean Self-BLEU score over a list of generated texts.

    Args:
        texts: generated responses (e.g. all watermarked responses from one run).

    Returns:
        Mean Self-BLEU in [0, 100] (BLEU-style percentage); NaN if fewer
        than 2 non-empty texts are given (Self-BLEU needs references).
    """
    _ensure_punkt()
    tokenized = [nltk.word_tokenize(t) for t in texts if t and t.strip()]

    if len(tokenized) < 2:
        return float("nan")

    smoothing = SmoothingFunction().method1
    scores = []
    for i, hypothesis in enumerate(tokenized):
        references = [tok for j, tok in enumerate(tokenized) if j != i]
        if not hypothesis:
            continue
        score = sentence_bleu(references, hypothesis, smoothing_function=smoothing)
        scores.append(score)

    return 100.0 * sum(scores) / len(scores) if scores else float("nan")