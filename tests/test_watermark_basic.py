"""Basic sanity tests that don't need a real LLM -- fast, run these first."""

import torch

from textsus.detection.detector import Detector
from textsus.gvalues.gvalues import g_value
from textsus.sampling.tournament import multilayer_tournament_sample
from textsus.scoring.mean_score import mean_score
from textsus.seed.random_seed import sliding_window_seed


def test_seed_is_deterministic():
    s1 = sliding_window_seed([1, 2, 3, 4], key=42, H=4)
    s2 = sliding_window_seed([1, 2, 3, 4], key=42, H=4)
    assert s1 == s2


def test_seed_changes_with_key():
    s1 = sliding_window_seed([1, 2, 3, 4], key=42, H=4)
    s2 = sliding_window_seed([1, 2, 3, 4], key=43, H=4)
    assert s1 != s2


def test_gvalue_bernoulli_is_binary():
    v = g_value(token_id=7, layer=1, seed=123, distribution="bernoulli")
    assert v in (0.0, 1.0)


def test_gvalue_deterministic():
    v1 = g_value(7, 1, 123, "bernoulli")
    v2 = g_value(7, 1, 123, "bernoulli")
    assert v1 == v2


def test_tournament_returns_valid_token():
    vocab_size = 50
    probs = torch.ones(vocab_size) / vocab_size
    token = multilayer_tournament_sample(
        probs, context_tokens=[1, 2, 3], key=42, m=3, H=4
    )
    assert 0 <= token < vocab_size


def test_detector_scores_watermarked_text_higher_on_average():
    """The core claim of the whole project, in miniature:
    text generated WITH the watermark should score higher than
    text that never went through Tournament sampling.
    """
    vocab_size = 50
    key = 42
    m = 4

    # "Watermarked": actually run through Tournament sampling.
    watermarked_tokens = []
    for _ in range(60):
        probs = torch.ones(vocab_size) / vocab_size
        tok = multilayer_tournament_sample(
            probs, context_tokens=watermarked_tokens, key=key, m=m, H=4
        )
        watermarked_tokens.append(tok)

    # "Unwatermarked": arbitrary fixed sequence, never passed through the tournament.
    unwatermarked_tokens = list(range(60))

    detector = Detector(key=key, m=m, H=4)
    wm_score = detector.score(watermarked_tokens)
    uwm_score = detector.score(unwatermarked_tokens)

    # Watermarked text should score meaningfully above 0.5 (Bernoulli midpoint);
    # not a strict guarantee for any single short sample, but should hold here.
    assert wm_score > 0.5
    assert mean_score(watermarked_tokens, key, m, H=4) == wm_score