import math

import pytest

from textsus.sampling.distortionary_tournament import distortionary_tournament_sample
from textsus.sampling.gumbel import gumbel_mean_score, gumbel_sample, gumbel_token_score
from textsus.sampling.soft_red_list import (
    apply_soft_red_list_bias,
    green_fraction_score,
    is_green,
    z_score,
)


# ---------- Gumbel ----------
def test_gumbel_sample_returns_a_candidate():
    probs = {1: 0.5, 2: 0.3, 3: 0.2}
    assert gumbel_sample(probs, seed=123) in probs


def test_gumbel_sample_is_deterministic_for_same_seed():
    probs = {1: 0.5, 2: 0.3, 3: 0.2}
    assert gumbel_sample(probs, 7) == gumbel_sample(probs, 7)


def test_gumbel_sample_ignores_zero_probability_tokens():
    probs = {1: 0.0, 2: 1.0}
    for seed in range(20):
        assert gumbel_sample(probs, seed) == 2


def test_gumbel_sample_rejects_empty_probs():
    with pytest.raises(ValueError):
        gumbel_sample({}, 1)


def test_gumbel_token_score_is_non_negative():
    assert gumbel_token_score(5, 99) >= 0.0


def test_gumbel_mean_score_rejects_empty_tokens():
    with pytest.raises(ValueError):
        gumbel_mean_score([], 42)


def test_gumbel_mean_score_is_deterministic():
    tokens = [3, 14, 15, 92, 65, 35]
    assert gumbel_mean_score(tokens, 42) == gumbel_mean_score(tokens, 42)


# ---------- Soft Red List ----------
def test_is_green_is_deterministic():
    assert is_green(10, 5) == is_green(10, 5)


def test_gamma_one_makes_everything_green_and_zero_nothing():
    assert all(is_green(t, 1, gamma=1.0) for t in range(50))
    assert not any(is_green(t, 1, gamma=0.0) for t in range(50))


def test_bias_only_raises_green_tokens_by_delta():
    logits = {t: 0.0 for t in range(30)}
    biased = apply_soft_red_list_bias(logits, seed=3, gamma=0.5, delta=2.0)
    for t in logits:
        expected = 2.0 if is_green(t, 3, 0.5) else 0.0
        assert biased[t] == expected


def test_zero_delta_leaves_logits_unchanged():
    logits = {1: 1.5, 2: -0.5}
    assert apply_soft_red_list_bias(logits, 9, delta=0.0) == logits


def test_green_fraction_is_between_zero_and_one():
    assert 0.0 <= green_fraction_score([1, 2, 3, 4, 5, 6], 42) <= 1.0


def test_z_score_rejects_empty_tokens():
    with pytest.raises(ValueError):
        z_score([], 42)


def test_z_score_is_finite():
    assert math.isfinite(z_score(list(range(40)), 42))


# ---------- Distortionary Tournament ----------
def test_distortionary_returns_one_of_the_candidates():
    tokens = list(range(27))
    assert distortionary_tournament_sample(tokens, 11, 3, 3) in tokens


def test_distortionary_rejects_wrong_candidate_count():
    with pytest.raises(ValueError):
        distortionary_tournament_sample(list(range(10)), 11, 3, 3)


def test_distortionary_rejects_too_few_competitors():
    with pytest.raises(ValueError):
        distortionary_tournament_sample([1], 11, 1, 1)


def test_distortionary_rejects_zero_layers():
    with pytest.raises(ValueError):
        distortionary_tournament_sample([1, 2], 11, 0, 2)