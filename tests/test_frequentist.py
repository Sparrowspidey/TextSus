import pytest

from textsus.scoring.frequentist import (
    frequentist_mean_score,
    frequentist_weighted_mean_score,
)


def test_frequentist_mean_score_returns_p_value():
    token_ids = [10, 20, 30, 40]

    p_value = frequentist_mean_score(
        token_ids=token_ids,
        watermarking_key=12345,
        num_layers=2,
    )

    assert isinstance(p_value, float)
    assert 0 <= p_value <= 1


def test_frequentist_mean_score_is_deterministic():
    token_ids = [10, 20, 30, 40]
    key = 12345

    p1 = frequentist_mean_score(token_ids, key, 2)
    p2 = frequentist_mean_score(token_ids, key, 2)

    assert p1 == p2


def test_frequentist_mean_score_rejects_empty_tokens():
    with pytest.raises(ValueError):
        frequentist_mean_score([], 12345, 2)


def test_frequentist_mean_score_rejects_invalid_layers():
    with pytest.raises(ValueError):
        frequentist_mean_score([10, 20], 12345, 0)


def test_frequentist_weighted_mean_score_returns_p_value():
    token_ids = [10, 20, 30, 40]

    p_value = frequentist_weighted_mean_score(
        token_ids=token_ids,
        watermarking_key=12345,
        layer_weights=[1.0, 0.5],
    )

    assert isinstance(p_value, float)
    assert 0 <= p_value <= 1


def test_frequentist_weighted_mean_score_is_deterministic():
    token_ids = [10, 20, 30, 40]
    key = 12345
    weights = [1.0, 0.5]

    p1 = frequentist_weighted_mean_score(token_ids, key, weights)
    p2 = frequentist_weighted_mean_score(token_ids, key, weights)

    assert p1 == p2


def test_frequentist_weighted_mean_score_rejects_empty_tokens():
    with pytest.raises(ValueError):
        frequentist_weighted_mean_score([], 12345, [1.0, 0.5])


def test_frequentist_weighted_mean_score_rejects_empty_weights():
    with pytest.raises(ValueError):
        frequentist_weighted_mean_score([10, 20], 12345, [])


def test_frequentist_weighted_mean_score_rejects_negative_weights():
    with pytest.raises(ValueError):
        frequentist_weighted_mean_score(
            [10, 20],
            12345,
            [1.0, -0.5],
        )


def test_frequentist_weighted_mean_score_rejects_zero_total_weight():
    with pytest.raises(ValueError):
        frequentist_weighted_mean_score(
            [10, 20],
            12345,
            [0.0, 0.0],
        )