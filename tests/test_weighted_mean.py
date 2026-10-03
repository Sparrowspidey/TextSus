import pytest

from textsus.scoring.weighted_mean import weighted_mean_score


def test_weighted_mean_returns_float():
    token_ids = [10, 20, 30, 40]
    score = weighted_mean_score(
        token_ids=token_ids,
        watermarking_key=12345,
        layer_weights=[3, 2, 1],
    )
    assert isinstance(score, float)


def test_weighted_mean_is_between_zero_and_one():
    token_ids = [10, 20, 30, 40]
    score = weighted_mean_score(
        token_ids=token_ids,
        watermarking_key=12345,
        layer_weights=[3, 2, 1],
    )
    assert 0 <= score <= 1


def test_weighted_mean_is_deterministic():
    token_ids = [10, 20, 30, 40]
    key = 12345
    weights = [3, 2, 1]

    score1 = weighted_mean_score(token_ids, key, weights)
    score2 = weighted_mean_score(token_ids, key, weights)

    assert score1 == score2


def test_weighted_mean_rejects_empty_tokens():
    with pytest.raises(ValueError):
        weighted_mean_score(
            token_ids=[],
            watermarking_key=12345,
            layer_weights=[3, 2, 1],
        )


def test_weighted_mean_rejects_empty_weights():
    with pytest.raises(ValueError):
        weighted_mean_score(
            token_ids=[10, 20],
            watermarking_key=12345,
            layer_weights=[],
        )


def test_weighted_mean_rejects_negative_weights():
    with pytest.raises(ValueError):
        weighted_mean_score(
            token_ids=[10, 20],
            watermarking_key=12345,
            layer_weights=[3, -1, 1],
        )


def test_weighted_mean_rejects_zero_total_weight():
    with pytest.raises(ValueError):
        weighted_mean_score(
            token_ids=[10, 20],
            watermarking_key=12345,
            layer_weights=[0, 0, 0],
        )