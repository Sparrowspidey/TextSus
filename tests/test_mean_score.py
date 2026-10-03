from textsus.scoring.mean_score import mean_score


def test_mean_score_returns_float():
    token_ids = [10, 20, 30, 40]
    score = mean_score(
        token_ids=token_ids,
        watermarking_key=12345,
        num_layers=2,
    )

    assert isinstance(score, float)


def test_mean_score_is_between_zero_and_one():
    token_ids = [10, 20, 30, 40]
    score = mean_score(
        token_ids=token_ids,
        watermarking_key=12345,
        num_layers=2,
    )

    assert 0 <= score <= 1


def test_mean_score_is_deterministic():
    token_ids = [10, 20, 30, 40]
    key = 12345

    score1 = mean_score(token_ids, key, 2)
    score2 = mean_score(token_ids, key, 2)

    assert score1 == score2


def test_mean_score_rejects_empty_tokens():
    import pytest

    with pytest.raises(ValueError):
        mean_score([], 12345, 2)


def test_mean_score_rejects_invalid_layers():
    import pytest

    with pytest.raises(ValueError):
        mean_score([10, 20], 12345, 0)
        