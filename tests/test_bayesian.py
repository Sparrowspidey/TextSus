import pytest

from textsus.scoring.bayesian import bayesian_score


def test_bayesian_score_returns_probability():
    token_ids = [10, 20, 30, 40]

    parameters = {
        "beta": [0.0, 0.0],
        "delta": [
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        "prior": 0.5,
    }

    posterior = bayesian_score(
        token_ids=token_ids,
        watermarking_key=12345,
        parameters=parameters,
    )

    assert isinstance(posterior, float)
    assert 0 <= posterior <= 1


def test_bayesian_score_is_deterministic():
    token_ids = [10, 20, 30, 40]

    parameters = {
        "beta": [0.0, 0.0],
        "delta": [
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        "prior": 0.5,
    }

    p1 = bayesian_score(token_ids, 12345, parameters)
    p2 = bayesian_score(token_ids, 12345, parameters)

    assert p1 == p2


def test_bayesian_score_rejects_empty_tokens():
    parameters = {
        "beta": [0.0, 0.0],
        "delta": [
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        "prior": 0.5,
    }

    with pytest.raises(ValueError):
        bayesian_score([], 12345, parameters)


def test_bayesian_score_rejects_invalid_prior():
    parameters = {
        "beta": [0.0, 0.0],
        "delta": [
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        "prior": 1.5,
    }

    with pytest.raises(ValueError):
        bayesian_score(
            [10, 20],
            12345,
            parameters,
        )


def test_bayesian_score_changes_with_parameters():
    token_ids = [10, 20, 30, 40]

    parameters_1 = {
        "beta": [0.0, 0.0],
        "delta": [
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        "prior": 0.5,
    }

    parameters_2 = {
        "beta": [2.0, 2.0],
        "delta": [
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        "prior": 0.5,
    }

    p1 = bayesian_score(token_ids, 12345, parameters_1)
    p2 = bayesian_score(token_ids, 12345, parameters_2)

    assert p1 != p2