import pytest

from textsus.gvalues.gvalues import generate_g_value
from textsus.sampling.tournament import (
    single_layer_tournament_sample,
    tournament_sample,
)


def test_single_layer_tournament_returns_one_of_the_candidates():
    token_ids = [10, 20, 30, 40]
    seed = 12345

    selected = single_layer_tournament_sample(
        token_ids,
        seed,
        num_competitors=4,
    )

    assert selected in token_ids


def test_single_layer_tournament_selects_only_maximum_g_candidates():
    token_ids = [10, 20, 30, 40]
    seed = 12345

    g_values = [
        generate_g_value(
            token_id,
            seed,
            layer=1,
        )
        for token_id in token_ids
    ]

    max_g_value = max(g_values)

    max_candidates = {
        token_id
        for token_id, g_value in zip(token_ids, g_values)
        if g_value == max_g_value
    }

    selected = single_layer_tournament_sample(
        token_ids,
        seed,
        num_competitors=4,
    )

    assert selected in max_candidates


def test_single_layer_tournament_requires_at_least_two_competitors():
    with pytest.raises(ValueError):
        single_layer_tournament_sample(
            [10],
            12345,
            num_competitors=1,
        )


def test_single_layer_tournament_requires_correct_number_of_candidates():
    with pytest.raises(ValueError):
        single_layer_tournament_sample(
            [10, 20, 30],
            12345,
            num_competitors=4,
        )


def test_two_competitor_tournament_returns_one_of_candidates():
    token_ids = list(range(8))
    seed = 12345

    selected = tournament_sample(
        token_ids,
        seed,
        num_layers=3,
        num_competitors=2,
    )

    assert selected in token_ids


def test_three_competitor_tournament_returns_one_of_candidates():
    token_ids = list(range(27))
    seed = 12345

    selected = tournament_sample(
        token_ids,
        seed,
        num_layers=3,
        num_competitors=3,
    )

    assert selected in token_ids


def test_four_competitor_tournament_returns_one_of_candidates():
    token_ids = list(range(64))
    seed = 12345

    selected = tournament_sample(
        token_ids,
        seed,
        num_layers=3,
        num_competitors=4,
    )

    assert selected in token_ids


def test_tournament_requires_power_of_n_candidates():
    with pytest.raises(ValueError):
        tournament_sample(
            list(range(10)),
            12345,
            num_layers=3,
            num_competitors=2,
        )


def test_tournament_requires_at_least_two_competitors():
    with pytest.raises(ValueError):
        tournament_sample(
            [10],
            12345,
            num_layers=1,
            num_competitors=1,
        )


def test_tournament_requires_at_least_one_layer():
    with pytest.raises(ValueError):
        tournament_sample(
            [10, 20],
            12345,
            num_layers=0,
            num_competitors=2,
        )


def test_tournament_uses_n_power_m_candidates():
    token_ids = list(range(25))
    seed = 12345

    selected = tournament_sample(
        token_ids,
        seed,
        num_layers=2,
        num_competitors=5,
    )

    assert selected in token_ids