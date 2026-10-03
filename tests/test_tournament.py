import pytest

from textsus.gvalues.gvalues import generate_g_value
from textsus.sampling.tournament import single_layer_tournament_sample


def test_tournament_returns_one_of_the_candidates():
    token_ids = [10, 20, 30, 40]
    seed = 12345

    selected = single_layer_tournament_sample(token_ids, seed)

    assert selected in token_ids


def test_tournament_requires_at_least_two_candidates():
    with pytest.raises(ValueError):
        single_layer_tournament_sample([10], 12345)


def test_tournament_selects_only_maximum_g_candidates():
    token_ids = [10, 20, 30, 40]
    seed = 12345

    g_values = [
        generate_g_value(token_id, seed, layer=1)
        for token_id in token_ids
    ]

    max_g_value = max(g_values)

    max_candidates = {
        token_id
        for token_id, g_value in zip(token_ids, g_values)
        if g_value == max_g_value
    }

    selected = single_layer_tournament_sample(token_ids, seed)

    assert selected in max_candidates