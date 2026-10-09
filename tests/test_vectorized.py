import numpy as np
import pytest

from textsus.sampling.vectorized_tournament import vectorized_tournament_sample


def identity_g_values(token_ids: np.ndarray, layer: int) -> np.ndarray:
    return token_ids.astype(np.float64)


def test_single_layer_selects_maximum_in_each_match() -> None:
    candidates = np.array([[1, 4]], dtype=np.int64)

    selected = vectorized_tournament_sample(
        candidates,
        num_layers=1,
        num_competitors=2,
        g_value_fn=identity_g_values,
        rng=np.random.default_rng(10),
    )

    np.testing.assert_array_equal(selected, np.array([4]))


def test_multiple_layers_reduce_to_one_candidate_per_batch() -> None:
    candidates = np.array(
        [[0, 1, 2, 3, 4, 5, 6, 7], [8, 7, 6, 5, 4, 3, 2, 1]],
        dtype=np.int64,
    )

    selected = vectorized_tournament_sample(
        candidates,
        num_layers=3,
        num_competitors=2,
        g_value_fn=identity_g_values,
        rng=np.random.default_rng(11),
    )

    np.testing.assert_array_equal(selected, np.array([7, 8]))


def test_supports_more_than_two_competitors() -> None:
    candidates = np.array([[2, 1, 5, 4, 3, 9, 8, 7, 6]], dtype=np.int64)

    selected = vectorized_tournament_sample(
        candidates,
        num_layers=2,
        num_competitors=3,
        g_value_fn=identity_g_values,
        rng=np.random.default_rng(12),
    )

    np.testing.assert_array_equal(selected, np.array([9]))


def test_tie_returns_one_of_the_maximum_score_candidates() -> None:
    candidates = np.array([[10, 11, 12]], dtype=np.int64)

    selected = vectorized_tournament_sample(
        candidates,
        num_layers=1,
        num_competitors=3,
        g_value_fn=lambda token_ids, layer: np.ones(token_ids.shape),
        rng=np.random.default_rng(13),
    )

    assert selected.item() in {10, 11, 12}


def test_rejects_wrong_candidate_count() -> None:
    candidates = np.array([[1, 2, 3]], dtype=np.int64)

    with pytest.raises(ValueError, match="last candidate axis"):
        vectorized_tournament_sample(
            candidates,
            num_layers=2,
            num_competitors=2,
            g_value_fn=identity_g_values,
            rng=np.random.default_rng(14),
        )


def test_rejects_g_value_shape_mismatch() -> None:
    candidates = np.array([[1, 2]], dtype=np.int64)

    with pytest.raises(ValueError, match="same shape"):
        vectorized_tournament_sample(
            candidates,
            num_layers=1,
            num_competitors=2,
            g_value_fn=lambda token_ids, layer: np.array([1.0]),
            rng=np.random.default_rng(15),
        )


@pytest.mark.parametrize(
    ("num_layers", "num_competitors"),
    [(0, 2), (1, 1)],
)
def test_rejects_invalid_tournament_configuration(
    num_layers: int,
    num_competitors: int,
) -> None:
    with pytest.raises(ValueError):
        vectorized_tournament_sample(
            np.array([[1, 2]], dtype=np.int64),
            num_layers=num_layers,
            num_competitors=num_competitors,
            g_value_fn=identity_g_values,
            rng=np.random.default_rng(16),
        )
