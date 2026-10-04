from textsus.gvalues.gvalues import generate_g_value


def test_g_value_is_deterministic():
    token_id = 42
    seed = 12345
    layer = 1

    g1 = generate_g_value(token_id, seed, layer)
    g2 = generate_g_value(token_id, seed, layer)

    assert g1 == g2


def test_bernoulli_g_value_is_zero_or_one():
    g_value = generate_g_value(
        token_id=42,
        seed=12345,
        layer=1,
        distribution="bernoulli",
    )

    assert g_value in (0, 1)


def test_uniform_g_value_is_between_zero_and_one():
    g_value = generate_g_value(
        token_id=42,
        seed=12345,
        layer=1,
        distribution="uniform",
    )

    assert 0 <= g_value < 1


def test_different_layer_can_produce_different_g_value():
    g1 = generate_g_value(
        token_id=42,
        seed=12345,
        layer=1,
        distribution="uniform",
    )

    g2 = generate_g_value(
        token_id=42,
        seed=12345,
        layer=2,
        distribution="uniform",
    )

    assert g1 != g2