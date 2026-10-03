from textsus.seed.random_seed import generate_random_seed


def test_seed_is_deterministic():
    tokens = [10, 20, 30, 40]
    key = 12345

    seed1 = generate_random_seed(tokens, key)
    seed2 = generate_random_seed(tokens, key)

    assert seed1 == seed2
def test_seed_uses_last_four_tokens():
    key = 12345

    tokens1 = [1, 2, 10, 20, 30, 40]
    tokens2 = [99, 88, 10, 20, 30, 40]

    seed1 = generate_random_seed(tokens1, key)
    seed2 = generate_random_seed(tokens2, key)

    assert seed1 == seed2
def test_different_key_produces_different_seed():
    tokens = [10, 20, 30, 40]

    seed1 = generate_random_seed(tokens, 12345)
    seed2 = generate_random_seed(tokens, 54321)

    assert seed1 != seed2
    