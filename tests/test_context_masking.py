from textsus.sampling.context_masking import is_context_repeated
from textsus.sampling.context_masking import (
    create_context_history,
    is_context_repeated,
)


def test_context_history_keeps_k_responses():
    history = create_context_history(2)

    history.append({(1, 2, 3, 4)})
    history.append({(5, 6, 7, 8)})

    assert len(history) == 2


def test_context_history_discards_old_responses():
    history = create_context_history(2)

    history.append({(1, 2, 3, 4)})
    history.append({(5, 6, 7, 8)})
    history.append({(9, 10, 11, 12)})

    assert len(history) == 2
    assert (1, 2, 3, 4) not in history[0]
    assert (1, 2, 3, 4) not in history[1]


def test_context_is_repeated_across_k_responses():
    history = create_context_history(3)

    history.append({(1, 2, 3, 4)})
    history.append({(5, 6, 7, 8)})
    history.append({(9, 10, 11, 12)})

    assert is_context_repeated((1, 2, 3, 4), history) is True

def test_repeated_context_is_detected():
    context = (1, 2, 3, 4)
    history = [
        {(1, 2, 3, 4)},
    ]

    assert is_context_repeated(context, history) is True


def test_new_context_is_not_repeated():
    context = (5, 6, 7, 8)
    history = [
        {(1, 2, 3, 4)},
    ]

    assert is_context_repeated(context, history) is False