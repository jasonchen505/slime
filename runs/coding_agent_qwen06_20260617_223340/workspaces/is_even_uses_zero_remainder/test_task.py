from numbers_local import is_even


def test_even_values():
    assert is_even(0)
    assert is_even(12)


def test_odd_values():
    assert not is_even(7)
