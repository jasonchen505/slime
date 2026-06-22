from compare import max_of_two


def test_max_of_two():
    assert max_of_two(1, 9) == 9
    assert max_of_two(10, 3) == 10
    assert max_of_two(4, 4) == 4
