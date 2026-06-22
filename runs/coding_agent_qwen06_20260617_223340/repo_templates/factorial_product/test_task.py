from factorial import factorial


def test_factorial_base_cases():
    assert factorial(0) == 1
    assert factorial(1) == 1


def test_factorial_values():
    assert factorial(5) == 120
