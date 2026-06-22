from arithmetic import safe_divide


def test_safe_divide_regular():
    assert safe_divide(8, 2) == 4


def test_safe_divide_zero():
    assert safe_divide(8, 0) is None
