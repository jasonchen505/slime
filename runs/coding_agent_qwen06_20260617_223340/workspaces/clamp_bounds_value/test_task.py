from math_utils import clamp


def test_clamp_inside():
    assert clamp(5, 1, 10) == 5


def test_clamp_low_high():
    assert clamp(-3, 1, 10) == 1
    assert clamp(30, 1, 10) == 10
