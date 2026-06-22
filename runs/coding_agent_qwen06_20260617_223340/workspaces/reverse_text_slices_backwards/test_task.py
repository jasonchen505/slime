from strings import reverse_text


def test_reverse_text():
    assert reverse_text('abc') == 'cba'
    assert reverse_text('racecar') == 'racecar'
