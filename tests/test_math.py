import pytest
from src.safe_math import calculate


def test_arithmetic():
    assert calculate('(1250 * 0.11) + 1250') == 1387.5
    assert calculate('4**2') == 16
    assert calculate('-5+7') == 2


@pytest.mark.parametrize('expression', [
    '__import__("os").system("whoami")',
    'open(".env").read()',
    '2**100',
    '[1, 2, 3]',
    'sum([1, 2])',
])
def test_unsafe_expression_rejected(expression):
    with pytest.raises((ValueError, SyntaxError)):
        calculate(expression)


def test_zero_division():
    with pytest.raises(ZeroDivisionError):
        calculate('1/0')
