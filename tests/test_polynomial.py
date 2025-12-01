import pytest
from algebra_lab import Polynomial


def test_str():
    p = Polynomial((0, 1, 2, 3))
    assert str(p) == "3x^3 + 2x^2 + x"


def test_str_zero():
    p = Polynomial((0,))
    assert str(p) == "0"


def test_str_negative():
    p = Polynomial((1, -2, 0, 3))
    assert str(p) == "3x^3 - 2x + 1"


def test_addition():
    p = Polynomial((1, 2))
    q = Polynomial((3, -2, 4))
    r = p + q
    assert r.coeffs == (4, 0, 4)


def test_scaler_addition():
    p = Polynomial((3, -2, 4))
    x = 5
    r = p + x
    assert r.coeffs == (8, -2, 4)


def test_subtraction():
    p = Polynomial((1, 2))
    q = Polynomial((3, -2, 4))
    r = p - q
    assert r.coeffs == (-2, 4, -4)


def test_scaler_subtraction():
    p = Polynomial((3, -2, 4))
    x = 5
    r = p - x
    assert r.coeffs == (-2, -2, 4)


def test_multiplication():
    p = Polynomial((1, 1))      # x + 1
    q = Polynomial((1, -1))     # -x + 1
    r = p * q                   # 1 - x^2
    assert r.coeffs == (1, 0, -1)


def test_power_0():
    p = Polynomial((2, 3))
    assert (p ** 0).coeffs == (1,)


def test_power_1():
    p = Polynomial((2, 3))
    assert (p ** 1) == p


def test_power_3():
    p = Polynomial((1, 1))  # x + 1
    assert (p ** 3).coeffs == (1, 3, 3, 1)


def test_power_negative():
    p = Polynomial((1, 1))
    with pytest.raises(ValueError):
        p ** -1


def test_derivative():
    p = Polynomial((2, -5, 3))   # 3x^2 - 5x + 2
    assert p.derivative().coeffs == (-5, 6)


def test_integral():
    p = Polynomial((3, -10, -2))   # -2x^2 - 10x + 3
    y_0 = 17
    assert p.integral(y_0).coeffs == (17, 3, -5, -2/3)
