from typing import Union
from collections.abc import Iterable

Number = Union[int, float]


"""Implements a Polynomial."""


class Polynomial:
    """
    Polynomial with coefficients a_i such that coeffs[i] is
    the coefficient of x**i
    """
    def __init__(self, coeffs: Iterable[Number]) -> None:
        """Initialise."""
        coeffs_tuple = tuple(coeffs)
        if not coeffs_tuple:
            coeffs_tuple = (0,)
        self.coeffs = coeffs_tuple
        self.__trim()

    def __trim(self) -> None:
        """
        Trim the coefficients of any leading zeros.

        This particular method allows for more efficient trimming since we
        don't need to repeatedly slice the tuple, it is only sliced once.
        """
        i = len(self.coeffs) - 1
        while i > 0 and self.coeffs[i] == 0:
            i -= 1
        self.coeffs = self.coeffs[: i + 1]

    @property
    def degree(self) -> int:
        """Return degree of polynomial."""
        return len(self.coeffs) - 1

    @property
    def leading(self) -> Number:
        return self.coeffs[-1]

    @property
    def isZero(self) -> bool:
        return self.coeffs == (0,)

    def __str__(self) -> str:
        """Return a readable string representation, highest degree first."""
        if self.isZero:
            return "0"

        parts: list[str] = []
        for d in range(self.degree, -1, -1):  # Iterate from highest degree
            c = self.coeffs[d]
            if not c:
                continue

            # ---- sign ----
            if not parts:
                sign = "-" if c < 0 else ""
            else:
                sign = " - " if c < 0 else " + "

            # ---- coefficient string ----
            abs_c = abs(c)
            if abs_c == 1 and d != 0:
                coeff_str = ""
            else:
                if abs_c % 1 == 0:
                    coeff_str = str(int(abs(c)))
                else:
                    coeff_str = str(abs(c))

            # ---- variable / power ----
            if d == 1:
                var_str = "x"
            elif d == 0:
                var_str = ""
            else:
                var_str = f"x^{d}"
            parts.append(f"{sign}{coeff_str}{var_str}")

        return "".join(parts)

    def __repr__(self) -> str:
        return f"Polynomial({self.coeffs!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Polynomial):
            return self.coeffs == other.coeffs
        else:
            return False

    def __neg__(self) -> "Polynomial":
        return -1*self

    def __call__(self, x: Number) -> Number:
        """Implement Horner method for evaluating polynomials efficiently."""

        result: Number = 0
        for c in reversed(self.coeffs):
            result = result * x + c
        return result

    def __add__(self, other: 'Polynomial' | Number) -> "Polynomial":
        """Add two polynomials or a polynomial and a scalar."""

        if isinstance(other, Polynomial):
            coeff_sum = [a + b for a, b in zip(self.coeffs, other.coeffs)]
            index = min(self.degree + 1, other.degree + 1)
            coeff_sum += self.coeffs[index:] + other.coeffs[index:]
            return Polynomial(coeff_sum)
        elif isinstance(other, (int, float)):
            return Polynomial((self.coeffs[0] + other,) + self.coeffs[1:])
        else:
            return NotImplemented

    def __radd__(self, other: Number) -> "Polynomial":
        return self + other

    def __sub__(self, other: 'Polynomial' | Number) -> "Polynomial":
        """Subtract two polynomials or a scalar from polynomial."""

        if isinstance(other, Polynomial):
            dif = [a - b for a, b in zip(self.coeffs, other.coeffs)]
            index = min(self.degree + 1, other.degree + 1)
            dif += self.coeffs[index:] + (-other).coeffs[index:]
            return Polynomial(dif)
        elif isinstance(other, (int, float)):
            return Polynomial((self.coeffs[0] - other,) + self.coeffs[1:])
        else:
            return NotImplemented

    def __rsub__(self, other: Number) -> "Polynomial":
        return (-self) + other

    def __mul__(self, other: 'Polynomial' | Number) -> "Polynomial":
        """
        Multiply two polynomials or a Polynomials and a scalar

        Does this via convolution of the two coefficient arrays.
        I have implemented the convolution manually, which means there is no
        dependency on other libraries such as numpy.
        """

        if isinstance(other, Polynomial):
            degree = self.degree + other.degree
            product: list[Number] = (degree + 1)*[0.0]
            for i, self_coeff in enumerate(self.coeffs):
                for j, other_coeff in enumerate(other.coeffs):
                    product[i+j] += self_coeff * other_coeff
            return Polynomial(product)
        elif isinstance(other, (int, float)):
            return Polynomial(other * c for c in self.coeffs)
        else:
            return NotImplemented

    def __rmul__(self, other: Number) -> "Polynomial":
        return self * other

    def __pow__(self, exponent: int) -> "Polynomial":
        """Return the polynomial raised to a non-negative integer power.

        Uses exponentiation by squaring, requiring O(log k) multiplications
        where k is the exponent.
        """
        if not isinstance(exponent, int):
            raise TypeError("Exponent must be integer, got a"
                            + f"{type(exponent)} instead.")

        if exponent < 0:
            raise ValueError("Exponent must be a non-negative integer.")

        if exponent == 0:
            return Polynomial((1,))

        result = Polynomial((1,))
        base = self
        n = exponent

        while n > 0:
            if n % 2 == 1:
                result = result * base
            base = base * base
            n //= 2

        return result

    def __divmod__(self, other: 'Polynomial' | Number) -> tuple["Polynomial",
                                                                "Polynomial"]:
        """Implement polynomial division, returning quotient and remainder."""
        if isinstance(other, (int, float)):
            if not other:
                raise ZeroDivisionError(
                    "Cannot divide polynomial by zero scalar.")
            return (Polynomial(coeff / other for coeff in self.coeffs),
                    Polynomial((0,)))
        if not isinstance(other, Polynomial):
            return NotImplemented
        if other.degree == 0:
            if other.isZero:
                raise ZeroDivisionError(
                    "Cannot divide polynomial by the zero polynomial.")
            return (Polynomial(coeff / other.leading for coeff in self.coeffs),
                    Polynomial((0,)))

        if self.isZero:
            return Polynomial((0,)), Polynomial((0,))
        numerator = Polynomial(self.coeffs)
        denominator = other
        quotient_coeffs = []
        x = Polynomial((0, 1))
        while numerator.degree >= denominator.degree:
            difference = (numerator.degree - denominator.degree)
            quotient_coeffs.append(numerator.coeffs[-1]/denominator.leading)
            numerator -= quotient_coeffs[-1] * x**(difference) * denominator
        return (Polynomial(reversed(quotient_coeffs)), numerator)

    def __floordiv__(self, other: 'Polynomial' | Number) -> "Polynomial":
        """Return quotient of polynomial division, self // other."""
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError(
                    "Cannot divide polynomial by zero scalar.")
            return Polynomial(coeff // other for coeff in self.coeffs)
        if not isinstance(other, Polynomial):
            return NotImplemented
        if other.degree == 0:
            if other.isZero:
                raise ZeroDivisionError(
                    "Cannot divide polynomial by the zero polynomial.")
            return Polynomial(coeff // other.leading for coeff in self.coeffs)
        q, _ = divmod(self, other)
        return q

    def __mod__(self, other: 'Polynomial' | Number) -> "Polynomial":
        """Return remainder of polynomial division, self % other."""
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError(
                    "Cannot divide polynomial by scalar zero.")
            return Polynomial((0,))
        _, r = divmod(self, other)
        return r

    def derivative(self) -> "Polynomial":
        coeffs = tuple((d+1)*c for d, c in enumerate(self.coeffs[1:]))
        return Polynomial(coeffs)

    def integral(self, y_0: Number = 0) -> "Polynomial":
        """Return an antiderivative of the polynomial.

        The constant term is chosen so that the resulting polynomial satisfies
        F(0) = y_0 (default 0).
        """
        return Polynomial((y_0,) + tuple(c/(d+1) for d, c in
                                         enumerate(self.coeffs)))


if __name__ == "__main__":
    x = Polynomial((0, 1))

    p = x**5 + 5*x**2 + 1
    q = x**2 + 2*x + 1

    print("p(x) =", p)
    print("q(x) =", q)
    print("q(2) =", q(2))
    print("p + q =", p + q)
    print("p * q =", p * q)
    print("p' =", p.derivative())
    print("∫p dx =", p.integral())
    print("p^3 =", p**3)
    quotient, remainder = divmod(p, q)
    print(f"p/q = {quotient} r {remainder}")
