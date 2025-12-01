from numbers import Number, Integral


"""Implements a Polynomial."""


class Polynomial:
    """
    Polynomial with coefficients a_i such that coeffs[i] is
    the coefficient of x**i
    """
    def __init__(self, coeffs) -> None:
        """Initialise."""
        coeffs_tuple = tuple(coeffs)
        if not coeffs_tuple:
            coeffs_tuple = (0,)
        self._coeffs = coeffs_tuple
        self._trim()

    def _trim(self) -> None:
        """
        Trim the coefficients of any leading zeros.

        This particular method allows for more efficient trimming since we
        don't need to repeatedly slice the tuple, it is only sliced once.
        """
        i = len(self._coeffs) - 1
        while i > 0 and self._coeffs[i] == 0:
            i -= 1
        self._coeffs = self._coeffs[: i + 1]

    @property
    def coeffs(self):
        """Allow coeffs tuple to be read-only."""
        return self._coeffs
    @property
    def degree(self):
        """Return degree of polynomial."""
        return len(self._coeffs) - 1

    def __str__(self):
        """Return a readable string representation, with the highest degree first."""
        if self._coeffs == (0,):
            return "0"
        poly = []
        sign = ""
        for d, c in enumerate(self._coeffs):
            if c:
                if c > 0:
                    sign = " + "
                elif c < 0:
                    sign = " - "
                if abs(c) == 1 and d != 0:
                    coeff = ""
                else:
                    coeff = abs(c)
                if 1 < d and d < self.degree:
                    poly.append(sign + f"{coeff}x^{d}")
                elif d == self.degree:
                    if c < 0:
                        poly.append(f"-{coeff}x^{d}")
                    else:
                        poly.append(f"{coeff}x^{d}")
                elif d == 1:
                    poly.append(sign + f"{coeff}x")
                else:
                    poly.append(sign + f"{coeff}")

        return "".join(reversed(poly))

    def __repr__(self):
        return f"Polynomial({self._coeffs!r})"

    def __eq__(self, other):
        if isinstance(other, Polynomial):
            return self._coeffs == other.coeffs
        else:
            return False

    def __neg__(self):
        return -1*self

    def __call__(self, x):
        """Implement Horner method for evaluating polynomials efficiently."""

        result = self._coeffs[-1]
        for i in range(self.degree):
            result = self._coeffs[(self.degree - 1) - i] + x * result
        return result

    def __add__(self, other):
        """Add two polynomials or a polynomial and a scaler."""

        if isinstance(other, Polynomial):
            sum = [a + b for a, b in zip(self._coeffs, other.coeffs)]
            index = min(self.degree + 1, other.degree + 1)
            sum += self._coeffs[index:] + other.coeffs[index:]
            return Polynomial(sum)
        elif isinstance(other, Number):
            return Polynomial((self._coeffs[0] + other,) + self._coeffs[1:])
        else:
            return NotImplemented

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        """Subtract two polynomials or a scaler from polynomial."""

        if isinstance(other, Polynomial):
            dif = [a - b for a, b in zip(self._coeffs, other.coeffs)]
            index = min(self.degree + 1, other.degree + 1)
            dif += self._coeffs[index:] + (-1*other).coeffs[index:]
            return Polynomial(dif)
        elif isinstance(other, Number):
            return Polynomial((self._coeffs[0] - other,) + self._coeffs[1:])
        else:
            return NotImplemented

    def __rsub__(self, other):
        return -1*(self - other)

    def __mul__(self, other):
        """
        Multiply two polynomials or a Polynomials and a scaler

        Does this via convolution of the two coefficient arrays. I have implemented the convolution manually,
        which means there is no dependency on other libraries such as numpy.
        """

        if isinstance(other, Polynomial):
            degree = self.degree + other.degree
            product = (degree + 1)*[0]
            for i, self_coeff in enumerate(self._coeffs):
                for j, other_coeff in enumerate(other.coeffs):
                    product[i+j] += self_coeff * other_coeff
            return Polynomial(product)
        elif isinstance(other, Number):
            return Polynomial(other * c for c in self._coeffs)
        else:
            return NotImplemented

    def __rmul__(self, other):
        return self * other

    def __pow__(self, exponent):
        """Return the polynomial raised to a non-negative integer power.

        Uses exponentiation by squaring, requiring O(log k) multiplications
        where k is the exponent.
        """
        if not isinstance(exponent, Integral):
            raise TypeError("Exponent must be integer.")

        exponent = int(exponent)
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

    def derivative(self):
        coeffs = tuple((d+1)*c for d, c in enumerate(self._coeffs[1:]))
        return Polynomial(coeffs)

    def integral(self, y_0: Number = 0) -> "Polynomial":
        """Return an antiderivative of the polynomial.

        The constant term is chosen so that the resulting polynomial satisfies
        F(0) = y_0 (default 0).
        """
        return Polynomial((y_0,) + tuple(c/(d+1) for d, c in
                                         enumerate(self._coeffs)))


if __name__ == "__main__":
    p = Polynomial((2, -5, 3))   # 3x^2 - 5x + 2
    q = Polynomial((1, 1))       # x + 1

    print("p(x) =", p)           # 3x^2 - 5x + 2
    print("q(x) =", q)           # x + 1
    print("p(2) =", p(2))        # 4
    print("p + q =", p + q)      # 3x^2 - 4x + 3
    print("p * q =", p * q)      # 3x^3 - 2x^2 - 3x + 2
    print("p'   =", p.derivative())
    print("∫p dx =", p.integral())
    print("p^3   =", p**3)
