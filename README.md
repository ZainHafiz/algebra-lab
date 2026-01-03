# algebra-lab

## Overview
A small pure-Python algebra library for univariate polynomials.

## Features
- Dense coefficient representation (coeffs[i] is the coefficient of x**i)
- Operator overloading: +, -, *, **, divmod, //, %, evaluation via p(x)
- Calculus: derivative and antiderivative (with initial condition)
- Manual convolution for multiplication (no external numerical dependencies)
- Fully tested with pytest and type-checked with mypy

## Usage

```py
>>> from algebra_lab import Polynomial
>>> x = Polynomial(0, 1)  # represents the variable x
>>> p = x**2 + 3*x + 2
>>> print(p)
x^2 + 3x + 2
>>> p(2)
12
>>> q = x + 1
>>> p + q
x^2 + 4x + 3
>>> p // q
x + 2
```

## Installation

```bash
pip install -e .
```

## Design Choices
- Dense coefficient representation is efficient and simple when polynomial degrees are small (e.g. <10), which is the intended use case for this library.
- Manual convolution was implemented to avoid external dependencies and to maintain full control and clarity over the algorithm.
