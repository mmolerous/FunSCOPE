"""Numerical integration helpers (Simpson's rule) shared by all models."""

import numpy as np
from scipy.integrate import simpson


def simpsons_weights(lb, ub, n):
    """Diagonal matrix of weights of Simpson's composite rule on n+1 equispaced points of [lb, ub]."""
    if n % 2 == 1:
        raise ValueError("Number of intervals must be even for Simpson's rule.")

    weights = np.ones(n + 1)
    weights[1:-1:2] = 4  # odd indices
    weights[2:-1:2] = 2  # even indices
    h = (ub - lb) / n
    return np.diag((h / 3) * weights)


def imse_simpson(t, f, g):
    """Integrated squared error between f and g over t (Simpson's rule), divided by the domain length."""
    y = (f - g) ** 2
    integral = simpson(y=y, x=t)
    return integral / (max(t) - min(t))
