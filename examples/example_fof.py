"""
Example: concurrent function-on-function regression with a monotonicity constraint.

Simulates one sample similar to the FOF scenario of the paper (tanh-shaped
beta(t), correlated errors), fits FunSCOPE with and without the constraint
beta'(t) >= 0, and saves a plot of the estimates.
Requires R with the package `refund` (see the README). Each fit takes ~10-20 s.
Run from the repository root:  python examples/example_fof.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # not needed after `pip install .`
from funscope import fit_fof

# -----------------------------------------------------------------------------
# Simulated data: Y_i(t) = X_i(t) beta(t) + c_i cos(t) + d_i sin(t) + w_i(t)
# -----------------------------------------------------------------------------
rng = np.random.default_rng(1)
N, m = 25, 50
t = np.linspace(0, 1, m)
beta = np.tanh(10 * (t - 0.5))

basis = np.polynomial.legendre.legvander(2 * t - 1, 4)                # polynomials of degree 0..4
scores = rng.normal(size=(N, 5))
X = scores @ basis.T
E = (rng.normal(0, 0.1, (N, 1)) * np.cos(t) + rng.normal(0, 0.1, (N, 1)) * np.sin(t)
     + rng.normal(0, 0.1, (N, m)))
Y = X * beta[None, :] + E

# -----------------------------------------------------------------------------
# Fit
# -----------------------------------------------------------------------------
fit_u = fit_fof(X, t, Y)                                  # FunSCOPE-U
fit_c = fit_fof(X, t, Y, constraints="increasing")        # FunSCOPE

for name, fit in [("unconstrained", fit_u), ("increasing", fit_c)]:
    increasing = bool(np.all(np.diff(fit["beta"]) >= -1e-8))
    print(f"{name:14s} IMSE(y) = {fit['imse_y']:.5f}   beta increasing: {increasing}   time = {fit['time']:.2f}s")

# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------
plt.plot(t, beta, "b-", label="true")
plt.plot(t, fit_u["beta"], "g--", label="FunSCOPE-U")
plt.plot(t, fit_c["beta"], "r--", label="FunSCOPE (increasing)")
plt.xlabel("t")
plt.ylabel(r"$\beta(t)$")
plt.legend()
plt.savefig(Path(__file__).with_suffix(".png"))
print("Plot saved in", Path(__file__).with_suffix(".png"))
