"""
Example: scalar-on-function regression with a non-negativity constraint.

Simulates one sample similar to the SOF scenario of the paper, fits FunSCOPE
with and without the constraint beta(t) >= 0, and saves a plot of the estimates.
Run from the repository root:  python examples/example_sof.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # not needed after `pip install .`
from funscope import fit_sof

# -----------------------------------------------------------------------------
# Simulated data: X_i(t) = sum_K psi_iK phi_K(t), y_i = 0.15 + int X_i beta + eps_i
# -----------------------------------------------------------------------------
rng = np.random.default_rng(1)
N, m = 50, 50
t = np.linspace(0, 1, m)
beta = np.exp(-((t - 0.5) ** 2) / 0.02) / np.sqrt(2 * np.pi * 0.01)  # N(0.5, 0.1^2) density

basis = np.polynomial.legendre.legvander(2 * t - 1, 19)               # polynomials of degree 0..19
scores = rng.normal(0, np.sqrt(np.arange(20, 0, -1)), size=(N, 20))
X = scores @ basis.T
y = 0.15 + X @ beta / m + rng.normal(0, 0.05, N)

# -----------------------------------------------------------------------------
# Fit
# -----------------------------------------------------------------------------
fit_u = fit_sof(X, t, y)                                  # unconstrained (FunSCOPE-U)
fit_c = fit_sof(X, t, y, constraints="nonnegative")       # beta(t) >= 0 (FunSCOPE)

for name, fit in [("unconstrained", fit_u), ("nonnegative", fit_c)]:
    print(f"{name:14s} MSE(y) = {fit['mse']:.5f}   min beta = {fit['beta'].min():+.4f}   time = {fit['time']:.2f}s")

# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------
plt.plot(t, beta, "b-", label="true")
plt.plot(t, fit_u["beta"], "g--", label="FunSCOPE-U")
plt.plot(t, fit_c["beta"], "r--", label="FunSCOPE (beta >= 0)")
plt.axhline(0, color="black", lw=0.5)
plt.xlabel("t")
plt.ylabel(r"$\beta(t)$")
plt.legend()
plt.savefig(Path(__file__).with_suffix(".png"))
print("Plot saved in", Path(__file__).with_suffix(".png"))
