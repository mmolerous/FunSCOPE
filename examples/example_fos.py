"""
Example: function-on-scalar regression with non-negativity and convexity constraints.

Simulates one sample similar to the FOS scenario of the paper (Weibull-shaped
beta(t), correlated errors), fits FunSCOPE with and without the constraints
beta(t) >= 0 and beta''(t) >= 0, and saves a plot of the estimates.
Requires R with the package `refund` (see the README).
Run from the repository root:  python examples/example_fos.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # not needed after `pip install .`
from funscope import fit_fos

# -----------------------------------------------------------------------------
# Simulated data: Y_i(t) = X_i beta(t) + c_i cos(t) + d_i sin(t) + w_i(t)
# -----------------------------------------------------------------------------
rng = np.random.default_rng(1)
N, m = 50, 50
t = np.linspace(0.1, 2.1, m)
k, lam = 0.75, 0.25
beta = (k / lam) * (t / lam) ** (k - 1) * np.exp(-((t / lam) ** k))   # Weibull(0.75, 0.25) density

X = rng.uniform(0.5, 1.5, N)
E = (rng.normal(0, 0.1, (N, 1)) * np.cos(t) + rng.normal(0, 0.1, (N, 1)) * np.sin(t)
     + rng.normal(0, 0.1, (N, m)))
Y = X[:, None] * beta[None, :] + E

# -----------------------------------------------------------------------------
# Fit
# -----------------------------------------------------------------------------
fit_u = fit_fos(X, t, Y)                                            # FunSCOPE-U
fit_c = fit_fos(X, t, Y, constraints=["nonnegative", "convex"])     # FunSCOPE

for name, fit in [("unconstrained", fit_u), ("nonneg.+convex", fit_c)]:
    print(f"{name:14s} IMSE(y) = {fit['imse_y']:.5f}   min beta = {fit['beta'].min():+.4f}   time = {fit['time']:.2f}s")

# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------
plt.plot(t, beta, "b-", label="true")
plt.plot(t, fit_u["beta"], "g--", label="FunSCOPE-U")
plt.plot(t, fit_c["beta"], "r--", label="FunSCOPE (beta >= 0, convex)")
plt.axhline(0, color="black", lw=0.5)
plt.xlabel("t")
plt.ylabel(r"$\beta(t)$")
plt.legend()
plt.savefig(Path(__file__).with_suffix(".png"))
print("Plot saved in", Path(__file__).with_suffix(".png"))
