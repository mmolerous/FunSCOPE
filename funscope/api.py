"""User-friendly interface to the FunSCOPE estimators.

`fit_sof`, `fit_fos` and `fit_fof` call `SOFR_basis`, `FOSR_basis` and
`FOFR_basis` (the functions used in the paper) and return the results as a
dictionary. Shape constraints can be given by name:

    "nonnegative", "nonpositive", "increasing", "decreasing", "convex", "concave"
"""

import numpy as np

from .fof import FOFR_basis
from .fos import FOSR_basis
from .sof import SOFR_basis
from .utils import imse_simpson

# name -> (derivative order, sign)
CONSTRAINTS = {
    "nonnegative": (0, "+"),
    "nonpositive": (0, "-"),
    "increasing": (1, "+"),
    "decreasing": (1, "-"),
    "convex": (2, "+"),
    "concave": (2, "-"),
}


def make_constraints(constraints):
    """
    Translate shape constraints into the dictionary used by the estimators.

    `constraints` can be None (unconstrained), a name or a list of names from
    `CONSTRAINTS`, or an already built dictionary, e.g. {"X": {0: {"+": 0}}}.

    >>> make_constraints(["nonnegative", "convex"])
    {'X': {0: {'+': 0}, 2: {'+': 0}}}
    """
    if constraints is None or isinstance(constraints, dict):
        return constraints
    if isinstance(constraints, str):
        constraints = [constraints]
    by_derivative = {}
    for name in constraints:
        if name not in CONSTRAINTS:
            raise ValueError(f"Unknown constraint '{name}'. Use one of {list(CONSTRAINTS)}.")
        deriv, sign = CONSTRAINTS[name]
        by_derivative.setdefault(deriv, {})[sign] = 0
    return {"X": dict(sorted(by_derivative.items()))}


def _imse_rows(t, Y, Y_hat):
    return np.mean([imse_simpson(t, Y[i], Y_hat[i]) for i in range(len(Y))])


def fit_sof(X, t, y, constraints=None, beta_true=None, degree=3, knots=14, g=2000, centered=True):
    """
    Scalar-on-function regression: y_i = beta_0 + int X_i(t) beta(t) dt + eps_i.

    Parameters
    ----------
    X : array (N x m)
        Functional covariate, each row observed on the common grid `t`.
    t : array (m,)
        Observation grid (equispaced).
    y : array (N,)
        Scalar responses.
    constraints : None, str, list of str or dict
        Shape constraints on beta(t) (see `make_constraints`).
    beta_true : array (m,), optional
        True beta(t) on `t`, only for simulations (to compute the IMSE).
    degree, knots, g, centered :
        B-spline degree, number of intervals, intervals of the integration
        grid and centering of y (defaults of the paper).

    Returns
    -------
    dict with keys
        beta        estimated beta(t) on `t`
        intercept   estimated beta_0 (on the original scale of y)
        fitted      fitted responses (original scale)
        mse         (1/N) sum (y_i - fitted_i)^2
        imse_beta   IMSE of beta (NaN if `beta_true` is None)
        time        computing time (seconds)
        model       fitted optimization model
    """
    X = np.asarray(X, dtype=float)
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float).reshape(1, -1)
    beta_real = None if beta_true is None else np.asarray(beta_true, dtype=float).reshape(-1, 1)
    model, delta, beta, y_est, mse, imse, total_time, _, _ = SOFR_basis(
        X, t, y, beta_real, degree, knots, g, centered, make_constraints(constraints), [])
    shift = np.mean(y) if centered else 0.0
    return {
        "beta": beta,
        "intercept": delta[-1] + shift,
        "fitted": y_est + shift,
        "mse": mse,
        "imse_beta": imse,
        "time": total_time,
        "model": model,
    }


def _functional_response_result(out, t, Y, Y_hat, keys, constraints):
    """Common output of fit_fos / fit_fof."""
    res = dict(zip(keys, out))
    constrained = constraints is not None
    est = "cons" if constrained else "unc2"
    return {
        "beta": res[f"Beta_est_{est}"],
        "beta_unconstrained": res["Beta_est_unc2"],
        "fitted": Y_hat,
        "imse_y": _imse_rows(t, Y, Y_hat),
        "imse_beta": res[f"imse_approx_{est}"],
        "time": res["time"],
        "covariance_failed": bool(res["identity_used"]),
        "model": res["FOSR_cons" if "FOSR_cons" in res else "FOFR_cons"],
    }


_FOS_KEYS = [
    "FOSR_unc", "FOSR_unc2", "FOSR_cons", "Delta_unc", "Delta_unc2", "Delta_cons",
    "Beta_est_unc", "Beta_est_unc2", "Beta_est_cons", "Beta_est_cons_Phi_inner",
    "y_est_unc", "y_est_unc2", "y_est_cons", "MSE_unc", "MSE_unc2", "MSE_cons",
    "time", "Phi", "knots_print", "tt", "mediaY",
    "imse_approx_unc", "imse_approx_unc2", "imse_approx_cons",
    "imse_mse_unc", "imse_mse_unc2", "imse_mse_cons", "identity_used",
]
_FOF_KEYS = [k.replace("FOSR", "FOFR") for k in _FOS_KEYS if k != "mediaY"]


def fit_fos(X, t, Y, constraints=None, beta_true=None, degree=3, knots=14, g=2000):
    """
    Function-on-scalar regression: Y_i(t) = beta_0(t) + X_i beta(t) + eps_i(t).

    The error correlation is taken into account by generalized least squares,
    with the covariance estimated by FPCA (requires R, see the README). The
    data are centered (Y column-wise and X).

    Parameters
    ----------
    X : array (N,)
        Scalar covariate.
    t : array (m,)
        Observation grid (equispaced).
    Y : array (N x m)
        Functional responses, each row observed on `t`.
    constraints, beta_true, degree, knots, g :
        As in `fit_sof`.

    Returns
    -------
    dict with keys
        beta                estimated beta(t) on `t` (constrained if `constraints` is given)
        beta_unconstrained  unconstrained GLS estimate (FunSCOPE-U)
        fitted              fitted curves on `t`: mean(Y)(t) + (X_i - mean(X)) beta(t)
        imse_y              (1/N) sum (1/T) int (Y_i - fitted_i)^2, w.r.t. the observed curves
        imse_beta           IMSE of beta (NaN if `beta_true` is None)
        time                computing time (seconds)
        covariance_failed   True if the FPCA covariance estimation failed (identity used)
        model               fitted optimization model
    """
    X = np.asarray(X, dtype=float).reshape(-1, 1)
    t = np.asarray(t, dtype=float)
    Y = np.asarray(Y, dtype=float)
    beta_real = None if beta_true is None else np.asarray(beta_true, dtype=float).reshape(-1, 1)
    out = FOSR_basis(X, t, Y, beta_real, degree, knots, g, True, make_constraints(constraints), [], "FPCA")
    beta = dict(zip(_FOS_KEYS, out))["Beta_est_cons" if constraints is not None else "Beta_est_unc2"]
    Y_hat = Y.mean(axis=0, keepdims=True) + (X - X.mean()) * beta[None, :]
    return _functional_response_result(out, t, Y, Y_hat, _FOS_KEYS, constraints)


def fit_fof(X, t, Y, constraints=None, beta_true=None, degree=3, knots=14, g=2000):
    """
    Concurrent function-on-function regression: Y_i(t) = beta_0(t) + X_i(t) beta(t) + eps_i(t).

    The error correlation is taken into account by generalized least squares,
    with the covariance estimated by FPCA (requires R, see the README). Y is
    centered column-wise; X is used as given.

    Parameters
    ----------
    X : array (N x m)
        Functional covariate, each row observed on `t`.
    t : array (m,)
        Observation grid (equispaced).
    Y : array (N x m)
        Functional responses, each row observed on `t`.
    constraints, beta_true, degree, knots, g :
        As in `fit_sof`.

    Returns
    -------
    Same dictionary as `fit_fos`, with fitted curves mean(Y)(t) + X_i(t) beta(t).
    """
    X = np.asarray(X, dtype=float)
    t = np.asarray(t, dtype=float)
    Y = np.asarray(Y, dtype=float)
    beta_real = None if beta_true is None else np.asarray(beta_true, dtype=float).reshape(-1, 1)
    out = FOFR_basis(X, t, Y, beta_real, degree, knots, g, True, make_constraints(constraints), [], "FPCA")
    beta = dict(zip(_FOF_KEYS, out))["Beta_est_cons" if constraints is not None else "Beta_est_unc2"]
    Y_hat = Y.mean(axis=0, keepdims=True) + X * beta[None, :]
    return _functional_response_result(out, t, Y, Y_hat, _FOF_KEYS, constraints)
