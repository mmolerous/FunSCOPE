"""Building blocks shared by the SOF, FOS and FOF estimators."""

import os
import sys
import warnings
from contextlib import contextmanager

import numpy as np
import pandas as pd
from cpsplines.fittings.fit_cpsplines import CPsplines
from cpsplines.psplines.bspline_basis import BsplineBasis

from .cpsplines_design import Fun_CPsplines_intercept
from .utils import imse_simpson, simpsons_weights


@contextmanager
def silenced():
    """Temporarily silence stdout and warnings (cpsplines and MOSEK are verbose)."""
    saved_stdout = sys.stdout
    with open(os.devnull, "w") as devnull, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = saved_stdout


def bspline_basis(x, degree, knots):
    """Cubic (by default) B-spline basis with `knots` equispaced intervals, evaluated at `x`."""
    basis = BsplineBasis(x=x, deg=degree, k=knots)
    basis.get_matrix_B()
    return basis


def presmooth(curves, t, degree, knots, n_coef):
    """
    Pre-smooth each row of `curves` (observed at `t`) with a P-spline.

    Returns the (n_curves x n_coef) matrix of B-spline coefficients.
    """
    coef = np.zeros((np.shape(curves)[0], n_coef))
    for j in range(np.shape(curves)[0]):
        data = pd.DataFrame({"x": t, "y": curves[j, :]})
        fit = CPsplines(
            deg=(degree,),
            ord_d=(2,),
            k=(knots,),
        )
        fit.fit(data=data, y_col="y")
        coef[j, :] = fit.sol
    return coef


def fine_grid(t, g):
    """Equispaced grid of g + 1 points on [min(t), max(t)], used for numerical integration."""
    return np.arange(min(t), max(t) + (max(t) - min(t)) / g, (max(t) - min(t)) / g)


def gram_matrix(Phi_inner, t, g):
    """Psi = integral of Phi(s)^T Phi(s) ds, approximated with Simpson's rule on the fine grid."""
    Sim_w = simpsons_weights(min(t), max(t), g)
    return np.transpose(Phi_inner) @ Sim_w @ Phi_inner


def sqrtm_eig(Psi):
    """Matrix square root of Psi through its (general) eigendecomposition. Used in FOS."""
    eigvals, eigvecs = np.linalg.eig(Psi)
    sqrt_eigvals = np.diag(np.sqrt(eigvals))
    return eigvecs @ sqrt_eigvals @ np.linalg.inv(eigvecs)


def sqrtm_sym(Psi):
    """Matrix square root of the symmetric PSD matrix Psi through `eigh`. Used in FOF."""
    Psi = (Psi + Psi.T) / 2  # enforce exact symmetry
    eigvals, eigvecs = np.linalg.eigh(Psi)
    eigvals = np.clip(eigvals, 0, None)  # guard against tiny negative eigenvalues
    sqrt_eigvals = np.diag(np.sqrt(eigvals))
    return eigvecs @ sqrt_eigvals @ eigvecs.T


def imse_beta(t, Beta_real, Beta_est):
    """IMSE of the estimated beta(t); NaN when the true beta(t) is unknown (Beta_real is None)."""
    if Beta_real is None:
        return np.nan
    return imse_simpson(t, np.array(Beta_real)[:, 0], Beta_est)


def fit_design(
    B,
    x_aux,
    y,
    degree,
    knots,
    int_constraint=None,
    alpha_constraint=None,
    retry_scaled=False,
    **kwargs,
):
    """
    Fit a (possibly shape-constrained) P-spline with design matrix `B`.

    The last column of `B` is an unpenalized intercept. `x_aux` is an auxiliary
    covariate: only its range is used, to build the knots on which the shape
    constraints are formulated. Extra keyword arguments (e.g. `sp_method`,
    `sp_args`) are passed to `Fun_CPsplines_intercept`. If `retry_scaled` is
    True and the solver fails, the fit is repeated with the response scaled to
    [0, 1].
    """
    if alpha_constraint is None:
        alpha_constraint = []
    data = pd.DataFrame({"X": x_aux, "Y": y})
    model = Fun_CPsplines_intercept(
        B=B,
        deg=(degree,),
        ord_d=(2,),
        n_int=(knots,),
        int_constraints=int_constraint,
        alpha_constraint=alpha_constraint,
        **kwargs,
    )
    if retry_scaled:
        try:
            model.fit(data=data, y_col="Y")
        except Exception:
            model.fit(data=data, y_col="Y", y_range=(0, 1))
    else:
        model.fit(data=data, y_col="Y")
    return model
