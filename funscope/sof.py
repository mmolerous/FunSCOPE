"""Scalar-on-function (SOF) regression with (shape-constrained) P-splines.

Model: y_i = beta_0 + int X_i(t) beta(t) dt + eps_i.
"""

import time

import numpy as np

from .building_blocks import bspline_basis, fine_grid, fit_design, gram_matrix, imse_beta, presmooth, silenced


def SOFR_basis(X_se, t, y, Beta_real=None, degree=3, knots=14, g=2000, centered=True, int_constraint=None, alpha_constraint=None):
    """
    Fit the SOF model for one sample.

    Parameters
    ----------
    X_se : np.ndarray (N x m)
        Functional covariate observed on the grid `t`.
    t : np.ndarray (m,)
        Observation grid.
    y : np.ndarray (1 x N)
        Scalar responses.
    Beta_real : array-like (m x 1) or None
        True beta(t) on `t`, only used to compute the IMSE of beta. Use None
        for real data (beta unknown); the IMSE is then returned as NaN.
    degree, knots : int
        Degree of the B-splines and number of equispaced intervals.
    g : int
        Number of intervals of the fine grid used for the integrals (even).
    centered : bool
        Whether to center the response.
    int_constraint : dict or None
        Shape constraints, e.g. {"X": {0: {"+": 0}}} for beta(t) >= 0
        (key = derivative order, "+"/"-" = sign, value = threshold).
    alpha_constraint : list or None
        None (or []) for a free intercept, [value] to fix it.

    Returns
    -------
    SOFR, Delta, Beta_est, y_est, MSE, imse_approx, total_time, smoothing_time, solving_time
    """
    if centered:
        y = y - np.mean(y)

    start_time = time.time()

    with silenced():
        # -----------------------------------------------------------------
        # 1. Pre-smoothing of the functional covariate: X_i(t) = A_i Phi(t)
        # -----------------------------------------------------------------
        n_coef = bspline_basis(t, degree, knots).matrixB.shape[1]
        A = presmooth(X_se, t, degree, knots, n_coef)
        smoothing_time = time.time() - start_time

        # -----------------------------------------------------------------
        # 2. Psi = int Phi_x(t)^T Phi_beta(t) dt (Simpson's rule)
        # -----------------------------------------------------------------
        tt = fine_grid(t, g)
        Phi_inner = bspline_basis(tt, degree, knots).matrixB
        Psi = gram_matrix(Phi_inner, t, g)
        Phi_beta = bspline_basis(t, degree, knots).matrixB

        # -----------------------------------------------------------------
        # 3. Design matrix [A Psi, 1] and (constrained) P-spline fit
        # -----------------------------------------------------------------
        Z = A @ Psi
        Z_2 = np.hstack((Z, np.ones((Z.shape[0], 1))))
        y_2 = y[0, :]
        x_aux = X_se[:, 3]  # auxiliary column, only its range is used by the fitter

        SOFR = fit_design(Z_2, x_aux, y_2, degree, knots, int_constraint, alpha_constraint)
        # If GCV selects an almost-zero smoothing parameter, use a grid search instead
        if SOFR.best_sp[0] < 1e-6:
            SOFR = fit_design(
                Z_2, x_aux, y_2, degree, knots, int_constraint, alpha_constraint,
                sp_method="grid_search",
                sp_args={"grid": (tuple(np.linspace(1e-6, 1e-2, num=20)),)},
            )
        Delta = SOFR.sol

    # ---------------------------------------------------------------------
    # 4. Estimates and errors
    # ---------------------------------------------------------------------
    Beta_est = Phi_beta @ Delta[:-1]
    y_est = Z_2 @ Delta
    MSE = np.mean((y_2 - y_est) ** 2)
    imse_approx = imse_beta(t, Beta_real, Beta_est)

    total_time = time.time() - start_time
    solving_time = total_time - smoothing_time

    return SOFR, Delta, Beta_est, y_est, MSE, imse_approx, total_time, smoothing_time, solving_time
