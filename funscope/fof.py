"""Concurrent function-on-function (FOF) regression with (shape-constrained) P-splines.

Model: Y_i(t) = beta_0(t) + X_i(t) beta(t) + eps_i(t), with correlated errors.

As in FOS, three estimates are computed for each sample: ``unc`` (ignoring
the error correlation), ``unc2`` (unconstrained GLS, FunSCOPE-U in the paper)
and ``cons`` (shape-constrained GLS, FunSCOPE in the paper).
"""

import time

import numpy as np

from .building_blocks import (
    bspline_basis,
    fine_grid,
    fit_design,
    gram_matrix,
    imse_beta,
    presmooth,
    silenced,
    sqrtm_sym,
)
from .covariance import fpca_covariance


def FOFR_basis(X, t, Y, Beta_real=None, degree=3, knots=14, g=2000, centered=True, int_constraint=None, alpha_constraint=None, covariance_estimation="FPCA"):
    """
    Fit the concurrent FOF model for one sample.

    Parameters
    ----------
    X : np.ndarray (N x m)
        Functional covariate observed on `t`.
    t : np.ndarray (m,)
        Observation grid.
    Y : np.ndarray (N x m)
        Functional responses observed on `t`.
    Beta_real : array-like (m x 1) or None
        True beta(t) on `t`, only used to compute the IMSE of beta. Use None
        for real data (beta unknown); the IMSE is then returned as NaN.
    degree, knots : int
        Degree of the B-splines and number of equispaced intervals.
    g : int
        Number of intervals of the fine grid used for the integrals (even).
    centered : bool
        Whether to center Y column-wise (X is not centered).
    int_constraint : dict or None
        Shape constraints, e.g. {"X": {1: {"+": 0}}} for beta'(t) >= 0.
    alpha_constraint : list or None
        None (or []) for a free intercept, [value] to fix it.
    covariance_estimation : "FPCA" or None
        "FPCA": GLS with the error covariance estimated by FPCA (paper).
        None: the error correlation is ignored (``unc2`` equals ``unc``).

    Returns
    -------
    Same tuple as `FOSR_basis`, without the mean of Y.
    """
    if covariance_estimation not in ("FPCA", None):
        raise ValueError("`covariance_estimation` must be 'FPCA' or None.")

    N = np.shape(X)[0]

    if centered:
        Y = Y - Y.mean(axis=0, keepdims=True)

    with silenced():
        start_time = time.time()

        # -----------------------------------------------------------------
        # 1. B-spline basis (same for X(t), Y(t) and beta(t))
        # -----------------------------------------------------------------
        Base = bspline_basis(t, degree, knots)
        Phi = Base.matrixB
        knots_print = Base.knots

        # -----------------------------------------------------------------
        # 2. Pre-smoothing: Y_i(t) = C_i Phi(t), X_i(t) = A_i Phi(t)
        # -----------------------------------------------------------------
        C = presmooth(Y, t, degree, knots, np.shape(Phi)[1])
        A = presmooth(X, t, degree, knots, np.shape(Phi)[1])

        # -----------------------------------------------------------------
        # 3. Psi = int (Phi(t) (x) Phi(t))^T (Phi(t) (x) Phi(t)) dt and its square root
        # -----------------------------------------------------------------
        tt = fine_grid(t, g)
        Phi_inner = bspline_basis(tt, degree, knots).matrixB
        Phi_inner_star = (Phi_inner[:, :, None] * Phi_inner[:, None, :]).reshape(np.shape(Phi_inner)[0], np.shape(Phi_inner)[1] * np.shape(Phi_inner)[1])
        Psi = gram_matrix(Phi_inner_star, t, g)
        Psiroot = sqrtm_sym(Psi)

        # -----------------------------------------------------------------
        # 4. Transformed response ystar_i = Psi^1/2 (1 (x) c_i) and design
        #    blocks Psi^1/2 (a_i (x) I_p), with p = knots + degree
        # -----------------------------------------------------------------
        p = np.shape(C)[1]
        C_tiled = np.tile(C, (1, p))
        ystar_unc_matrix = Psiroot @ C_tiled.T
        ystar_unc = ystar_unc_matrix.flatten(order="F").tolist()

        I_p = np.eye(p)
        A_kron_I = (A[:, :, None, None] * I_p[None, None, :, :]).reshape(N, p * p, p)
        Z1 = (Psiroot @ A_kron_I).reshape(N * p**2, p)
        Z2 = np.ones((N * p**2, 1))
        Zstar_unc = np.hstack((Z1, Z2))

        # -----------------------------------------------------------------
        # 5. Unconstrained fit ignoring the error correlation (unc)
        # -----------------------------------------------------------------
        FOFR_unc = fit_design(Zstar_unc, ystar_unc, ystar_unc, degree, knots, alpha_constraint=alpha_constraint)
        Delta_unc = FOFR_unc.sol
        lambda_optimo = FOFR_unc.best_sp

        Beta_est_unc = Phi @ Delta_unc[:-1]
        y_est_unc = Zstar_unc @ Delta_unc
        MSE_unc = np.mean((ystar_unc - y_est_unc) ** 2)

        # -----------------------------------------------------------------
        # 6. Error covariance (FPCA) and unconstrained GLS fit (unc2)
        # -----------------------------------------------------------------
        FOFR_unc2 = []
        Delta_unc2 = []
        identity_used = 0
        if covariance_estimation == "FPCA":
            U_hat = np.reshape(ystar_unc - y_est_unc, (N, (knots + degree) ** 2), order="F")
            # Each knot is repeated (knots + degree) times to match the p**2 columns
            # of U_hat; a tiny increasing jitter avoids exact ties in fpca.sc
            jitter = np.tile(np.arange(knots + degree), len(knots_print[2:-2])) * 1e-6
            knots_evals = np.repeat(knots_print[2:-2], knots + degree) + jitter
            _, sigmat, sigmat_invsqrt, identity_used = fpca_covariance(U_hat, knots_evals, (knots + degree) ** 2, N)

            ystar_cons = (sigmat_invsqrt @ ystar_unc_matrix).flatten(order="F")
            Z1_cons = (sigmat_invsqrt @ Psiroot @ A_kron_I).reshape(N * p**2, p)
            Z2_cons = np.tile(sigmat_invsqrt @ np.ones((p**2, 1)), (N, 1))
            Zstar_cons = np.hstack((Z1_cons, Z2_cons))

            FOFR_unc2 = fit_design(Zstar_cons, ystar_cons, ystar_cons, degree, knots, alpha_constraint=alpha_constraint)
            lambda_optimo = FOFR_unc2.best_sp
            Delta_unc2 = FOFR_unc2.sol
            Beta_est_unc2 = Phi @ Delta_unc2[:-1]
            y_est_unc2 = Zstar_unc @ Delta_unc2
        else:
            Zstar_cons = Zstar_unc.copy()
            ystar_cons = ystar_unc.copy()
            Beta_est_unc2 = Beta_est_unc
            y_est_unc2 = y_est_unc

        MSE_unc2 = np.mean((ystar_unc - y_est_unc2) ** 2)

        # -----------------------------------------------------------------
        # 7. Shape-constrained GLS fit (cons), same smoothing parameter as unc2
        # -----------------------------------------------------------------
        FOFR_cons = fit_design(
            Zstar_cons, ystar_cons, ystar_cons, degree, knots, int_constraint, alpha_constraint,
            retry_scaled=True,
            sp_method="grid_search",
            sp_args={"grid": (lambda_optimo,)},
        )
        Delta_cons = FOFR_cons.sol
        Beta_est_cons = Phi @ Delta_cons[:-1]
        Beta_est_cons_Phi_inner = Phi_inner @ Delta_cons[:-1]
        y_est_cons = Zstar_unc @ Delta_cons
        MSE_cons = np.mean((ystar_unc - y_est_cons) ** 2)

        total_time = time.time() - start_time

        # -----------------------------------------------------------------
        # 8. Errors: IMSE of beta(t) and IMSE of Y(t) w.r.t. the pre-smoothed
        #    curves, (1/N) sum (1/T) int (Y_i - fitted_i)^2. The squared norm of
        #    the transformed residual is that integral, so this is MSE_* * p**2 / T
        # -----------------------------------------------------------------
        imse_approx_unc = imse_beta(t, Beta_real, Beta_est_unc)
        imse_approx_unc2 = imse_beta(t, Beta_real, Beta_est_unc2)
        imse_approx_cons = imse_beta(t, Beta_real, Beta_est_cons)
        T = max(t) - min(t)
        imse_mse_unc = np.sum((ystar_unc - y_est_unc) ** 2) / (N * T)
        imse_mse_unc2 = np.sum((ystar_unc - y_est_unc2) ** 2) / (N * T)
        imse_mse_cons = np.sum((ystar_unc - y_est_cons) ** 2) / (N * T)

    return (
        FOFR_unc, FOFR_unc2, FOFR_cons,
        Delta_unc, Delta_unc2, Delta_cons,
        Beta_est_unc, Beta_est_unc2, Beta_est_cons, Beta_est_cons_Phi_inner,
        y_est_unc, y_est_unc2, y_est_cons,
        MSE_unc, MSE_unc2, MSE_cons,
        total_time, Phi, knots_print, tt,
        imse_approx_unc, imse_approx_unc2, imse_approx_cons,
        imse_mse_unc, imse_mse_unc2, imse_mse_cons,
        identity_used,
    )
