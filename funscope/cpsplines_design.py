"""Constrained P-spline fit with a user-supplied design matrix and an unpenalized intercept.

Adapted from ``cpsplines.fittings.fit_cpsplines.CPsplines``. The differences
with the original class are:

* the design matrix ``B`` is given by the user (the transformed functional
  design matrix of the SOF/FOS/FOF model), instead of being a B-spline basis
  evaluated at the covariate;
* the last column of ``B`` is an intercept, which is excluded from the
  second-order difference penalty;
* the intercept can optionally be fixed to a value (``alpha_constraint``).
"""

import itertools
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import mosek.fusion
import numpy as np
import pandas as pd
import scipy
from joblib import Parallel, delayed
from scipy.spatial import Delaunay
from statsmodels.genmod.families.family import Family, Gaussian

from cpsplines.fittings.fit_cpsplines import NumericalError
from cpsplines.mosek_functions.obj_function import ObjectiveFunction
from cpsplines.mosek_functions.point_constraints import PointConstraints
from cpsplines.utils.fast_kron import matrix_by_transpose
from cpsplines.utils.gcv import GCV
from cpsplines.utils.normalize_data import DataNormalizer
from cpsplines.utils.rearrange_data import RearrangingError, scatter_to_grid
from cpsplines.utils.simulator_grid_search import print_grid_search_results
from cpsplines.utils.simulator_optimize import Simulator
from cpsplines.utils.timer import timer

from .bspline_basis import Fun_BsplineBasis
from .int_constraints import Fun_IntConstraints


class Fun_CPsplines_intercept:
    def __init__(
        self,
        deg: Iterable[int] = (3,),
        ord_d: Iterable[int] = (2,),
        n_int: Iterable[int] = (40,),
        B: np.ndarray = None,
        x_range: Optional[Dict[str, Tuple[Union[int, float]]]] = None,
        sp_method: str = "optimizer",
        sp_args: Optional[Dict[str, Any]] = None,
        family: str = "gaussian",
        int_constraints: Optional[
            Dict[str, Dict[int, Dict[str, Union[int, float]]]]
        ] = None,
        pt_constraints: Optional[Dict[Tuple[int], Dict[str, pd.DataFrame]]] = None,
        pdf_constraint: bool = False,
        alpha_constraint: list = [],
    ):
        self.deg = deg
        self.ord_d = ord_d
        self.n_int = n_int
        self.x_range = x_range
        self.sp_method = sp_method
        self.sp_args = sp_args
        self.family = self._get_family(family)
        self.int_constraints = int_constraints
        self.pt_constraints = pt_constraints
        self.pdf_constraint = pdf_constraint
        self.alpha_constraint = alpha_constraint
        self.B = B

    @staticmethod
    def _get_family(family: str) -> Family:
        if family == "gaussian":
            family_statsmodels = Gaussian()
        else:
            raise ValueError(f"Family {family} is not implemented.")
        return family_statsmodels

    def _get_bspline_bases(self, x: Iterable[np.ndarray]) -> List[Fun_BsplineBasis]:
        bspline_bases = []
        if self.x_range is None:
            self.x_range = {}
        for deg, xsample, n_int, name in zip(
            self.deg, x, self.n_int, self.feature_names
        ):
            x_min, x_max = np.min(xsample), np.max(xsample)
            prediction_dict = {}
            if name in self.x_range:
                # Values of `x_range` outside the fitting region are passed to
                # the `prediction` argument of the basis
                pred_min, pred_max = min(self.x_range[name]), max(self.x_range[name])
                if pred_max > x_max:
                    prediction_dict["forward"] = pred_max
                if pred_min < x_min:
                    prediction_dict["backwards"] = pred_min
            bsp = Fun_BsplineBasis(
                deg=deg, xsample=xsample, n_int=n_int, prediction=prediction_dict, B=self.B
            )
            bsp.get_matrix_B()
            if self.int_constraints is not None or self.pdf_constraint:
                bsp.get_matrices_S()
            bspline_bases.append(bsp)
        return bspline_bases

    def _fill_sp_args(self):
        """Fill `sp_args` with default values when they are not provided."""
        if self.sp_args is None:
            self.sp_args = {}
        if self.sp_method == "grid_search":
            self.sp_args["grid"] = self.sp_args.get(
                "grid", [(0.01, 0.1, 1, 10) for _ in range(len(self.deg))]
            )
            self.sp_args["parallel"] = self.sp_args.get("parallel", False)
            self.sp_args["n_jobs"] = self.sp_args.get("n_jobs", -2)
            self.sp_args["top_n"] = self.sp_args.get("top_n", None)
        else:
            self.sp_args["verbose"] = self.sp_args.get("verbose", False)
            self.sp_args["x0"] = self.sp_args.get("x0", np.ones(len(self.deg)))
            self.sp_args["method"] = self.sp_args.get("method", "SLSQP")
            self.sp_args["bounds"] = self.sp_args.get(
                "bounds", [(1e-10, 1e16) for _ in range(len(self.deg))]
            )
        return None

    def _get_obj_func_arrays(self, y: np.ndarray) -> Dict[str, np.ndarray]:
        """Design matrix and second-order difference penalty (the intercept is not penalized)."""
        obj_matrices = {}
        obj_matrices["B"] = []
        obj_matrices["D"] = []
        obj_matrices["D_mul"] = []

        B = self.B
        obj_matrices["B"].append(B)
        n = np.shape(B)[1] - 1
        D = np.diag(np.ones(n - 1), k=-1) + np.diag(-2 * np.ones(n), k=0) + np.diag(np.ones(n - 1), k=1)
        D = D[1:-1, :]
        P = np.transpose(D) @ D
        # Extend penalty matrices with a zero row/column for the intercept
        P_2 = np.zeros((n + 1, n + 1))
        P_2[:-1, :-1] = P
        n_2 = n + 1
        D_2 = np.zeros((n_2 - 2, n_2))
        D_2[:-1, :-1] = D

        obj_matrices["D"].append(D_2)
        obj_matrices["D_mul"].append(P_2)
        obj_matrices["y"] = y.copy()
        return obj_matrices

    def _initialize_model(
        self,
        obj_matrices: Union[np.ndarray, Iterable[np.ndarray]],
        y_col: str,
        data_normalizer: Optional[DataNormalizer] = None,
    ) -> mosek.fusion.Model:
        M = mosek.fusion.Model()
        # Decision variables and objective function
        mos_obj_f = ObjectiveFunction(bspline=self.bspline_bases, model=M)
        sp = [M.parameter(f"sp_{i}", 1) for i, _ in enumerate(self.deg)]
        mos_obj_f.create_obj_function(
            obj_matrices=obj_matrices,
            sp=sp,
            family=self.family,
            data_arrangement=self.data_arrangement,
        )

        # Optionally fix the intercept (last coefficient)
        if self.alpha_constraint != []:
            sh = mos_obj_f.var_dict["theta"].getShape()[0]
            M.constraint(mos_obj_f.var_dict["theta"].index(sh - 1), mosek.fusion.Domain.equalsTo(self.alpha_constraint[0]))

        # Shape constraints over the whole domain
        if self.int_constraints is not None:
            matrices_S = {
                name: bsp.matrices_S
                for name, bsp in zip(self.feature_names, self.bspline_bases)
            }
            for var_name in self.int_constraints.keys():
                for deriv, constraints in self.int_constraints[var_name].items():
                    if list(constraints.values())[0] != 0 and not isinstance(
                        self.family, Gaussian
                    ):
                        raise ValueError(
                            "No threshold is allowed in the shape constraints for non Gaussian data."
                        )
                    # Scale the thresholds if the data is scaled
                    if data_normalizer is not None:
                        derivative = True if deriv != 0 else False
                        constraints = {
                            k: data_normalizer.transform(y=v, derivative=derivative)
                            for k, v in constraints.items()
                        }
                    matrices_S_copy = matrices_S.copy()
                    cons = Fun_IntConstraints(
                        bspline={
                            name: bsp
                            for name, bsp in zip(self.feature_names, self.bspline_bases)
                        },
                        var_name=var_name,
                        derivative=deriv,
                        constraints=constraints,
                    )
                    cons.interval_cons(
                        var_dict=mos_obj_f.var_dict, model=M, matrices_S=matrices_S_copy
                    )
        else:
            self.int_constraints = {}

        # Point constraints
        if self.pt_constraints is not None:
            for deriv, dict_deriv in self.pt_constraints.items():
                for sense, data in dict_deriv.items():
                    if data_normalizer is not None:
                        derivative = any(v != 0 for v in deriv)
                        data = data.assign(
                            y=data_normalizer.transform(
                                y=data[y_col], derivative=derivative
                            )
                        )
                        if "tol" in data.columns:
                            data = data.assign(
                                tol=data_normalizer.transform(
                                    y=data["tol"], derivative=False
                                )
                                - data_normalizer.transform(y=0, derivative=False)
                            )
                    cons2 = PointConstraints(
                        derivative=deriv,
                        sense=sense,
                        bspline=self.bspline_bases,
                    )
                    cons2.point_cons(
                        data=data,
                        y_col=y_col,
                        var_dict=mos_obj_f.var_dict,
                        model=M,
                    )
        else:
            self.pt_constraints = {}
        return M

    def _get_sp_grid_search(
        self,
        obj_matrices: Dict[str, Union[np.ndarray, Iterable[np.ndarray]]],
    ) -> Tuple[Union[int, float]]:
        """Smoothing parameter minimizing the GCV over the grid in `sp_args["grid"]`."""
        iter_sp = list(itertools.product(*self.sp_args["grid"]))
        if self.sp_args["parallel"]:
            gcv = Parallel(n_jobs=self.sp_args["n_jobs"])(
                delayed(GCV)(sp, obj_matrices, self.family, self.data_arrangement)
                for sp in iter_sp
            )
        else:
            gcv = [
                GCV(
                    sp=sp,
                    obj_matrices=obj_matrices,
                    family=self.family,
                    data_arrangement=self.data_arrangement,
                )
                for sp in iter_sp
            ]
        if self.sp_args["top_n"] is not None:
            print_grid_search_results(
                x_val=iter_sp, obj_val=gcv, top_n=self.sp_args["top_n"]
            )
        return iter_sp[gcv.index(min(gcv))]

    def _get_sp_optimizer(
        self,
        obj_matrices: Dict[str, Union[np.ndarray, Iterable[np.ndarray]]],
    ) -> Tuple[Union[int, float]]:
        """Smoothing parameter minimizing the GCV with `scipy.optimize.minimize`."""
        # All arguments in `sp_args` except `verbose` are passed to scipy
        scipy_optimize_params = self.sp_args.copy()
        scipy_optimize_params.pop("verbose", None)
        if self.sp_args["verbose"]:
            gcv_sim = Simulator(GCV)
        best_sp = scipy.optimize.minimize(
            gcv_sim.simulate if self.sp_args["verbose"] else GCV,
            args=(obj_matrices, self.family, self.data_arrangement),
            callback=gcv_sim.callback if self.sp_args["verbose"] else None,
            **scipy_optimize_params,
        ).x
        return best_sp

    def _preprocessor(
        self, data: pd.DataFrame, y_col: str
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        self.data_arrangement = "scattered"
        x = [row for row in data[data.columns.drop(y_col).tolist()].values.T]
        y = data[y_col].values
        try:
            z, t = scatter_to_grid(data=data, y_col=y_col)
            if len(data) == np.prod(t.shape) and np.isnan(t).sum() == 0:
                self.data_arrangement = "gridded"
                x, y = z.copy(), t.copy()
                logging.info("Data is rearranged into a grid.")
        except RearrangingError:
            pass
        return x, y

    def fit(
        self,
        data: pd.DataFrame,
        y_col: str,
        y_range: Optional[Iterable[Union[int, float]]] = None,
        **kwargs,
    ):
        """
        Compute the fitted coefficients (stored in ``self.sol``; the last one is
        the intercept). The kwargs are passed to MOSEK as solver parameters.

        Parameters
        ----------
        data : pd.DataFrame
            Two columns: an auxiliary covariate column (only its range is used,
            to build the knots on which the shape constraints are formulated)
            and the response column.
        y_col : str
            The column name of the response.
        y_range : Optional[Iterable[Union[int, float]]]
            If not None, the response is scaled to this range before solving,
            which helps MOSEK when the response has a very large norm.
        """
        if len({len(i) for i in [self.deg, self.ord_d, self.n_int]}) != 1:
            raise ValueError("The lengths of `deg`, `ord_d`, `n_int` must agree.")

        if self.sp_method not in ["grid_search", "optimizer"]:
            raise ValueError(f"Invalid `sp_method`: {self.sp_method}.")

        self.feature_names = data.drop(columns=y_col).columns

        if data.shape[1] > 2:
            df_pred = [data.drop(columns=y_col)]
            if self.x_range:
                for key, value in self.x_range.items():
                    column_name = data.loc[:, key].name
                    for v in value:
                        df_pred.append(
                            data.drop(columns=[y_col, column_name])
                            .agg(["min", "max"])
                            .assign(**{column_name: v})
                        )
            self.data_hull = Delaunay(pd.concat(df_pred))

        x, y = self._preprocessor(data=data, y_col=y_col)

        self.bspline_bases = self._get_bspline_bases(x=x)

        _ = self._fill_sp_args()
        if y_range is not None:
            if not isinstance(self.family, Gaussian):
                raise ValueError(
                    "The argument `y_range` is only available for Gaussian data."
                )
            if len(y_range) != 2:
                raise ValueError("The range for `y` must be an interval.")
            data_normalizer = DataNormalizer(feature_range=y_range)
            _ = data_normalizer.fit(y)
            y = data_normalizer.transform(y)
        else:
            data_normalizer = None

        obj_matrices = self._get_obj_func_arrays(y=y)
        obj_matrices["B_mul"] = list(map(matrix_by_transpose, obj_matrices["B"]))

        M = self._initialize_model(
            obj_matrices=obj_matrices, y_col=y_col, data_normalizer=data_normalizer
        )
        model_params = {"theta": M.getVariable("theta")}
        for i, _ in enumerate(self.deg):
            model_params[f"sp_{i}"] = M.getParameter(f"sp_{i}")

        # Smoothing parameter selection (GCV on the unconstrained problem)
        if self.sp_method == "grid_search":
            self.best_sp = self._get_sp_grid_search(
                obj_matrices=obj_matrices,
            )
        else:
            self.best_sp = self._get_sp_optimizer(
                obj_matrices=obj_matrices,
            )
        theta_shape = model_params["theta"].getShape()
        for i, sp in enumerate(self.best_sp):
            model_params[f"sp_{i}"].setValue(sp)
        for key, v in kwargs.items():
            _ = M.setSolverParam(key, v)
        try:
            with timer(
                tag=f"Solve the problem with smoothing parameters {tuple(self.best_sp)}: "
            ):
                M.solve()
            self.sol = model_params["theta"].level().reshape(theta_shape)
            if y_range is not None:
                self.sol = data_normalizer.inverse_transform(y=self.sol)
        except mosek.fusion.SolutionError as e:
            raise NumericalError(
                f"The solution for the smoothing parameter {self.best_sp} "
                f"could not be found due to numerical issues. The original error "
                f"was: {e}"
            )

        return None
