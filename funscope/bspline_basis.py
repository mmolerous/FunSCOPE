"""B-spline basis wrapper that accepts an arbitrary, user-supplied design matrix.

Adapted from ``cpsplines.psplines.bspline_basis.BsplineBasis``. The knot
sequence is still built from ``xsample`` (it is needed to formulate the shape
constraints through the matrices S), but the design matrix is replaced by the
matrix ``B`` given by the user (e.g. the transformed functional design matrix).
"""

import math
from typing import Dict, Union

import numpy as np
from scipy.interpolate import BSpline


class Fun_BsplineBasis:
    def __init__(
        self,
        xsample: np.ndarray,
        B: np.ndarray,
        deg: int = 3,
        n_int: int = 40,
        prediction: Dict[str, Union[int, float]] = {},
    ):
        self.xsample = xsample
        self.deg = deg
        self.n_int = n_int
        self.prediction = prediction
        self.B = B

    def get_matrix_B(self) -> None:
        """Build the knot sequence from ``xsample`` and set ``matrixB`` to the user-supplied ``B``."""
        min_x = self.xsample.min()
        max_x = self.xsample.max()
        if self.deg < 0:
            raise ValueError("The degree of the B-spline basis must be at least 0.")
        if self.xsample.ndim != 1:
            raise ValueError("Regressor vector must be one-dimensional.")
        if self.n_int < 2:
            raise ValueError(
                "The fitting regions must be split in at least 2 intervals."
            )
        if len(set(self.prediction) - set(["backwards", "forward"])) > 0:
            raise ValueError(
                "Prediction only admits as keys `forward` and `backwards`."
            )

        if "backwards" in self.prediction:
            if self.prediction["backwards"] >= min_x:
                raise ValueError(
                    (
                        "Backwards prediction limit must stand on the "
                        "left-hand side of the regressor vector."
                    )
                )
        if "forward" in self.prediction:
            if self.prediction["forward"] <= max_x:
                raise ValueError(
                    (
                        "Forward prediction limit must stand on the "
                        "right-hand side of the regressor vector."
                    )
                )

        # Distance between adjacent knots
        step_length = (max_x - min_x) / self.n_int

        # Number of extra knots needed to extend the basis backwards/forward
        self.int_back = (
            math.ceil((min_x - self.prediction["backwards"]) / step_length)
            if "backwards" in self.prediction
            else 0
        )
        self.int_forw = (
            math.ceil((self.prediction["forward"] - max_x) / step_length)
            if "forward" in self.prediction
            else 0
        )

        # Knot sequence: `n_int` + 2 * `deg` + 1 equally spaced knots
        knots = np.linspace(
            min_x - (self.int_back + self.deg) * step_length,
            max_x + (self.int_forw + self.deg) * step_length,
            self.n_int + self.int_back + self.int_forw + 2 * self.deg + 1,
        )
        # Avoid floating point errors at the boundaries of the fitting region
        knots[self.int_back + self.deg] = min_x
        knots[-(self.int_forw + self.deg + 1)] = max_x
        self.knots = knots

        self.matrixB = self.B
        return None

    def get_matrices_S(self):
        """Matrices S_k mapping B-spline coefficients to polynomial coefficients on each interval."""
        C = np.zeros(shape=(self.deg + 1, self.deg + 1))
        for i in range(self.deg + 1):
            C[:, i] = BSpline.basis_element(t=self.knots[i : self.deg + 2 + i])(
                x=np.linspace(
                    self.knots[self.deg], self.knots[self.deg + 1], self.deg + 1
                )
            )
        S = []
        # S_k satisfies S_k @ T_k = C, where T_k has as columns (1, x, ..., x**deg)
        # evaluated at linspace(knots[k + deg], knots[k + deg + 1], deg + 1)
        for k in range(self.n_int + self.int_back + self.int_forw):
            T_k = np.vander(
                np.linspace(
                    self.knots[k + self.deg], self.knots[k + self.deg + 1], self.deg + 1
                ),
                increasing=True,
            )
            S_k = np.linalg.solve(T_k, C)
            S.append(S_k)
        self.matrices_S = S
        return None
