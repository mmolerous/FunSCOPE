"""FunSCOPE: shape-constrained functional regression with P-splines and conic optimization.

User interface (results returned as dictionaries):
    fit_sof : scalar-on-function regression
    fit_fos : function-on-scalar regression
    fit_fof : concurrent function-on-function regression

Estimators used in the paper (full output, as tuples):
    SOFR_basis, FOSR_basis, FOFR_basis
"""

from .api import CONSTRAINTS, fit_fof, fit_fos, fit_sof, make_constraints
from .fof import FOFR_basis
from .fos import FOSR_basis
from .sof import SOFR_basis
from .utils import imse_simpson, simpsons_weights

__all__ = [
    "fit_sof", "fit_fos", "fit_fof", "make_constraints", "CONSTRAINTS",
    "SOFR_basis", "FOSR_basis", "FOFR_basis", "imse_simpson", "simpsons_weights",
]
