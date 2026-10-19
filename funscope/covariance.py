"""Estimation of the error covariance matrix by FPCA (``refund::fpca.sc`` in R).

R is called through ``Rscript``. Its location is taken from the environment
variable ``FUNSCOPE_RSCRIPT`` if defined, otherwise ``Rscript`` must be on the
PATH. Required R packages: ``refund``, ``Matrix`` and ``lme4``.
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

R_SCRIPT = Path(__file__).resolve().parent / "R" / "fpca_covariance.R"


def find_rscript():
    """Path to the ``Rscript`` executable."""
    rscript = os.environ.get("FUNSCOPE_RSCRIPT") or shutil.which("Rscript")
    if rscript is None:
        raise RuntimeError(
            "Rscript was not found. Add R to the PATH or set the environment "
            "variable FUNSCOPE_RSCRIPT to the full path of Rscript."
        )
    return rscript


def fpca_covariance(E, T, m, n):
    """
    Estimate the covariance of the residuals ``E`` (n x m), observed at the points ``T``.

    Returns
    -------
    Linv : pd.DataFrame
        Inverse of the Cholesky factor of Sigma.
    sigmat : np.ndarray
        Estimated covariance matrix Sigma (m x m).
    sigmat_invsqrt : np.ndarray
        Sigma^{-1/2}.
    identity_used : int
        1 if the decomposition failed and the identity was returned instead.
    """
    with tempfile.TemporaryDirectory() as tmp:
        TtoR_file = os.path.join(tmp, "T.txt")
        EtoR_file = os.path.join(tmp, "E.txt")
        output_file = os.path.join(tmp, "output.csv")

        np.savetxt(TtoR_file, T, delimiter=",")
        np.savetxt(EtoR_file, E, delimiter=",")

        subprocess.run(
            [find_rscript(), str(R_SCRIPT), output_file, TtoR_file, EtoR_file, str(m), str(n)],
            check=True,
        )

        Linv = pd.read_csv(output_file)
        sigmat = pd.read_csv(os.path.join(tmp, "sigmat.csv")).values
        sigmat_invsqrt = pd.read_csv(os.path.join(tmp, "sigmat_invsqrt.csv")).values
        identity_used = int(pd.read_csv(os.path.join(tmp, "identity_used.csv")).values.flatten()[0])

    return Linv, sigmat, sigmat_invsqrt, identity_used
