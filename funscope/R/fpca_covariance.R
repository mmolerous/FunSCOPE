# =============================================================================
# FunSCOPE - Error covariance estimation by FPCA (called from Python)
# =============================================================================
# Usage (done automatically by funscope.covariance.fpca_covariance):
#   Rscript fpca_covariance.R <output.csv> <T.txt> <E.txt> <m> <n>
#
# Reads the residual matrix E (n x m) evaluated at the points T, estimates the
# covariance Sigma with refund::fpca.sc and writes, next to <output.csv>:
#   output.csv          L^{-1}, with Sigma = L L^T (Cholesky)
#   sigmat.csv          Sigma
#   sigmat_invsqrt.csv  Sigma^{-1/2}
#   identity_used.csv   1 if the decomposition failed and the identity was used
# =============================================================================

args <- commandArgs(trailingOnly = TRUE)
output_file <- args[1]
TtoR_file   <- args[2]
EtoR_file   <- args[3]
m <- as.numeric(args[4])
n <- as.numeric(args[5])

# Silence R output and messages
sink(nullfile())
msg_con <- file(nullfile(), open = "wt")
sink(msg_con, type = "message")

for (pkg in c("Matrix", "lme4", "refund")) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(sprintf("R package '%s' is required. Install it with install.packages('%s').", pkg, pkg))
  }
}
library(Matrix)
library(lme4)
library(refund)

# -----------------------------------------------------------------------------
# Covariance estimation
# -----------------------------------------------------------------------------
transformR <- function(Res, T, m) {
  if (length(T) < 15) {
    n_basis <- length(T) - 1
  } else {
    n_basis <- 15
  }
  fpca1 <- fpca.sc(Res, center = TRUE, nbasis = n_basis, argvals = T, pve = 0.99, var = TRUE)
  if (length(fpca1$evalues) > 1) {
    sigmat <- fpca1$efunctions %*% diag(fpca1$evalues) %*% t(fpca1$efunctions) + fpca1$sigma2 * diag(m)
  } else {
    sigmat <- fpca1$evalues * fpca1$efunctions %*% t(fpca1$efunctions) + fpca1$sigma2 * diag(m)
  }

  # Cholesky factor; fall back to the identity if it fails
  R <- tryCatch(chol(sigmat), error = function(e) NULL)
  if (is.null(R)) {
    return(list(sigmat = diag(m), Linv = diag(m), sigmat_invsqrt = diag(m), identity_used = 1))
  }
  L <- t(R)
  Linv <- forwardsolve(L, diag(dim(L)[1]))

  # Sigma^{-1/2} from the eigendecomposition given by the FPCA
  V <- fpca1$efunctions
  evals_full <- fpca1$evalues + fpca1$sigma2
  sigmat_invsqrt <- V %*% diag(1 / sqrt(evals_full), nrow = length(evals_full)) %*% t(V) +
                    (1 / sqrt(fpca1$sigma2)) * (diag(m) - V %*% t(V))
  if (any(is.infinite(sigmat_invsqrt)) || any(is.nan(sigmat_invsqrt))) {
    return(list(sigmat = diag(m), Linv = diag(m), sigmat_invsqrt = diag(m), identity_used = 1))
  }
  return(list(sigmat = sigmat, Linv = Linv, sigmat_invsqrt = sigmat_invsqrt, identity_used = 0))
}

# -----------------------------------------------------------------------------
# Read input, compute and write output
# -----------------------------------------------------------------------------
E_try <- as.matrix(read.table(EtoR_file, sep = ",", dec = "."))
T_try <- read.table(TtoR_file, sep = ",", dec = ".")
T_try <- c(T_try$V1)
output <- transformR(E_try, T_try, m)

write.csv(output$Linv, output_file, row.names = FALSE)
write.csv(output$sigmat, sub("output\\.csv$", "sigmat.csv", output_file), row.names = FALSE)
write.csv(output$sigmat_invsqrt, sub("output\\.csv$", "sigmat_invsqrt.csv", output_file), row.names = FALSE)
write.csv(output$identity_used, sub("output\\.csv$", "identity_used.csv", output_file), row.names = FALSE)
