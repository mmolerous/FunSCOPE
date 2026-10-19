# =============================================================================
# SOF regression - Normal scenario
# Step 1: generate the simulated data
# =============================================================================
# y_i = 0.15 + int X_i(t) beta(t) dt + eps_i,  t in [0, 1]
#   beta(t)  = density of N(0.5, 0.1^2)  (non-negative)
#   X_i(t)   = sum_{K=1}^{20} psi_iK phi_K(t), phi_K orthogonal polynomials of
#              degree K-1, psi_iK ~ N(0, 20 - K + 1)
#   eps_i    ~ N(0, 0.05^2)
# N in {25, 50, 100}, 200 Monte Carlo samples, m = 50 grid points
# (beta is also stored on a fine grid of 500 points).
#
# Output (in ./output): X_, Y_, T1_, BetaT1_, T2_, BetaT2_ScenarioB_N_<N>_m_<m>.csv
#   X: (N * 200) x (m + 1), the last column is the sample index
#   Y: 200 x N, one row per sample
# Run from this folder:  Rscript 01_generate_data.R
# =============================================================================

get_script_dir <- function() {
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg) == 1) return(dirname(normalizePath(sub("^--file=", "", file_arg))))
  getwd()
}
out_dir <- file.path(get_script_dir(), "output")
dir.create(out_dir, showWarnings = FALSE)

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
set.seed(100)
Simulaciones <- 200          # Monte Carlo samples
Nvalues <- c(25, 50, 100)    # sample sizes
m1 <- 50                     # grid size
m2 <- 500                    # fine grid size

T1 <- seq(0, 1, length = m1)
T2 <- seq(0, 1, length = m2)
betaT1 <- dnorm(T1, mean = 1/2, sd = 0.1)
betaT2 <- dnorm(T2, mean = 1/2, sd = 0.1)

rankX <- 20
Phi <- cbind(1 / sqrt(m1), poly(T1, degree = rankX - 1))  # orthogonal polynomial basis
lambda <- rankX:1

alpha <- 0.15  # intercept

# -----------------------------------------------------------------------------
# Simulation
# -----------------------------------------------------------------------------
for (N in Nvalues) {
  set.seed(N)
  X_se_Muestras <- array(0, dim = c(N, Simulaciones, m1))
  y_Muestras <- matrix(data = 0, nrow = Simulaciones, ncol = N)
  X_se_Muestras_big <- array(0, dim = c(N * Simulaciones, ncol = m1 + 1))

  for (i in 1:Simulaciones) {
    eps <- rnorm(N, 0, 0.05)
    Xi <- sapply(lambda, function(l) scale(rnorm(N, sd = sqrt(l)), scale = FALSE))
    X <- Xi %*% t(Phi)
    lfterm <- as.numeric((1 / m1) * X %*% betaT1)
    y <- alpha + lfterm + eps

    y_Muestras[i, ] <- y
    X_se_Muestras[, i, ] <- matrix(X, nrow = N, ncol = m1)
    X_se_Muestras_big[(1 + (i - 1) * N):(N * i), 1:m1] <- X_se_Muestras[, i, ]
    X_se_Muestras_big[(1 + (i - 1) * N):(N * i), m1 + 1] <- i
  }

  # ---------------------------------------------------------------------------
  # Save
  # ---------------------------------------------------------------------------
  out_file <- function(prefix, m) file.path(out_dir, paste0(prefix, "_ScenarioB_N_", N, "_m_", m, ".csv"))
  write.csv(X_se_Muestras_big, file = out_file("X", m1), row.names = FALSE)
  write.csv(y_Muestras, file = out_file("Y", m1), row.names = FALSE)
  write.csv(T1, file = out_file("T1", m1), row.names = FALSE)
  write.csv(betaT1, file = out_file("BetaT1", m1), row.names = FALSE)
  write.csv(T2, file = out_file("T2", m2), row.names = FALSE)
  write.csv(betaT2, file = out_file("BetaT2", m2), row.names = FALSE)
}
