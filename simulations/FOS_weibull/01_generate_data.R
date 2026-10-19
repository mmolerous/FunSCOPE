# =============================================================================
# FOS regression - Weibull scenario
# Step 1: generate the simulated data
# =============================================================================
# Y_i(t) = beta_0(t) + X_i beta(t) + eps_i(t),  t in [0.1, 2.1]
#   beta_0(t) = 0
#   beta(t)   = Weibull density, shape 0.75, scale 0.25 (non-negative, convex)
#   X_i       ~ Unif(0.5, 1.5)
#   eps_i(t)  = c_i cos(t) + d_i sin(t) + w_i(t),
#               c_i, d_i ~ N(0, 0.1^2), w_i(t) ~ N(0, 0.1^2)
# N in {25, 50, 100}, m = 50 grid points (beta also on a fine grid of 500).
# 300 samples are generated per N: the FunSCOPE script uses the first 200 for
# which the error covariance estimation does not fail.
#
# Output (in ./output): X_, Y_, T1_, Beta0T1_, Beta1T1_, T2_, Beta0T2_,
#                       Beta1T2_ScenarioWeibullFOSR_N_<N>_m_<m>.csv
#   X: N x 300, one column per sample
#   Y: (N * 300) x (m + 1), the last column is the sample index
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
Simulaciones <- 300          # samples generated (200 are used)
Nvalues <- c(25, 50, 100)    # sample sizes
m1 <- 50                     # grid size
m2 <- 500                    # fine grid size

T1 <- seq(0.1, 2.1, length = m1)
T2 <- seq(0.1, 2.1, length = m2)
beta0T1 <- rep(0, length(T1))
beta0T2 <- rep(0, length(T2))
shape <- 0.75
scale <- 0.25
beta1T1 <- dweibull(T1, shape = shape, scale = scale)
beta1T2 <- dweibull(T2, shape = shape, scale = scale)

# -----------------------------------------------------------------------------
# Simulation
# -----------------------------------------------------------------------------
for (N in Nvalues) {
  set.seed(1234)
  X_se_Muestras <- array(0, dim = c(N, Simulaciones))
  y_Muestras <- array(0, dim = c(N, Simulaciones, m1))
  y_Muestras_big <- array(0, dim = c(N * Simulaciones, ncol = m1 + 1))

  for (i in 1:Simulaciones) {
    # Correlated error process
    c <- rnorm(N, 0, .1)
    d <- rnorm(N, 0, .1)
    sigma <- .1
    wn <- matrix(rnorm(m1 * N, 0, sigma), N, m1)
    E <- matrix(0, N, m1)
    for (i2 in 1:N) {
      for (l in 1:length(T1)) {
        E[i2, l] <- c[i2] * cos(T1[l]) + d[i2] * sin(T1[l]) + wn[i2, l]
      }
    }

    # Scalar covariate and functional response
    X <- runif(N, 0.5, 1.5)
    y <- matrix(0, N, m1)
    for (i3 in 1:N) {
      for (l2 in 1:length(T1)) {
        y[i3, l2] <- beta0T1[l2] + X[i3] * beta1T1[l2] + E[i3, l2]
      }
    }

    y_Muestras[, i, ] <- y
    X_se_Muestras[, i] <- as.vector(X)
    y_Muestras_big[(1 + (i - 1) * N):(N * i), 1:m1] <- y_Muestras[, i, ]
    y_Muestras_big[(1 + (i - 1) * N):(N * i), m1 + 1] <- i
  }

  # ---------------------------------------------------------------------------
  # Save
  # ---------------------------------------------------------------------------
  out_file <- function(prefix, m) file.path(out_dir, paste0(prefix, "_ScenarioWeibullFOSR_N_", N, "_m_", m, ".csv"))
  write.csv(X_se_Muestras, file = out_file("X", m1), row.names = FALSE)
  write.csv(y_Muestras_big, file = out_file("Y", m1), row.names = FALSE)
  write.csv(T1, file = out_file("T1", m1), row.names = FALSE)
  write.csv(beta1T1, file = out_file("Beta1T1", m1), row.names = FALSE)
  write.csv(T2, file = out_file("T2", m2), row.names = FALSE)
  write.csv(beta1T2, file = out_file("Beta1T2", m2), row.names = FALSE)
  write.csv(beta0T1, file = out_file("Beta0T1", m1), row.names = FALSE)
  write.csv(beta0T2, file = out_file("Beta0T2", m2), row.names = FALSE)
}
