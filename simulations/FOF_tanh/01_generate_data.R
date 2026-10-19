# =============================================================================
# Concurrent FOF regression - Hyperbolic tangent scenario
# Step 1: generate the simulated data
# =============================================================================
# Y_i(t) = beta_0(t) + X_i(t) beta(t) + eps_i(t),  t in [0, 1]
#   beta_0(t) = 0
#   beta(t)   = tanh(10 (t - 0.5))  (increasing)
#   X_i(t)    = sum_{K=1}^{5} psi_iK phi_K(t), phi_K orthogonal polynomials of
#               degree K-1, psi_iK normal draws standardized to zero sample
#               mean and unit sample variance (so all K have the same variance)
#   eps_i(t)  = c_i cos(t) + d_i sin(t) + w_i(t),
#               c_i, d_i ~ N(0, 0.1^2), w_i(t) ~ N(0, 0.1^2)
# N in {25, 50, 100}, 200 Monte Carlo samples, m = 50 grid points
# (beta is also stored on a fine grid of 500 points).
#
# Output (in ./output): X_, Y_, T1_, Beta0T1_, Beta1T1_, T2_, Beta0T2_,
#                       Beta1T2_ScenarioTanhFOFR_N_<N>_m_<m>.csv
#   X, Y: (N * 200) x (m + 1), the last column is the sample index
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
beta0T1 <- rep(0, length(T1))
beta0T2 <- rep(0, length(T2))
k <- 10     # steepness of the transition
x0 <- 0.5   # center of the transition
beta1T1 <- tanh(k * (T1 - x0))
beta1T2 <- tanh(k * (T2 - x0))

rankX <- 5
Phi <- cbind(1 / sqrt(m1), poly(T1, degree = rankX - 1))  # orthogonal polynomial basis
lambda <- rankX:1

# -----------------------------------------------------------------------------
# Simulation
# -----------------------------------------------------------------------------
for (N in Nvalues) {
  set.seed(N)
  X_se_Muestras <- array(0, dim = c(N, Simulaciones, m1))
  y_Muestras <- array(0, dim = c(N, Simulaciones, m1))
  X_se_Muestras_big <- array(0, dim = c(N * Simulaciones, ncol = m1 + 1))
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

    # Functional covariate.
    # NOTE: the first draw of Xi is overwritten by the second one, but it is
    # kept because it advances the random number generator (needed to
    # reproduce the data of the paper exactly).
    Xi <- sapply(lambda, function(l) scale(rnorm(N, sd = sqrt(l)), scale = FALSE))
    Xi <- sapply(lambda, function(l) scale(rnorm(N, sd = sqrt(l))))
    X <- Xi %*% t(Phi)

    # Functional response (concurrent model)
    y <- matrix(0, N, m1)
    for (i3 in 1:N) {
      for (l2 in 1:length(T1)) {
        y[i3, l2] <- beta0T1[l2] + X[i3, l2] * beta1T1[l2] + E[i3, l2]
      }
    }

    y_Muestras[, i, ] <- y
    X_se_Muestras[, i, ] <- matrix(X, nrow = N, ncol = m1)
    X_se_Muestras_big[(1 + (i - 1) * N):(N * i), 1:m1] <- X_se_Muestras[, i, ]
    X_se_Muestras_big[(1 + (i - 1) * N):(N * i), m1 + 1] <- i
    y_Muestras_big[(1 + (i - 1) * N):(N * i), 1:m1] <- y_Muestras[, i, ]
    y_Muestras_big[(1 + (i - 1) * N):(N * i), m1 + 1] <- i
  }

  # ---------------------------------------------------------------------------
  # Save
  # ---------------------------------------------------------------------------
  out_file <- function(prefix, m) file.path(out_dir, paste0(prefix, "_ScenarioTanhFOFR_N_", N, "_m_", m, ".csv"))
  write.csv(X_se_Muestras_big, file = out_file("X", m1), row.names = FALSE)
  write.csv(y_Muestras_big, file = out_file("Y", m1), row.names = FALSE)
  write.csv(T1, file = out_file("T1", m1), row.names = FALSE)
  write.csv(beta1T1, file = out_file("Beta1T1", m1), row.names = FALSE)
  write.csv(T2, file = out_file("T2", m2), row.names = FALSE)
  write.csv(beta1T2, file = out_file("Beta1T2", m2), row.names = FALSE)
  write.csv(beta0T1, file = out_file("Beta0T1", m1), row.names = FALSE)
  write.csv(beta0T2, file = out_file("Beta0T2", m2), row.names = FALSE)
}
