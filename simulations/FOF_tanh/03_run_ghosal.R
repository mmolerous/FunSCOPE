# =============================================================================
# Concurrent FOF regression - Hyperbolic tangent scenario
# Step 3: fit the shape-constrained Bernstein polynomial method of
#         Ghosal et al. (2023) with the constraint beta'(t) >= 0
# =============================================================================
# X and Y are centered column-wise in each sample. The degree of the Bernstein
# basis is selected by 5-fold cross-validation over 3:8.
#
# Output (in ./output), one file per N:
#   CB-R_time_ScenarioTanhFOFR_N_<N>_m_50.csv       computing time (seconds)
#   CB-R_Beta0_est_ScenarioTanhFOFR_N_<N>_m_50.csv  estimated beta_0(t), one row per sample
#   CB-R_Beta1_est_ScenarioTanhFOFR_N_<N>_m_50.csv  estimated beta(t), one row per sample
#   CB-R_y_est_ScenarioTanhFOFR_N_<N>_m_50.csv      fitted curves, (N * 200) x (m + 1)
# Run from this folder:  Rscript 03_run_ghosal.R
# =============================================================================

get_script_dir <- function() {
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg) == 1) return(dirname(normalizePath(sub("^--file=", "", file_arg))))
  getwd()
}
script_dir <- get_script_dir()
out_dir <- file.path(script_dir, "output")

ghosal_file <- file.path(script_dir, "..", "..", "competitor", "ghosal2023", "SFDABP.R")
if (!file.exists(ghosal_file)) {
  stop("SFDABP.R not found. Download it from the supplementary material of Ghosal et al. (2023), ",
       "see competitor/ghosal2023/README.md.")
}
source(ghosal_file)
library(readr)

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
Scenario <- "TanhFOFR"
Nvalues <- c(25, 50, 100)
m1 <- 50
Simulaciones <- 200
Ngrid <- c(3:8)  # candidate degrees of the Bernstein basis

cv.FLCM.con <- Vectorize(cv.FLCM.con, vectorize.args = "N1")

in_file <- function(prefix, N) file.path(out_dir, paste0(prefix, "_Scenario", Scenario, "_N_", N, "_m_", m1, ".csv"))

for (N in Nvalues) {
  n <- N  # SFDABP.R uses the global variable n

  # ---------------------------------------------------------------------------
  # Read data
  # ---------------------------------------------------------------------------
  X_se <- as.matrix(read_csv(in_file("X", N), show_col_types = FALSE))
  y <- as.matrix(read_csv(in_file("Y", N), show_col_types = FALSE))
  t <- as.vector(as.matrix(read_csv(in_file("T1", N), show_col_types = FALSE)))

  time <- rep(0, Simulaciones)
  Beta0_est <- matrix(0, nrow = Simulaciones, ncol = m1)
  Beta1_est <- matrix(0, nrow = Simulaciones, ncol = m1)
  y_est <- array(0, dim = c(N, Simulaciones, m1))
  y_est_big <- array(0, dim = c(N * Simulaciones, ncol = m1 + 1))

  # ---------------------------------------------------------------------------
  # Fit each Monte Carlo sample
  # ---------------------------------------------------------------------------
  set.seed(1)
  for (i in 1:Simulaciones) {
    # Center data
    y_centered <- y[((i - 1) * N + 1):(i * N), 1:m1]
    X_centered <- X_se[((i - 1) * N + 1):(i * N), 1:m1]
    y_centered <- sweep(y_centered, 2, colMeans(y_centered), "-")
    X_centered <- sweep(X_centered, 2, colMeans(X_centered), "-")

    print(paste0("Simulacion ", i, ", N=", N))
    start_time <- Sys.time()
    cv <- cv.FLCM.con(Ngrid, y_centered, X_centered, t, constr = "inc")
    N1 <- Ngrid[which.min(cv)]  # selected degree
    out <- FLCM.con(N1, y_centered, X_centered, t, m1, constr = "inc")
    end_time <- Sys.time()
    time[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))

    Beta0_est[i, ] <- out$beta0t
    Beta1_est[i, ] <- out$beta1t
    y_est[, i, ] <- out$Yhat
    y_est_big[(1 + (i - 1) * N):(N * i), 1:m1] <- y_est[, i, ]
    y_est_big[(1 + (i - 1) * N):(N * i), m1 + 1] <- i
  }

  # ---------------------------------------------------------------------------
  # Save
  # ---------------------------------------------------------------------------
  write.csv(time, file = in_file("CB-R_time", N), row.names = FALSE)
  write.csv(Beta0_est, file = in_file("CB-R_Beta0_est", N), row.names = FALSE)
  write.csv(Beta1_est, file = in_file("CB-R_Beta1_est", N), row.names = FALSE)
  write.csv(y_est_big, file = in_file("CB-R_y_est", N), row.names = FALSE)
}
