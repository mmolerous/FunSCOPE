# =============================================================================
# SOF regression - Normal scenario
# Step 3: fit the shape-constrained Bernstein polynomial method of
#         Ghosal et al. (2023) with the constraint beta(t) >= 0
# =============================================================================
# The degree of the Bernstein basis is selected by 5-fold cross-validation over
# 3:8. Data are not centered (the model includes an intercept).
#
# Output (in ./output), one file per N:
#   CB-R_time_ScenarioB_N_<N>_m_50.csv      computing time (seconds)
#   CB-R_Beta_est_ScenarioB_N_<N>_m_50.csv  estimated beta(t), one row per sample
#   CB-R_y_est_ScenarioB_N_<N>_m_50.csv     fitted responses, one row per sample
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
Scenario <- "B"
Nvalues <- c(25, 50, 100)
m1 <- 50
Simulaciones <- 200
Ngrid <- c(3:8)  # candidate degrees of the Bernstein basis

cv.SOFR.con <- Vectorize(cv.SOFR.con, vectorize.args = "N1")

in_file <- function(prefix, N) file.path(out_dir, paste0(prefix, "_Scenario", Scenario, "_N_", N, "_m_", m1, ".csv"))

for (N in Nvalues) {
  # ---------------------------------------------------------------------------
  # Read data
  # ---------------------------------------------------------------------------
  X_se <- as.matrix(read_csv(in_file("X", N), show_col_types = FALSE))
  y <- as.matrix(read_csv(in_file("Y", N), show_col_types = FALSE))
  t <- as.vector(as.matrix(read_csv(in_file("T1", N), show_col_types = FALSE)))

  time <- rep(0, Simulaciones)
  Beta_est <- matrix(0, nrow = Simulaciones, ncol = m1)
  y_est <- matrix(0, nrow = Simulaciones, ncol = N)

  # ---------------------------------------------------------------------------
  # Fit each Monte Carlo sample
  # ---------------------------------------------------------------------------
  set.seed(1)
  for (i in 1:Simulaciones) {
    print(paste0("Simulacion ", i, ", N=", N))
    X_i <- X_se[((i - 1) * N + 1):(i * N), 1:m1]
    start_time <- Sys.time()
    cv <- cv.SOFR.con(Ngrid, y[i, ], X_i, t, constr = "positive")
    N1 <- Ngrid[which.min(cv)]  # selected degree
    out <- SOFR.con(N1, y[i, ], X_i, t, m1, constr = "positive")
    end_time <- Sys.time()
    time[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
    Beta_est[i, ] <- out$betaSOFR
    y_est[i, ] <- out$yhat
  }

  # ---------------------------------------------------------------------------
  # Save
  # ---------------------------------------------------------------------------
  write.csv(time, file = in_file("CB-R_time", N), row.names = FALSE)
  write.csv(Beta_est, file = in_file("CB-R_Beta_est", N), row.names = FALSE)
  write.csv(y_est, file = in_file("CB-R_y_est", N), row.names = FALSE)
}
