"""
Concurrent FOF regression - Hyperbolic tangent scenario
Step 4: compute the metrics and plots of the Ghosal et al. (2023) estimates (step 3),
in the same way as for FunSCOPE. Responses are centered column-wise, as in step 3.

Writes in ./output: X_ScenarioTanhFOFR_N_<N>_m_50_Ghosal.txt / .png / _beta0_Ghosal.png / .dat
The paper reports IMSE(y) as MSE_avg / MSE_std of these reports.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from funscope.experiments import (
    bias_variance,
    data_file,
    mean_std,
    plot_band,
    read_array,
    read_csv,
    read_grid,
    save_shelf,
    write_report,
)
from funscope.utils import imse_simpson

OUTPUT = HERE / "output"

# =============================================================================
# Settings
# =============================================================================
SCENARIO = "TanhFOFR"
N_VALUES = [25, 50, 100]
M1 = 50
N_SIM = 200
CONSTRAINTS_NAME = "First_Positive"
YLIM = (-1.5, 1.5)
YLIM_BETA0 = (-2, 2)


def sample_metrics(y, y_est, t, beta_true, Beta_est, N, i):
    """MSE of Y, IMSE of beta and IMSE of Y for the i-th sample."""
    y_centered = y[(i * N):(N * (i + 1)), 0:M1]
    y_centered = y_centered - y_centered.mean(axis=0, keepdims=True)
    y_estimated = y_est[(i * N):(N * (i + 1)), 0:M1]
    MSE = np.mean((y_centered - y_estimated) ** 2)
    imse_beta = imse_simpson(t, beta_true, Beta_est[i, :])
    imse_y = np.mean([imse_simpson(t, y_centered[idx, :], y_estimated[idx, :]) for idx in range(N)])
    return MSE, imse_beta, imse_y


def main():
    for N in N_VALUES:
        # ---------------------------------------------------------------------
        # Read data and Ghosal et al. estimates
        # ---------------------------------------------------------------------
        Beta0_est = read_array(data_file(OUTPUT, "CB-R_Beta0_est", SCENARIO, N, M1))
        Beta_est = read_array(data_file(OUTPUT, "CB-R_Beta1_est", SCENARIO, N, M1))
        y_est = read_array(data_file(OUTPUT, "CB-R_y_est", SCENARIO, N, M1))
        time = read_array(data_file(OUTPUT, "CB-R_time", SCENARIO, N, M1))
        y = read_array(data_file(OUTPUT, "Y", SCENARIO, N, M1))
        t = read_grid(data_file(OUTPUT, "T1", SCENARIO, N, M1))
        beta_true = np.array(read_csv(data_file(OUTPUT, "Beta1T1", SCENARIO, N, M1)))[:, 0]
        beta0_true = np.array(read_csv(data_file(OUTPUT, "Beta0T1", SCENARIO, N, M1)))[:, 0]

        # ---------------------------------------------------------------------
        # Metrics
        # ---------------------------------------------------------------------
        metrics = [sample_metrics(y, y_est, t, beta_true, Beta_est, N, i) for i in range(N_SIM)]
        MSE, imse_approx, imse_mse = (list(v) for v in zip(*metrics))

        MSE_avg, MSE_std = mean_std(MSE)
        IMSE_avg, IMSE_std = mean_std(imse_approx)
        IMSE_MSE_avg, IMSE_MSE_std = mean_std(imse_mse)
        time_avg = np.mean(time)
        Beta_media, IntSquBias, IVar = bias_variance(t, beta_true, Beta_est)

        basename = f"X_Scenario{SCENARIO}_N_{N}_m_{M1}"
        write_report(
            OUTPUT / f"{basename}_Ghosal.txt",
            title=("RESULTS Constrained Bernstein", CONSTRAINTS_NAME),
            raw=[("MSE", MSE), ("IMSE Beta", imse_approx), ("IMSE Y", imse_mse), ("Time", time)],
            averaged=[
                ("MSE_avg", MSE_avg),
                ("MSE_std", MSE_std),
                ("IMSE_avg", IMSE_avg),
                ("IMSE_std", IMSE_std),
                ("IMSE_Y_avg", IMSE_MSE_avg),
                ("IMSE_Y_std", IMSE_MSE_std),
                ("ISB", IntSquBias),
                ("IVar", IVar),
                ("Time", time_avg),
            ],
        )

        # ---------------------------------------------------------------------
        # Plots and storage
        # ---------------------------------------------------------------------
        Beta_low, Beta_upp = plot_band(t, beta_true, Beta_est, YLIM, OUTPUT / f"{basename}_Ghosal.png")
        plot_band(t, beta0_true, Beta0_est, YLIM_BETA0, OUTPUT / f"{basename}_beta0_Ghosal.png")

        save_shelf(OUTPUT / f"{basename}_Ghosal.dat", {
            "Beta_est": Beta_est, "y_est": y_est, "MSE": MSE, "imse_approx": imse_approx, "time": time,
            "Beta_media": Beta_media, "Beta_upp": Beta_upp, "Beta_low": Beta_low,
            "Scenario": SCENARIO, "N": N, "m1": M1,
            "MSE_avg": MSE_avg, "IMSE_avg": IMSE_avg, "IMSE_MSE_avg": IMSE_MSE_avg,
            "MSE_std": MSE_std, "IMSE_std": IMSE_std, "IMSE_MSE_std": IMSE_MSE_std,
            "IntSquBias": IntSquBias, "IVar": IVar, "time_avg": time_avg,
        })


if __name__ == "__main__":
    main()
