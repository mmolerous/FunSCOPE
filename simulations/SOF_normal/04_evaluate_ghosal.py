"""
SOF regression - Normal scenario
Step 4: compute the metrics and plots of the Ghosal et al. (2023) estimates (step 3),
in the same way as for FunSCOPE.

Writes in ./output: X_ScenarioB_N_<N>_m_50_Ghosal.txt / .png / .dat
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
SCENARIO = "B"
N_VALUES = [25, 50, 100]
M1 = 50
N_SIM = 200
CONSTRAINTS_NAME = "Zero_Positive"
YLIM = (-0.1, 4.5)


def main():
    for N in N_VALUES:
        # ---------------------------------------------------------------------
        # Read data and Ghosal et al. estimates
        # ---------------------------------------------------------------------
        Beta_est = read_array(data_file(OUTPUT, "CB-R_Beta_est", SCENARIO, N, M1))
        y_est = read_array(data_file(OUTPUT, "CB-R_y_est", SCENARIO, N, M1))
        time = read_array(data_file(OUTPUT, "CB-R_time", SCENARIO, N, M1))
        y = read_array(data_file(OUTPUT, "Y", SCENARIO, N, M1))
        t = read_grid(data_file(OUTPUT, "T1", SCENARIO, N, M1))
        beta_true = np.array(read_csv(data_file(OUTPUT, "BetaT1", SCENARIO, N, M1)))[:, 0]

        # ---------------------------------------------------------------------
        # Metrics
        # ---------------------------------------------------------------------
        MSE = [np.mean((y[i, :] - y_est[i, :]) ** 2) for i in range(N_SIM)]
        imse_approx = [imse_simpson(t, beta_true, Beta_est[i, :]) for i in range(N_SIM)]

        MSE_avg, MSE_std = mean_std(MSE)
        IMSE_avg, IMSE_std = mean_std(imse_approx)
        time_avg = np.mean(time)
        Beta_media, IntSquBias, IVar = bias_variance(t, beta_true, Beta_est)

        basename = f"X_Scenario{SCENARIO}_N_{N}_m_{M1}_Ghosal"
        write_report(
            OUTPUT / f"{basename}.txt",
            title=("RESULTS Constrained Bernstein", CONSTRAINTS_NAME),
            raw=[("MSE", MSE), ("IMSE", imse_approx), ("Time", time)],
            averaged=[
                ("MSE_avg", MSE_avg),
                ("MSE_std", MSE_std),
                ("IMSE_avg", IMSE_avg),
                ("IMSE_std", IMSE_std),
                ("ISB", IntSquBias),
                ("IVar", IVar),
                ("Time", time_avg),
            ],
        )

        # ---------------------------------------------------------------------
        # Plot and storage
        # ---------------------------------------------------------------------
        Beta_low, Beta_upp = plot_band(t, beta_true, Beta_est, YLIM, OUTPUT / f"{basename}.png")

        save_shelf(OUTPUT / f"{basename}.dat", {
            "Beta_est": Beta_est, "y_est": y_est, "MSE": MSE, "imse_approx": imse_approx, "time": time,
            "Beta_media": Beta_media, "Beta_upp": Beta_upp, "Beta_low": Beta_low,
            "Scenario": SCENARIO, "N": N, "m1": M1,
            "MSE_avg": MSE_avg, "IMSE_avg": IMSE_avg, "MSE_std": MSE_std, "IMSE_std": IMSE_std,
            "IntSquBias": IntSquBias, "IVar": IVar, "time_avg": time_avg,
        })


if __name__ == "__main__":
    main()
