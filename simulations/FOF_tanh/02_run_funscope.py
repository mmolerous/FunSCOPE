"""
Concurrent FOF regression - Hyperbolic tangent scenario
Step 2: fit FunSCOPE-U (unconstrained) and FunSCOPE (beta'(t) >= 0) to every simulated sample.

In the output, the suffixes refer to the three estimates of each run:
    unc   unconstrained, ignoring the error correlation
    unc2  unconstrained GLS                 -> FunSCOPE-U in the paper
    cons  constrained GLS                   -> FunSCOPE in the paper (run "First_Positive")
The paper reports IMSE(y) as MSE_avg_* / MSE_std_* of these reports.

Writes in ./output, for each N and constraint setting
(X_ScenarioTanhFOFR_N_<N>_m_50_int_constraints_<name>_centered_True_knots_14 + suffix):
    _FOFR.txt             metrics
    _FOFR.png             mean and 95% band of the unc estimate
    _FOFR_cons.png        mean and 95% band of the cons estimate
    _beta0_FOFR.png       intercept of the cons estimate
    _somebetas_FOFR.png   some individual cons estimates
    _FOFR.dat             all results (shelve)
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from funscope import FOFR_basis
from funscope.experiments import (
    bias_variance,
    data_file,
    mean_std,
    plot_band,
    plot_some,
    read_array,
    read_csv,
    read_grid,
    save_shelf,
    write_report,
)

OUTPUT = HERE / "output"

# =============================================================================
# Settings
# =============================================================================
SCENARIO = "TanhFOFR"
N_VALUES = [25, 50, 100]
M1 = 50
N_SIM = 200

DEGREE = 3
KNOTS = 14
G = 2000           # fine grid used to compute the integrals
CENTERED = True
CONSTRAINTS = {    # name used in the output files -> shape constraints
    "First_Positive": {"X": {1: {"+": 0}}},  # beta'(t) >= 0
    "None": None,
}
ALPHA_CONSTRAINTS = []
COVARIANCE_ESTIMATION = "FPCA"
YLIM = (-1.5, 1.5)
YLIM_BETA0 = (-2, 2)

# Outputs of FOFR_basis, in order
OUTPUT_KEYS = [
    "FOFR_unc", "FOFR_unc2", "FOFR_cons",
    "Delta_unc", "Delta_unc2", "Delta_cons",
    "Beta_est_unc", "Beta_est_unc2", "Beta_est_cons", "Beta_est_cons_Phi_inner",
    "y_est_unc", "y_est_unc2", "y_est_cons",
    "MSE_unc", "MSE_unc2", "MSE_cons",
    "time", "Phi", "knots_print", "tt",
    "imse_approx_unc", "imse_approx_unc2", "imse_approx_cons",
    "imse_mse_unc", "imse_mse_unc2", "imse_mse_cons",
    "identity_used",
]
PER_SIMULATION = [k for k in OUTPUT_KEYS if k not in ("Beta_est_cons_Phi_inner", "Phi", "knots_print", "tt")]


# =============================================================================
# Steps
# =============================================================================
def load_data(N):
    X_se = read_array(data_file(OUTPUT, "X", SCENARIO, N, M1))
    y = read_array(data_file(OUTPUT, "Y", SCENARIO, N, M1))
    t = read_grid(data_file(OUTPUT, "T1", SCENARIO, N, M1))
    Beta0_real = read_csv(data_file(OUTPUT, "Beta0T1", SCENARIO, N, M1))
    Beta1_real = read_csv(data_file(OUTPUT, "Beta1T1", SCENARIO, N, M1))
    return X_se, y, t, Beta0_real, Beta1_real


def run_simulations(X_se, y, t, Beta1_real, N, name, int_constraints):
    """Fit the model to each Monte Carlo sample."""
    res = {key: [None] * N_SIM for key in PER_SIMULATION}
    res["Beta0_est_cons"] = [None] * N_SIM
    for i in range(N_SIM):
        print(f"centered_{CENTERED} constraints_{name} knots_{KNOTS} N_{N} Simulacion_{i}")
        out = FOFR_basis(
            X_se[(i * N):(N * (i + 1)), 0:M1], t, y[(i * N):(N * (i + 1)), 0:M1], Beta1_real,
            DEGREE, KNOTS, G, CENTERED, int_constraints, ALPHA_CONSTRAINTS, COVARIANCE_ESTIMATION,
        )
        out = dict(zip(OUTPUT_KEYS, out))
        for key in PER_SIMULATION:
            res[key][i] = out[key]
        res["Phi"] = out["Phi"]
        res["Beta0_est_cons"][i] = out["Delta_cons"][-1] * np.ones(M1)
        print("\n")
    return res


def summarize(res, t, beta_true, int_constraints, basename):
    """Write the metrics report and return the summary statistics."""
    summary = {}
    averaged = []
    for s in ["unc", "unc2", "cons"]:
        MSE_avg, MSE_std = mean_std(res[f"MSE_{s}"])
        IMSE_avg, IMSE_std = mean_std(res[f"imse_approx_{s}"])
        IMSE_MSE_avg, IMSE_MSE_std = mean_std(res[f"imse_mse_{s}"])
        Beta_media, IntSquBias, IVar = bias_variance(t, beta_true, res[f"Beta_est_{s}"])
        summary.update({
            f"MSE_avg_{s}": MSE_avg, f"IMSE_avg_{s}": IMSE_avg, f"IMSE_MSE_avg_{s}": IMSE_MSE_avg,
            f"MSE_std_{s}": MSE_std, f"IMSE_std_{s}": IMSE_std, f"IMSE_MSE_std_{s}": IMSE_MSE_std,
            f"IntSquBias_{s}": IntSquBias, f"IVar_{s}": IVar, f"Beta_media_{s}": Beta_media,
        })
        averaged += [
            (f"MSE_avg_{s}", MSE_avg),
            (f"MSE_std_{s}", MSE_std),
            (f"IMSE_Beta_avg_{s}", IMSE_avg),
            (f"IMSE_Beta_std_{s}", IMSE_std),
            (f"IMSE_Y_avg_{s}", IMSE_MSE_avg),
            (f"IMSE_Y_std_{s}", IMSE_MSE_std),
            (f"ISB_{s}", IntSquBias),
            (f"IVar_{s}", IVar),
        ]
    summary["time_avg"] = np.mean(res["time"])
    averaged.append(("Time", summary["time_avg"]))

    write_report(
        OUTPUT / f"{basename}_FOFR.txt",
        title=("RESULTS int_constraints", int_constraints),
        raw=[
            ("MSE unconstrained", res["MSE_unc"]),
            ("MSE unconstrained 2", res["MSE_unc2"]),
            ("MSE constrained", res["MSE_cons"]),
            ("IMSE Beta unconstrained", res["imse_approx_unc"]),
            ("IMSE Beta unconstrained 2", res["imse_approx_unc2"]),
            ("IMSE Beta constrained", res["imse_approx_cons"]),
            ("IMSE Y unconstrained", res["imse_mse_unc"]),
            ("IMSE Y unconstrained 2", res["imse_mse_unc2"]),
            ("IMSE Y constrained", res["imse_mse_cons"]),
            ("Time", res["time"]),
        ],
        averaged=averaged,
    )
    return summary


def main():
    for N in N_VALUES:
        X_se, y, t, Beta0_real, Beta1_real = load_data(N)
        beta_true = np.array(Beta1_real)[:, 0]
        beta0_true = np.array(Beta0_real)[:, 0]
        for name, int_constraints in CONSTRAINTS.items():
            basename = f"X_Scenario{SCENARIO}_N_{N}_m_{M1}_int_constraints_{name}_centered_{CENTERED}_knots_{KNOTS}"
            print("\n")
            res = run_simulations(X_se, y, t, Beta1_real, N, name, int_constraints)
            for s in ["unc", "unc2", "cons"]:
                res[f"Beta_est_{s}"] = np.array(res[f"Beta_est_{s}"])
            summary = summarize(res, t, beta_true, int_constraints, basename)

            # Plots
            Beta_low, Beta_upp = plot_band(t, beta_true, res["Beta_est_unc"], YLIM, OUTPUT / f"{basename}_FOFR.png")
            plot_band(t, beta_true, res["Beta_est_cons"], YLIM, OUTPUT / f"{basename}_FOFR_cons.png")
            plot_band(t, beta0_true, res["Beta0_est_cons"], YLIM_BETA0, OUTPUT / f"{basename}_beta0_FOFR.png")
            plot_some(t, beta_true, res["Beta_est_cons"], YLIM, OUTPUT / f"{basename}_somebetas_FOFR.png")

            save_shelf(OUTPUT / f"{basename}_FOFR.dat", {
                **{k: v for k, v in res.items() if k != "Beta0_est_cons"},
                **summary,
                "imse_mse_approx_unc": res["imse_mse_unc"],
                "imse_mse_approx_unc2": res["imse_mse_unc2"],
                "imse_mse_approx_cons": res["imse_mse_cons"],
                "Beta_upp": Beta_upp, "Beta_low": Beta_low,
                "Scenario": SCENARIO, "N": N, "m1": M1,
                "int_constraints": int_constraints, "alpha_constraints": ALPHA_CONSTRAINTS,
            })


if __name__ == "__main__":
    main()
