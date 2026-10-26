# FunSCOPE

Shape-constrained functional regression with P-splines and conic optimization.

FunSCOPE estimates functional regression models with P-splines while imposing
shape constraints (sign, monotonicity and/or curvature) on the functional
coefficient $\beta(t)$. The constraints are necessary and sufficient: they
hold on the whole domain, not only on a grid. They are formulated as
semidefinite constraints and solved with MOSEK. The following models are
supported:

| Model | Equation | Function |
|---|---|---|
| Scalar-on-function (SOF) | $y_i = \beta_0 + \int X_i(t)\beta(t)\,dt + \varepsilon_i$ | `fit_sof` |
| Function-on-scalar (FOS) | $Y_i(t) = \beta_0(t) + X_i\,\beta(t) + \varepsilon_i(t)$ | `fit_fos` |
| Concurrent function-on-function (FOF) | $Y_i(t) = \beta_0(t) + X_i(t)\,\beta(t) + \varepsilon_i(t)$ | `fit_fof` |

For FOS and FOF the error correlation is taken into account by generalized
least squares, with the error covariance estimated by FPCA
(`refund::fpca.sc`).

This repository contains the code for the paper:

> M. Durbán, V. Guerrero and C. Molero-Río. *A conic optimization framework for shape-constrained functional regression analysis.*

It can be used in two ways:

* **to fit the models to your own data** (see [Quick start](#quick-start));
* **to reproduce the simulation study of the paper**, including the
  comparison with the method of Ghosal et al. (2023) (see
  [Reproducing the results of the paper](#reproducing-the-results-of-the-paper)).

---

## Repository structure

```
FunSCOPE/
├── funscope/                     Python package
│   ├── api.py                    fit_sof, fit_fos, fit_fof: user interface
│   ├── sof.py                    SOFR_basis: scalar-on-function regression
│   ├── fos.py                    FOSR_basis: function-on-scalar regression
│   ├── fof.py                    FOFR_basis: concurrent function-on-function regression
│   ├── building_blocks.py        steps shared by the three models (bases, pre-smoothing, Psi, fits)
│   ├── cpsplines_design.py       constrained P-spline fit with a given design matrix + intercept
│   ├── int_constraints.py        shape constraints (semidefinite formulation)
│   ├── bspline_basis.py          B-spline basis wrapper used by the constraints
│   ├── covariance.py             error covariance by FPCA (calls R)
│   ├── R/fpca_covariance.R       R side of the covariance estimation
│   ├── utils.py                  Simpson's rule weights and IMSE
│   └── experiments.py            helpers for the simulations (I/O, summaries, plots)
├── examples/                     one runnable example per model, with simulated data
├── simulations/                  scripts reproducing the simulation study of the paper
│   ├── SOF_normal/               SOF, normal-density beta(t), constraint beta >= 0
│   ├── FOS_weibull/              FOS, Weibull-density beta(t), constraints beta >= 0 and beta'' >= 0
│   └── FOF_tanh/                 concurrent FOF, tanh beta(t), constraint beta' >= 0
└── competitor/ghosal2023/        place for the R code of Ghosal et al. (2023) (not included, see below)
```

---

## Installation

### Python

Python 3.12 and a [MOSEK](https://www.mosek.com/) license are required
(free academic licenses are available). FunSCOPE builds on
[`cpsplines`](https://github.com/ManuelNavarroGarcia/cpsplines) 0.3.0, which
is installed automatically.

```bash
git clone https://github.com/mmolerous/FunSCOPE.git
cd FunSCOPE
pip install .
```

To reproduce the paper with the exact package versions we used, create the
conda environment instead (it installs the pinned versions of
`requirements.txt`), and then install FunSCOPE in it:

```bash
conda env create -f environment.yml
conda activate funscope
pip install --no-deps .
```

### R

R (≥ 4.3) is needed for the error covariance estimation (FOS and FOF), for
the generation of the simulated data and for the competing method:

```r
install.packages(c("refund", "Matrix", "lme4",                         # FunSCOPE (FOS, FOF)
                   "readr", "caret", "restriktor", "quadprog", "MASS"))  # Ghosal et al. (2023)
```

FunSCOPE calls R through `Rscript`. If `Rscript` is not on the `PATH`, set
the environment variable `FUNSCOPE_RSCRIPT` to its full path, e.g.

```bash
# Windows (PowerShell)
$env:FUNSCOPE_RSCRIPT = "C:/Program Files/R/R-4.3.0/bin/Rscript.exe"
# Linux / macOS
export FUNSCOPE_RSCRIPT=/usr/local/bin/Rscript
```

SOF only needs Python; R is not required.

---

## Quick start

```python
import numpy as np
from funscope import fit_sof, fit_fos, fit_fof

# SOF: X (N x m) functional covariate on the grid t (m,), y (N,) scalar responses
res = fit_sof(X, t, y, constraints="nonnegative")

# FOS: X (N,) scalar covariate, Y (N x m) functional responses on t
res = fit_fos(X, t, Y, constraints=["nonnegative", "convex"])

# Concurrent FOF: X (N x m) and Y (N x m), both on t
res = fit_fof(X, t, Y, constraints="increasing")

res["beta"]      # estimated beta(t) on t
res["fitted"]    # fitted responses
```

The folder `examples/` contains a complete, runnable example for each model
(`python examples/example_sof.py`, …). Each one simulates a sample, fits the
model with and without constraints, and saves a plot of the estimates.

### Input data

* All curves (functional covariate and/or response) must be observed on a
  **common grid** `t` of `m` points. The paper uses equispaced grids.
* SOF: `X` is an `N x m` array and `y` a vector of length `N`.
* FOS: `X` is a vector of length `N` (one scalar covariate) and `Y` an
  `N x m` array.
* FOF: `X` and `Y` are `N x m` arrays.

### Shape constraints

`constraints` is `None` (unconstrained) or one or more of the following names:

| Name | Constraint |
|---|---|
| `"nonnegative"` / `"nonpositive"` | $\beta(t) \ge 0$ / $\beta(t) \le 0$ |
| `"increasing"` / `"decreasing"` | $\beta'(t) \ge 0$ / $\beta'(t) \le 0$ |
| `"convex"` / `"concave"` | $\beta''(t) \ge 0$ / $\beta''(t) \le 0$ |

They hold on the whole domain of `t`. The underlying dictionary format,
`{"X": {derivative_order: {"+" or "-": threshold}}}`, can also be passed
directly; it allows non-zero thresholds, e.g. `{"X": {0: {"+": 1}}}` for
$\beta(t) \ge 1$.

### Options and output

The defaults are those used in the paper: cubic B-splines (`degree=3`) with
`knots=14` equispaced intervals, and `g=2000` (even) intervals for the
numerical integration. The smoothing parameter is selected automatically by
generalized cross-validation. Data are centered before fitting. The argument
`beta_true` is only used in simulations, to report the IMSE of
$\hat\beta(t)$.

The functions return a dictionary:

| Key | Content |
|---|---|
| `beta` | estimated $\beta(t)$ on `t` (constrained, if `constraints` is given) |
| `beta_unconstrained` | (FOS, FOF) unconstrained estimate, FunSCOPE-U |
| `intercept` | (SOF) estimated $\beta_0$ |
| `fitted` | fitted responses: scalars (SOF) or curves on `t` (FOS, FOF) |
| `mse` / `imse_y` | (SOF) $\frac1N\sum(y_i-\hat y_i)^2$ / (FOS, FOF) $\frac1N\sum\frac1T\int(Y_i-\hat Y_i)^2$ |
| `imse_beta` | IMSE of $\hat\beta(t)$ (`NaN` if `beta_true` is not given) |
| `time` | computing time (seconds) |
| `covariance_failed` | (FOS, FOF) `True` if the FPCA covariance estimation failed and the identity was used |
| `model` | fitted optimization model |

Typical computing times per fit are below 1 s for SOF, a few seconds for FOS
and 10–20 s for FOF with $m=50$ and $N \le 100$.

### Limitations

* One covariate per model; FOF refers to the concurrent model only.
* All curves must share the same observation grid.
* There is no `predict` method for new observations yet.
* FOS and FOF require R for the covariance estimation.

### Lower-level functions

`fit_sof`, `fit_fos` and `fit_fof` call `SOFR_basis`, `FOSR_basis` and
`FOFR_basis`, which are the functions used in the paper and return all the
intermediate results as a tuple (see their docstrings). For FOS and FOF they
compute three estimates:

* `unc`: unconstrained, ignoring the error correlation;
* `unc2`: unconstrained, generalized least squares (**FunSCOPE-U** in the paper);
* `cons`: shape-constrained, generalized least squares (**FunSCOPE** in the paper).

---

## Reproducing the results of the paper

Each folder in `simulations/` contains numbered scripts that are run in
order. Every script reads from and writes to the `output/` subfolder of its
scenario, which is created automatically.

| Step | SOF_normal | FOS_weibull | FOF_tanh |
|---|---|---|---|
| 1. Generate data (R) | `01_generate_data.R` | `01_generate_data.R` | `01_generate_data.R` |
| 2. FunSCOPE / FunSCOPE-U (Python) | `02_run_funscope.py` | `02_run_funscope.py` | `02_run_funscope.py` |
| 3. Ghosal et al. (2023) (R) | `03_run_ghosal.R` | – | `03_run_ghosal.R` |
| 4. Metrics for Ghosal et al. (Python) | `04_evaluate_ghosal.py` | – | `04_evaluate_ghosal.py` |

The method of Ghosal et al. (2023) does not support FOS regression in its
open-source implementation, so for FOS only FunSCOPE is run.

**Before step 3**, download the R code of Ghosal et al. (2023): it is the
supplementary file *MMC 2* of the article
([doi:10.1016/j.csda.2022.107614](https://doi.org/10.1016/j.csda.2022.107614)).
Copy `SFDABP.R` from that zip into `competitor/ghosal2023/` (see
`competitor/ghosal2023/README.md`). It is not included in this repository
because it is not distributed under an open-source license.

Example (SOF):

```bash
cd simulations/SOF_normal
Rscript 01_generate_data.R
python 02_run_funscope.py
Rscript 03_run_ghosal.R
python 04_evaluate_ghosal.py
```

All settings (sample sizes, number of Monte Carlo samples, number of knots,
constraints, …) are defined at the top of each script.

### Computing time

Each script fits 200 Monte Carlo samples for $N \in \{25, 50, 100\}$.
Approximate running times on the computer used in the paper:

| Step | SOF | FOS | FOF |
|---|---|---|---|
| 1. Data generation | seconds | seconds | seconds |
| 2. FunSCOPE (constrained and unconstrained runs) | ~15 min | ~1 h | ~4 h |
| 3. Ghosal et al. (2023) | ~50 min | – | ~6 h |
| 4. Metrics | seconds | – | seconds |

To make a quick test, reduce `N_SIM` (Python) / `Simulaciones` (R) and
`N_VALUES` / `Nvalues` at the top of the scripts.

### Outputs

For each sample size and each constraint setting, the scripts write:

* `*.txt`: per-sample values and averages (mean and standard deviation over
  the Monte Carlo samples) of the error measures and computing times;
* `*.png`: true $\beta(t)$, Monte Carlo mean of the estimates and pointwise
  2.5%–97.5% percentile band;
* `*.dat`: all results (estimates, fitted values, metrics), stored with
  Python's `shelve`.

Correspondence with the tables and figures of the paper:

| Paper | SOF | FOS / FOF |
|---|---|---|
| FunSCOPE | `MSE_avg`, `IMSE_avg` of the constrained run (`Zero_Positive`) | `*_cons` entries of the constrained run |
| FunSCOPE-U | `MSE_avg`, `IMSE_avg` of the `None` run | `*_unc2` entries |
| Ghosal et al. | `*_Ghosal.txt` | `*_Ghosal.txt` (FOF only) |
| $\mathrm{MSE}(\hat y)$ / $\mathrm{IMSE}(\hat y)$ | `MSE_avg` / `MSE_std` | `IMSE_Y_avg_*` / `IMSE_Y_std_*` (Ghosal et al.: `MSE_avg` / `MSE_std`) |
| $\mathrm{IMSE}(\hat\beta)$ | `IMSE_avg` / `IMSE_std` | `IMSE_Beta_avg_*` / `IMSE_Beta_std_*` |
| Time | `Total Time` | `Time` |
| Figures (FunSCOPE / FunSCOPE-U) | `.png` of the constrained / `None` run | FOS: `.png`; FOF: `_FOFR_cons.png`, of the constrained / `None` run |

### Reproducibility

Steps 1 and 2 reproduce the data and the FunSCOPE results of the paper.
Results on other platforms or solver versions may differ slightly: in our
checks, the generated data agreed up to $10^{-9}$ and the estimates up to
about $10^{-6}$ (relative) in SOF and FOF and about 1% per sample in FOS
(where the FPCA covariance estimation in R is involved). Computing times
depend on the machine. The results in the paper were obtained on a MacBook
Pro (Apple M2 Pro, 16 GB RAM) with Python 3.12, `cpsplines` 0.3.0 and MOSEK
11.0.13.

---

## License

MIT License (see `LICENSE`). Parts of `funscope/` are adapted from
[`cpsplines`](https://github.com/ManuelNavarroGarcia/cpsplines) (MIT). The
code of Ghosal et al. (2023) is not covered by this license and is not
distributed here.

---

## References

* Ghosal, R., Ghosh, S., Urbanek, J., Schrack, J. A. and Zipunnikov, V. (2023).
  Shape-constrained estimation in functional regression with Bernstein polynomials.
  *Computational Statistics & Data Analysis*, 178, 107614.
* Navarro-García, M., Guerrero, V. and Durbán, M. (2023). On constrained smoothing
  and out-of-range prediction using P-splines: a conic optimization approach.
  *Applied Mathematics and Computation*, 441, 127679.
