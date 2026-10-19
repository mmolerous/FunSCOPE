# Ghosal et al. (2023)

The simulation study compares FunSCOPE with the shape-constrained Bernstein
polynomial estimators of

> Ghosal, R., Ghosh, S., Urbanek, J., Schrack, J. A. and Zipunnikov, V. (2023).
> Shape-constrained estimation in functional regression with Bernstein polynomials.
> *Computational Statistics & Data Analysis*, 178, 107614.
> [doi:10.1016/j.csda.2022.107614](https://doi.org/10.1016/j.csda.2022.107614)

Their R code is distributed by the authors as supplementary material of the
article and is **not included in this repository**. To run the comparison:

1. On the article page, go to *Appendix A. Supplementary material* and
   download **MMC 2** (zip file, "Software implementation via R").
2. Copy the file `SFDABP.R` from the zip into this folder
   (`competitor/ghosal2023/SFDABP.R`).

The file is used, without modifications, by
`simulations/SOF_normal/03_run_ghosal.R` (`cv.SOFR.con`, `SOFR.con`) and
`simulations/FOF_tanh/03_run_ghosal.R` (`cv.FLCM.con`, `FLCM.con`). These
functions read the sample size from the global variables `N` / `n`, which
the calling scripts define.
