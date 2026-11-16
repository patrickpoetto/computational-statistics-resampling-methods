########################################################################
# Package requirements for this repository.
#
# Run once:   Rscript requirements.R
# Then:       Rscript scripts_build_data.R   # builds data/*.csv
#             Rscript 01_bagging_decision_trees.R
#             Rscript 02_bootstrap_pca_multiple_testing.R
#
# Tested with R 4.5.2 (2025-10-31) on macOS (Homebrew R).
########################################################################

pkgs <- c(
  "dplyr", # data manipulation (both scripts)
  "purrr", # map() in 01 (bagging-of-OLS motivating example)
  "rpart", # CART trees (01)
  "mlbench", # Friedman simulators + Ozone dataset (01; also used to build data/ozone.csv)
  "MASS", # Boston Housing, mvrnorm() (used to build data/housing.csv, data/boardgames.csv)
  "knitr", # kable() (02)
  "kableExtra" # kable_styling() (02)
)

# NB: the original 02_bootstrap_pca_multiple_testing.R also loaded `car`
# (almost certainly for `car::logit()`). `car` pulls in `lme4`, which on
# some machines fails to compile from source (its `nloptr` dependency
# needs a Fortran toolchain not always present with a bare Homebrew R
# install). Since `car` was used here for exactly one function, the
# script no longer depends on it: a small inline `logit()` helper
# (matching `car::logit(x, percents = TRUE, adjust = 0.025)`) is defined
# directly in 02_bootstrap_pca_multiple_testing.R instead. If you want to
# use `car` anyway: install.packages("car", type = "binary") usually
# sidesteps the compilation issue on CRAN-provided binaries.

installed <- rownames(installed.packages())
missing <- setdiff(pkgs, installed)
if (length(missing) > 0) {
  install.packages(missing, repos = "https://cloud.r-project.org")
} else {
  cat("All required packages already installed.\n")
}
