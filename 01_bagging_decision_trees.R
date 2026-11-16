########################################################################
# Bagging on Decision Trees
#
# Computational Statistics coursework (university group project)
# Authors: Manuel Bottino, Patrick Poetto, Jacopo Spagliardi
# Original date: 2023-06-28
#
# References
#   [1] C. M. Bishop. Pattern Recognition and Machine Learning.
#       Ed. by M. Jordan. Information Science and Statistics. Springer, 2006.
#   [2] L. Breiman. "Bagging predictors". Machine Learning 24.2 (1996), pp. 123-140.
#   [3] Han, Jiawei, et al. Data Mining Concepts and Techniques.
#       Morgan Kaufmann Publishers, 2023.
#   [4] Hastie, Trevor, et al. The Elements of Statistical Learning.
#       Vol. 27. 2. 2009. Chap. 9, p. 745.
#   [5] Kuhn, Max, et al. Applied Predictive Modeling. Springer, 2016.
#   [6] Seni, Giovanni, et al. Ensemble Methods in Data Mining: Improving
#       Accuracy Through Combining Predictions. Ed. by C. Robert Grossman,
#       University of Illinois. Synthesis Lectures on Data Mining and
#       Knowledge Discovery. Morgan & Claypool Publishers, 2010.
########################################################################

library(dplyr)
library(purrr)
library(rpart)
library(mlbench)

########################################################################
# 1. Review
#
# Bagging (Bootstrap AGGregatING, Breiman 1996) creates B instances of a
# base learner, each trained on an independent bootstrap sample (sampled
# with replacement) of the training data. The B predictions are then
# combined by averaging (regression) or majority vote (classification).
#
# 1.1 Why it works
#   For a base predictor phi(x, L) trained on learning set L, the bagged
#   predictor is phi_B(x) = E_L[phi(x, L)]. By Jensen's inequality,
#   E[Z^2] >= (E[Z])^2, so aggregating over bootstrap replicates of L can
#   only reduce (or leave unchanged) the expected squared error relative
#   to a single fit -- the gain is driven by the variance of phi(x, L)
#   across replicates. The more "unstable" the base learner (the more its
#   fit changes with small perturbations of the training data), the more
#   bagging helps.
#
# 1.2 Instability
#   Decision trees, neural networks and subset-selection regression are
#   classic examples of unstable procedures (small changes in the
#   training sample can produce very different fitted models), while
#   e.g. k-NN and OLS with no variable selection are comparatively stable.
#   Bagging is expected to help the former much more than the latter.
########################################################################

########################################################################
# 2. Motivating example: bagging is not a free lunch for a STABLE learner
#
# Applied to the Kaggle "bike sharing" dataset with a simple linear
# regression (OLS), a stable procedure, to illustrate that bagging brings
# little to no improvement (and can even add noise) when the base learner
# already has low variance.
########################################################################

# `bike_data` is the bike-sharing data frame (date, weather, season, etc. -> cnt).
# [Reproducibility note: the original script named this data frame `data`,
#  which shadows base R's `data()` function; renamed to `bike_data` here.
#  The dataset itself was not bundled with the reconstructed script -- see
#  data/README.md / scripts_build_data.R: this is a real public bike-sharing
#  dataset (Fanaee-T & Gama, UCI), cleaned to drop the id/date columns and
#  the casual/registered columns (which sum to `cnt` and would otherwise
#  give the OLS a trivially perfect fit).]
bike_data <- read.csv("data/bike_sharing.csv")

ols <- lm(cnt ~ ., data = bike_data)
summary(ols)

# Train/test split (90% / 10%)
train_indices <- sample(1:nrow(bike_data), 0.9 * nrow(bike_data))
train_data <- bike_data[train_indices, ]
test_data <- bike_data[-train_indices, ]

# Bootstrap the training data B = 100 times
n_bootstraps <- 100
train_datasets <- lapply(
  1:n_bootstraps,
  function(i) train_data[sample(1:nrow(train_data), replace = TRUE), ]
)

# Fit one OLS model per bootstrap replicate
models <- map(train_datasets, function(data) lm(cnt ~ ., data = data))

# Extract coefficients (for inspection of their variability across replicates)
coefficients <- lapply(models, coef)

# Bagged predictions: average the predictions of the 100 bootstrap models
predictions <- list()
for (i in 1:length(models)) {
  predictions[[i]] <- predict(models[[i]], newdata = test_data)
}
predictions_bag <- rowMeans(do.call(cbind, predictions))

# Single-model (non-bagged) OLS predictions, for comparison
predictions_ols <- predict(ols, newdata = test_data)

# Compare residual standard errors
residuals_bag <- test_data$cnt - predictions_bag
residuals_ols <- test_data$cnt - predictions_ols

std_error_ols <- sqrt(sum(residuals_ols^2) / (length(residuals_ols) - length(ols$coefficients)))
std_error_bag <- sqrt(sum(residuals_bag^2) / (length(residuals_bag) - length(ols$coefficients)))
# -> std_error_bag ~= std_error_ols: bagging an already-stable linear model
#    does not meaningfully reduce error, consistent with the theory above.

########################################################################
# 3. Model: CART regression trees
#
# The base learner used for the main experiment is a CART regression
# tree (Breiman et al., 1984), grown greedily: at each node, choose the
# splitting variable j and split point s that minimize the sum of squared
# residuals in the two resulting half-planes, recursively, until a
# stopping rule is hit. The tree is then pruned back via cost-complexity
# pruning, with the pruning parameter (tree size) chosen by k-fold cross
# validation. In R this is implemented by `rpart::rpart` / `rpart::prune`.
# (A simple illustrative tree can be grown on e.g. a Spotify features
# dataset with `rpart(formula, data = spotify)`.)
########################################################################

########################################################################
# 4. Algorithm setup -- bagging regression trees (Breiman, 1996)
#
# For each of 100 repetitions:
#   1. Randomly split the data into a 10% test set T and a 90% learning
#      set L (for the simulated Friedman datasets, an 80/20 or 85/15
#      split is used instead, since simulated data is regenerated afresh
#      for each repetition and a larger test share is affordable).
#   2. Grow a single regression tree on L (choosing tree size via 10-fold
#      cross-validation) and compute its squared prediction error on T:
#      eS(L, T).
#   3. Draw 25 bootstrap samples LB from L; for each, grow and prune a
#      tree, producing predictors phi_1, ..., phi_25.
#   4. The bagged predictor is the average of the 25 trees; compute its
#      squared prediction error on T: eB(L, T).
#   5. Average eS and eB over the 100 repetitions to obtain eS_bar and
#      eB_bar, and report the relative decrease in error.
########################################################################

set.seed(655)

# Helper 1: random train/test split.
#   bin_80_20 = 0 -> ~90/10 split (one of 10 folds held out as test)
#   bin_80_20 = 1 -> ~80/20 split (two of 10 folds held out as test)
random_training_test <- function(df, k, bin_80_20 = 0) {
  fold_indices <- sample(cut(seq(1, nrow(df)), breaks = k, labels = FALSE, ordered_result = FALSE))
  if (bin_80_20 == 1) {
    test_indices <- which(fold_indices == 1 | fold_indices == 2)
  } else {
    test_indices <- which(fold_indices == 1)
  }
  train_i <- df[-test_indices, ]
  test_i <- df[test_indices, ]
  return(list(train_i, test_i))
}

# Helper 2: single regression tree, tree size chosen by k-fold CV on the
# learning set, evaluated on the held-out test set -> eS(L, T).
k_fold_reg_tree <- function(df, k, y) {
  formula <- as.formula(paste(y, "~", "."))
  train_i <- df[[1]]
  test_i <- df[[2]]
  fold_indices <- sample(cut(seq(1, nrow(train_i)), breaks = k, labels = FALSE, ordered_result = FALSE))
  cv_result <- Inf
  for (i in 1:k) {
    test_indices <- which(fold_indices == i)
    train_data <- train_i[-test_indices, ]
    test_data <- train_i[test_indices, ]
    model <- rpart(formula, data = train_data)
    predictions <- predict(model, newdata = test_data)
    result <- mean((predictions - test_data[[y]])^2)
    if (result < cv_result) {
      cv_result <- result
      predictions <- predict(model, newdata = test_i)
      final_res <- mean((predictions - test_i[[y]])^2)
    }
  }
  return(final_res)
}

# Helper 3: bagged regression tree (B = 25 bootstrap trees) -> eB(L, T).
reg_tree_boot <- function(df, b, y) {
  formula <- as.formula(paste(y, "~", "."))
  train_i <- df[[1]]
  test_i <- df[[2]]
  predictions <- rep(0, nrow(test_i))
  for (i in 1:b) {
    train_data <- slice_sample(train_i, n = nrow(train_i), replace = TRUE)
    model <- rpart(formula, data = train_data)
    # [Code-quality note: `rpart::prune()`'s signature is `prune(tree, cp,
    #  ...)` -- it has no `newdata` argument, so `newdata = train_i` below
    #  is silently swallowed by `...` and has no effect (not an error, but
    #  not doing anything either; harmless here since prune only needs
    #  `tree`/`cp`, but likely a leftover from an earlier draft). Left as
    #  in the extracted source since removing it does not change results.]
    pruned_tree <- rpart::prune(model, cp = 0.01, newdata = train_i)
    predictions <- predictions + predict(pruned_tree, newdata = test_i)
  }
  predictions <- predictions / b
  final_res <- mean((predictions - test_i[[y]])^2)
  return(final_res)
}

########################################################################
# 5. Data: Boston Housing, Ozone, and 3 simulated Friedman benchmarks
########################################################################

# Boston Housing (UCI).
# [Reproducibility note: housing.csv was not bundled with the original
#  script. data/housing.csv is MASS::Boston written out in the classic
#  UCI column order (see scripts_build_data.R) -- real data, not synthetic.]
housing <- read.table("data/housing.csv", quote = "\"", comment.char = "")
colnames(housing) <- c(
  "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS",
  "RAD", "TAX", "PTRATIO", "B", "LSTAT", "y"
)

# Ozone (response column renamed to "y" for the generic formula helper).
# [Reproducibility note: the `ozone` object was not bundled with the
#  original script either. data/ozone.csv is mlbench::Ozone with the
#  response (V4, daily max one-hour ozone) already in column 1 and
#  restricted to complete cases (see scripts_build_data.R) -- real data.]
ozone <- read.csv("data/ozone.csv")
colnames(ozone)[1] <- "y"

# Simulated Friedman benchmark datasets (mlbench), B = 1000 draws each,
# n = 1200 observations per draw, matching Breiman's (1996) simulation
# setup for the 3 artificial regression problems.
B <- 1000

friedman1 <- t(replicate(B, {
  fried1 <- mlbench.friedman1(n = 1200, sd = 1)
  data.frame(cbind(fried1$x, fried1$y))
}))
colnames(friedman1) <- c("x_1", "x_2", "x_3", "x_4", "x_5", "x_6", "x_7", "x_8", "x_9", "x_10", "y")

friedman2 <- t(replicate(B, {
  fried2 <- mlbench.friedman2(n = 1200)
  data.frame(cbind(fried2$x, fried2$y))
}))
colnames(friedman2) <- c("x_1", "x_2", "x_3", "x_4", "y")

friedman3 <- t(replicate(B, {
  fried3 <- mlbench.friedman3(n = 1200)
  data.frame(cbind(fried3$x, fried3$y))
}))
colnames(friedman3) <- c("x_1", "x_2", "x_3", "x_4", "y")

datasets <- list(housing, ozone, friedman1, friedman2, friedman3)

########################################################################
# 6. Run the experiment: 100 repetitions per dataset, average eS and eB
########################################################################

e_bar_global <- lapply(datasets, function(data) {
  colMeans(t(replicate(B, {
    if (ncol(data) == 5 | ncol(data) == 11) {
      # Simulated Friedman data: draw one of the B pre-generated samples
      # and use an 80/20 split (test set is drawn fresh each repetition).
      i <- sample(1:B, 1)
      df <- random_training_test(data.frame(data[i, ]), 10, 1)
    } else {
      # Boston Housing / Ozone: fixed dataset, 90/10 split.
      df <- random_training_test(data, 10)
    }
    mse_cv <- k_fold_reg_tree(df, 10, "y")
    mse_bag <- reg_tree_boot(df, 25, "y")
    c(mse_cv, mse_bag)
  })))
})

# Add the relative % decrease in error (eS_bar - eB_bar) / eS_bar * 100
e_bar_global <- lapply(e_bar_global, function(data) {
  data[3] <- ((data[1] - data[2]) / data[1]) * 100
  round(data, 2)
})

########################################################################
# 7. Results
#
# | Dataset        | eS_bar     | eB_bar     | Decrease |
# |----------------|------------|------------|----------|
# | Boston Housing | 23.16      | 16.58      | 28.40%   |
# | Ozone          | 23.11      | 18.46      | 20.14%   |
# | Friedman #1    | 9.64       | 6.94       | 28.05%   |
# | Friedman #2    | 3.017648e4 | 2.241992e4 | 25.70%   |
# | Friedman #3    | 0.03       | 0.02       | 32.51%   |
#
# Breiman (1996), original results, for comparison:
#
# | Dataset        | eS_bar | eB_bar | Decrease |
# |----------------|--------|--------|----------|
# | Boston Housing | 20.0   | 11.6   | 42%      |
# | Ozone          | 23.9   | 18.8   | 21%      |
# | Friedman #1    | 11.4   | 6.1    | 46%      |
# | Friedman #2    | 31100  | 22100  | 29%      |
# | Friedman #3    | 0.0403 | 0.0242 | 40%      |
########################################################################

########################################################################
# 8. Conclusion
#
# Bagging reduces the variance of an unstable base model by averaging
# predictions from multiple bootstrap-trained instances (Jensen's
# inequality guarantees this cannot increase squared error in
# expectation). Applied here to CART regression trees following
# Breiman's (1996) procedure -- split train/test, grow & prune a single
# tree via cross-validation, bootstrap-aggregate 25 trees, repeat over
# 100 train/test splits and average -- bagging produced a consistent
# 20-33% reduction in mean squared error across all 5 benchmark
# datasets (Boston Housing, Ozone, Friedman #1/#2/#3), in the same
# direction as Breiman's original findings, though with smaller
# magnitude. The gap to the original numbers is attributable to dataset
# version differences, the stochastic nature of the simulated Friedman
# generators, and 27+ years of differences in computing environment,
# default `rpart` settings and random number generation relative to the
# 1996 study.
########################################################################
