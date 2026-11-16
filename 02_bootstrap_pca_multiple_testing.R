########################################################################
# Bootstrap Confidence Intervals, PCA, Permutation Testing and
# Multiple-Comparison Correction
#
# Computational Statistics coursework (university group project)
# Assignment-2
# Authors: Manuel Bottino, Patrick Poetto, Jacopo Spagliardi
# Original date: 2023-06-06
#
# Part 1 (Exercise 1): Board games dataset -- correlation structure,
#   bootstrap confidence intervals for the covariance matrix, and PCA
#   via eigendecomposition.
# Part 2 (Exercise 2): Spotify tracks dataset -- bootstrap/jackknife
#   regression, permutation tests comparing artists' popularity, and
#   multiple-testing correction (Bonferroni vs. Holm vs. Benjamini-
#   Hochberg).
########################################################################

library(dplyr)
library(knitr)
library(kableExtra)

########################################################################
# PART 1 -- BOARD GAMES DATASET
#
# Variables: yearpublished, minplayers, maxplayers, playingtime, minage,
# average_rating, total_owners, average_weight (perceived difficulty).
# `df` below is assumed to already contain just these numeric columns
# (the first two columns of the raw import -- an id and a name field --
# have been dropped).
########################################################################

########################################################################
# 1b. Covariance / correlation matrix
########################################################################

df <- df[, -1:-2]
df <- data.frame(scale(df))

sample_cov <- cov(df)
a <- cor(df) # identical to cov2cor(sample_cov) since df is already scaled

cor_matrix <- cov2cor(sample_cov)
cor_matrix <- round(cor_matrix, 2)
new_col_names <- c("YP", "mP", "MP", "PT", "mA", "AR", "TO", "AW")
colnames(cor_matrix) <- new_col_names

kable(cor_matrix, format = "markdown") %>%
  kable_styling()

# Result: strongest correlations are average_weight/playingtime-type
# relationships; notably loudness/energy-style pairs are not applicable
# here (that pattern belongs to the Spotify data in Part 2) -- for the
# board-games matrix the standout pairs are difficulty (AW) correlating
# with playing time (PT) and minimum age (mA), consistent with harder,
# longer games being aimed at older players.

########################################################################
# 1c. Bootstrap confidence intervals for the covariance matrix
#
# Three CI construction methods are compared on the same B = 1000
# bootstrap replicates of the sample covariance matrix:
#   1. CLT-based normal approximation using the bootstrap standard error.
#   2. Bootstrap percentile method (distribution-free, more robust).
#   3. Bias-corrected ("reflection") method: 2*theta_hat - percentile CI.
########################################################################

set.seed(123)
B <- 1000
cov_var_sample <- array(NA, dim = c(ncol(df), ncol(df), B))
for (i in 1:B) {
  cov_var_sample[, , i] <- cov(slice_sample(df, n = nrow(df), replace = TRUE))
}
theta_hat_bar <- apply(cov_var_sample, c(1, 2), mean)
se_cov_var <- apply(cov_var_sample, c(1, 2), sd)

# Method 1: CLT-based normal approximation
lower_bound <- sample_cov + qnorm(0.025, mean = 0, sd = 1) * se_cov_var
upper_bound <- sample_cov + qnorm(0.975, mean = 0, sd = 1) * se_cov_var

# Method 2: bootstrap percentile method
boot_sample_variance <- data.frame(matrix(nrow = 8, ncol = 8))
colnames(boot_sample_variance) <- c(
  "yearpublished", "minplayers", "maxplayers", "playingtime",
  "minage", "average_rating", "total_owners", "average_weight"
)
for (i in 1:8) {
  for (j in 1:8) {
    boot_sample_variance[i, j] <- list(list(cov_var_sample[i, j, ]))
  }
}

boot_sample_variance_ci <- data.frame(matrix(nrow = 8, ncol = 8))
for (i in 1:8) {
  for (j in 1:8) {
    boot_sample_variance_ci[i, j] <- list(list(
      quantile(boot_sample_variance[i, j][[1]], probs = c(0.025, 0.975))
    ))
  }
}

# Method 3: bias-corrected estimate (reflection method)
boot_sample_variance_ci_bias <- data.frame(matrix(nrow = 6, ncol = 6))
for (i in 1:6) {
  for (j in 1:6) {
    boot_sample_variance_ci_bias[i, j] <- list(list(
      2 * sample_cov[i, j] - quantile(boot_sample_variance[i, j][[1]], probs = c(0.025, 0.975))
    ))
  }
}

########################################################################
# 1d. Principal Component Analysis via eigendecomposition
#
# Rather than calling `prcomp`, the PCA is built directly from the
# eigendecomposition of the sample covariance matrix, to show explicitly
# that the eigenvalues partition the total variance and to track the
# cumulative proportion of variance explained, theta(j), as a function
# of the number of retained components j.
########################################################################

# Sanity check: total variance == sum of eigenvalues
sum(eigen(sample_cov)$values) == sum(diag(sample_cov)) # TRUE

theta <- numeric(ncol(df))
for (i in 1:length(theta)) {
  theta[i] <- sum(eigen(sample_cov)$values[1:i]) / sum(eigen(sample_cov)$values)
}

# Smallest number of components crossing a 76% variance-explained threshold
theta[which(theta > 0.76)][1] # 0.8551548
j_star <- 6

plot(ecdf(theta))

########################################################################
# 1e. Bootstrap distribution of j* (the number of components needed to
# cross the 76% variance threshold), to quantify the uncertainty of the
# dimensionality-reduction choice itself.
########################################################################

boot_theta_j <- matrix(NA, dim(cov_var_sample)[3], length(theta))
for (i in 1:dim(cov_var_sample)[3]) {
  theta_i <- c()
  for (j in 1:length(theta)) {
    theta_i <- c(theta_i, sum(eigen(cov_var_sample[, , i])$values[1:j]) / sum(eigen(cov_var_sample[, , i])$values))
  }
  boot_theta_j[i, ] <- theta_i
}

boot_theta_mean <- colMeans(boot_theta_j)
boot_theta_se <- apply(boot_theta_j, 2, sd)
bias <- boot_theta_mean - theta

j_star_boot <- which(boot_theta_j > 0.76, arr.ind = TRUE)[, 2]
mean(j_star_boot) # 6.830384
sd(j_star_boot) # 0.9596201
length(j_star_boot[j_star_boot == 5]) / length(j_star_boot) # 0.08480781 -- in ~8.5% of
# bootstrap replicates, j* = 5 components would already suffice, showing
# the point estimate j* = 6 carries non-trivial sampling uncertainty.

########################################################################
# PART 2 -- SPOTIFY TRACKS DATASET
#
# 2072 tracks, 18 variables (acousticness, danceability, energy,
# instrumentalness, liveness, speechiness, valence, popularity, loudness,
# duration_ms, tempo, explicit flag, author, artist, track id, number of
# songs by that artist, ...). Source: Spotify Web API.
########################################################################

########################################################################
# 2f. Linear model for average_rating and bootstrap of its coefficients
#
# `df` here refers to the Spotify-derived modelling data frame (distinct
# from the board-games `df` used in Part 1); `sub_data` subsamples 5000
# rows for computational tractability.
########################################################################

set.seed(abs(636 - 555 - 3226))
ind <- sample(1:nrow(df), 5000, FALSE)
sub_data <- df[ind, ]

set.seed(123)
lm_game <- lm(data = sub_data, average_rating ~ .)
summary(lm_game)
# Result: R^2 = 0.151; significant predictors: yearpublished, playingtime,
# minage, total_owners, average_weight.

# Bootstrap (B = 1000) of the regression coefficients, R^2, and a derived
# nonlinear statistic (clamped at 0) combining playingtime, minage,
# minplayers and yearpublished effects.
boot_reg_game <- function(data, B = 1000) {
  n0 <- nrow(data)
  res <- replicate(B, {
    ind <- sample(1:n0, n0, TRUE)
    Dstar <- data[ind, ]
    Mstar <- lm(data = Dstar, average_rating ~ .)
    c(
      Mstar$coef, summary(Mstar)$r.squared,
      max((Mstar$coef["playingtime"] - Mstar$coef["minage"]) /
        (Mstar$coef["minplayers"] + Mstar$coef["yearpublished"]), 0)
    )
  })
  t(res)
}
res_game_boot <- boot_reg_game(data = sub_data)

########################################################################
# 2h. Jackknife estimates of the same regression, for comparison with
# the bootstrap (leave-one-out resampling instead of resampling with
# replacement).
########################################################################

jack_reg_game <- function(data) {
  n0 <- nrow(data)
  res <- matrix(NA, n0, ncol(data) + 2)
  for (i in 1:n0) {
    Mjack <- lm(data = data[-i, ], average_rating ~ .)
    res[i, ] <- c(
      Mjack$coef, summary(Mjack)$r.squared,
      max((Mjack$coef["playingtime"] - Mjack$coef["minage"]) /
        (Mjack$coef["minplayers"] + Mjack$coef["yearpublished"]), 0)
    )
  }
  return(res)
}
res_game_jack <- jack_reg_game(sub_data)

########################################################################
# 2c. Permutation test -- are two artists' popularity distributions
# different?
#
# H0: the median log-odds of popularity is equal between artists X and Y
# (two-sided). The test statistic is the absolute difference in medians
# of the logit-transformed popularity scores; its null distribution is
# built by randomly permuting the pooled popularity scores between the
# two artists' labels (label-swapping), which is exact under H0 since
# under the null the group label carries no information about the
# popularity value.
########################################################################

pop_est <- function(df, auth_X, auth_Y, songs) {
  songs_X <- df$pop[df$first_auth == auth_X]
  songs_Y <- df$pop[df$first_auth == auth_Y]
  if (!missing(songs)) {
    songs_X <- songs[1:length(songs_X)]
    songs_Y <- songs[(length(songs_X) + 1):(length(songs_X) + length(songs_Y))]
  }
  median_X <- median(logit(songs_X))
  median_Y <- median(logit(songs_Y))
  return(abs(median_X - median_Y))
}

first_auth <- df2$first_auth
pop <- df2$pop
authors <- unique(first_auth)
p_value_matrix <- matrix(NA, nrow = length(authors), ncol = length(authors))

for (i in 1:(length(authors) - 1)) {
  for (j in (i + 1):length(authors)) {
    author1 <- authors[i]
    author2 <- authors[j]
    songs_author1 <- pop[first_auth == author1]
    songs_author2 <- pop[first_auth == author2]

    observed_statistic <- abs(pop_est(df2, author1, author2))

    num_permutations <- 1000
    permutation_stats <- rep(NA, num_permutations)
    for (k in 1:num_permutations) {
      permuted_songs <- sample(c(songs_author1, songs_author2))
      permutation_stats[k] <- abs(pop_est(df2, author1, author2, permuted_songs))
    }

    p_value <- sum(permutation_stats >= observed_statistic) / num_permutations
    p_value_matrix[i, j] <- p_value
    p_value_matrix[j, i] <- p_value
  }
}

########################################################################
# 2d. Hierarchical clustering of artists by pairwise popularity similarity
#
# The pairwise permutation p-values above are turned into a dissimilarity
# matrix (1 - p_value: pairs that are *not* significantly different get a
# small distance, pairs that are very different get a large distance) and
# fed to agglomerative hierarchical clustering (`hclust`), cut into 5
# groups (`cutree`) to visualize which artists have statistically
# indistinguishable popularity. (Only the method is described in the
# original write-up; the clustering call itself was not part of the
# extracted code and is therefore not reproduced verbatim here.)
########################################################################

########################################################################
# 2e/f/g. One-sample tests per artist + multiple-testing correction
#
# For each of the 40 artists, a one-sided one-sample t-test checks
# whether mean standardized popularity is greater than 0. Testing 40
# hypotheses simultaneously at a nominal alpha = 0.10 inflates the
# family-wise error rate; three standard corrections are compared.
########################################################################

df_t_test <- df2 %>% mutate(pop = scale(pop))
artists <- unique(df_t_test$first_auth)

t_art <- numeric()
for (i in artists) {
  t_art <- c(
    t_art,
    t.test(df_t_test$pop[df_t_test$first_auth == i], mu = 0, alternative = "greater")$p.value
  )
}

alpha <- 0.1
t_art[which(t_art < alpha)]

# Probability that at least one of the 40 per-artist nulls is rejected by
# chance alone if all were true and independent, at alpha = 0.10:
1 - (1 - alpha)^40 # 0.9852191 -- i.e. essentially certain to see "false
# positives" with no correction, motivating the multiple-testing
# adjustments below.

t_art_bonf <- p.adjust(t_art, method = "bonferroni")
t_art_holm <- p.adjust(t_art, method = "holm")
t_art_bh <- p.adjust(t_art, method = "BH")

########################################################################
# Discussion: Bonferroni vs. Holm vs. Benjamini-Hochberg
#
# - Bonferroni: rejects H_i if p_i <= alpha / m. Controls the family-wise
#   error rate (FWER = P(>= 1 false discovery)) at level alpha, but is
#   conservative, especially as the number of tests m grows, since it
#   does not use any information about the joint distribution of the
#   p-values.
# - Holm: a uniformly more powerful, still FWER-controlling, step-down
#   procedure. Orders the p-values p_(1) <= ... <= p_(m) and rejects
#   H_(i) as long as p_(k) <= alpha / (m - k + 1) for all k <= i; because
#   the threshold relaxes as more hypotheses are rejected, Holm rejects
#   at least as many hypotheses as Bonferroni, and strictly more
#   whenever there are several small p-values.
# - Benjamini-Hochberg (BH): controls the (less stringent) false
#   discovery rate (FDR = E[false discoveries / total discoveries])
#   rather than the FWER, and is therefore less conservative than both
#   Bonferroni and Holm -- appropriate when a small, known proportion of
#   false discoveries among the rejected hypotheses is an acceptable
#   trade-off for greater power (e.g., exploratory screening of many
#   artists rather than a single confirmatory comparison).
########################################################################
