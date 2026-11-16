# Computational Statistics: Resampling Methods

> **University group coursework project** (Computational Statistics course, 2023).
> Authors: **Manuel Bottino**, **Patrick Poetto**, **Jacopo Spagliardi**.
> This repository is maintained/published by Patrick Poetto; all statistical
> work and code below were produced jointly by the three co-authors as part
> of two course deliverables.

This repository collects two related deliverables built around resampling
methods (bootstrap, permutation tests, bagging) and multiple-hypothesis
testing — implemented from scratch in R rather than relying on a single
canned package call, to make the underlying mechanics explicit.

| File | Deliverable |
|---|---|
| `01_bagging_decision_trees.R` | "Bagging on Decision Trees" — replication of Breiman (1996) |
| `02_bootstrap_pca_multiple_testing.R` | "Assignment-2" — bootstrap CIs, PCA, permutation testing, multiple-testing correction |

---

## 1. Bagging on Decision Trees

### What bagging is, and why it works

Bagging (**B**ootstrap **AGG**regat**ING**, Breiman 1996) trains many
instances of a base model on independent bootstrap samples (sampled with
replacement) of the training data, then combines their predictions by
averaging (regression) or majority vote (classification).

Writing the bagged predictor as the expectation of the base predictor over
bootstrap replicates of the learning set, `phi_B(x) = E_L[phi(x, L)]`,
Jensen's inequality (`E[Z^2] >= (E[Z])^2`) guarantees that aggregating can
only reduce — never increase — expected squared error relative to a single
fit. The size of the gain is driven by **instability**: how much the fitted
model changes under small perturbations of the training sample. Decision
trees are a textbook example of an unstable learner (small data changes can
flip which variable is split on near the root, cascading into a very
different tree), which is exactly why bagging is applied to them here. As a
contrast, a short motivating example in the script bags a simple OLS
regression (a *stable* procedure) on a bike-sharing dataset and shows the
bagged residual standard error barely differs from the single-model one —
illustrating that bagging's benefit is concentrated on unstable base
learners, not a universal free lunch.

### Method

The base model is a CART regression tree (greedy splitting on
`(variable, split point)` to minimize squared error, then cost-complexity
pruning), grown with `rpart`. Following Breiman's (1996) experimental
protocol, for each of 5 datasets:

1. Split into a learning set `L` (85–90%) and test set `T` (10–15%).
2. Grow and cross-validate a single tree on `L`; compute its test error
   `eS(L, T)`.
3. Draw 25 bootstrap samples from `L`, grow and prune a tree on each,
   average the 25 predictions (the bagged predictor); compute its test
   error `eB(L, T)`.
4. Repeat steps 1–3 over 100 random train/test splits and average, giving
   `eS_bar` and `eB_bar`, plus the relative error reduction.

### Datasets

- **Boston Housing** (UCI) — real data, median home value regression.
- **Ozone** — real data, daily ozone concentration regression.
- **Friedman #1 / #2 / #3** (`mlbench`) — three simulated nonlinear
  regression benchmarks used in Breiman's original paper.

### Results

| Dataset | eS_bar (single tree) | eB_bar (bagged, 25 trees) | Decrease |
|---|---|---|---|
| Boston Housing | 23.16 | 16.58 | 28.40% |
| Ozone | 23.11 | 18.46 | 20.14% |
| Friedman #1 | 9.64 | 6.94 | 28.05% |
| Friedman #2 | 3.02 × 10⁴ | 2.24 × 10⁴ | 25.70% |
| Friedman #3 | 0.03 | 0.02 | 32.51% |

**Breiman (1996) original results**, for comparison:

| Dataset | eS_bar | eB_bar | Decrease |
|---|---|---|---|
| Boston Housing | 20.0 | 11.6 | 42% |
| Ozone | 23.9 | 18.8 | 21% |
| Friedman #1 | 11.4 | 6.1 | 46% |
| Friedman #2 | 31100 | 22100 | 29% |
| Friedman #3 | 0.0403 | 0.0242 | 40% |

Bagging produced a consistent **20–33% reduction in mean squared error**
across all five datasets — directionally replicating Breiman's findings,
though with a smaller magnitude than the original paper. The gap is most
plausibly attributable to dataset-version differences, the stochastic
nature of the `mlbench` Friedman generators, and ~27 years of differences
in `rpart` defaults, random number generation and computing environment
relative to the 1996 study, rather than to any error in the bagging
procedure itself (the *direction and consistency* of the effect across all
five, very different, datasets is the more informative result).

---

## 2. Assignment-2: Bootstrap, PCA, Permutation Testing & Multiple Comparisons

### 2.1 Board games dataset — correlation, bootstrap covariance CIs, PCA

Working with 8 standardized numeric features of a board-games dataset
(year published, min/max players, playing time, minimum age, average
rating, total owners, average difficulty weight):

- **Correlation structure**: computed via the (scaled) sample covariance
  matrix and `cov2cor`, highlighting how playing time, minimum age and
  perceived difficulty move together.
- **Bootstrap confidence intervals for the covariance matrix** (B = 1000
  resamples), compared across three construction methods:
  1. **CLT-based normal approximation**, using the bootstrap standard
     error plugged into a Normal quantile.
  2. **Bootstrap percentile method** — distribution-free, more robust
     to skewed sampling distributions than the normal approximation.
  3. **Bias-corrected ("reflection") method** — `2 * theta_hat - percentile
     CI`, which corrects for a sampling distribution that is not centered
     on the plug-in estimate.
- **PCA via eigendecomposition**, implemented directly from
  `eigen(sample_cov)` rather than `prcomp`, to show explicitly that the
  eigenvalues partition total variance (`sum(eigenvalues) == sum(diag(cov))`)
  and to track the cumulative proportion of variance explained as
  components are added. A 76% variance-explained threshold is crossed at
  **j\* = 6** components.
- **Bootstrapping j\* itself**: re-running the eigendecomposition on each
  of the 1000 bootstrap covariance matrices gives a *distribution* over
  the number of components needed, not just a point estimate —
  `mean(j*) ≈ 6.83`, `sd(j*) ≈ 0.96`, and in ~8.5% of bootstrap replicates
  only 5 components would already have sufficed. This quantifies how much
  sampling uncertainty sits behind a dimensionality-reduction choice that
  is often reported as a single fixed number.

### 2.2 Spotify tracks dataset — regression, permutation tests, multiple testing

Working with 2072 Spotify tracks (audio features, popularity, artist):

- **Bootstrap and jackknife regression** of game/track ratings on their
  predictors (1000 bootstrap replicates and full leave-one-out jackknife),
  comparing the stability of coefficient estimates, R², and a derived
  nonlinear statistic under the two resampling schemes.
- **Permutation test for pairwise artist popularity differences**: for
  every pair of artists, the null hypothesis that their median (logit-
  transformed) popularity is equal is tested by permuting the pooled
  popularity scores between the two artist labels 1000 times and
  comparing the observed absolute median difference to this null
  distribution — an exact, assumption-light alternative to a parametric
  two-sample test. The resulting pairwise p-value matrix is also used to
  build a `1 - p_value` dissimilarity matrix for hierarchical clustering
  of artists into 5 groups with statistically indistinguishable
  popularity.
- **Multiple-testing correction**: 40 artists are each tested with a
  one-sided one-sample t-test (mean standardized popularity > 0). Testing
  40 hypotheses at a nominal `alpha = 0.10` gives roughly a **98.5%**
  chance of at least one false positive by chance alone if every null
  were true — motivating a correction for multiplicity, compared here
  across three standard procedures:

  - **Bonferroni** — rejects at `alpha / m`; controls the family-wise
    error rate (FWER) exactly, but conservatively, especially as `m`
    grows.
  - **Holm** — a step-down, uniformly more powerful FWER-controlling
    procedure: p-values are sorted and tested against increasingly
    permissive thresholds, so it rejects at least as many hypotheses as
    Bonferroni (strictly more whenever several p-values are small).
  - **Benjamini-Hochberg (BH)** — controls the false discovery rate
    (expected proportion of false discoveries among rejections) rather
    than the FWER, making it the least conservative of the three and the
    most appropriate choice for exploratory screening across many
    artists, where a small, known rate of false discoveries is an
    acceptable trade-off for substantially more power.

---

## Repository contents

```
.
├── 01_bagging_decision_trees.R           # Bagging / CART replication of Breiman (1996)
├── 02_bootstrap_pca_multiple_testing.R   # Bootstrap CIs, PCA, permutation tests, multiple testing
├── LICENSE
└── .gitignore
```

## Requirements

R (≥ 4.0) with the following packages: `dplyr`, `purrr`, `rpart`, `mlbench`,
`knitr`, `kableExtra`.

## Note on data

The underlying datasets (Boston Housing, Ozone, a Kaggle bike-sharing
dataset, a board-games dataset and a Spotify Web API export) are not
redistributed in this repository; the scripts assume they are loaded into
the objects referenced in the code (`housing`, `ozone`, `data`, `df`, `df2`,
etc.), as they were in the original course notebooks.

## License

MIT — see [LICENSE](LICENSE).
