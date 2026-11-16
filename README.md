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

> **Reconstruction note.** Both scripts were reconstructed from OCR'd text
> extracted from the original PDF course reports — the `.R` files
> themselves were not part of what survived. Neither of the original
> coursework datasets was bundled with that extracted text either, and
> some library calls / helper definitions were not captured in the OCR.
> This pass verified the reconstruction by actually **running both
> scripts end-to-end with Rscript**, fixed what broke, and added a
> `data/` folder plus a `scripts_build_data.R` generator so anyone can
> reproduce every number below from scratch. See
> [**How to run**](#how-to-run) and [**Data**](#data) for exactly what
> changed and why.

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

**Verified during this review**: re-running the (fixed, now-reproducible)
script end-to-end with `B = 200` against the `MASS::Boston` /
`mlbench::Ozone` stand-ins in `data/` (see [Data](#data)) gives:

| Dataset | eS_bar | eB_bar | Decrease |
|---|---|---|---|
| Boston Housing | 22.91 | 16.37 | 28.54% |
| Ozone | 26.08 | 20.74 | 20.46% |
| Friedman #1 | 9.64 | 6.96 | 27.76% |
| Friedman #2 | 2.97 × 10⁴ | 2.23 × 10⁴ | 24.83% |
| Friedman #3 | 0.03 | 0.02 | 31.88% |

— essentially the same story (the Friedman numbers are near-identical,
since `mlbench`'s simulators are literally the same generators either
way; Boston Housing and Ozone differ by a few percentage points, as
expected from real-but-not-bit-identical data and a different `B`/seed).
This is strong evidence the bagging/CART pipeline itself is correct, not
just that it runs without erroring.

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

> **Mislabeling caught during this review.** The original report
> introduces the bootstrap/jackknife regression below under the "PART 2
> — SPOTIFY" heading, but its formula (`average_rating ~ .`) and its own
> reported "significant predictors" — `yearpublished`, `playingtime`,
> `minage`, `total_owners`, `average_weight` — are board-games column
> names, not Spotify audio features (there is no `average_rating` in the
> Spotify data described anywhere else in the report). The code's
> variable reuse (`df` is silently reassigned) makes this easy to read
> past. `02_bootstrap_pca_multiple_testing.R` now has this section
> explicitly re-use the **raw, unscaled board-games data** (consistent
> with what the formula and output actually are), and the Spotify data
> (`df2`) is used only where the code references `first_auth`/`pop`
> (the permutation tests and multiple-testing correction, 2c–2g below).

- **Bootstrap and jackknife regression** of board-game ratings on their
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
  popularity. *(The original report only described this clustering step
  in prose — the clustering call itself was not present in the extracted
  text. This repository now includes a direct, clearly-labelled
  implementation — average-linkage `hclust` + `cutree(k = 5)` on the
  `1 - p_value` distance matrix — added during this review rather than
  recovered from the original source.)*
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
├── scripts_build_data.R                  # Builds everything in data/ (run once)
├── requirements.R                        # Installs required R packages
├── data/                                 # Datasets used by the two scripts (see below)
│   ├── housing.csv                       # REAL  — MASS::Boston, UCI column order
│   ├── ozone.csv                         # REAL  — mlbench::Ozone, complete cases, response in col 1
│   ├── bike_sharing.csv                  # REAL  — public UCI bike-sharing daily counts
│   ├── boardgames.csv                    # SYNTHETIC — stand-in, see caveat below
│   └── spotify.csv                       # SYNTHETIC — stand-in, see caveat below
├── LICENSE
└── .gitignore
```

## How to run

Both scripts were verified by actually executing them end-to-end with
`Rscript` (R 4.5.2, macOS/Homebrew) as part of this review; here is exactly
what that takes from a clean checkout:

```bash
# 1. Install required packages (one-time)
Rscript requirements.R

# 2. Build the datasets both scripts read from data/ (one-time)
Rscript scripts_build_data.R

# 3. Run either script from the repository root (both use relative
#    "data/..." paths, so the working directory matters)
Rscript 01_bagging_decision_trees.R
Rscript 02_bootstrap_pca_multiple_testing.R
```

`01_bagging_decision_trees.R` with the original `B = 1000` outer
repetitions across all 5 datasets takes roughly 10–20 minutes on a laptop
(it is doing on the order of ~175,000 `rpart` fits); reduce `B` near the
top of section 6 for a quick smoke test. `02_bootstrap_pca_multiple_testing.R`
runs in well under a minute.

## Requirements

R (≥ 4.0) with: `dplyr`, `purrr`, `rpart` (ships with base R), `mlbench`,
`MASS` (ships with base R), `knitr`, `kableExtra`. See `requirements.R`
(installs anything missing) and the note there about why this version no
longer depends on `car` (only ever used for one function, `logit()`, which
is now a small inline helper — `car`'s `lme4` dependency fails to compile
from source on a bare toolchain).

## Data

**None of the original coursework datasets were bundled with the
reconstructed scripts** — the `.R` files were recovered from OCR'd PDF
text, not from the original project folder, and the datasets were never
part of that text. This review prioritized getting both scripts to
actually run over reproducing the original numbers exactly, per the
following approach (full generation code in `scripts_build_data.R`):

| File | Status | Source |
|---|---|---|
| `data/housing.csv` | **Real data** | `MASS::Boston`, written out in the classic UCI column order (`CRIM`…`LSTAT`, response last) so the original `read.table` + `colnames()` call in the script works unmodified. |
| `data/ozone.csv` | **Real data** | `mlbench::Ozone`, response column (`V4`, daily max one-hour ozone) moved to column 1, restricted to complete cases (330→203 rows after `na.omit`), matching the classic Breiman/Hastie-style treatment of this dataset. |
| `data/bike_sharing.csv` | **Real data** | Public UCI bike-sharing daily-counts dataset (Fanaee-T & Gama), used only in script 1's short "bagging a stable OLS" motivating example; `casual`/`registered` (which sum to `cnt`) and the date/id columns were dropped to avoid a trivially perfect / rank-deficient fit. |
| `data/boardgames.csv` | **Synthetic stand-in** | The original board-games export (BoardGameGeek-style: `yearpublished`, `minplayers`, `maxplayers`, `playingtime`, `minage`, `average_rating`, `total_owners`, `average_weight`) is not public/available, so 6000 synthetic rows were generated from a designed multivariate-normal correlation structure chosen to reproduce the *qualitative* pattern the original report describes (difficulty correlating with playtime and minimum age, etc.). **This is not the real coursework data** — treat the correlation/PCA/regression numbers in this script's output as illustrative of the method, not as the original report's actual findings. |
| `data/spotify.csv` | **Synthetic stand-in** | Likewise not public; 2072 synthetic tracks across 40 synthetic artists with a popularity score, built only so the permutation-test and multiple-testing-correction machinery (which needs `first_auth` and `pop`) runs end-to-end. **Not the real Spotify data.** |

If you have access to the original datasets, drop them into `data/` under
the same filenames/column names and both scripts will use them unchanged
(remove or ignore `scripts_build_data.R` in that case).

## What this review changed

Beyond the data/packaging work above, running both scripts surfaced and
fixed a handful of real issues in the reconstructed code (see inline
`[Reproducibility note]` / `[Bug fix]` comments in the scripts for exact
locations):

- **01**: the bike-sharing motivating example referenced an undefined
  `data` frame (also a poor name, shadowing base R's `data()`); it is now
  loaded from `data/bike_sharing.csv` as `bike_data`. `housing` and `ozone`
  were likewise undefined; both now load from `data/`.
- **02**: `car::logit()` was called but `car` was never loaded, and failed
  to install on a bare toolchain regardless (see Requirements); replaced
  with a small inline `logit()` matching `car`'s default behaviour.
  `df`/`df2` (board games / Spotify) were undefined; both now load from
  `data/`. The "PART 2 — SPOTIFY" regression section was found to
  actually operate on board-games columns (see §2.2 above) and was
  corrected to use the board-games data explicitly rather than silently.
  The hierarchical-clustering step, previously undocumented beyond prose,
  now has a real implementation. A floating-point `==` sanity check
  (`sum(eigenvalues) == trace(cov)`) that reliably evaluates to `FALSE`
  due to floating-point rounding was replaced with `all.equal`-based
  comparison.

## License

MIT — see [LICENSE](LICENSE).
