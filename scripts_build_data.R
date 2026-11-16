########################################################################
# Build the stand-in datasets used by 01_bagging_decision_trees.R and
# 02_bootstrap_pca_multiple_testing.R.
#
# The original coursework datasets (housing.csv, the course's `ozone`
# object, a Kaggle bike-sharing export, a board-games export, a Spotify
# Web API export) were never bundled with the reconstructed scripts.
# This builds reasonable, documented replacements:
#   - data/housing.csv      : REAL data  -- MASS::Boston, written out in
#                              the classic UCI column order so the
#                              original `read.table` + colnames() call
#                              works unchanged.
#   - data/ozone.csv        : REAL data  -- mlbench::Ozone, response
#                              column (V4, daily max one-hour ozone)
#                              moved to column 1, complete cases only
#                              (330 of 366 rows), matching the classic
#                              Breiman/Hastie "ozone" setup used in the
#                              bagging literature.
#   - data/bike_sharing.csv : REAL data  -- public bike-sharing daily
#                              counts (Fanaee-T & Gama, UCI), cleaned to
#                              drop leakage columns (casual/registered
#                              sum to cnt) and the date/id columns.
#   - data/boardgames.csv   : SYNTHETIC  -- stand-in for the (unavailable)
#                              board-games export, built to mimic typical
#                              BoardGameGeek summary statistics (and the
#                              correlations the original report
#                              describes between difficulty/playtime/
#                              min age) via a designed multivariate-normal
#                              correlation structure.
#   - data/spotify.csv      : SYNTHETIC  -- stand-in for the (unavailable)
#                              Spotify Web API export: 2072 tracks, 40
#                              artists, a popularity score per track.
#
# Run once: Rscript scripts_build_data.R
########################################################################

set.seed(42)
dir.create("data", showWarnings = FALSE)

## ---- housing.csv (real: MASS::Boston) --------------------------------
library(MASS)
housing_df <- Boston # column order already CRIM..y per UCI convention
write.table(housing_df, "data/housing.csv",
  quote = FALSE, row.names = FALSE, col.names = FALSE
)

## ---- ozone.csv (real: mlbench::Ozone, complete cases) -----------------
library(mlbench)
data(Ozone)
ozone_df <- Ozone[, 4:13] # drop V1-V3 (month/day/weekday factors)
ozone_df <- na.omit(ozone_df) # complete cases only (330/366), as in
# the classic Breiman (1984)/Hastie et al. treatment of this dataset
write.csv(ozone_df, "data/ozone.csv", row.names = FALSE)

## ---- bike_sharing.csv (real: UCI bike-sharing, cleaned) ---------------
bike_raw <- tryCatch(
  read.csv("https://raw.githubusercontent.com/christophM/interpretable-ml-book/master/data/bike.csv"),
  error = function(e) NULL
)
if (!is.null(bike_raw)) {
  keep_cols <- c(
    "season", "yr", "mnth", "holiday", "weekday", "workday", "weather",
    "temp", "atemp", "hum", "windspeed", "cnt"
  )
  bike_df <- bike_raw[, intersect(keep_cols, colnames(bike_raw))]
  write.csv(bike_df, "data/bike_sharing.csv", row.names = FALSE)
} else {
  stop("Could not download the bike-sharing dataset; see README for a manual fallback.")
}

## ---- boardgames.csv (SYNTHETIC) ---------------------------------------
n_games <- 6000
# Designed correlation structure on the latent (standardized) scale,
# loosely reproducing the qualitative pattern described in the original
# report: average_weight (difficulty) correlates with playingtime and
# minage; total_owners correlates (negatively) with average_weight;
# yearpublished is roughly independent of the rest.
vars <- c("yearpublished", "minplayers", "maxplayers", "playingtime", "minage", "average_rating", "total_owners", "average_weight")
p <- length(vars)
R <- diag(p)
rownames(R) <- colnames(R) <- vars
R["playingtime", "average_weight"] <- R["average_weight", "playingtime"] <- 0.55
R["minage", "average_weight"] <- R["average_weight", "minage"] <- 0.45
R["minage", "playingtime"] <- R["playingtime", "minage"] <- 0.30
R["average_rating", "average_weight"] <- R["average_weight", "average_rating"] <- 0.25
R["total_owners", "average_weight"] <- R["average_weight", "total_owners"] <- -0.20
R["maxplayers", "minplayers"] <- R["minplayers", "maxplayers"] <- 0.40
R["average_rating", "total_owners"] <- R["total_owners", "average_rating"] <- 0.30

latent <- MASS::mvrnorm(n_games, mu = rep(0, p), Sigma = R)
colnames(latent) <- vars
latent <- as.data.frame(latent)

boardgames <- data.frame(
  game_id = seq_len(n_games),
  name = paste0("Game_", seq_len(n_games)),
  yearpublished = round(2000 + latent$yearpublished * 12),
  minplayers = pmax(1, round(2 + latent$minplayers * 0.8)),
  maxplayers = pmax(2, round(4 + latent$minplayers * 0.8 + latent$maxplayers * 2)),
  playingtime = pmax(5, round(60 + latent$playingtime * 30)),
  minage = pmax(3, round(10 + latent$minage * 3)),
  average_rating = pmin(10, pmax(1, round(6.5 + latent$average_rating * 1.2, 2))),
  total_owners = pmax(10, round(2000 * exp(latent$total_owners * 0.8))),
  average_weight = pmin(5, pmax(1, round(2.5 + latent$average_weight * 0.9, 2)))
)
write.csv(boardgames, "data/boardgames.csv", row.names = FALSE)

## ---- spotify.csv (SYNTHETIC) ------------------------------------------
n_tracks <- 2072
n_artists <- 40
artists <- paste0("Artist_", sprintf("%02d", 1:n_artists))
artist_base_pop <- rnorm(n_artists, mean = 55, sd = 15) # per-artist typical popularity
track_artist <- sample(seq_len(n_artists), n_tracks, replace = TRUE)

pop <- pmin(100, pmax(0, round(
  artist_base_pop[track_artist] + rnorm(n_tracks, 0, 10)
)))

spotify <- data.frame(
  track_id = seq_len(n_tracks),
  first_auth = artists[track_artist],
  pop = pop,
  acousticness = runif(n_tracks),
  danceability = runif(n_tracks),
  energy = runif(n_tracks),
  instrumentalness = runif(n_tracks),
  liveness = runif(n_tracks),
  speechiness = runif(n_tracks),
  valence = runif(n_tracks),
  loudness = rnorm(n_tracks, -8, 4),
  duration_ms = round(rnorm(n_tracks, 210000, 40000)),
  tempo = rnorm(n_tracks, 120, 25),
  explicit = sample(c(TRUE, FALSE), n_tracks, replace = TRUE, prob = c(0.2, 0.8))
)
write.csv(spotify, "data/spotify.csv", row.names = FALSE)

cat("Built: data/housing.csv, data/ozone.csv, data/bike_sharing.csv, data/boardgames.csv, data/spotify.csv\n")
