# Random Neighbors Clustering — algorithm

## Problem
Given a data matrix `X ∈ R^{n×d}` with large `d`, we want to find a **small subset of features** that yields a
**useful clustering** under a chosen clustering kernel (DBSCAN/OPTICS/KMeans/etc.).

The main idea is analogous to random forests:
- random forests: bootstrap rows + features → fit many trees → aggregate
- random neighbors: bootstrap rows + features → fit many clusterers → select/aggregate

## RandomNeighbors (this repo)

### Inputs
- `X`: numeric array, shape `(n_samples, n_features)`
- `clusterer`: an sklearn-style clusterer implementing `fit(X)` and exposing `labels_`
- sampling policies for rows/cols (sqrt/log2/percentile/random or custom)
- `score_fn`: by default silhouette (requires ≥2 clusters and not all points in one label)

### Core loop
For iterations `t = 1..T`:
1. Sample row indices `R_t` and feature indices `C_t`.
2. Fit clusterer on `X[R_t, C_t]`.
3. Compute a score (default silhouette).
4. Store history: score, sampled features, cluster counts, noise count.

### Outputs
- best score and best iteration
- history dict (per-iteration metadata)
- optional aggregate feature importance (frequency-weighted by score)

### Notes
- Silhouette is biased toward globular clusters and can be undefined for some clusterers.
- For density clusterers (DBSCAN/OPTICS), be careful: many `-1` noise labels makes silhouette tricky.

## UnsupervisedRFProximity (baseline)
This is a standard unsupervised random forest trick:
1) Generate a synthetic dataset by sampling each feature independently from its empirical marginal.
2) Train a classifier to distinguish real vs synthetic.
3) Compute a proximity matrix: samples are close if they land in the same leaf often.
4) Cluster using distances `D = 1 - proximity`.

This gives a clustering metric that can capture complex interactions.

See `docs/related_work.md` for references.
