from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import AgglomerativeClustering


@dataclass
class URFResult:
    proximity: np.ndarray  # (n, n)
    rf: RandomForestClassifier


class UnsupervisedRFProximity:
    """Unsupervised random forest proximity baseline.

    Standard trick:
      - Build synthetic samples by independently sampling each feature from its marginal.
      - Train RF classifier to discriminate real vs synthetic.
      - Define proximity between two real samples as fraction of trees where they share a leaf.

    This is expensive O(n^2 * n_estimators) in the naive full-matrix form; intended for
    n up to a few 10k at most (often you subsample for proximity).
    """

    def __init__(
        self,
        *,
        n_estimators: int = 200,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 1,
        max_features: str | int | float = "sqrt",
        random_state: Optional[int] = None,
        n_jobs: int = -1,
    ):
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs

        self._rf: Optional[RandomForestClassifier] = None

    def _make_synthetic(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n, d = X.shape
        Xs = np.empty_like(X)
        for j in range(d):
            col = X[:, j]
            Xs[:, j] = rng.choice(col, size=n, replace=True)
        return Xs

    def fit(self, X: np.ndarray) -> "UnsupervisedRFProximity":
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be 2D")

        rng = np.random.default_rng(self.random_state)
        Xs = self._make_synthetic(X, rng)

        X_all = np.vstack([X, Xs])
        y_all = np.hstack([np.ones(X.shape[0]), np.zeros(X.shape[0])])

        rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        rf.fit(X_all, y_all)
        self._rf = rf
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._rf is None:
            raise RuntimeError("call fit() first")

        X = np.asarray(X)
        n = X.shape[0]

        # leaf_ids: (n_samples, n_trees)
        leaf_ids = self._rf.apply(X)

        # Proximity: fraction of trees where leaf_ids match
        prox = np.zeros((n, n), dtype=np.float32)
        for t in range(leaf_ids.shape[1]):
            ids = leaf_ids[:, t]
            # group by leaf id
            order = np.argsort(ids)
            ids_sorted = ids[order]
            # find groups
            starts = np.r_[0, np.flatnonzero(ids_sorted[1:] != ids_sorted[:-1]) + 1]
            ends = np.r_[starts[1:], len(ids_sorted)]
            for a, b in zip(starts, ends):
                grp = order[a:b]
                prox[np.ix_(grp, grp)] += 1.0

        prox /= float(leaf_ids.shape[1])
        np.fill_diagonal(prox, 1.0)
        return prox

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def cluster(self, proximity: np.ndarray, *, n_clusters: int = 5) -> np.ndarray:
        # Distance for agglomerative
        D = 1.0 - proximity
        # AgglomerativeClustering wants condensed? It can accept precomputed with metric="precomputed" in newer sklearn.
        model = AgglomerativeClustering(n_clusters=n_clusters, metric="precomputed", linkage="average")
        return model.fit_predict(D)
