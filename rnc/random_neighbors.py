from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import silhouette_score


ScoreFn = Callable[[np.ndarray, np.ndarray], float]


@dataclass
class FitResult:
    best_score: float
    best_iter: int
    best_features: List[int]
    history: Dict[str, Dict[str, Any]]


class RandomNeighbors:
    """Random-forest-style feature bagging for clustering.

    Compared to the original version in this repo, this implementation:
    - adds `random_state` for reproducibility
    - fixes log sampling to use log2
    - makes normalization/scaling robust to zero-variance columns
    - supports a pluggable scoring function
    - exposes a simple feature-importance summary

    The scoring step is what defines "good" clusters; silhouette is default.
    """

    def __init__(
        self,
        *,
        use_custom_axis_samples: bool = False,
        select_columns: str = "log2",
        select_rows: str = "percentile",
        sample_iter: int = 20,
        custom_feature_sample_list: Optional[Sequence[int]] = None,
        random_axis_max_pct: float = 0.2,
        normalize_data: bool = True,
        scale_data: bool = True,
        random_state: Optional[int] = None,
        verbose: bool = False,
        score_fn: Optional[ScoreFn] = None,
    ):
        self.use_custom_axis_samples = use_custom_axis_samples
        self.sample_iter = int(sample_iter)
        self.select_columns = select_columns
        self.select_rows = select_rows
        self.custom_feature_sample_list = list(custom_feature_sample_list) if custom_feature_sample_list else None
        self.random_axis_max_pct = float(random_axis_max_pct)
        self.normalize_data = bool(normalize_data)
        self.scale_data = bool(scale_data)
        self.random_state = random_state
        self.verbose = verbose
        self.score_fn = score_fn or (lambda Xs, labels: float(silhouette_score(Xs, labels)))

        self._rng = np.random.default_rng(random_state)
        self._last_result: Optional[FitResult] = None

    def __repr__(self) -> str:
        return (
            "RandomNeighbors(" 
            f"sample_iter={self.sample_iter}, select_columns={self.select_columns}, select_rows={self.select_rows}, "
            f"normalize={self.normalize_data}, scale={self.scale_data}, random_state={self.random_state})"
        )

    @staticmethod
    def _safe_normalize(X: np.ndarray) -> np.ndarray:
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std = np.where(std == 0, 1.0, std)
        return (X - mean) / std

    @staticmethod
    def _safe_minmax_scale(X: np.ndarray) -> np.ndarray:
        mn = X.min(axis=0)
        mx = X.max(axis=0)
        denom = mx - mn
        denom = np.where(denom == 0, 1.0, denom)
        return (X - mn) / denom

    def _sample_axis(self, axis_n: int, num_samples: int) -> List[List[int]]:
        if not (isinstance(axis_n, int) and axis_n > 0):
            raise ValueError("axis_n must be positive int")
        if not (isinstance(num_samples, int) and 0 < num_samples <= axis_n):
            raise ValueError("num_samples must be in [1, axis_n]")
        return [self._rng.choice(axis_n, size=num_samples, replace=False).tolist() for _ in range(self.sample_iter)]

    def build_sample_index(self, axis_n: int, max_axis_selector: str) -> List[List[int]]:
        if self.sample_iter <= 0:
            raise ValueError("sample_iter must be > 0")

        if self.use_custom_axis_samples:
            if not self.custom_feature_sample_list:
                raise ValueError("custom_feature_sample_list must be provided when use_custom_axis_samples=True")
            out: List[List[int]] = []
            for k in self.custom_feature_sample_list:
                out.extend(self._sample_axis(axis_n, int(k)))
            return out[: self.sample_iter]

        if max_axis_selector == "sqrt":
            k = max(1, int(np.sqrt(axis_n)))
            return self._sample_axis(axis_n, k)

        if max_axis_selector == "log2":
            k = max(1, int(np.log2(axis_n)))
            return self._sample_axis(axis_n, k)

        if max_axis_selector == "percentile":
            k = max(1, int(axis_n * 0.1))
            return self._sample_axis(axis_n, k)

        if max_axis_selector == "random":
            max_k = max(1, int(axis_n * self.random_axis_max_pct))
            sizes = self._rng.integers(low=1, high=max_k + 1, size=self.sample_iter)
            return [self._rng.choice(axis_n, size=int(k), replace=False).tolist() for k in sizes]

        raise ValueError("Invalid selector. Use one of: sqrt, log2, percentile, random")

    def fit(self, X: np.ndarray, *, clusterer: Any) -> FitResult:
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if X.shape[0] <= 1 or X.shape[1] <= 1:
            raise ValueError("X must have at least 2 rows and 2 cols")

        Xp = X
        if self.normalize_data:
            Xp = self._safe_normalize(Xp)
        if self.scale_data:
            Xp = self._safe_minmax_scale(Xp)

        n_rows, n_cols = Xp.shape
        row_samples = self.build_sample_index(n_rows, self.select_rows)
        col_samples = self.build_sample_index(n_cols, self.select_columns)

        best_score = -np.inf
        best_iter = -1
        best_cols: List[int] = []
        history: Dict[str, Dict[str, Any]] = {}

        for i in range(self.sample_iter):
            rows = np.asarray(row_samples[i], dtype=int)
            cols = np.asarray(col_samples[i], dtype=int)
            Xs = Xp[rows[:, None], cols]

            fit = clusterer.fit(Xs)
            labels = np.asarray(fit.labels_)

            # Reject degenerate clusterings
            uniq = set(labels.tolist())
            n_clusters = len(uniq) - (1 if -1 in uniq else 0)
            n_noise = int((labels == -1).sum())

            rec: Dict[str, Any] = {
                "columns": cols.tolist(),
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "labels": labels,
                "score": None,
            }

            if len(uniq) <= 1 or n_clusters < 2:
                history[f"iteration_{i}"] = rec
                continue

            try:
                score = float(self.score_fn(Xs, labels))
            except Exception:
                history[f"iteration_{i}"] = rec
                continue

            rec["score"] = score
            history[f"iteration_{i}"] = rec

            if self.verbose:
                print(f"iter {i}: score={score:.4f} n_clusters={n_clusters} n_noise={n_noise} n_features={len(cols)}")

            if score > best_score:
                best_score = score
                best_iter = i
                best_cols = cols.tolist()

        res = FitResult(best_score=float(best_score), best_iter=int(best_iter), best_features=best_cols, history=history)
        self._last_result = res
        return res

    def feature_importance(self, *, top_k: int = 25) -> List[Tuple[int, float]]:
        """Score-weighted feature frequency from the last fit.

        Not a principled importance metric; a quick heuristic to see which features
        show up in high-scoring iterations.
        """
        if self._last_result is None:
            raise RuntimeError("call fit() first")

        scores: Dict[int, float] = {}
        for rec in self._last_result.history.values():
            s = rec.get("score")
            cols = rec.get("columns")
            if s is None or not cols:
                continue
            w = max(0.0, float(s))
            for j in cols:
                scores[j] = scores.get(j, 0.0) + w

        items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return items[: int(top_k)]
