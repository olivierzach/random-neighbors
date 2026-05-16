"""Benchmark harness for RandomNeighbors + URF baselines.

This script runs a small suite of sklearn datasets and writes:
- artifacts/benchmarks/<run_id>/results.csv
- artifacts/benchmarks/<run_id>/plots/*.png

Metrics:
- silhouette on the full dataset using the chosen feature subset / representation
- ARI/NMI when ground-truth labels are available (many toy datasets)

Note: These are *clustering* benchmarks; ground truth is used only for evaluation.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.datasets import load_digits, load_iris, load_wine, make_blobs, make_moons
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

from rnc import RandomNeighbors, UnsupervisedRFProximity


@dataclass
class Row:
    dataset: str
    n: int
    d: int
    k_true: Optional[int]
    method: str
    seed: int
    fit_seconds: float
    silhouette: Optional[float]
    ari: Optional[float]
    nmi: Optional[float]
    details: Dict[str, Any]


def _now_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _maybe_pca(X: np.ndarray, max_d: int, seed: int) -> Tuple[np.ndarray, bool]:
    if X.shape[1] <= max_d:
        return X, False
    pca = PCA(n_components=max_d, random_state=seed)
    return pca.fit_transform(X), True


def eval_labels(X: np.ndarray, labels: np.ndarray, y_true: Optional[np.ndarray]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    uniq = set(np.asarray(labels).tolist())
    n_clusters = len(uniq) - (1 if -1 in uniq else 0)
    if len(uniq) <= 1 or n_clusters < 2:
        sil = None
    else:
        try:
            sil = float(silhouette_score(X, labels))
        except Exception:
            sil = None

    if y_true is None:
        return sil, None, None

    try:
        ari = float(adjusted_rand_score(y_true, labels))
    except Exception:
        ari = None
    try:
        nmi = float(normalized_mutual_info_score(y_true, labels))
    except Exception:
        nmi = None

    return sil, ari, nmi


def run_one_dataset(
    *,
    name: str,
    X: np.ndarray,
    y_true: Optional[np.ndarray],
    k_true: Optional[int],
    seed: int,
    out_dir: Path,
) -> List[Row]:
    rows: List[Row] = []

    # --- Baseline: KMeans on full X (optionally PCA) ---
    X_base, used_pca = _maybe_pca(X, max_d=50, seed=seed)
    k = int(k_true or 5)

    t0 = datetime.now()
    km = KMeans(n_clusters=k, n_init="auto", random_state=seed)
    labels_km = km.fit_predict(X_base)
    dt = (datetime.now() - t0).total_seconds()
    sil, ari, nmi = eval_labels(X_base, labels_km, y_true)
    rows.append(
        Row(
            dataset=name,
            n=int(X.shape[0]),
            d=int(X.shape[1]),
            k_true=k_true,
            method="kmeans_full" + ("_pca50" if used_pca else ""),
            seed=seed,
            fit_seconds=float(dt),
            silhouette=sil,
            ari=ari,
            nmi=nmi,
            details={"k": k},
        )
    )

    # --- RandomNeighbors: pick features, then run KMeans on full X[:, feats] ---
    rn = RandomNeighbors(
        select_columns="log2",
        select_rows="sqrt",
        sample_iter=200,
        random_state=seed,
        verbose=False,
        normalize_data=True,
        scale_data=True,
    )

    t0 = datetime.now()
    res = rn.fit(X, clusterer=KMeans(n_clusters=k, n_init="auto", random_state=seed))
    dt = (datetime.now() - t0).total_seconds()

    feats = rn.stable_features(top_frac=0.2, min_freq=0.3, top_k=min(25, X.shape[1]))
    if len(feats) == 0:
        feats = res.best_features

    X_sel = X[:, feats]
    labels_sel = KMeans(n_clusters=k, n_init="auto", random_state=seed).fit_predict(X_sel)
    sil, ari, nmi = eval_labels(X_sel, labels_sel, y_true)

    rows.append(
        Row(
            dataset=name,
            n=int(X.shape[0]),
            d=int(X.shape[1]),
            k_true=k_true,
            method="random_neighbors+kmeans",
            seed=seed,
            fit_seconds=float(dt),
            silhouette=sil,
            ari=ari,
            nmi=nmi,
            details={
                "k": k,
                "best_score_subsample": res.best_score,
                "best_iter": res.best_iter,
                "n_features": int(len(feats)),
                "features": [int(j) for j in feats],
            },
        )
    )

    # --- URF landmarks: proximity-to-landmarks features + KMeans ---
    urf = UnsupervisedRFProximity(n_estimators=200, random_state=seed)
    t0 = datetime.now()
    urf.fit(X_base)
    P, lm_idx = urf.transform_landmarks(X_base, n_landmarks=min(256, X_base.shape[0]))
    dt = (datetime.now() - t0).total_seconds()

    # KMeans in landmark-proximity space
    labels_p = KMeans(n_clusters=k, n_init="auto", random_state=seed).fit_predict(P)
    sil, ari, nmi = eval_labels(P, labels_p, y_true)

    rows.append(
        Row(
            dataset=name,
            n=int(X.shape[0]),
            d=int(X.shape[1]),
            k_true=k_true,
            method="urf_landmarks+kmeans" + ("_pca50" if used_pca else ""),
            seed=seed,
            fit_seconds=float(dt),
            silhouette=sil,
            ari=ari,
            nmi=nmi,
            details={"k": k, "n_landmarks": int(P.shape[1])},
        )
    )

    # Write a quick 2D plot for the dataset if feasible
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    X2, _ = _maybe_pca(X, max_d=2, seed=seed)

    def _scatter(ax, X2, labels, title):
        ax.scatter(X2[:, 0], X2[:, 1], c=labels, s=8, cmap="tab20", alpha=0.8)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    _scatter(axes[0], X2, labels_km, "kmeans_full")
    _scatter(axes[1], X2, labels_sel, "random_neighbors")
    _scatter(axes[2], X2, labels_p, "urf_landmarks")
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(plots_dir / f"{name}.png", dpi=160)
    plt.close(fig)

    return rows


def main() -> None:
    seed = int(os.getenv("RNC_BENCH_SEED", "0"))
    run_id = os.getenv("RNC_RUN_ID", _now_run_id())

    out_dir = Path("artifacts") / "benchmarks" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets: List[Tuple[str, np.ndarray, Optional[np.ndarray], Optional[int]]] = []

    X, y = load_iris(return_X_y=True)
    datasets.append(("iris", X, y, 3))

    X, y = load_wine(return_X_y=True)
    datasets.append(("wine", X, y, 3))

    X, y = load_digits(return_X_y=True)
    datasets.append(("digits", X, y, 10))

    X, y = make_moons(n_samples=2000, noise=0.07, random_state=seed)
    datasets.append(("moons", X, y, 2))

    X, y = make_blobs(n_samples=6000, n_features=50, centers=6, cluster_std=2.8, random_state=seed)
    # add noise dimensions
    rng = np.random.default_rng(seed)
    X = np.hstack([X, rng.normal(size=(X.shape[0], 200))])
    datasets.append(("blobs_highdim", X, y, 6))

    rows: List[Row] = []
    for name, X, y_true, k_true in datasets:
        rows.extend(run_one_dataset(name=name, X=X, y_true=y_true, k_true=k_true, seed=seed, out_dir=out_dir))

    df = pd.DataFrame([{
        **{k: v for k, v in asdict(r).items() if k != "details"},
        "details": json.dumps(r.details, sort_keys=True),
    } for r in rows])

    df.to_csv(out_dir / "results.csv", index=False)

    # Aggregate bar plot by dataset/method for ARI (if present)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if df["ari"].notna().any():
        pivot = df.pivot_table(index=["dataset"], columns=["method"], values="ari", aggfunc="mean")
        ax = pivot.plot(kind="bar", figsize=(12, 5), rot=0)
        ax.set_ylabel("ARI (higher is better)")
        ax.set_title("Clustering ARI by dataset")
        plt.tight_layout()
        plt.savefig(plots_dir / "ari_bar.png", dpi=160)
        plt.close()

    if df["silhouette"].notna().any():
        pivot = df.pivot_table(index=["dataset"], columns=["method"], values="silhouette", aggfunc="mean")
        ax = pivot.plot(kind="bar", figsize=(12, 5), rot=0)
        ax.set_ylabel("Silhouette (higher is better)")
        ax.set_title("Silhouette by dataset")
        plt.tight_layout()
        plt.savefig(plots_dir / "silhouette_bar.png", dpi=160)
        plt.close()

    # Convenience pointer
    latest = Path("artifacts") / "benchmarks" / "LATEST"
    try:
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(out_dir)
    except Exception:
        pass

    print(f"Wrote artifacts to: {out_dir}")


if __name__ == "__main__":
    main()
