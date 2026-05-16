import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from rnc import RandomNeighbors


def main() -> None:
    X, y = make_blobs(n_samples=5000, n_features=50, centers=5, cluster_std=3.0, random_state=0)

    # Add a bunch of pure-noise features
    rng = np.random.default_rng(0)
    X = np.hstack([X, rng.normal(size=(X.shape[0], 200))])

    rn = RandomNeighbors(
        select_columns="log2",
        select_rows="sqrt",
        sample_iter=200,
        random_state=0,
        verbose=False,
        normalize_data=True,
        scale_data=True,
    )

    best = rn.fit(X, clusterer=KMeans(n_clusters=5, n_init="auto", random_state=0))
    print("best_score:", best.best_score)
    print("best_iter:", best.best_iter)
    print("best_n_features:", len(best.best_features))

    imp = rn.feature_importance(top_k=20)
    print("top_features:")
    for j, s in imp:
        print(f"  f{j}: {s:.4f}")


if __name__ == "__main__":
    main()
