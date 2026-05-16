from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

from rnc import RandomNeighbors, UnsupervisedRFProximity


def main() -> None:
    X, y = load_iris(return_X_y=True)

    print("== RandomNeighbors (feature bagging + clustering) ==")
    rn = RandomNeighbors(sample_iter=200, select_columns="sqrt", select_rows="percentile", random_state=0)
    res = rn.fit(X, clusterer=KMeans(n_clusters=3, n_init="auto", random_state=0))
    print("best_score:", res.best_score)
    print("best_features:", res.best_features)

    print("\n== UnsupervisedRFProximity (baseline) ==")
    urf = UnsupervisedRFProximity(n_estimators=300, random_state=0)
    prox = urf.fit_transform(X)
    labels = urf.cluster(prox, n_clusters=3)
    print("labels_counts:", {i: (labels == i).sum() for i in set(labels)})


if __name__ == "__main__":
    main()
