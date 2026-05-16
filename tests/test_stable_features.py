import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from rnc import RandomNeighbors


def test_stable_features_runs():
    X, _ = make_blobs(n_samples=300, n_features=20, centers=3, random_state=0)
    rn = RandomNeighbors(sample_iter=50, select_rows="sqrt", select_columns="sqrt", random_state=0)
    rn.fit(X, clusterer=KMeans(n_clusters=3, n_init="auto", random_state=0))

    feats = rn.stable_features(top_frac=0.3, min_freq=0.1)
    assert isinstance(feats, list)
