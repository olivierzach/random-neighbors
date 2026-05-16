import numpy as np
from sklearn.cluster import KMeans

from rnc import RandomNeighbors


def test_fit_runs_and_returns_result():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 30))
    rn = RandomNeighbors(sample_iter=20, select_rows="sqrt", select_columns="sqrt", random_state=0)
    res = rn.fit(X, clusterer=KMeans(n_clusters=3, n_init="auto", random_state=0))

    assert isinstance(res.best_score, float)
    assert isinstance(res.best_iter, int)
    assert isinstance(res.best_features, list)
    assert isinstance(res.history, dict)


def test_feature_importance_requires_fit():
    rn = RandomNeighbors()
    try:
        rn.feature_importance()
        assert False, "expected error"
    except RuntimeError:
        assert True
