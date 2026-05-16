import numpy as np
from sklearn.datasets import make_blobs

from rnc import UnsupervisedRFProximity


def test_transform_landmarks_shape_and_range():
    X, _ = make_blobs(n_samples=500, n_features=10, centers=4, random_state=0)
    urf = UnsupervisedRFProximity(n_estimators=50, random_state=0)
    urf.fit(X)

    P, idx = urf.transform_landmarks(X, n_landmarks=64)
    assert P.shape == (X.shape[0], idx.shape[0])
    assert idx.shape[0] == 64
    assert np.all((P >= 0.0) & (P <= 1.0))
