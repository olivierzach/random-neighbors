import numpy as np
from sklearn.datasets import make_blobs

from rnc import UnsupervisedRFProximity


def main() -> None:
    X, _ = make_blobs(n_samples=20000, n_features=40, centers=8, random_state=0)

    urf = UnsupervisedRFProximity(n_estimators=200, random_state=0)
    urf.fit(X)

    P, lm_idx = urf.transform_landmarks(X, n_landmarks=512)
    print("P shape:", P.shape)
    print("landmarks:", lm_idx.shape)

    # Simple sanity checks
    assert P.shape[0] == X.shape[0]
    assert P.shape[1] == lm_idx.shape[0]
    assert np.all((P >= 0.0) & (P <= 1.0))


if __name__ == "__main__":
    main()
