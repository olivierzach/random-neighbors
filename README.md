# Random Neighbors

Random Neighbors is a small research repo for **random-forest-style clustering / feature selection** on high-dimensional data.

It contains:
- **`RandomNeighbors`**: repeatedly subsample rows + features, fit a clustering kernel, and keep the feature subsets that
  produce the best cluster separation (silhouette by default).
- **`UnsupervisedRFProximity`** (baseline): an *unsupervised random forest* trick that learns a proximity matrix from
  leaf co-occurrence (Breiman-style proximities), then clusters in proximity space.

This makes it easy to run experiments like:
- "Can I find a small feature subset that yields decent clusters?"
- "Does feature bagging beat an unsupervised RF proximity baseline on this dataset?"

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Quick start

```python
import numpy as np
from sklearn.cluster import KMeans
from rnc import RandomNeighbors

X = np.random.default_rng(0).normal(size=(1000, 500))

rn = RandomNeighbors(
    select_columns="log2",
    select_rows="sqrt",
    sample_iter=200,
    random_state=0,
)

res = rn.fit(X, clusterer=KMeans(n_clusters=5, n_init="auto", random_state=0))
print(res.best_score, res.best_features)

print(rn.feature_importance(top_k=10))
```

## Experiments

See `experiments/`.

```bash
python -m experiments.run_blobs
python -m experiments.run_iris
```

## Docs
- `docs/algorithm.md` — what the code is doing
- `docs/related_work.md` — references and keywords

## Status / TODO
- Add more principled feature importance (stability selection / permutation)
- Add benchmark harness + plots
- Add scalable approximations for URF proximity (n can get big quickly)
