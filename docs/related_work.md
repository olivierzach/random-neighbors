# Related work / references

This repo’s main algorithm is inspired by the "bagging" structure of random forests, but it is **not** the same as
"unsupervised random forests". This doc collects references for both.

## Random forest proximities / unsupervised RF
- Leo Breiman, *Random Forests* (2001). Proximities are defined by co-occurrence in terminal nodes.
  - Breiman’s random forests site (manual + examples): https://www.stat.berkeley.edu/~breiman/RandomForests/
- Unsupervised random forests (review / implementation patterns):
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC8025042/
- Geometry- and Accuracy-Preserving Random Forest Proximities (2022):
  - https://arxiv.org/pdf/2201.12682

## Feature bagging + clustering / stability selection
Key keywords if you want to expand this:
- clustering stability / consensus clustering
- sparse clustering / feature selection for clustering
- random subspace methods

## Density-based clustering kernels
The repo includes PDFs for DBSCAN and OPTICS in `docs/`.
