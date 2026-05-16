"""Random Neighbors Clustering (rnc).

Two related ideas live here:

1) RandomNeighbors: feature/row bagging + repeated clustering, selecting feature subsets that
   yield high cluster separation (silhouette by default).

2) UnsupervisedRFProximity: an unsupervised random forest baseline that builds a proximity
   matrix from leaf co-occurrence, then clusters in proximity space.
"""

from .random_neighbors import RandomNeighbors
from .unsupervised_rf import UnsupervisedRFProximity

__all__ = ["RandomNeighbors", "UnsupervisedRFProximity"]
