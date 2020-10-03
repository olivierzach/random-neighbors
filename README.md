# Random Neighbors

Random Neighbors Clustering performs "random forest" style clustering on high-dimensional data. Provides sampling strategies
to bootstrap input rows and columns. Method iteratively fits clustering models to the 
samples and optimizes for the features that provide the best distinct cluster separation.


## Algorithm

Implementation is straight-forward:
- Bootstrap rows and columns based on inputted sampling strategy for a predefined set of iterations
- For each iteration on the sampled data, fit a clustering model and extract performance metrics
- Output the features that provide the most distinct clusters across all iterations

Like a random forest iteratively builds decision trees on bootstrapped data, random neighbors
iteratively builds clusters on bootstrapped data. Random Neighbors iterations can be used to extract the
single set of columns that provide largest cluster separation, or features can be averaged across iterations in a 
psuedo-feature importance metric. Dropout can be optionally applied to the history of iterations to avoid bias in 
the sampled fitting procedure. 

## Benefits

Main benefits include:
- Faster fitting on high-dimensional data through bootstrapping
- Iterations search for a low-dimensional representation of the data that produces distinct clusters
- Random sampling guards against correlation in input features, which can skew clustering results

## Case Study: Clustering Product Usage

Product logs contain a wealth of information on every action with a business application. Naturally,
business would like to segment users based on this activity in order to drive product strategy including pricing, new features,
churn reduction, and in-app upgrades. 

Product usage data can be extremely high-dimensional with a feature rich application. Size of 
input data grows with the total numbers of actions possible in the app and the total numbers
of users. The size and dimension of this data makes fitting clustering methods on the entire dataset challenging. 
Fitting on all rows is time-consuming, and a large amount of columns, possibly correlated, can build disoriented results. 

Product data like this drove the development of `RandomNeighbors`. Fitting this method produced a 5-feature, 5-cluster
representation of the entire dataset based on a strong `silhouette score`. Resulting representation enabled real-time 
product clustering that was not possible applying model to entire feature-set. 

Details of the implementation: 
- `DBSCAN` kernel, the "decision tree" of `RandomNeighbors`
- 50 total model iterations
- Column sampling of at most `log2(features)` for each iteration
- Row sampling of `sqrt(rows)` for each iteration
- Extract `silhouette score` from each iteration, keep only iterations with positive score and at least two clusters
- Keep history of columns that provided the best score
- Re-fit model on the full dataset using the best features selected from the method

## Usage

```python
from rnc.random_neighbors import RandomNeighbors
from sklearn.cluster import DBSCAN
import numpy as np

# high dimensional data
data = np.random.rand(100000, 10000)

# use vanilla DBSCAN as random neighbors kernel
# any sklearn.cluster method that uses sklearn.metrics `silhouette_score` is supported
cluster = DBSCAN()

# initialize with bootstrap strategy
rnc = RandomNeighbors(
    select_columns='sqrt',
    select_rows='log2',
    sample_iter=200,
    normalize_data=True,
    scale_data=True
)

# fit method to our data
best_metric, best_iter, fit_history = rnc.fit_random_neighbors(x=data, cluster=cluster)

# see results
print(best_metric, best_iter)

# history shows scores and bootstrap indices by iteration
scores = [(k, v['score']) for (k, v) in fit_history.items()]
print(scores)
```

## Method Documentation

Available methods for `fit_random_neighbors()`:
- `use_custom_axis_samples`: supply your own index samples for columns and rows
- `select_columns`: sample policy for columns (`sqrt`, `log2`, `percentile`, `random`)
- `select_rows`: sample policy for rows (`sqrt`, `log2`, `percentile`, `random`)
- `sample_iter`: number of model fit iterations
- `custom_feature_sample_list`: list of custom sample indices
- `random_axis_max_pct`: threshold for percentile sampling strategy
- `normalize_data`: normalize input data
- `scale_data`: min-max scale input data

Method Returns:
- `best_metric`: best overall `silhouette score` across all iterations
- `best_iter`: iteration that provided the best overall score
- `history_dict`: dictionary of rows, columns, score, labels, and fit object by model iteration

## Roadmap

- Psuedo-Feature Importance Metrics
- Expanded Testing
- History dropout 
- Data Examples
