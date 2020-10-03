from rnc.random_neighbors import RandomNeighbors
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from collections import Counter

data = np.random.rand(10000, 10000)
print(data.shape)
print(data.shape)

max_rows, max_cols = data.shape
print(max_rows, max_cols)

rnc = RandomNeighbors(select_columns='sqrt', select_rows='sqrt')
scan = DBSCAN()

best_metric, best_metric_iter, fit_history = rnc.fit_random_neighbors(x=data, cluster=scan)
[(k, v['n_clusters']) for (k, v) in fit_history.items()]

row_samples = rnc.build_sample_index(max_rows, max_axis_selector='sqrt')
col_samples = rnc.build_sample_index(max_cols, max_axis_selector='sqrt')
print(len(row_samples), len(col_samples))
bootstrap_list = [(row_samples[i], col_samples[i]) for i in range(20)]
bootstrap_list[0][0]

for i, v in enumerate(bootstrap_list):
    boot_rows = np.array(v[0])
    boot_cols = np.array(v[1])
    sampled_x = data[boot_rows[:, None], boot_cols]

    cluster_fit = scan.fit(sampled_x)
    labels = cluster_fit.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    if len(set(labels)) == 1:
        print(f'No Label Differentiation - skipping iteration {i}')

    else:
        sil_score = silhouette_score(sampled_x, labels)

        if sil_score > 0.0:
            print(f"iteration {i}: n_clusters {n_clusters}, n_noise {n_noise}")
            print(f"{Counter(labels)}")
            print(f"silhouette coefficient: {sil_score} \n")
