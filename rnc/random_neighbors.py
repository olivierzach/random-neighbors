import numpy as np
from collections import Counter
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import random


class RandomNeighbors(object):
    
    def __init__(
            self,
            use_custom_feature_samples=False,
            max_features='log2',
            sample_iter=20,
            custom_feature_sample_list=None,
            kernel='DBSCAN',
            eps_list=None,
            min_samples_list=None,
            metric_list=None,
            data=None
    ):

        self.use_custom_feature_samples = use_custom_feature_samples
        self.sample_iter = sample_iter
        self.max_features = max_features
        self.custom_feature_sample_list = custom_feature_sample_list
        self.kernel = kernel
        self.eps_list = eps_list
        self.min_samples_list = min_samples_list
        self.metric_list = metric_list
        self.X = data

    def brute_dbscan_search(self):
        """
        Brute grid search through various clustering methods that output a silhouette score.
        Support DBSCAN, KNN, FAST-DBSCAN kernels.

        Parameters
        ----------

        Returns
        -------
        Object
        """

        best_sil = 0.0
        best_eps = 0.0
        best_ms = 0.0
        best_metric = None

        # search and fit clustering algorithm and show results
        for i in self.eps_list:
            for j in self.min_samples_list:
                for k in self.metric_list:

                    try:
                        # cluster on predictions, features, and meta data flags
                        cluster = DBSCAN(
                            eps=i,
                            min_samples=j,
                            metric=k,
                            algorithm='ball_tree'
                        )

                        # fit and check outputs
                        cluster_fit = cluster.fit(self.X)
                        labels = cluster_fit.labels_
                        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                        n_noise = list(labels).count(-1)
                        sil_score = silhouette_score(self.X, labels)

                        if sil_score > 0.0:
                            print(f"eps {i}, ms {j}: clusters {n_clusters}, noise {n_noise}, dist {k}")
                            print(f"{Counter(labels)}")
                            print(f"Silhouette Coefficient: {sil_score} \n")

                        if (sil_score > best_sil) & (n_clusters > 1.0):
                            best_sil = sil_score
                            best_eps = i
                            best_ms = j
                            best_metric = k

                    except Exception as e:
                        print(e)

        return best_eps, best_ms, best_metric, best_sil

    @staticmethod
    def brute_knn_search():
        # TODO: build the KNN search here
        return None

    @staticmethod
    def sample_axis(max_axis, sample_iter, num_samples):
        """
        Sample from range of 0-max_cols, num_cols without replacement, for sample_n iterations

        Parameters
        ----------
        max_axis : number of total rows or total columns in the dataset
        sample_iter: number of iterations to sampling from an axis
        num_samples: number of observations from axis to sample

        Returns
        -------
        list
        """

        return [random.sample(range(max_axis), num_samples) for _ in sample_iter]

    def build_sample_index(self, total_features):
        """
        Route the sample_columns method through num_cols options. If custom list provided, build a list of
        columns based on size provided in the list. If no custom list provided, use the sqrt, log2, or
        10th percentile of total columns to build list of columns to sample.

        Parameters
        ----------

        Returns
        -------
        list
        """

        max_cols = total_features

        if self.use_custom_feature_samples:

            custom_samples = []

            for i in self.custom_feature_sample_list:

                idx_samples = self.sample_axis(
                    max_axis=max_cols,
                    sample_iter=1,
                    num_samples=i
                )

                custom_samples.append(idx_samples)

            idx_samples = custom_samples

        elif self.max_features == 'sqrt':

            sqrt_ = int(np.sqrt(max_cols))

            idx_samples = self.sample_axis(
                max_axis=max_cols,
                sample_iter=self.sample_iter,
                num_samples=sqrt_
            )

        elif self.max_features == 'log2':

            log_ = int(np.log(max_cols))

            idx_samples = self.sample_axis(
                max_axis=max_cols,
                sample_iter=self.sample_iter,
                num_samples=log_
            )

        else:

            percentile_ = int(max_cols * .1)

            idx_samples = self.sample_axis(
                max_axis=max_cols,
                sample_iter=self.sample_iter,
                num_samples=percentile_
            )

        return idx_samples

    def randomize_clusters(self, df, eps_list, min_samples_list, metric_list, col_samples):
        """
        Random forest style clustering
        Uses sample of data frame columns and rows to find the best breakage
        DBSCAN clustering kernel using brute cluster search
        Parameters
        ----------
        df : data frame to cluster
        eps_list : radius of cluster parameter - list of values to search through
        min_samples_list : number of samples needed to form cluster - list of values to search through
        metric_list : list of distance metrics to search through
        col_samples : random sample of df columns
        Returns
        -------
        multiple
        """

        # define global storage for best breakage parameters
        global_eps = None
        global_ms = None
        global_metric = None
        global_sil = 0.0
        global_col_samples = None

        # TODO: initialize the sample rows and columns here
        # TODO: initialize with sqrt, log2, and distribution based sampling of rows
        # TODO: use the brute grid search or search through static parameters
        # TODO: route the cluster kernel here: DBSCAN, KNN

        # loop through randomized columns
        for c in col_samples:

            # sample rows for clustering
            # row_sample = np.random.uniform(.3, .7, [1, 1])[0][0]

            # run clustering search
            best_eps, best_ms, best_metric, best_sil = self.brute_dbscan_search()

            # update metrics
            if best_sil > global_sil:
                global_sil = best_sil
                global_eps = best_eps
                global_ms = best_ms
                global_metric = best_metric
                global_col_samples = c

        print(f"best eps: {global_eps} best ms: {global_ms} best metric: {global_metric} best score{global_sil}")
        print(global_col_samples)

        return global_col_samples, global_eps, global_ms, global_metric
