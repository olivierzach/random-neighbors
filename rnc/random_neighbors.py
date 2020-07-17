import numpy as np
from collections import Counter
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import random


class RandomNeighbors:
    
    def __init__(
            self,
            use_custom_axis_samples=False,
            select_columns='log2',
            select_rows='percentile',
            sample_iter=20,
            custom_feature_sample_list=None,
            kernel='dbscan',
            eps=None,
            min_samples=None,
            metric=None
    ):

        self.use_custom_axis_samples = use_custom_axis_samples
        self.sample_iter = sample_iter
        self.select_columns = select_columns
        self.select_rows = select_rows
        self.custom_feature_sample_list = custom_feature_sample_list
        self.kernel = kernel
        self.metric = metric
        self.eps = eps
        self.min_samples = min_samples

    @staticmethod
    def sample_axis(axis_n, sample_iter, num_samples):
        """
        Sample from range of (0, max_cols), num_samples without replacement, for each sample iteration

        Parameters
        ----------
        axis_n : number of total rows or total columns in the dataset
        sample_iter: number of iterations to sampling from an axis
        num_samples: number of observations from axis to sample

        Returns
        -------
        list
        """

        return [random.sample(range(axis_n), num_samples) for _ in range(sample_iter)]

    def build_sample_index(self, axis_n, max_axis_selector='log2'):
        """
        Route the sample_axis method through sample type options.
        If custom list provided, build a list of columns based on size provided in the list.
        If no custom list provided, use one of [sqrt, log2, percentile, random] to build list of sampled axis indexes

        Parameters
        ----------
        axis_n : number of total rows or total columns in the dataset
        max_axis_selector : metric to determine number of axis to sample

        Returns
        -------
        list
        """

        idx_samples = None

        # route for a custom list of axis sample sizes
        if self.use_custom_axis_samples:

            custom_samples = []

            for i in self.custom_feature_sample_list:

                idx_samples = self.sample_axis(
                    axis_n=axis_n,
                    sample_iter=1,
                    num_samples=i
                )

                custom_samples.append(idx_samples)

            idx_samples = custom_samples

        # sqrt selection of axis for all iterations - each set will have sqrt(axis_n) samples
        if max_axis_selector == 'sqrt':

            sqrt_ = int(np.sqrt(axis_n))

            idx_samples = self.sample_axis(
                axis_n=axis_n,
                sample_iter=self.sample_iter,
                num_samples=sqrt_
            )

        # log selection of axis for all iterations - each set will have log(axis_n) samples
        if max_axis_selector == 'log2':

            log_ = int(np.log(axis_n))

            idx_samples = self.sample_axis(
                axis_n=axis_n,
                sample_iter=self.sample_iter,
                num_samples=log_
            )

        # percentile selection of axis for all iterations - each set will have .1*axis_n samples
        if max_axis_selector == 'percentile':

            percentile_ = int(axis_n * .1)

            idx_samples = self.sample_axis(
                axis_n=axis_n,
                sample_iter=self.sample_iter,
                num_samples=percentile_
            )

        # random selection of axis - each set will have different size samples
        if max_axis_selector == 'random':

            # grab a set of random numbers sample iter items wide
            random_ = random.sample(range(int(axis_n * .2)), self.sample_iter)

            # grab list of randomly sizes axis indexes
            idx_samples = [list(random.sample(range(axis_n), i)) for i in random_]

        if idx_samples:
            return idx_samples

        else:
            raise ValueError("Invalid parameters. Valid parameters are ['sqrt', 'log2', 'percentile', 'random']")

    def fit_random_neighbors(self, x):
        """
        Recusively fit clustering algorithm using bootstrapped rows and columns for each fit.
        Output the best scores and a history objectRoute the sample_axis method through sample type options.
        If custom list provided, build a list of columns based on size provided in the list.
        If no custom list provided, use one of [sqrt, log2, percentile, random] to build list of sampled axis indexes

        Parameters
        ----------
        axis_n : number of total rows or total columns in the dataset
        max_axis_selector : metric to determine number of axis to sample

        Returns
        -------
        list
        """

        # define global storage to hold best parameters
        best_sil = 0.0
        best_sil_iter = 0.0
        history_dict = {}

        # build the bootstrap of columns, rows
        max_rows, max_cols = x.shape
        row_samples = self.build_sample_index(axis_n=max_rows, max_axis_selector=self.select_rows)
        col_samples = self.build_sample_index(axis_n=max_cols, max_axis_selector=self.select_columns)
        bootstrap_list = [(row_samples[i], col_samples[i]) for i in range(max_rows)]

        if self.kernel == 'dbscan':

            for i, v in enumerate(bootstrap_list):

                boot_rows = np.array(v[0])
                boot_cols = np.array(v[1])

                sampled_x = x[boot_rows[:, None], boot_cols]

                # cluster on predictions, features, and meta data flags
                cluster = DBSCAN(
                    eps=self.eps,
                    min_samples=self.min_samples,
                    metric=self.metric,
                    algorithm='ball_tree'
                )

                # fit and check outputs
                cluster_fit = cluster.fit(sampled_x)
                labels = cluster_fit.labels_
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                sil_score = silhouette_score(sampled_x, labels)

                if sil_score > 0.0:
                    print(f"iteration {i}: n_clusters {n_clusters}, n_noise {n_noise}")
                    print(f"{Counter(labels)}")
                    print(f"silhouette coefficient: {sil_score} \n")

                    history_dict['iteration_' + str(i)] = {
                        'score': sil_score,
                        'columns': boot_cols,
                        'labels': labels,
                        'n_clusters': n_clusters,
                        'n_noise': n_noise,
                        'fit_': cluster_fit
                    }

                    if (sil_score > best_sil) & (n_clusters > 1.0):
                        best_sil = sil_score
                        best_sil_iter = i

                return best_sil, best_sil_iter, history_dict
