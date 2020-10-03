import numpy as np
from collections import Counter
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
            random_axis_max_pct=.2,
            normalize_data=True,
            scale_data=True
    ):

        self.use_custom_axis_samples = use_custom_axis_samples
        self.sample_iter = sample_iter
        self.select_columns = select_columns
        self.select_rows = select_rows
        self.custom_feature_sample_list = custom_feature_sample_list
        self.random_axis_max_pct = random_axis_max_pct
        self.normalize_data = normalize_data
        self.scale_data = scale_data

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

        assert isinstance(axis_n, int)
        assert axis_n > 0
        assert isinstance(sample_iter, int)
        assert sample_iter > 0
        assert isinstance(num_samples, int)
        assert num_samples > 0

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

        assert isinstance(axis_n, int)
        assert axis_n > 0
        assert self.sample_iter > 0
        assert isinstance(max_axis_selector, str)
        assert isinstance(self.random_axis_max_pct, float)

        # TODO: find a better way to route than if blocks...
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
        elif max_axis_selector == 'sqrt':

            sqrt_ = int(np.sqrt(axis_n))

            idx_samples = self.sample_axis(
                axis_n=axis_n,
                sample_iter=self.sample_iter,
                num_samples=sqrt_
            )

        # log selection of axis for all iterations - each set will have log(axis_n) samples
        elif max_axis_selector == 'log2':

            log_ = int(np.log(axis_n))

            idx_samples = self.sample_axis(
                axis_n=axis_n,
                sample_iter=self.sample_iter,
                num_samples=log_
            )

        # percentile selection of axis for all iterations - each set will have .1*axis_n samples
        elif max_axis_selector == 'percentile':

            percentile_ = int(axis_n * .1)

            idx_samples = self.sample_axis(
                axis_n=axis_n,
                sample_iter=self.sample_iter,
                num_samples=percentile_
            )

        # random selection of axis - each set will have different size samples
        elif max_axis_selector == 'random':

            # grab a set of random numbers sample iter items wide
            random_ = random.sample(range(int(axis_n * self.random_axis_max_pct)), self.sample_iter)

            # grab list of randomly sizes axis indexes
            idx_samples = [list(random.sample(range(axis_n), i)) for i in random_]

        else:
            raise ValueError("Invalid parameters. Valid parameters are ['sqrt', 'log2', 'percentile', 'random']")

        return idx_samples

    @staticmethod
    def normalize_input_data(x):
        """
        Normalize column axis by subtracting mean and dividing by standard deviation.

        Parameters
        ----------
        x : array of data [rows, cols]

        Returns
        -------
        array
        """

        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)

        x -= mean
        x /= std

        return x

    @staticmethod
    def scale_input_data(x):
        """
        Scale columns by subtracting column from column min and dividing by column max minus column min.
        Standard min-max scaling.

        Parameters
        ----------
        x : array of data [rows, cols]

        Returns
        -------
        array
        """
        max_ = np.amax(x, axis=0)
        min_ = np.amin(x, axis=0)
        denominator = max_ - min_

        x -= min_
        x /= denominator

        return x

    def fit_random_neighbors(self, x, cluster=None):
        """
        Recursively fit clustering algorithm using bootstrapped rows and columns for each fit.
        Output the best scores and a history objectRoute the sample_axis method through sample type options.
        If custom list provided, build a list of columns based on size provided in the list.
        If no custom list provided, use one of [sqrt, log2, percentile, random] to build list of sampled axis indexes

        Parameters
        ----------
        x : array of data [rows, cols] to cluster
        cluster: sci-kit learn cluster object

        Returns
        -------
        tuple
        """

        assert isinstance(x, (np.ndarray, np.generic))
        assert x.shape[0] > 0
        assert x.shape[1] > 0
        assert isinstance(self.normalize_data, bool)
        assert isinstance(self.scale_data, bool)

        if self.normalize_data:
            x = self.normalize_input_data(x)

        if self.scale_data:
            x = self.scale_input_data(x)

        # define global storage to hold best parameters
        best_metric = 0.0
        best_metric_iter = 0.0
        history_dict = {}

        # build the bootstrap of columns, rows
        max_rows, max_cols = x.shape
        row_samples: list = self.build_sample_index(axis_n=max_rows, max_axis_selector=self.select_rows)
        col_samples: list = self.build_sample_index(axis_n=max_cols, max_axis_selector=self.select_columns)
        bootstrap_list = [(row_samples[i], col_samples[i]) for i in range(self.sample_iter)]

        for i, v in enumerate(bootstrap_list):

            boot_rows = np.array(v[0])
            boot_cols = np.array(v[1])
            sampled_x = x[boot_rows[:, None], boot_cols]

            # fit and check outputs
            cluster_fit = cluster.fit(sampled_x)
            labels = cluster_fit.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)

            if len(set(labels)) == 1:
                print(f'No Label Differentiation - skipping iteration {i}')
                history_dict['iteration_' + str(i)] = {
                    'score': None,
                    'columns': boot_cols,
                    'labels': labels,
                    'n_clusters': n_clusters,
                    'n_noise': n_noise,
                    'fit_': cluster_fit
                }

            else:
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

                    if (sil_score > best_metric) & (n_clusters > 1.0):
                        best_metric = sil_score
                        best_metric_iter = i

        return best_metric, best_metric_iter, history_dict
