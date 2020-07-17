from rnc.random_neighbors import RandomNeighbors
import numpy as np

rnc = RandomNeighbors()

# TODO: explicit test suite for all rnc methods

max_cols = 2000
max_rows = 1000000
col_samples = rnc.build_sample_index(axis_n=max_cols, max_axis_selector='log2')
print(col_samples)
print(len(col_samples))

row_samples = rnc.build_sample_index(axis_n=max_rows, max_axis_selector='random')
print(row_samples)
print(len(row_samples))

for i in row_samples:
    print(len(i))

len(row_samples[1])

bootstrap_list = [(row_samples[i], col_samples[i]) for i in range(50)]
len(bootstrap_list[0][0])

for i in bootstrap_list:
    print(i[0])
    print(i[1])


a = np.random.rand(max_rows, max_cols)
a.shape
row_idx = np.array(row_samples[15])
col_idx = np.array(col_samples[15])

sampled_a = a[row_idx[:, None], col_idx]
print(sampled_a)
print(sampled_a.shape)