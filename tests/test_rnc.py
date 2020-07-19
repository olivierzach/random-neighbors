from rnc.random_neighbors import RandomNeighbors
import numpy as np

rnc = RandomNeighbors()


def test_rnc_sample_axis(axis_n=1000, num_samples=35, sample_iter=20):
    res = rnc.sample_axis(
        axis_n=axis_n,
        num_samples=num_samples,
        sample_iter=sample_iter
    )

    assert isinstance(res, list)
    assert len(res) == sample_iter

    for s in res:
        assert len(s) == num_samples
        assert np.max(s) <= axis_n

    return True


def test_rnc_build_sample_index(axis_n=1000, max_axis_selector='log2', sample_iter=20):

    res = rnc.build_sample_index(
        axis_n=axis_n,
        max_axis_selector=max_axis_selector
    )

    assert isinstance(res, list)
    assert len(res) == sample_iter

    for s in res:
        if max_axis_selector == 'log2':
            assert len(s) == int(np.log(axis_n))

        if max_axis_selector == 'sqrt':
            assert len(s) == int(np.sqrt(axis_n))

        if max_axis_selector == 'percentile':
            assert len(s) == int(axis_n * .1)

        if max_axis_selector == 'random':
            assert len(s) <= int(axis_n * .2)

    return True
