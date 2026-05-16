import numpy as np

from rnc.random_neighbors import RandomNeighbors


def test_build_sample_index_shapes(axis_n=1000, sample_iter=20):
    rn = RandomNeighbors(sample_iter=sample_iter, random_state=0)

    for selector in ["log2", "sqrt", "percentile", "random"]:
        res = rn.build_sample_index(axis_n=axis_n, max_axis_selector=selector)
        assert isinstance(res, list)
        assert len(res) == sample_iter

        for s in res:
            assert isinstance(s, list)
            assert 1 <= len(s) <= axis_n
            assert np.max(s) < axis_n

            if selector == "log2":
                assert len(s) == max(1, int(np.log2(axis_n)))
            if selector == "sqrt":
                assert len(s) == max(1, int(np.sqrt(axis_n)))
            if selector == "percentile":
                assert len(s) == max(1, int(axis_n * 0.1))
            if selector == "random":
                assert len(s) <= max(1, int(axis_n * 0.2))
