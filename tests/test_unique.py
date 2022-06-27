import numpy as np
import pytest

from skrough.unique import get_uniques_index


@pytest.mark.parametrize(
    "values, expected",
    [
        ([1, 2, 3, 4], [0, 1, 2, 3]),
        ([4, 3, 2, 1], [3, 2, 1, 0]),
        ([1, 1, 1, 1], [0]),
        ([], []),
        ([1, 1, 2, 1], [0, 2]),
        ([2, 2, 1, 2], [2, 0]),
        ([1, 2, 1, 2, 1, 2], [0, 1]),
    ],
)
def test_get_uniques_index(values, expected):
    values = np.asarray(values)
    expected = np.asarray(expected)
    assert np.array_equal(get_uniques_index(values), expected)
