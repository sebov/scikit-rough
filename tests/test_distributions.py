import numpy as np
import pytest

from skrough.distributions import get_dec_distribution
from skrough.structs.group_index import GroupIndex


@pytest.mark.parametrize(
    "group_index, y, y_count, expected",
    [
        ([0, 0, 0, 0], [0, 0, 0, 1], 2, [[3, 1]]),
        ([0, 0, 1, 1], [0, 0, 0, 1], 2, [[2, 0], [1, 1]]),
        ([0, 0, 1, 1], [0, 1, 2, 3], 4, [[1, 1, 0, 0], [0, 0, 1, 1]]),
    ],
)
def test_get_dec_distribution(group_index, y, y_count, expected):
    group_index = GroupIndex.create_from_index(group_index)
    y = np.asarray(y)
    expected = np.asarray(expected)
    dec_distribution = get_dec_distribution(group_index, y, y_count)
    assert all(
        [
            np.array_equal(dec_distribution, expected),
        ]
    )
