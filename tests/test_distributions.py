"""Test reducts"""

import numpy as np
import pytest

from skrough.distributions import get_dec_distribution
from skrough.structs.group_index import GroupIndex


@pytest.mark.parametrize(
    "group_index,factorized_dec_values,dec_values_count_distinct,expected",
    [
        ([0, 0, 0, 0], [0, 0, 0, 1], 2, ([[3, 1]], [4])),
        ([0, 0, 1, 1], [0, 0, 0, 1], 2, ([[2, 0], [1, 1]], [2, 2])),
        ([0, 0, 1, 1], [0, 1, 2, 3], 4, ([[1, 1, 0, 0], [0, 0, 1, 1]], [2, 2])),
    ],
)
def test_compute_dec_distribution(
    group_index, factorized_dec_values, dec_values_count_distinct, expected
):
    group_index = GroupIndex.create_from_index(group_index)
    factorized_dec_values = np.asarray(factorized_dec_values)
    expected_dec_distribution, expected_counts = expected
    expected_dec_distribution = np.asarray(expected_dec_distribution)
    # TODO: do we need expected_counts?
    expected_counts = np.asarray(expected_counts)
    result_dec_distribution = get_dec_distribution(
        group_index, factorized_dec_values, dec_values_count_distinct
    )
    assert all(
        [
            np.array_equal(result_dec_distribution, expected_dec_distribution),
        ]
    )
