import numpy as np
import pytest

from skrough.chaos_measures import conflicts_number, entropy, gini_impurity
from skrough.dataprep import prepare_factorized_values
from skrough.structs.group_index import GroupIndex


@pytest.mark.parametrize(
    "index, values, expected_distribution",
    [
        (
            [0, 0, 0, 0],
            [0, 0, 0, 42],
            [
                [3, 1],
            ],
        ),
        (
            [0, 1, 1, 1],
            [0, 1, 0, 1],
            [
                [1, 0],
                [1, 2],
            ],
        ),
        (
            [0, 1, 0, 1],
            [0, 0, 1, 1],
            [
                [1, 1],
                [1, 1],
            ],
        ),
        (
            [0, 0, 1, 1],
            [0, 0, 0, 1],
            [
                [2, 0],
                [1, 1],
            ],
        ),
        (
            [0, 0, 1, 1],
            [0, 1, 2, 3],
            [
                [1, 1, 0, 0],
                [0, 0, 1, 1],
            ],
        ),
        (
            [0, 2, 0, 3],
            [0, 1, 2, 3],
            [
                [1, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ],
        ),
        (
            [5, 4, 3, 2, 1, 0],
            [0, 1, 0, 1, 0, 1],
            [
                [1, 0],
                [0, 1],
                [1, 0],
                [0, 1],
                [1, 0],
                [0, 1],
            ],
        ),
    ],
)
def test_get_distribution_and_chaos_score(index, values, expected_distribution):
    group_index = GroupIndex.create_from_index(index, compress=True)
    y, y_count = prepare_factorized_values(values)
    result_distribution = group_index.get_distribution(y, y_count)
    assert np.array_equal(result_distribution, expected_distribution)
    for chaos_measure in [conflicts_number, entropy, gini_impurity]:
        result_chaos_score = group_index.get_chaos_score(y, y_count, chaos_measure)
        expected_chaos_score = chaos_measure(
            result_distribution,
            result_distribution.sum(),
        )
        assert result_chaos_score == expected_chaos_score
