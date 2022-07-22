import numpy as np
import pytest

from skrough.chaos_measures import conflicts_number, entropy, gini_impurity
from skrough.chaos_score import get_chaos_score_for_data
from skrough.dataprep import prepare_factorized_array, prepare_factorized_vector
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
    y, y_count = prepare_factorized_vector(values)
    result_distribution = group_index.get_distribution(y, y_count)
    assert np.array_equal(result_distribution, expected_distribution)
    for chaos_measure in [conflicts_number, entropy, gini_impurity]:
        result_chaos_score = group_index.get_chaos_score(y, y_count, chaos_measure)
        expected_chaos_score = chaos_measure(
            result_distribution,
            result_distribution.sum(),
        )
        assert result_chaos_score == expected_chaos_score


@pytest.mark.parametrize(
    "input_index, values",
    [
        ([0, 0], []),
        ([0, 0], [0]),
        ([0, 0], [0, 0, 0]),
        ([0, 0, 0, 0], []),
        ([0, 0, 0, 0], [0]),
        ([0, 0, 0, 0], [0, 0, 1]),
        ([0, 0, 0, 0], [0, 0, 0, 42, 1]),
        ([0, 1, 0, 1], []),
        ([0, 1, 0, 1], [0]),
        ([0, 1, 0, 1], [0, 0, 1]),
        ([0, 1, 0, 1], [0, 0, 1, 1, 0]),
        ([0, 2, 0, 3], []),
        ([0, 2, 0, 3], [1]),
        ([0, 2, 0, 3], [1, 1]),
        ([0, 2, 0, 3], [0, 1, 2, 1, 3]),
    ],
)
def test_get_distribution_and_chaos_score_mismatch(input_index, values):
    group_index = GroupIndex.create_from_index(input_index)
    values, values_count = prepare_factorized_vector(np.asarray(values))
    with pytest.raises(ValueError, match="length does not match the group index"):
        group_index.get_distribution(
            values,
            values_count,
        )
    with pytest.raises(ValueError, match="length does not match the group index"):
        group_index.get_chaos_score(
            values,
            values_count,
            conflicts_number,
        )


@pytest.mark.parametrize(
    "data",
    [
        np.zeros(shape=(4, 3)),
        np.ones(shape=(10, 10)),
        np.empty(shape=(6, 4)),
        np.eye(5),
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 1],
            [1, 0, 0],
            [1, 0, 0],
        ],
    ],
)
def test_get_chaos_score_after_split(data):
    # let last column be the decision
    data = np.asarray(data)
    x, x_count = prepare_factorized_array(data[:, 0:-2])
    group_index = GroupIndex.create_from_data(x, x_count)

    split_values, split_values_count = prepare_factorized_vector(data[:, -2])
    values, values_count = prepare_factorized_vector(data[:, -1])
    all_x, all_x_counts = prepare_factorized_array(data[:, 0:-1])

    for chaos_measure in [conflicts_number, entropy, gini_impurity]:
        result_chaos_score = group_index.get_chaos_score_after_split(
            split_values=split_values,
            split_values_count=split_values_count,
            values=values,
            values_count=values_count,
            chaos_fun=chaos_measure,
        )

        expected_chaos_score = get_chaos_score_for_data(
            x=all_x,
            x_counts=all_x_counts,
            y=values,
            y_count=values_count,
            chaos_fun=chaos_measure,
        )

        assert result_chaos_score == expected_chaos_score
