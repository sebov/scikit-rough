import numpy as np
import pandas as pd
import pytest

from skrough.containers import GroupIndex


@pytest.mark.parametrize(
    "input_index, values, expected_group_index, expected_n_groups, compress",
    [
        (
            [0, 0, 0, 0],
            [0, 0, 0, 42],
            [0, 0, 0, 1],
            2,
            True,
        ),
        (
            [0, 1, 1, 1],
            [0, 1, 0, 1],
            [0, 1, 2, 1],
            3,
            True,
        ),
        (
            [0, 1, 0, 1],
            [0, 0, 1, 1],
            [0, 1, 2, 3],
            4,
            True,
        ),
        (
            [5, 4, 3, 2, 1, 0],
            [0, 1, 0, 1, 0, 1],
            [0, 1, 2, 3, 4, 5],
            6,
            True,
        ),
        (
            [0, 2, 0, 3],
            [0, 1, 2, 3],
            [0, 9, 2, 15],
            16,
            False,
        ),
    ],
)
def test_split_groups(
    input_index, values, expected_group_index, expected_n_groups, compress
):
    group_index = GroupIndex(
        index=np.asarray(input_index),
        count=max(input_index) + 1,
    )
    factorized_values, uniques = pd.factorize(values)
    expected_group_index = np.asarray(expected_group_index)
    group_index = group_index.split(
        factorized_values,
        len(uniques),
        compress=compress,
    )
    assert all(
        [
            np.array_equal(group_index.index, expected_group_index),
            np.array_equal(group_index.count, expected_n_groups),
        ]
    )


@pytest.mark.parametrize(
    "xx, xx_counts, attrs, expected_group_index, expected_n_groups",
    [
        (
            [
                [0, 0],
                [0, 0],
                [0, 1],
                [1, 0],
            ],
            [2, 2],
            [],
            [0, 0, 0, 0],
            1,
        ),
        (
            [
                [0, 0],
                [0, 0],
                [0, 1],
                [1, 0],
            ],
            [2, 2],
            [0],
            [0, 0, 0, 1],
            2,
        ),
        (
            [
                [0, 0],
                [0, 0],
                [0, 1],
                [1, 0],
            ],
            [2, 2],
            [1],
            [0, 0, 1, 0],
            2,
        ),
        (
            [
                [0, 0],
                [0, 0],
                [0, 1],
                [1, 0],
            ],
            [2, 2],
            [0, 1],
            [0, 0, 1, 2],
            3,
        ),
    ],
)
def test_create_from_data(
    xx, xx_counts, attrs, expected_group_index, expected_n_groups
):
    xx = np.array(xx)
    xx_counts = np.array(xx_counts)

    group_index = GroupIndex.create_from_data(xx, xx_counts, attrs)
    np.array_equal(group_index.index, expected_group_index)
    np.array_equal(group_index.count, expected_n_groups)
