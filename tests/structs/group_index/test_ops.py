import numpy as np
import pandas as pd
import pytest

from skrough.structs.group_index import GroupIndex
from tests.structs.group_index.helpers import _assert_group_index


@pytest.mark.parametrize(
    "input_index, values, expected_index, expected_n_groups, compress",
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
def test_split(input_index, values, expected_index, expected_n_groups, compress):
    group_index = GroupIndex.create_from_index(input_index)
    factorized_values, uniques = pd.factorize(values)
    result = group_index.split(
        factorized_values,
        len(uniques),
        compress=compress,
    )
    _assert_group_index(result, expected_index, expected_n_groups)


@pytest.mark.parametrize(
    "index, expected_index, expected_n_groups",
    [
        ([], [], 0),
        ([0, 1, 2], [0, 1, 2], 3),
        ([10, 20, 30], [0, 1, 2], 3),
    ],
)
def test_compress(index, expected_index, expected_n_groups):
    group_index = GroupIndex.create_from_index(index)
    result = group_index.compress()
    _assert_group_index(result, expected_index, expected_n_groups)


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
def test_get_distribution(index, values, expected_distribution):
    group_index = GroupIndex.create_from_index(index, compress=True)
    factorized_values, uniques = pd.factorize(values)
    result = group_index.get_distribution(factorized_values, len(uniques))
    assert np.array_equal(result, expected_distribution)
