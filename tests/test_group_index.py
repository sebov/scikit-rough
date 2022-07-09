import numpy as np
import pandas as pd
import pytest

from skrough.structs.group_index import GroupIndex


def _assert_group_index(group_index: GroupIndex, expected_index, expected_n_groups):
    assert np.array_equal(group_index.index, expected_index)
    assert group_index.n_objs == len(expected_index)
    assert group_index.n_groups == expected_n_groups


def _assert_empty_group_index(group_index: GroupIndex):
    _assert_group_index(group_index, [], 0)


def test_create_empty():
    result = GroupIndex.create_empty()
    _assert_empty_group_index(result)


@pytest.mark.parametrize(
    "size",
    [0, 1, 2, 5, 10],
)
def test_create_uniform(n_objs):
    result = GroupIndex.create_uniform(n_objs)
    if n_objs == 0:
        _assert_empty_group_index(result)
    else:
        assert np.array_equal(result.index, np.repeat(0, n_objs))
        assert result.n_objs == n_objs
        assert result.n_groups == 1


@pytest.mark.parametrize(
    "size",
    [-1, -10],
)
def test_create_uniform_wrong_size(size):
    with pytest.raises(ValueError, match="less than zero"):
        GroupIndex.create_uniform(size)


@pytest.mark.parametrize(
    "index, compress, expected_index, expected_n_groups",
    [
        ([], False, [], 0),
        ([], True, [], 0),
        ([0], False, [0], 1),
        ([0], True, [0], 1),
        ([0, 0, 0], False, [0, 0, 0], 1),
        ([0, 0, 0], True, [0, 0, 0], 1),
        ([0, 1, 2], False, [0, 1, 2], 3),
        ([0, 1, 2], True, [0, 1, 2], 3),
        ([1, 2, 3], False, [1, 2, 3], 4),
        ([1, 2, 3], True, [0, 1, 2], 3),
        ([1], False, [1], 2),
        ([1], True, [0], 1),
        ([1, 9, 0], False, [1, 9, 0], 10),
        ([1, 9, 0], True, [0, 1, 2], 3),
        ((1, 9, 0), False, [1, 9, 0], 10),
        ((1, 9, 0), True, [0, 1, 2], 3),
    ],
)
def test_create_from_index(index, compress, expected_index, expected_n_groups):
    result = GroupIndex.create_from_index(index, compress=compress)
    _assert_group_index(result, expected_index, expected_n_groups)

    index2 = np.asarray(index)
    result2 = GroupIndex.create_from_index(index2, compress=compress)
    _assert_group_index(result2, expected_index, expected_n_groups)


@pytest.mark.parametrize(
    "index",
    [
        [-1],
        [-10],
        [-1, -2],
        [0, 1, 2, -10],
        [-1, 0, 1, 2, 3],
    ],
)
def test_create_from_index_wrong_args(index):
    with pytest.raises(ValueError, match="less than zero"):
        GroupIndex.create_from_index(index)


@pytest.mark.parametrize(
    "x, x_counts, attrs, expected_index, expected_n_groups",
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
        (
            [
                [0, 0],
                [1, 0],
                [2, 1],
            ],
            [3, 2],
            [0, 1],
            [0, 1, 2],
            3,
        ),
    ],
)
def test_create_from_data(x, x_counts, attrs, expected_index, expected_n_groups):
    x = np.array(x)
    x_counts = np.array(x_counts)
    result = GroupIndex.create_from_data(x, x_counts, attrs)
    _assert_group_index(result, expected_index, expected_n_groups)


@pytest.mark.parametrize(
    "x, x_counts, attrs",
    [
        (
            np.empty(shape=(0, 0)),
            [],
            [],
        ),
        (
            np.empty(shape=(0, 1)),
            [],
            [],
        ),
        (
            np.empty(shape=(0, 2)),
            [],
            [],
        ),
        (
            np.empty(shape=(0, 3)),
            [],
            [],
        ),
        (
            np.empty(shape=(0, 2)),
            [2, 3],
            [0, 1],
        ),
    ],
)
def test_create_from_data_empty(x, x_counts, attrs):
    x = np.array(x)
    x_counts = np.array(x_counts)
    result = GroupIndex.create_from_data(x, x_counts, attrs)
    _assert_empty_group_index(result)


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
