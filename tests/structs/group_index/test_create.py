import numpy as np
import pytest

from skrough.structs.group_index import GroupIndex
from tests.helpers import generate_data
from tests.structs.group_index.helpers import (
    _assert_empty_group_index,
    _assert_group_index,
)


def test_create_empty():
    result = GroupIndex.create_empty()
    _assert_empty_group_index(result)


@pytest.mark.parametrize(
    "n_objs",
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
def test_group_index_from_index(index, compress, expected_index, expected_n_groups):
    result = GroupIndex.from_index(index, compress=compress)
    _assert_group_index(result, expected_index, expected_n_groups)

    index2 = np.asarray(index)
    result2 = GroupIndex.from_index(index2, compress=compress)
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
def test_group_index_from_index_wrong_args(index):
    with pytest.raises(ValueError, match="less than zero"):
        GroupIndex.from_index(index)


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
                [0, 0],
                [0, 1],
                [1, 0],
            ],
            [2, 2],
            None,
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
        (
            [
                [0, 0],
                [1, 0],
                [2, 1],
            ],
            [3, 2],
            None,
            [0, 1, 2],
            3,
        ),
    ],
)
def test_group_index_from_data(x, x_counts, attrs, expected_index, expected_n_groups):
    x = np.array(x)
    x_counts = np.array(x_counts)
    result = GroupIndex.from_data(x, x_counts, attrs)
    _assert_group_index(result, expected_index, expected_n_groups)


@pytest.mark.parametrize(
    "x, x_counts, attrs",
    [
        (
            generate_data(size=(0, 0)),
            [],
            [],
        ),
        (
            generate_data(size=(0, 1)),
            [],
            [],
        ),
        (
            generate_data(size=(0, 2)),
            [],
            [],
        ),
        (
            generate_data(size=(0, 3)),
            [],
            [],
        ),
        (
            generate_data(size=(0, 2)),
            [2, 3],
            [0, 1],
        ),
    ],
)
def test_group_index_from_data_empty(x, x_counts, attrs):
    x = np.array(x)
    x_counts = np.array(x_counts)
    result = GroupIndex.from_data(x, x_counts, attrs)
    _assert_empty_group_index(result)
