import numpy as np
import pytest

from skrough.dataprep import prepare_factorized_vector
from skrough.structs.group_index import GroupIndex
from tests.structs.group_index.helpers import _assert_group_index


@pytest.mark.parametrize(
    "group_index, values",
    [
        (GroupIndex.create_empty(), []),
        (GroupIndex.create_uniform(0), []),
        (GroupIndex.create_uniform(1), [0]),
        (GroupIndex.create_uniform(3), [0, 1, 0]),
        (GroupIndex.create_uniform(3), [0, 1, 2]),
        (GroupIndex.create_from_index([]), []),
        (GroupIndex.create_from_index([1]), [2]),
        (GroupIndex.create_from_index([0, 2]), [0, 0]),
    ],
)
def test_check_values(group_index: GroupIndex, values):
    group_index._check_values(values)  # pylint: disable=protected-access


@pytest.mark.parametrize(
    "group_index, values",
    [
        (GroupIndex.create_empty(), [0]),
        (GroupIndex.create_empty(), [0, 1]),
        (GroupIndex.create_uniform(0), [0]),
        (GroupIndex.create_uniform(0), [1, 0]),
        (GroupIndex.create_uniform(1), []),
        (GroupIndex.create_uniform(1), [0, 1]),
        (GroupIndex.create_uniform(3), []),
        (GroupIndex.create_uniform(3), [0]),
        (GroupIndex.create_uniform(3), [0, 1]),
        (GroupIndex.create_uniform(3), [0, 1, 0, 1]),
        (GroupIndex.create_from_index([]), [0]),
        (GroupIndex.create_from_index([]), [1, 0]),
        (GroupIndex.create_from_index([1]), []),
        (GroupIndex.create_from_index([1]), [1, 2]),
        (GroupIndex.create_from_index([0, 2]), []),
        (GroupIndex.create_from_index([0, 2]), [0]),
        (GroupIndex.create_from_index([0, 2]), [1, 0, 0]),
    ],
)
def test_check_values_mismatch(group_index: GroupIndex, values):
    with pytest.raises(ValueError, match="length does not match the group index"):
        group_index._check_values(values)  # pylint: disable=protected-access


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
    values, values_count = prepare_factorized_vector(np.asarray(values))
    result = group_index.split(
        values,
        values_count,
        compress=compress,
    )
    _assert_group_index(result, expected_index, expected_n_groups)


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
def test_split_mismatch(input_index, values):
    group_index = GroupIndex.create_from_index(input_index)
    values, values_count = prepare_factorized_vector(np.asarray(values))
    with pytest.raises(ValueError, match="length does not match the group index"):
        group_index.split(
            values,
            values_count,
        )


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
