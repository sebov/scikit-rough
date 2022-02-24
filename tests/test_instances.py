import numpy as np
import pytest

from skrough.containers import GroupIndex
from skrough.instances import choose_objects


@pytest.mark.parametrize(
    "group_index,dec_values,permutation,expected",
    [
        (
            [0, 1, 2, 1, 1, 1, 2, 2, 0, 0],
            [0, 0, 0, 1, 0, 0, 1, 1, 1, 0],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [0, 1, 2, 4, 5, 9],
        ),
        (
            [0, 1, 2, 1, 1, 1, 2, 2, 0, 0],
            [0, 0, 0, 1, 0, 0, 1, 1, 1, 0],
            [0, 3, 2, 1, 4, 5, 6, 7, 8, 9],
            [0, 2, 3, 9],
        ),
        (
            [0, 1, 2, 1, 1, 1, 2, 2, 0, 0],
            [0, 1, 1, 1, 1, 0, 1, 1, 1, 1],
            [0, 3, 2, 1, 4, 5, 6, 7, 8, 9],
            [0, 1, 2, 3, 4, 6, 7],
        ),
    ],
)
def test_draw_objects(group_index, dec_values, permutation, expected):
    group_index = GroupIndex.create_from_index(np.asarray(group_index))
    dec_values = np.asarray(dec_values)
    dec_values_count = len(np.unique(dec_values))
    permutation = np.asarray(permutation)
    expected = np.asarray(expected)
    result = choose_objects(group_index, dec_values, dec_values_count, permutation)
    assert all([np.array_equal(result, expected)])


@pytest.mark.parametrize(
    "group_index,dec_values,expected",
    [
        (
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        ),
        (
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        ),
    ],
)
def test_draw_objects_random(group_index, dec_values, expected):
    group_index = GroupIndex.create_from_index(np.asarray(group_index))
    dec_values = np.asarray(dec_values)
    dec_values_count = len(np.unique(dec_values))
    expected = np.asarray(expected)
    result = choose_objects(group_index, dec_values, dec_values_count, objs=None)
    assert all([np.array_equal(result, expected)])


def test_draw_objects_random_2():
    group_index = GroupIndex.create_from_index(
        np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    )
    dec_values = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    dec_values_count = len(np.unique(dec_values))
    result = choose_objects(group_index, dec_values, dec_values_count, objs=None)
    assert len(result) == 2
    assert result[0] < 5
    assert 5 <= result[1] < 10
