import numpy as np
import pytest

from skrough.instances import choose_objects
from skrough.structs.group_index import GroupIndex


@pytest.mark.parametrize(
    "group_index,dec_values,objs_permutation,expected_all,expected_representatives",
    [
        (
            [0, 1, 2, 1, 1, 1, 2, 2, 0, 0],
            [0, 0, 0, 1, 0, 0, 1, 1, 1, 0],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [0, 1, 2, 4, 5, 9],
            [0, 1, 2],
        ),
        (
            [0, 1, 2, 1, 1, 1, 2, 2, 0, 0],
            [0, 0, 0, 1, 0, 0, 1, 1, 1, 0],
            [0, 3, 2, 1, 4, 5, 6, 7, 8, 9],
            [0, 2, 3, 9],
            [0, 2, 3],
        ),
        (
            [0, 1, 2, 1, 1, 1, 2, 2, 0, 0],
            [0, 1, 1, 1, 1, 0, 1, 1, 1, 1],
            [0, 3, 2, 1, 4, 5, 6, 7, 8, 9],
            [0, 1, 2, 3, 4, 6, 7],
            [0, 2, 3],
        ),
        (
            [],
            [],
            [],
            [],
            [],
        ),
    ],
)
def test_choose_objects(
    group_index, dec_values, objs_permutation, expected_all, expected_representatives
):
    group_index = GroupIndex.create_from_index(np.asarray(group_index))
    dec_values = np.asarray(dec_values)
    dec_values_count = len(np.unique(dec_values))
    objs_permutation = np.asarray(objs_permutation)
    expected_all = np.asarray(expected_all)
    expected_representatives = np.asarray(expected_representatives)
    result_all = choose_objects(
        group_index,
        dec_values,
        dec_values_count,
        objs_permutation,
    )
    assert np.array_equal(result_all, expected_all)
    result_representatives = choose_objects(
        group_index,
        dec_values,
        dec_values_count,
        objs_permutation,
        return_representatives_only=True,
    )
    assert np.array_equal(result_representatives, expected_representatives)


@pytest.mark.parametrize(
    "group_index,dec_values,expected_all,expected_representatives",
    [
        (
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        ),
        (
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            None,
        ),
        (
            [],
            [],
            [],
            [],
        ),
    ],
)
def test_choose_objects_random(
    group_index, dec_values, expected_all, expected_representatives
):
    group_index = GroupIndex.create_from_index(np.asarray(group_index))
    dec_values = np.asarray(dec_values)
    # TODO: shouldn't prepare_factorized_x be used here instead?, i.e., in its current
    # form where there are no gaps present and it starts from 0 this looks ok
    dec_values_count = len(np.unique(dec_values))
    expected_all = np.asarray(expected_all)
    result_all = choose_objects(
        group_index,
        dec_values,
        dec_values_count,
    )
    assert np.array_equal(result_all, expected_all)
    if expected_representatives is not None:
        expected_representatives = np.asarray(expected_representatives)
        result_representatives = choose_objects(
            group_index,
            dec_values,
            dec_values_count,
            return_representatives_only=True,
        )
        assert np.array_equal(result_representatives, expected_representatives)


def test_choose_objects_random_2():
    group_index = GroupIndex.create_from_index(
        np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    )
    dec_values = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # TODO: shouldn't prepare_factorized_x be used here instead?, i.e., in its current
    # form where there are no gaps present and it starts from 0 this looks ok
    dec_values_count = len(np.unique(dec_values))
    result = choose_objects(group_index, dec_values, dec_values_count)
    assert len(result) == 2
    assert result[0] < 5
    assert 5 <= result[1] < 10
