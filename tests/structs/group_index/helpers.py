import numpy as np

from skrough.structs.group_index import GroupIndex


def _assert_group_index(group_index: GroupIndex, expected_index, expected_n_groups):
    assert np.array_equal(group_index.index, expected_index)
    assert group_index.n_objs == len(expected_index)
    assert group_index.n_groups == expected_n_groups


def _assert_empty_group_index(group_index: GroupIndex):
    _assert_group_index(group_index, [], 0)
