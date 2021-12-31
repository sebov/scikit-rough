# import numpy as np

# import pandas as pd
# import pytest

# TODO: more tests

# @pytest.mark.parametrize('distribution,expected', [
#     ([[2, 0], [0, 0], [1, 1]], [True, True, False]),
#     ([[1, 1], [3, 2], [2, 2]], [False, False, False]),
#     ])
# def test_compute_homogeneity(distribution, expected):
#     distribution = np.asarray(distribution)
#     result = compute_homogeneity(distribution)
#     assert all([
#         np.array_equal(result, expected)
#         ])
#
#
# @pytest.mark.parametrize('group_index,values,expected', [
#     ([0, 0, 0, 0], [0, 0, 0, 42], ([0, 0, 0, 1], 2)),
#     ([0, 1, 1, 1], [0, 1, 0, 1], ([0, 1, 2, 1], 3)),
#     ([0, 1, 0, 1], [0, 0, 1, 1], ([0, 1, 2, 3], 4)),
#     ([5, 4, 3, 2, 1, 0], [0, 1, 0, 1, 0, 1], ([0, 1, 2, 3, 4, 5], 6)),
#     ])
# def test_split_groups(group_index, values, expected):
#     group_index = np.asarray(group_index)
#     n_groups = group_index.max() + 1
#     factorized_values, uniques = pd.factorize(values)
#     expected_group_index, expected_n_groups = expected
#     expected_group_index = np.asarray(expected_group_index)
#     group_index, n_groups = split_groups(group_index,
#                                          n_groups,
#                                          factorized_values,
#                                          len(uniques),
#                                          compress_group_index=True)
#     assert all([np.array_equal(group_index, expected_group_index),
#                 np.array_equal(n_groups, expected_n_groups),
#                 ])
#
#
