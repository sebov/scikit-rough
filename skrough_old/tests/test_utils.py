'''Test reducts'''

import numpy as np
import pandas as pd
import pytest

from skrough.utils.group_index import compute_dec_distribution, compute_homogeneity, split_groups, draw_objects
from skrough.metrics.gini_impurity import gini_impurity


@pytest.mark.parametrize('distribution,expected', [
    ([[4]], 0),
    ([[1, 1]], 2 / 4),
    ([[1, 1, 1]], 6 / 9),
    ([[4, 3, 3]], 66 / 100),
    ([[3, 1]], 6 / 16),
    ([[2, 0], [1, 1]], (0 / 4) * (2 / 4) + (2 / 4) * (2 / 4)),
    ([[1, 1, 0, 0], [0, 0, 1, 1]], (2 / 4) * (2 / 4) + (2 / 4) * (2 / 4)),
    ([[1, 1, 1, 1], [0, 0, 1, 1]], (12 / 16) * (4 / 6) + (2 / 4) * (2 / 6)),
    ([[0, 0], [1, 1]], 0 * (0 / 2) + (2 / 4) * (2 / 2)),
    ([[0, 1], [5, 1], [3, 5]], (0 / 1) * (0 / 15) + (10 / 36) * (6 / 15) + (30 / 64) * (8 / 15)),
    ([[1, 1]] * 10, 2 / 4),
    ([[9999, 1]], (10000 * 10000 - 9999 * 9999 - 1 * 1) / (10000 * 10000)),
    ])
def test_compute_gini_impurity(distribution, expected):
    distribution = np.asarray(distribution)
    n = distribution.sum()
    result = gini_impurity(distribution, n)
    assert np.allclose(result, expected)


@pytest.mark.parametrize('group_index,factorized_dec_values,dec_values_count_distinct,expected', [
    ([0, 0, 0, 0], [0, 0, 0, 1], 2, ([[3, 1]], [4])),
    ([0, 0, 1, 1], [0, 0, 0, 1], 2, ([[2, 0], [1, 1]], [2, 2])),
    ([0, 0, 1, 1], [0, 1, 2, 3], 4, ([[1, 1, 0, 0], [0, 0, 1, 1]], [2, 2])),
    ])
def test_compute_dec_distribution(group_index,
                                  factorized_dec_values,
                                  dec_values_count_distinct,
                                  expected):
    group_index = np.asarray(group_index)
    n_groups = group_index.max() + 1
    factorized_dec_values = np.asarray(factorized_dec_values)
    expected_dec_distribution, expected_counts = expected
    expected_dec_distribution = np.asarray(expected_dec_distribution)
    # TODO: is expected_counts needed?
    expected_counts = np.asarray(expected_counts)
    result_dec_distribution = compute_dec_distribution(group_index,
                                                       n_groups,
                                                       factorized_dec_values,
                                                       dec_values_count_distinct)
    assert all([
        np.array_equal(result_dec_distribution, expected_dec_distribution),
        ])


@pytest.mark.parametrize('distribution,expected', [
    ([[2, 0], [0, 0], [1, 1]], [True, True, False]),
    ([[1, 1], [3, 2], [2, 2]], [False, False, False]),
    ])
def test_compute_homogeneity(distribution, expected):
    distribution = np.asarray(distribution)
    result = compute_homogeneity(distribution)
    assert all([
        np.array_equal(result, expected)
        ])


@pytest.mark.parametrize('group_index,values,expected', [
    ([0, 0, 0, 0], [0, 0, 0, 42], ([0, 0, 0, 1], 2)),
    ([0, 1, 1, 1], [0, 1, 0, 1], ([0, 1, 2, 1], 3)),
    ([0, 1, 0, 1], [0, 0, 1, 1], ([0, 1, 2, 3], 4)),
    ([5, 4, 3, 2, 1, 0], [0, 1, 0, 1, 0, 1], ([0, 1, 2, 3, 4, 5], 6)),
    ])
def test_split_groups(group_index, values, expected):
    group_index = np.asarray(group_index)
    n_groups = group_index.max() + 1
    factorized_values, uniques = pd.factorize(values)
    expected_group_index, expected_n_groups = expected
    expected_group_index = np.asarray(expected_group_index)
    group_index, n_groups = split_groups(group_index,
                                         n_groups,
                                         factorized_values,
                                         len(uniques),
                                         compress_group_index=True)
    assert all([np.array_equal(group_index, expected_group_index),
                np.array_equal(n_groups, expected_n_groups),
                ])


@pytest.mark.parametrize('group_index,dec_values,permutation,expected', [
    ([0, 1, 2, 1, 1, 1, 2, 2, 0, 0],
     [0, 0, 0, 1, 0, 0, 1, 1, 1, 0],
     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
     [0, 1, 2, 4, 5, 9]),
    ([0, 1, 2, 1, 1, 1, 2, 2, 0, 0],
     [0, 0, 0, 1, 0, 0, 1, 1, 1, 0],
     [0, 3, 2, 1, 4, 5, 6, 7, 8, 9],
     [0, 2, 3, 9]),
    ([0, 1, 2, 1, 1, 1, 2, 2, 0, 0],
     [0, 1, 1, 1, 1, 0, 1, 1, 1, 1],
     [0, 3, 2, 1, 4, 5, 6, 7, 8, 9],
     [0, 1, 2, 3, 4, 6, 7]),
    ])
def test_draw_objects(group_index, dec_values, permutation, expected):
    group_index = np.asarray(group_index)
    dec_values = np.asarray(dec_values)
    permutation = np.asarray(permutation)
    expected = np.asarray(expected)
    result = draw_objects(group_index, dec_values, permutation)
    assert all([
        np.array_equal(result, expected)
        ])
