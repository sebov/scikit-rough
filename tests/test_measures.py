"""Test reducts"""

import numpy as np
import pytest

from skrough.measures import entropy, gini_impurity


@pytest.mark.parametrize(
    "distribution,expected",
    [
        ([[4]], 0),
        ([[1, 1]], 2 / 4),
        ([[1, 1, 1]], 6 / 9),
        ([[4, 3, 3]], 66 / 100),
        ([[3, 1]], 6 / 16),
        ([[2, 0], [1, 1]], (0 / 4) * (2 / 4) + (2 / 4) * (2 / 4)),
        ([[1, 1, 0, 0], [0, 0, 1, 1]], (2 / 4) * (2 / 4) + (2 / 4) * (2 / 4)),
        ([[1, 1, 1, 1], [0, 0, 1, 1]], (12 / 16) * (4 / 6) + (2 / 4) * (2 / 6)),
        ([[0, 0], [1, 1]], 0 * (0 / 2) + (2 / 4) * (2 / 2)),  # NOSONAR
        (
            [[0, 1], [5, 1], [3, 5]],
            (0 / 1) * (0 / 15) + (10 / 36) * (6 / 15) + (30 / 64) * (8 / 15),
        ),
        ([[1, 1]] * 10, 2 / 4),
        ([[9999, 1]], (10000 * 10000 - 9999 * 9999 - 1 * 1) / (10000 * 10000)),
    ],
)
def test_compute_gini_impurity(distribution, expected):
    distribution = np.asarray(distribution)
    n = distribution.sum()
    result = gini_impurity(distribution, n)
    assert np.allclose(result, expected)


# TODO: add more tests for entropy
@pytest.mark.parametrize(
    "distribution,expected",
    [
        (
            [[4]],
            0,
        ),
        (
            [[1, 1]],
            1,
        ),
        (
            [
                [0, 2],
                [1, 1],
            ],
            0.5,
        ),
        (
            [
                [0, 0],
                [1, 1],
            ],
            1,
        ),
    ],
)
def test_compute_entropy(distribution, expected):
    distribution = np.asarray(distribution)
    n = distribution.sum()
    result = entropy(distribution, n)
    assert np.allclose(result, expected)
