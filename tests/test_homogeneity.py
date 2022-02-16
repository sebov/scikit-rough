"""Test reducts"""

import numpy as np
import pytest

from skrough.homogeneity import compute_homogeneity


@pytest.mark.parametrize(
    "distribution,expected",
    [
        ([[2, 0], [0, 0], [1, 1]], [True, True, False]),
        ([[1, 1], [3, 2], [2, 2]], [False, False, False]),
    ],
)
def test_compute_homogeneity(distribution, expected):
    distribution = np.asarray(distribution)
    result = compute_homogeneity(distribution)
    assert all([np.array_equal(result, expected)])
