import numpy as np
import pytest

from skrough.weights import normalize_weights


@pytest.mark.parametrize(
    "weights, expected",
    [
        ([10], [1]),
        ([0], [1]),
        ([], []),
        ([10, 10, 10, 10], [0.25, 0.25, 0.25, 0.25]),
        ([1, 1, 1, 1], [0.25, 0.25, 0.25, 0.25]),
        ([1, 1, 0], [0.5, 0.5, 0]),
        ([0, 0, 0], [1 / 3, 1 / 3, 1 / 3]),
        ([0, 1, 0], [0, 1, 0]),
    ],
)
def test_normalize_weights(weights, expected):
    normalized = normalize_weights(weights)
    # all should be > 0
    assert (normalized > 0).all()
    # should be close to expected
    assert np.allclose(normalized, expected)
