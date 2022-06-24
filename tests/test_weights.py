import numpy as np
import pytest

from skrough.weights import normalize_weights, prepare_weights


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


@pytest.mark.parametrize(
    "weights, size, expand_none",
    [
        (None, None, True),
        (10, None, True),
        (10, None, False),
        (4.5, None, True),
        (4.5, None, False),
    ],
)
def test_prepare_weights_wrong_args(weights, size, expand_none):
    with pytest.raises(ValueError):
        prepare_weights(weights=weights, size=size, expand_none=expand_none)
