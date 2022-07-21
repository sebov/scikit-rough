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
        ([-1], [-1]),
        ([-1, 1], [-0.5, 0.5]),
        ([-1, -1], [-0.5, -0.5]),
    ],
)
def test_normalize_weights(weights, expected):
    weights = np.asarray(weights)
    expected = np.asarray(expected)
    result = normalize_weights(weights)
    # when there are no negative values in input then result elements should be > 0
    if (weights >= 0).all():
        assert (result > 0).all()
    # result should be close to expected
    assert result.shape == expected.shape
    assert np.allclose(result, expected)


@pytest.mark.parametrize(
    "weights, size, expand_none",
    [
        (None, None, True),
        (10, None, True),
        (10, None, False),
        (4.5, None, True),
        (4.5, None, False),
        (None, -1, True),
        (None, -10, True),
        (10, -1, True),
        (10, -10, True),
        (10, -1, False),
        (10, -10, False),
        (4.5, -1, True),
        (4.5, -10, True),
        (4.5, -1, False),
        (4.5, -10, False),
    ],
)
def test_prepare_weights_wrong_size(weights, size, expand_none):
    with pytest.raises(ValueError, match="`size` cannot be"):
        prepare_weights(weights=weights, size=size, expand_none=expand_none)


@pytest.mark.parametrize(
    "size, normalize",
    [
        (0, False),
        (0, True),
        (1, False),
        (1, True),
        (2, False),
        (2, True),
        (10, False),
        (10, True),
    ],
)
def test_prepare_weights_none(size, normalize):
    result = prepare_weights(
        weights=None,
        size=size,
        expand_none=False,
        normalize=normalize,
    )
    assert result is None


@pytest.mark.parametrize(
    "weights, size, normalize, expected",
    [
        (None, 0, False, []),
        (None, 0, True, []),
        (None, 1, False, [1]),
        (None, 1, True, [1]),
        (None, 2, False, [1, 1]),
        (None, 2, True, [0.5, 0.5]),
        (0, 0, False, []),
        (0, 0, True, []),
        (2, 0, False, []),
        (2, 0, True, []),
        (0.5, 0, False, []),
        (0.5, 0, True, []),
        (2.5, 0, False, []),
        (2.5, 0, True, []),
        (0, 1, False, [0]),
        (0, 1, True, [1]),
        (0, 2, False, [0, 0]),
        (0, 2, True, [0.5, 0.5]),
        (1, 1, False, [1]),
        (1, 1, True, [1]),
        (1, 2, False, [1, 1]),
        (1, 2, True, [0.5, 0.5]),
        (2, 2, False, [2, 2]),
        (2, 2, True, [0.5, 0.5]),
        (2, 3, False, [2, 2, 2]),
        (2, 3, True, [1 / 3, 1 / 3, 1 / 3]),
        (10, 5, False, [10, 10, 10, 10, 10]),
        (10, 5, True, [0.2, 0.2, 0.2, 0.2, 0.2]),
        (0.0, 1, False, [0.0]),
        (0.0, 1, True, [1.0]),
        (0.0, 2, False, [0.0, 0.0]),
        (0.0, 2, True, [0.5, 0.5]),
        (1.5, 1, False, [1.5]),
        (1.5, 1, True, [1]),
        (1.5, 2, False, [1.5, 1.5]),
        (1.5, 2, True, [0.5, 0.5]),
        (2.5, 2, False, [2.5, 2.5]),
        (2.5, 2, True, [0.5, 0.5]),
        (2.5, 3, False, [2.5, 2.5, 2.5]),
        (2.5, 3, True, [1 / 3, 1 / 3, 1 / 3]),
        (10.5, 5, False, [10.5, 10.5, 10.5, 10.5, 10.5]),
        (10.5, 5, True, [0.2, 0.2, 0.2, 0.2, 0.2]),
        (-2, 0, False, []),
        (-2, 0, True, []),
        (-0.5, 0, False, []),
        (-0.5, 0, True, []),
        (-2.5, 0, False, []),
        (-2.5, 0, True, []),
        (-1, 1, False, [-1]),
        (-1, 1, True, [-1]),
        (-1, 2, False, [-1, -1]),
        (-1, 2, True, [-0.5, -0.5]),
        (-2, 2, False, [-2, -2]),
        (-2, 2, True, [-0.5, -0.5]),
        (-2, 3, False, [-2, -2, -2]),
        (-2, 3, True, [-1 / 3, -1 / 3, -1 / 3]),
        (-10, 5, False, [-10, -10, -10, -10, -10]),
        (-10, 5, True, [-0.2, -0.2, -0.2, -0.2, -0.2]),
        (-1.5, 1, False, [-1.5]),
        (-1.5, 1, True, [-1]),
        (-1.5, 2, False, [-1.5, -1.5]),
        (-1.5, 2, True, [-0.5, -0.5]),
        (-2.5, 2, False, [-2.5, -2.5]),
        (-2.5, 2, True, [-0.5, -0.5]),
        (-2.5, 3, False, [-2.5, -2.5, -2.5]),
        (-2.5, 3, True, [-1 / 3, -1 / 3, -1 / 3]),
        (-10.5, 5, False, [-10.5, -10.5, -10.5, -10.5, -10.5]),
        (-10.5, 5, True, [-0.2, -0.2, -0.2, -0.2, -0.2]),
    ],
)
def test_prepare_weights_scalar(weights, size, normalize, expected):
    expected = np.asarray(expected)
    result = prepare_weights(
        weights=weights,
        size=size,
        expand_none=True,
        normalize=normalize,
    )
    assert result.shape == expected.shape
    assert np.allclose(result, expected)


@pytest.mark.parametrize(
    "weights, size, normalize, expected",
    [
        ([1], 1, False, [1]),
        ([1], 4, False, [1]),
        ([2], 1, False, [2]),
        ([2], 4, False, [2]),
        ([-1], 4, False, [-1]),
        ([1, 2, 3], 3, False, [1, 2, 3]),
        ([1, -2, 3], 3, False, [1, -2, 3]),
        ([-1, -2, -3], None, False, [-1, -2, -3]),
        ([1], 1, True, [1]),
        ([1], 4, True, [1]),
        ([2], 1, True, [1]),
        ([2], 4, True, [1]),
        ([-1], 1, True, [-1]),
        ([-1], 4, True, [-1]),
        ([-2], 1, True, [-1]),
        ([-2], 4, True, [-1]),
        ([1, 2, 7], None, True, [0.1, 0.2, 0.7]),
        ([1, 3, 0], None, True, [0.25, 0.75, 0]),
        ([0, 0, 0], None, True, [1 / 3, 1 / 3, 1 / 3]),
        ([-1, 2, 7], None, True, [-0.1, 0.2, 0.7]),
        ([1, -3, 0], None, True, [0.25, -0.75, 0]),
    ],
)
def test_prepare_weights_array(weights, size, normalize, expected):
    weights = np.asarray(weights)
    expected = np.asarray(expected)
    result = prepare_weights(
        weights=weights,
        size=size,
        normalize=normalize,
    )
    assert result.shape == expected.shape
    assert np.allclose(result, expected)
