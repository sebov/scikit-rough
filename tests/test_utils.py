import numpy as np
import pytest

from skrough.utils import minmax


@pytest.mark.parametrize(
    "values, expected_min_max",
    [
        ([1, 2, 3], (1, 3)),
        ([3, 1, 1, 1, -1, 1], (-1, 3)),
        ([1, 1, 1], (1, 1)),
        ([1, 1, np.inf], (1, np.inf)),
        ([-np.inf, 1, 1], (-np.inf, 1)),
        ([-np.inf, 1, np.inf], (-np.inf, np.inf)),
        ([-np.inf, 1, np.inf], (-np.inf, np.inf)),
    ],
)
def test_minmax(values, expected_min_max):
    assert minmax(np.asarray(values)) == expected_min_max


@pytest.mark.parametrize(
    "values",
    [
        [],
        [np.inf, np.nan],
        [1, 1, 1, np.nan],
    ],
)
def test_minmax_raise(values):
    with pytest.raises(ValueError):
        minmax(np.asarray(values))
