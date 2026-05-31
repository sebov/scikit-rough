import numpy as np
import pytest

from skrough.utils import get_positions_where_values_in


def get_positions_where_values_in_alternative_impl(
    values: np.ndarray,
    reference: np.ndarray,
):
    return np.isin(values, reference).nonzero()[0]


@pytest.mark.parametrize(
    "values, reference",
    [
        ([], []),
        ([], [10, 20]),
        ([10, 20], []),
        ([2, 7, 1, 8, 2, 8, 1], [1, 2]),
        ([2, 7, 1, 8, 2, 8, 1], [1]),
        ([2, 7, 1, 8, 2, 8, 1], [1, 8]),
        ([2, 7, 1, 8, 2, 8, 1], [1, 10]),
        ([2, 7, 1, 8, 2, 8, 1], [10, 20]),
    ],
)
def test_get_positions_where_values_in(values, reference):
    values = np.asarray(values)
    reference = np.asarray(reference)
    assert np.array_equal(
        get_positions_where_values_in(values, reference),
        get_positions_where_values_in_alternative_impl(values, reference),
    )
