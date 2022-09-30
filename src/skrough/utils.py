"""Utils module.

The :mod:`skrough.utils` module delivers miscellaneous functions related to values,
positions and indices.
"""

from typing import Any, List, Tuple

import numba
import numpy as np

EMPTY_ARRAY_MSG = "Empty array specified"
NAN_VALUES_PRESENT_MSG = "There is a nan value present in the array"


@numba.njit
def minmax(values: np.ndarray) -> Tuple[Any, Any]:
    """Return ``min`` and ``max`` values.

    Return ``min`` and ``max`` values from a given array. The function operates on
    non-empty arrays which do not contain :obj:`numpy.nan` values.

    Args:
        values: Input array.

    Returns:
        A pair of ``min`` and ``max`` values from the input array.

    Raises:
        ValueError: When the input array is empty or contains :obj:`numpy.nan` value.
    """
    length = len(values)
    if length == 0:
        raise ValueError(EMPTY_ARRAY_MSG)
    if np.isnan(values[0]):
        raise ValueError(NAN_VALUES_PRESENT_MSG)
    _min = _max = values[0]
    for i in range(1, length):
        if np.isnan(values[i]):
            raise ValueError(NAN_VALUES_PRESENT_MSG)
        _min = min(_min, values[i])
        _max = max(_max, values[i])
    return _min, _max


@numba.njit
def get_positions_where_values_in(
    values: np.ndarray,
    reference: np.ndarray,
) -> List[int]:
    """Compute positions for which values are in the ``reference`` collection.

    Compute positions for which ``values`` elements are in the ``reference`` collection.
    It is equivalent to ``np.isin(values, reference).nonzero()[0]`` but should be faster
    for larger ``reference`` collections.

    Args:
        values: An input collection of values for which the check if its elements are in
            the ``reference`` collection.
        reference: A collection of reference values for which the ``values`` elements
            are checked against.

    Returns:
        A collection of indices for which the elements of ``values`` on the given
        positions are in the ``reference`` collection.

    Examples:
        >>> get_positions_where_values_in(np.asarray([2, 7, 1, 8]), np.asarray([1, 8]))
        [2, 3]
        >>> get_positions_where_values_in(np.asarray([1, 2, 1, 1]), np.asarray([1, 8]))
        [0, 1, 3]
    """
    reference_set = set(reference)
    return [i for i in range(len(values)) if values[i] in reference_set]
