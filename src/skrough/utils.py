from typing import Any, Tuple

import numba
import numpy as np

EMPTY_ARRAY_MSG = "Empty array specified"
NAN_VALUES_PRESENT_MSG = "There is a nan value present in the array"


@numba.njit
def minmax(values: np.ndarray) -> Tuple[Any, Any]:
    """Returns min and max values.

    Returns min and max values from a given array. Checks if the array is not empty
    and does not contain ``nan`` values.

    Args:
        ar: Input array.

    Raises:
        ValueError: When the input array is empty or contains nan values.

    Returns:
        A pair of min and max values from the input array.
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
def get_positions_where_values_in(values: np.ndarray, reference: np.ndarray):
    """Get positions for which values are in the reference collection.

    Get positions for which values are in the reference collection. It is equivalent to
    `np.isin(values, reference).nonzero()[0]`.

    Args:
        values: A collection of values for which to check if its elements are in the
            reference collection.
        reference: A collection of reference values that the values are checked against.

    Returns:
        A collection of indices for which a value on the given position is in
        the reference collection.
    """
    reference_set = set(reference)
    return [i for i in range(len(values)) if values[i] in reference_set]
