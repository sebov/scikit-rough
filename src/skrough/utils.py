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
