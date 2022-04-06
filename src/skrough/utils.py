from typing import Any, Tuple

import numba
import numpy as np

EMPTY_ARRAY_MSG = "Empty array specified"
NAN_VALUES_PRESENT_MSG = "There is a nan value present in the array"


@numba.njit
def minmax(ar: np.ndarray) -> Tuple[Any, Any]:
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
    length = len(ar)
    if length == 0:
        raise ValueError(EMPTY_ARRAY_MSG)
    if np.isnan(ar[0]):
        raise ValueError(NAN_VALUES_PRESENT_MSG)
    _min = _max = ar[0]
    for i in range(1, length):
        if np.isnan(ar[i]):
            raise ValueError(NAN_VALUES_PRESENT_MSG)
        _min = min(_min, ar[i])
        _max = max(_max, ar[i])
    return _min, _max
