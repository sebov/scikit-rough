import numba
import numpy as np


@numba.njit
def minmax(ar: np.ndarray):
    length = len(ar)
    if length == 0:
        raise ValueError("empty array specified")
    _min = _max = ar[0]
    for i in range(1, length):
        _min = min(ar[i], _min)
        _max = max(ar[i], _max)
    return _min, _max
