import numpy as np


def get_uniques_index(values: np.ndarray) -> np.ndarray:
    """Get positions of first occurrences of unique values.

    Get positions in the input array for which the given unique values appeared
    for the first time. Unique values are considered in the ascending order of their
    values, i.e., from the lowest to the highest unique values occurring in the input
    array, positions of their first occurrence is returned.

    Args:
        values: Input array.

    Returns:
        The positions in the input array for which given values appeared for
        the first time.

    Examples:
        >>> get_uniques_index(np.array([1, 2, 3]))
        [0, 1, 2]
        >>> get_uniques_index(np.array([3, 2, 1]))
        [2, 1, 0]
        >>> get_uniques_index(np.array([1, 1, 1]))
        [0]
    """
    _, idx = np.unique(values, return_index=True)
    return idx
