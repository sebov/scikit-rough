import numpy as np


def get_uniques_index(values: np.ndarray) -> np.ndarray:
    """Get positions of first occurrences of unique values.

    Get positions/indices for which unique values in the input array appear for the
    first time. The indices are reported in the order corresponding to the ascending
    order of unique values, i.e., the first index indicates the first occurrence of the
    lowest unique value, the second index indicates the first occurrence of the second
    lowest unique value, etc.

    Args:
        values: Input array.

    Returns:
        The positions/indices in the input array for which unique values (reported in
        ascending order) appear for the first time.

    Examples:
        >>> get_uniques_index(np.array([1, 2, 3]))
        array([0, 1, 2])
        >>> get_uniques_index(np.array([3, 2, 1]))
        array([2, 1, 0])
        >>> get_uniques_index(np.array([1, 1, 1]))
        array([0])
        >>> get_uniques_index(np.array([1, 1, 2, 1]))
        array([0, 2])
        >>> get_uniques_index(np.array([2, 2, 1, 2]))
        array([2, 0])
        >>> get_uniques_index(np.array([]))
        array([])
    """
    _, idx = np.unique(values, return_index=True)
    return idx
