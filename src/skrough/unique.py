"""Unique-related operations.

The :mod:`skrough.unique` module delivers helper functions for unique-related
computations. Currently all operations are simple wrappers around :func:`np.unique` but
they are here to provide interfaces that the rest of the code uses.
"""

from typing import Tuple

import numpy as np


def get_rows_nunique(x: np.ndarray) -> int:
    """Compute the number of unique rows.

    Compute the number of unique rows. Degenerated tables are handled accordingly,
    i.e., a table with no columns has 1 unique rows if only it has at least one row,
    otherwise it is 0.

    Args:
        x: Input data table.

    Returns:
        Number of unique rows.
    """
    return np.unique(x, axis=0).shape[0]


def get_uniques_positions(values: np.ndarray) -> np.ndarray:
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
    _, idx = get_uniques_and_positions(values)
    return idx


# TODO: add docstring
def get_uniques_and_positions(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    uniques, uniques_index = np.unique(values, return_index=True)
    return uniques, uniques_index


# TODO: add docstring
def get_uniques_and_compacted(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    uniques, uniques_inverse = np.unique(values, return_inverse=True)
    return uniques, uniques_inverse
