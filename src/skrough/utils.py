"""Utils module.

The :mod:`skrough.utils` module delivers miscellaneous functions related to values, positions and
indices.
"""

from typing import Any, List, Tuple

import numba
import numpy as np


@numba.njit(cache=True)
def get_positions_where_values_in(
    values: np.ndarray,
    reference: np.ndarray,
) -> List[int]:
    """Find indices of elements that match any value from a reference collection.

    Compute positions for which ``values`` elements are in the ``reference`` collection. It is
    equivalent to ``np.isin(values, reference).nonzero()[0]`` but should be faster for larger
    ``reference`` collections.

    Args:
        values: An input collection of values for which the check if its elements are in the
            ``reference`` collection.
        reference: A collection of reference values for which the ``values`` elements are checked
            against.

    Returns:
        A collection of indices for which the elements of ``values`` on the given positions are in
        the ``reference`` collection.

    Examples:
        >>> get_positions_where_values_in(np.asarray([2, 7, 1, 8]), np.asarray([1, 8]))
        [2, 3]
        >>> get_positions_where_values_in(np.asarray([1, 2, 1, 1]), np.asarray([1, 8]))
        [0, 2, 3]
    """
    reference_set = set(reference)
    return [i for i in range(len(values)) if values[i] in reference_set]
