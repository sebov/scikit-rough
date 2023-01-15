"""Unify utils module."""

import numpy as np

import skrough.typing as rght


def unify_locations(items: rght.LocationsLike) -> rght.Locations:
    """Unify locations-like input.

    Unify locations-like input, i.e., :class:`numpy.ndarray` or a sequence of
    integer-location based indexing of objects/attributes/elements to a common form of
    :class:`~skrough.typing.Locations`.

    Args:
        items: Locations-like input to be unified into a common form of
        :class:`~skrough.typing.Locations`.

    Returns:
        Unified locations.

    Examples:
        >>> unify_locations(np.arange(5))
        array([0, 1, 2, 3, 4])
        >>> unify_locations([1,2,3])
        array([1, 2, 3])
        >>> unify_locations([])
        array([], dtype=int64)
    """
    return np.asarray(items, dtype=np.int64)
