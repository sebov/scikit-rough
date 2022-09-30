"""Unify utils module."""

import numpy as np

import skrough.typing as rght


def unify_locations(objs: rght.LocationsLike) -> rght.Locations:
    """Unify locations-like input.

    Unify locations-like input, i.e., :class:`numpy.ndarray` or a sequence of
    integer-location based indexing of objects/attributes/elements to common form
    of :class:`~skrough.typing.Locations`.

    Args:
        objs: _description_

    Returns:
        _description_
    """
    return np.asarray(objs, dtype=np.int64)
