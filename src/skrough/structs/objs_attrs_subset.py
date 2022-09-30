"""Dual objects-attrs subset structures."""

from typing import List

from attrs import define

import skrough.typing as rght


@define
class ObjsAttrsSubset:
    """A class to represent a subset of objects and attributes.

    A class to represent a subset of objects and attributes. They are both stored
    separately in a form of integer-location based indexing sequence of objects and
    attributes, respectively.
    """

    objs: List[int]
    """Subset of objects - integer-location based indexing sequence of objects."""

    attrs: List[int]
    """Subset of attributes - integer-location based indexing sequence of attributes."""

    @classmethod
    def from_objs_attrs_like(
        cls, objs_like: rght.LocationsLike, attrs_like: rght.LocationsLike
    ):
        """Create a new instance.

        Create a new instance using the ``objs_like`` and ``attrs_like`` input values.

        Args:
            objs_like: A base value that should be used to initialize objects subset
                when creating a new instance.
            attrs_like: A base value that should be used to initialize attributes subset
                when creating a new instance.

        Returns:
            A new instance created using the ``objs_like`` and ``attrs_like`` arguments.
        """
        return cls(objs=list(objs_like), attrs=list(attrs_like))
