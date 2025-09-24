"""Objects-attrs subset structures."""

from dataclasses import dataclass
from typing import List

import skrough.typing as rght


@dataclass
class ObjsAttrsSubset:
    """Represents a subset of objects and attributes.

    A class to represent a subset of objects and attributes. The subsets are defined by
    two separate lists of 0-based integer indices, one for objects and one for
    attributes.
    """

    objs: List[int]
    """A subset of objects, defined by a list of 0-based integer indices."""

    attrs: List[int]
    """A subset of attributes, defined by a list of 0-based integer indices."""

    @classmethod
    def from_objs_attrs_like(
        cls, objs_like: rght.IndexListLike, attrs_like: rght.IndexListLike
    ):
        """Create a new instance.

        Args:
            objs_like: An input value to initialize the object subset.
            attrs_like: An input value to initialize the attrs subset.

        Returns:
            A subset of objects and attributes.
        """
        return cls(objs=list(objs_like), attrs=list(attrs_like))
