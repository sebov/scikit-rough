"""Attrs subset structures."""

from typing import List, Union

from attrs import define

import skrough.typing as rght


@define
class AttrsSubset:
    """A class to represent a subset of attributes.

    A class to represent a subset of attributes. They are stored in a form of
    integer-location based indexing sequence of attributes.
    """

    attrs: List[int]
    """Subset of attributes - integer-location based indexing sequence of attributes."""

    @classmethod
    def from_attrs_like(
        cls, attrs_subset_like: Union["AttrsSubset", rght.LocationsLike]
    ):
        """Create a new instance.

        Create a new instance using the ``attrs_subset_like`` input value.

        Args:
            attrs_subset_like: A base value that should be used to create a new
                instance.

        Returns:
            A new instance created using the ``attrs_subset_like`` argument.
        """
        attrs = (
            attrs_subset_like.attrs
            if isinstance(attrs_subset_like, AttrsSubset)
            else attrs_subset_like
        )
        return cls(attrs=list(attrs))
