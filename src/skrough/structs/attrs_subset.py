from typing import List, Union

from attrs import define

import skrough.typing as rght


@define
class AttrsSubset:
    attrs: List[int]

    @classmethod
    def create_from(cls, attrs_subset_like: Union["AttrsSubset", rght.Attrs]):
        attrs = (
            attrs_subset_like.attrs
            if isinstance(attrs_subset_like, AttrsSubset)
            else attrs_subset_like
        )
        return cls(list(attrs))
