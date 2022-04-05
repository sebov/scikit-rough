from typing import List, Union

from attrs import define

AttrsSubsetLike = Union["AttrsSubset", List[int]]


@define
class AttrsSubset:
    attrs: List[int]

    @classmethod
    def create_from(cls, reduct_like: AttrsSubsetLike):
        attrs = (
            reduct_like.attrs if isinstance(reduct_like, AttrsSubset) else reduct_like
        )
        return cls(list(attrs))
