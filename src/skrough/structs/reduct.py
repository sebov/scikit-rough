from typing import List, Union

from attrs import define

ReductLike = Union["Reduct", List[int]]


@define
class Reduct:
    attrs: List[int]

    @classmethod
    def create_from(cls, reduct_like: ReductLike):
        attrs = reduct_like.attrs if isinstance(reduct_like, Reduct) else reduct_like
        return cls(list(attrs))
