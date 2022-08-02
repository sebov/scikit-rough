from typing import List

from attrs import define

import skrough.typing as rght


@define
class ObjsAttrsSubset:
    objs: List[int]
    attrs: List[int]

    @classmethod
    def from_objs_attrs_like(cls, objs_like: rght.ObjsLike, attrs_like: rght.AttrsLike):
        return cls(objs=list(objs_like), attrs=list(attrs_like))
