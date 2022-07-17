from typing import List

from attrs import define


@define
class ObjsAttrsSubset:
    objs: List[int]
    attrs: List[int]
