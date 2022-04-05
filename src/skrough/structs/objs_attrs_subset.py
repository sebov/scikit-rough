from typing import List

from attrs import define

from skrough.structs.attrs_subset import AttrsSubset


@define
class ObjsAttrsSubset(AttrsSubset):
    objs: List[int]
