from typing import List

from attrs import define

from skrough.structs.reduct import Reduct


@define
class Bireduct(Reduct):
    objs: List[int]
