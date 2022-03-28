from typing import List, Union

import numpy as np
import numpy.typing as npt
import pandas.core.sorting
from attrs import define


@define
class GroupIndex:
    index: npt.NDArray[np.int64]
    count: int

    @classmethod
    def create_empty(cls):
        return cls(
            index=np.empty(0, dtype=np.int64),
            count=0,
        )

    @classmethod
    def create_one_group(cls, size):
        return cls(
            index=np.zeros(size, dtype=np.int64),
            count=1,
        )

    @classmethod
    def create_from_index(cls, index: npt.ArrayLike):
        index = np.asarray(index, dtype=np.int64)
        return cls(
            index=index,
            count=len(np.unique(index)),
        )

    def compress(self):
        index, uniques = pandas.core.sorting.compress_group_index(
            self.index,
            sort=False,
        )
        self.index = index
        self.count = len(uniques)
        return self


ReductLike = Union["Reduct", List[int]]


@define
class Reduct:
    attrs: List[int]

    @classmethod
    def create_from(cls, reduct_like: ReductLike):
        attrs = reduct_like.attrs if isinstance(reduct_like, Reduct) else reduct_like
        return cls(list(attrs))


@define
class Bireduct(Reduct):
    objs: List[int]
