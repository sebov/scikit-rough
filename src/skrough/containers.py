import numpy as np
import numpy.typing as npt
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


@define
class Reduct:

    attrs: list[int]


@define
class Bireduct(Reduct):

    objects: list[int]
