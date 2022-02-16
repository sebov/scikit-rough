import numpy as np
from attrs import define


@define
class GroupIndex:

    index: np.ndarray
    count: np.int64

    @classmethod
    def create_empty(cls):
        return cls(
            index=np.empty(0, dtype=np.int64),
            count=np.int64(0),
        )

    @classmethod
    def create_one_group(cls, size):
        return cls(
            index=np.zeros(size, dtype=np.int64),
            count=np.int64(1),
        )


@define
class Reduct:

    attrs: list[int]


@define
class Bireduct(Reduct):

    objects: list[int]
