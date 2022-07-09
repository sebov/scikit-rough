from typing import Sequence, Union

import numba
import numpy as np
import numpy.typing as npt
import pandas.core.sorting
from attrs import define

import skrough.typing as rght
from skrough.typing_utils import unify_attrs
from skrough.utils import minmax


@numba.njit
def _get_distribution(
    groups: npt.NDArray[np.int64],
    groups_count: int,
    values: np.ndarray,
    values_count: int,
) -> npt.NDArray[np.int64]:
    """
    Compute decision distribution within groups of objects
    """
    result = np.zeros((groups_count, values_count), dtype=np.int64)
    nrow = groups.shape[0]
    for i in range(nrow):
        result[groups[i], values[i]] += 1
    return result


@define
class GroupIndex:
    index: npt.NDArray[np.int64]
    """index that assigns objects (by their positions in the index) to groups"""
    n_groups: int
    """number of groups"""

    @property
    def size(self) -> int:
        """Number of objects described by this group index.

        Returns:
            Number of objects.
        """
        return len(self.index)

    @classmethod
    def create_empty(cls) -> "GroupIndex":
        return cls(
            index=np.empty(0, dtype=np.int64),
            n_groups=0,
        )

    @classmethod
    def create_uniform(cls, size: int) -> "GroupIndex":
        if size < 0:
            raise ValueError("Size less than zero")

        if size == 0:
            result = cls.create_empty()
        else:
            result = cls(
                index=np.zeros(size, dtype=np.int64),
                n_groups=1,
            )
        return result

    @classmethod
    def create_from_index(
        cls,
        index: Union[Sequence[int], np.ndarray],
        compress: bool = False,
    ) -> "GroupIndex":
        index = np.asarray(index, dtype=np.int64)
        if len(index) == 0:
            result = cls.create_empty()
        else:
            _min, _max = minmax(index)
            if _min < 0:
                raise ValueError("Index value less than zero")
            result = cls(
                index=index,
                n_groups=_max + 1,
            )
        if compress:
            result = result.compress()
        return result

    @classmethod
    def create_from_data(
        cls,
        x: np.ndarray,
        x_counts: np.ndarray,
        attrs: rght.AttrsLike,
    ):
        """
        Split objects into groups according to values on given attributes
        """
        unified_attrs = unify_attrs(attrs)
        if len(unified_attrs) == 0:
            result = cls.create_uniform(size=len(x))
        else:
            result = cls.create_empty()
            result.index = pandas.core.sorting.get_group_index(
                labels=x[:, unified_attrs].T,
                shape=x_counts[unified_attrs],
                sort=False,
                xnull=False,
            )
            result = result.compress()
        return result

    def split(
        self,
        values: np.ndarray,
        values_count: int,
        compress: bool = True,
    ) -> "GroupIndex":
        """
        Split groups of objects into finer groups according to values on
        a single splitting attribute
        """
        result = self.create_empty()
        result.index = self.index * values_count + values
        result.n_groups = self.n_groups * values_count
        if compress:
            result = result.compress()
        return result

    def compress(self) -> "GroupIndex":
        result = self.create_empty()
        index, uniques = pandas.core.sorting.compress_group_index(
            self.index,
            sort=False,
        )
        result.index = index
        result.n_groups = len(uniques)
        return result

    def get_distribution(
        self,
        values: np.ndarray,
        values_count: int,
    ) -> npt.NDArray[np.int64]:
        return _get_distribution(
            self.index,
            self.n_groups,
            values,
            values_count,
        )
