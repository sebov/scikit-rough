from typing import Optional, Sequence, Union

import numba
import numpy as np
import numpy.typing as npt
import pandas.core.sorting
from attrs import define

import skrough.typing as rght
from skrough.unify import unify_locations
from skrough.utils import minmax


@numba.njit
def _get_distribution(
    groups: npt.NDArray[np.int64],
    groups_count: int,
    values: npt.NDArray[np.int64],
    values_count: int,
) -> npt.NDArray[np.int64]:
    """
    Compute decision distribution within groups of objects
    """
    result = np.zeros(shape=(groups_count, values_count), dtype=np.int64)
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
    def n_objs(self) -> int:
        """Number of objects described by this group index.

        Returns:
            Number of objects.
        """
        return len(self.index)

    @classmethod
    def create_empty(cls) -> "GroupIndex":
        return cls(
            index=np.empty(shape=0, dtype=np.int64),
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
                index=np.zeros(shape=size, dtype=np.int64),
                n_groups=1,
            )
        return result

    @classmethod
    def from_index(
        cls,
        index: Union[Sequence[int], npt.NDArray[np.int64]],
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
    def from_data(
        cls,
        x: npt.NDArray[np.int64],
        x_counts: npt.NDArray[np.int64],
        attrs: Optional[rght.LocationsLike] = None,
    ):
        """
        Split objects into groups according to values on given attributes
        """
        if attrs is None:
            attrs = range(x.shape[1])
        unified_attrs = unify_locations(attrs)
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

    def _check_values(self, values):
        if len(values) != self.n_objs:
            raise ValueError("Values vector length does not match the group index")

    def split(
        self,
        values: npt.NDArray[np.int64],
        values_count: int,
        compress: bool = False,
    ) -> "GroupIndex":
        """
        Split groups of objects into finer groups according to values on
        a single splitting attribute

        It is up to the user to ensure that ``values_count`` correctly represents
        ``values``. Otherwise, the behavior is unspecified.

        """
        self._check_values(values)

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
        values: npt.NDArray[np.int64],
        values_count: int,
    ) -> npt.NDArray[np.int64]:
        """
        It is up to the user to ensure that ``values_count`` correctly represents
        ``values``. Otherwise, the behavior is unspecified.
        """
        self._check_values(values)

        return _get_distribution(
            self.index,
            self.n_groups,
            values,
            values_count,
        )

    def get_chaos_score(
        self,
        values: npt.NDArray[np.int64],
        values_count: int,
        chaos_fun: rght.ChaosMeasure,
    ) -> rght.ChaosMeasureReturnType:
        """
        Compute chaos score for the given grouping of objects (into equivalence
        classes).

        It is up to the user to ensure that ``values_count`` correctly represents
        ``values``. Otherwise, the behavior is unspecified.
        """
        self._check_values(values)

        distribution = self.get_distribution(values, values_count)
        return chaos_fun(distribution, self.n_objs)

    def get_chaos_score_after_split(
        self,
        split_values: npt.NDArray[np.int64],
        split_values_count: int,
        values: npt.NDArray[np.int64],
        values_count: int,
        chaos_fun: rght.ChaosMeasure,
    ):
        split_group_index = self.split(split_values, split_values_count, compress=False)
        return split_group_index.get_chaos_score(values, values_count, chaos_fun)
