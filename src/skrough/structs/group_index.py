import numpy as np
import numpy.typing as npt
import pandas.core.sorting
from attrs import define

import skrough.typing as rght
from skrough.distributions import get_values_distribution
from skrough.typing_utils import unify_attrs
from skrough.utils import minmax


@define
class GroupIndex:
    index: npt.NDArray[np.int64]
    """index that assigns objects (by their positions in the index) to groups"""
    count: int
    """number of groups"""

    @property
    def n_objects(self) -> int:
        """Number of objects described by this group index.

        Returns:
            Number of objects.
        """
        return len(self.index)

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
    def create_from_index(cls, index: npt.ArrayLike, compress: bool = False):
        index = np.asarray(index, dtype=np.int64)
        if len(index) == 0:
            raise ValueError("Empty index specified")
        _min, _max = minmax(index)
        if _min < 0:
            raise ValueError("Index value less than zero")
        result = cls(
            index=index,
            count=_max + 1,
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
        if not unified_attrs:
            result = cls.create_one_group(size=len(x))
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
    ):
        """
        Split groups of objects into finer groups according to values on
        a single splitting attribute
        """
        result = self.create_empty()
        result.index = self.index * values_count + values
        result.count = self.count * values_count
        if compress:
            result = result.compress()
        return result

    def compress(self):
        result = self.create_empty()
        index, uniques = pandas.core.sorting.compress_group_index(
            self.index,
            sort=False,
        )
        result.index = index
        result.count = len(uniques)
        return result

    def get_distribution(
        self,
        values: np.ndarray,
        values_count: int,
    ) -> npt.NDArray[np.int64]:
        return get_values_distribution(
            self.index,
            self.count,
            values,
            values_count,
        )
