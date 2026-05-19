"""Group index implementation using numba-accelerated distribution."""

import numba
import numpy as np
import numpy.typing as npt

from skrough.structs.group_index._base import GroupIndexBase


@numba.njit
def _get_distribution(
    groups: npt.NDArray[np.int64],
    groups_count: int,
    values: npt.NDArray[np.int64],
    values_count: int,
) -> npt.NDArray[np.int64]:
    """Compute decision distribution within groups of objects."""
    result = np.zeros(shape=(groups_count, values_count), dtype=np.int64)
    nrow = groups.shape[0]
    for i in range(nrow):
        result[groups[i], values[i]] += 1
    return result


class GroupIndexNumba(GroupIndexBase):
    """Group index with numba-accelerated distribution computation.

    Uses ``@numba.njit`` for the inner distribution loop, providing
    significant speedups on large datasets.
    """

    def get_distribution(
        self,
        values: npt.NDArray[np.int64],
        values_count: int,
    ) -> npt.NDArray[np.int64]:
        self._check_values(values)
        return _get_distribution(
            self.index,
            self.n_groups,
            values,
            values_count,
        )
