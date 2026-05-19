"""Group index implementation using pure numpy (no numba dependency)."""

import numpy as np
import numpy.typing as npt

from skrough.structs.group_index._base import GroupIndexBase


def _get_distribution(
    groups: npt.NDArray[np.int64],
    groups_count: int,
    values: npt.NDArray[np.int64],
    values_count: int,
) -> npt.NDArray[np.int64]:
    """Compute decision distribution within groups of objects.

    Uses ``numpy.add.at`` for unbuffered in-place addition, which is
    vectorized and avoids Python-level loops entirely.
    """
    result = np.zeros((groups_count, values_count), dtype=np.int64)
    np.add.at(result, (groups, values), 1)
    return result


class GroupIndexPure(GroupIndexBase):
    """Group index with pure-numpy distribution computation.

    Uses ``numpy.add.at`` instead of a numba-jitted loop. Suitable as a
    baseline for benchmarking numba speedups or for environments where
    numba is unavailable.
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
