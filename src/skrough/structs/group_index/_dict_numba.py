"""Dict-based group index with numba-jitted disorder-score computation.

Inherits the dict-based ``split`` with on-the-fly compactification from
:class:`GroupIndexDict` but replaces ``get_disorder_score`` with a
numa-jitted streaming implementation.  Objects are sorted by group index
and the per-group counting + disorder-function calls run entirely inside
numa, eliminating the Python-numba boundary crossings that dominate the
pure-Python dict iteration.
"""

import numpy as np
import numpy.typing as npt

import numba

import skrough.typing as rght
from skrough.structs.group_index._dict import GroupIndexDict


@numba.njit(cache=True)
def _streaming_disorder(
    sorted_groups: npt.NDArray[np.int64],
    sorted_values: npt.NDArray[np.int64],
    values_count: int,
    n_objs: int,
    disorder_fun: rght.DisorderMeasure,
) -> float:
    """Streaming disorder score from pre-sorted index and values.

    Iterates contiguous groups inside a numb-jitted loop, calling
    ``disorder_fun`` on per-group 1xV rows.  Because the entire loop is
    compiled, there is no Python overhead per group and no numba boundary
    crossing per ``disorder_fun`` call.
    """
    total = 0.0
    i = 0
    while i < n_objs:
        j = i + 1
        while j < n_objs and sorted_groups[j] == sorted_groups[i]:
            j += 1

        cnt = np.zeros(values_count, dtype=np.int64)
        for k in range(i, j):
            cnt[sorted_values[k]] += 1

        total += disorder_fun(cnt.reshape(1, -1), n_objs)
        i = j
    return total


class GroupIndexDictNumba(GroupIndexDict):
    """Dict-based group index with numba-jitted disorder computation.

    ``split`` uses the same dict-based algorithm as
    :class:`GroupIndexDict` (implicit compactification via the ``M``
    helper dict).  ``get_disorder_score`` sorts objects by group index
    and runs the per-group counting and disorder-function calls entirely
    inside a ``@numba.njit``-compiled loop.
    """

    def get_disorder_score(
        self,
        values: npt.NDArray[np.int64],
        values_count: int,
        disorder_fun: rght.DisorderMeasure,
    ) -> rght.DisorderMeasureReturnType:
        self._check_values(values)

        n = self.n_objs
        if n == 0:
            return disorder_fun(
                np.zeros((0, values_count), dtype=np.int64),
                n,
            )

        order = np.argsort(self.index)
        return _streaming_disorder(
            self.index[order],
            values[order],
            values_count,
            n,
            disorder_fun,
        )

    def get_distribution(
        self,
        values: npt.NDArray[np.int64],
        values_count: int,
    ) -> npt.NDArray[np.int64]:
        """Build distribution matrix from the groups dict."""
        self._check_values(values)
        result = np.zeros((self.n_groups, values_count), dtype=np.int64)
        for group_key, obj_indices in self._groups.items():
            group_vals = values[obj_indices]
            result[group_key] = np.bincount(group_vals, minlength=values_count)
        return result
