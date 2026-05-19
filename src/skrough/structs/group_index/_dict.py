"""Group index using per-group dictionaries and on-the-fly compactification.

Instead of storing a flat index array and building a full distribution
matrix, this implementation maintains a dict ``group_key -> list[object_id]``
for each group.  During ``split`` a helper map ``M`` translates
``(old_key * attr_max + attr_val)`` into fresh sequential keys,
compacting groups implicitly without a post-hoc ``compress`` pass.

Disorder scores are computed by iterating group dicts and calling the
disorder function on per-group 1xV rows -- no full ``n_groups x n_values``
matrix is ever materialised, and no ``argsort`` is needed (the dict
already delineates group boundaries).
"""

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pandas.core.sorting

import skrough.typing as rght
from skrough.structs.group_index._base import GroupIndexBase
from skrough.unify import unify_index_list


@dataclass
class GroupIndexDict(GroupIndexBase):
    """Dict-based group index with implicit compactification.

    Maintains a sparse ``_groups: dict[int, list[int]]`` mapping each
    sequential group key to the list of object indices belonging to that
    group.  The ``split`` method uses a helper dict ``M`` that maps
    ``old_key * attr_max + attr_val`` to fresh sequential keys, so the
    resulting group keys are always compact (0 .. n_groups-1).  No
    explicit ``compress`` is needed.

    ``get_disorder_score`` iterates the group dicts directly, building
    per-group 1xV rows and passing each to the disorder function with
    ``n_elements = n_objs``.  This avoids both the full matrix allocation
    and the ``argsort`` required by the hash-based streaming
    implementations.
    """

    _groups: dict[int, list[int]] | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self._groups is None:
            self._rebuild_groups()

    def _rebuild_groups(self):
        """Rebuild the sparse groups dict from ``self.index``."""
        groups: dict[int, list[int]] = defaultdict(list)
        for i, g in enumerate(self.index):
            groups[int(g)].append(i)
        self._groups = dict(groups)

    @classmethod
    def from_data(
        cls,
        x: npt.NDArray[np.int64],
        x_counts: npt.NDArray[np.int64],
        attrs: rght.IndexListLike | None = None,
    ):
        if attrs is None:
            attrs = range(x.shape[1])
        unified_attrs = unify_index_list(attrs)
        if len(unified_attrs) == 0:
            return cls.create_uniform(size=len(x))

        raw_index = pandas.core.sorting.get_group_index(
            labels=x[:, unified_attrs].T,
            shape=x_counts[unified_attrs],
            sort=False,
            xnull=False,
        )
        index, uniques = pandas.core.sorting.compress_group_index(
            raw_index,
            sort=False,
        )
        n_groups = len(uniques)

        groups: dict[int, list[int]] = defaultdict(list)
        for i, g in enumerate(index):
            groups[int(g)].append(i)

        result = cls(index=index, n_groups=n_groups)
        result._groups = dict(groups)
        return result

    def split(
        self,
        values: npt.NDArray[np.int64],
        values_count: int,
        compress: bool = False,
    ):
        self._check_values(values)

        new_groups: dict[int, list[int]] = {}
        M: dict[int, int] = {}
        max_key = 0
        new_index = np.zeros(self.n_objs, dtype=np.int64)

        for old_key, obj_indices in self._groups.items():
            offset = old_key * values_count
            for i in obj_indices:
                v = int(values[i])
                tmp = offset + v
                if tmp in M:
                    nk = M[tmp]
                else:
                    nk = max_key
                    M[tmp] = nk
                    new_groups[nk] = []
                    max_key += 1
                new_groups[nk].append(i)
                new_index[i] = nk

        result = type(self)(index=new_index, n_groups=max_key)
        result._groups = new_groups
        return result

    def compress(self):
        """No-op -- group keys are always sequential."""
        return type(self)(index=self.index.copy(), n_groups=self.n_groups)

    def get_distribution(
        self,
        values: npt.NDArray[np.int64],
        values_count: int,
    ) -> npt.NDArray[np.int64]:
        self._check_values(values)
        result = np.zeros((self.n_groups, values_count), dtype=np.int64)
        for group_key, obj_indices in self._groups.items():
            group_vals = values[obj_indices]
            result[group_key] = np.bincount(group_vals, minlength=values_count)
        return result

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

        total: float = 0.0
        for obj_indices in self._groups.values():
            if not obj_indices:
                continue

            group_vals = values[obj_indices]
            counts = np.bincount(group_vals, minlength=values_count)

            per_group_row = counts.reshape(1, -1)
            total += disorder_fun(per_group_row, n)

        return total
