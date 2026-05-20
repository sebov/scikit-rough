"""Group index storing concatenated attribute-value strings as ``index``.

Instead of maintaining sequential integer group IDs, this implementation
stores the raw concatenated string representation of each object's
attribute values in ``self.index`` (``npt.NDArray[np.str_]``). Group
membership is computed lazily, on demand, by hashing each concatenated
string with xxhash and mapping unique hashes to sequential IDs.

``split`` is thin: it merely appends ``#value`` to each string.
Actual hash-based group-identifier derivation happens only inside
``get_distribution`` / ``get_disorder_score``.
"""

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import numpy.typing as npt
import xxhash

import skrough.typing as rght
from skrough.structs.group_index._base import GroupIndexBase
from skrough.unify import unify_index_list


@dataclass
class GroupIndexLazy(GroupIndexBase):
    """Group index with concatenated string representation.

    ``self.index`` stores per-object concatenated strings (e.g.
    ``"0#1#2"``).  ``self.n_groups`` is a placeholder (0) until the
    first call to ``get_distribution`` / ``get_disorder_score``, at
    which point strings are hashed, mapped to sequential group IDs, and
    the distribution is built (analogous to ``GroupIndexPure``).
    """

    index: npt.NDArray[np.str_] = field(default_factory=lambda: np.array([], dtype=str))
    n_groups: int = 0

    @classmethod
    def create_empty(cls):
        return cls(
            index=np.array([], dtype=str),
            n_groups=len(np.array([], dtype=str)),
        )

    @classmethod
    def create_uniform(cls, size: int):
        if size < 0:
            raise ValueError("Size less than zero")
        if size == 0:
            return cls.create_empty()
        concatenated = np.zeros(size, dtype=str)
        return cls(index=concatenated, n_groups=len(concatenated))

    @classmethod
    def from_index(
        cls,
        index: Sequence[int] | npt.NDArray[np.int64],
        compress: bool = False,
    ):
        index_arr = np.asarray(index, dtype=np.int64)
        if len(index_arr) == 0:
            return cls.create_empty()
        if np.min(index_arr) < 0:
            raise ValueError("Index value less than zero")
        concatenated = index_arr.astype(str)
        return cls(index=concatenated, n_groups=len(concatenated))

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
            concatenated = np.zeros(len(x), dtype=str)
        else:
            concatenated = x[:, unified_attrs[0]].astype(str)
            for a in unified_attrs[1:]:
                concatenated = np.char.add(
                    np.char.add(concatenated, "#"),
                    x[:, a].astype(str),
                )

        return cls(index=concatenated, n_groups=len(concatenated))

    def split(
        self,
        values: npt.NDArray[np.int64],
        values_count: int,
        compress: bool = False,
    ):
        self._check_values(values)
        new_concatenated = np.char.add(
            np.char.add(self.index, "#"),
            np.asarray(values).astype(str),
        )
        return type(self)(index=new_concatenated, n_groups=len(new_concatenated))

    def compress(self):
        return type(self)(index=self.index.copy(), n_groups=len(self.index))

    def get_distribution(
        self,
        values: npt.NDArray[np.int64],
        values_count: int,
    ) -> npt.NDArray[np.int64]:
        self._check_values(values)

        n = len(self.index)
        if n == 0:
            return np.zeros((0, values_count), dtype=np.int64)

        hashes = np.zeros(n, dtype=np.uint64)
        for i in range(n):
            hashes[i] = xxhash.xxh64(str(self.index[i])).intdigest()

        unique_hashes, group_ids = np.unique(hashes, return_inverse=True)
        n_groups = len(unique_hashes)

        result = np.zeros((n_groups, values_count), dtype=np.int64)
        np.add.at(result, (group_ids, values), 1)
        return result
