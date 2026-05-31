"""Group index using numba-jitted boost::hash_combine.

Replaces xxhash (which has no vectorised Python API) with a simple
numa-jittable hash function based on the well-known boost::hash_combine
pattern.  All per-object hashing runs inside a ``@numba.njit`` loop,
removing the Python-level overhead while keeping the same architecture:
raw hash values, no compactification, streaming disorder computation.
"""

import numpy as np
import numpy.typing as npt

import numba

import skrough.typing as rght
from skrough.structs.group_index._hash import GroupIndexHash
from skrough.unify import unify_index_list

GOLDEN_U64 = np.uint64(0x9E3779B9) << np.uint64(32) | np.uint64(0x7F4A7C15)
"""64-bit golden ratio for boost::hash_combine.

Constructed from two 32-bit halves to avoid large-Python-int issues in
numa's ``np.uint64()`` constructor.
"""


@numba.njit(cache=True, inline="always")
def _hash_combine_u64(h: np.uint64, v: np.int64) -> np.uint64:
    """Boost-style hash combine of a uint64 accumulator and an int64 value."""
    h ^= np.uint64(v) + GOLDEN_U64 + (h << np.uint64(6)) + (h >> np.uint64(2))
    return h


@numba.njit(cache=True)
def _hash_rows(
    x: npt.NDArray[np.int64],
    unified_attrs: npt.NDArray[np.int64],
    n_objs: int,
) -> npt.NDArray[np.uint64]:
    """Hash each row (object) across the selected attributes."""
    n_attrs = len(unified_attrs)
    hashes = np.zeros(n_objs, dtype=np.uint64)
    for i in range(n_objs):
        h = np.uint64(0)
        for a in range(n_attrs):
            h = _hash_combine_u64(h, x[i, unified_attrs[a]])
        hashes[i] = h
    return hashes


@numba.njit(cache=True)
def _hash_split(
    index: npt.NDArray[np.int64],
    values: npt.NDArray[np.int64],
    n_objs: int,
) -> npt.NDArray[np.uint64]:
    """Hash (old_group_id, attribute_value) pairs for a split."""
    new_hashes = np.zeros(n_objs, dtype=np.uint64)
    idx_u64 = index.view(np.uint64)
    for i in range(n_objs):
        new_hashes[i] = _hash_combine_u64(idx_u64[i], values[i])
    return new_hashes


class GroupIndexHashNumba(GroupIndexHash):
    """Group index with numba-jitted hash functions.

    Uses boost::hash_combine (a simple bitwise mixing function) instead
    of xxhash so that the per-object hashing loop can be compiled by
    numba.  Inherits the streaming ``get_disorder_score`` from
    :class:`GroupIndexHash`.
    """

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

        hashes = _hash_rows(x, np.asarray(unified_attrs, dtype=np.int64), len(x))
        n_groups = len(np.unique(hashes))
        return cls(index=hashes.view(np.int64), n_groups=n_groups)

    def split(
        self,
        values: npt.NDArray[np.int64],
        values_count: int,
        compress: bool = False,
    ):
        self._check_values(values)
        new_hashes = _hash_split(self.index, values, self.n_objs)
        n_groups = len(np.unique(new_hashes))
        return type(self)(index=new_hashes.view(np.int64), n_groups=n_groups)
