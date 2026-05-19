"""Group index implementation using xxhash for group assignment.

Uses non-cryptographic hashing (xxhash) to assign objects to groups
incrementally, avoiding integer-overflow issues of multiplication-based
approaches and the need for explicit compactification.

Raw hash values are stored directly as ``index`` -- no mapping to
sequential IDs is performed.  The number of unique hashes is computed once
during construction and cached as ``n_groups``.

The streaming ``get_disorder_score`` never builds the full
``n_groups x n_values`` distribution matrix. Instead it sorts objects by
group, iterates groups sequentially, and calls the disorder function on
per-group 1xV rows with ``n_elements = total_n``. Because all built-in
disorder measures (entropy, gini impurity, conflicts count) are
decomposable with group-weight averaging, summing these per-group results
gives the exact total score.
"""

import numpy as np
import numpy.typing as npt
import xxhash

import skrough.typing as rght
from skrough.structs.group_index._base import GroupIndexBase
from skrough.unify import unify_index_list


def _hash_pair_u64(group_id: np.int64, value: np.int64) -> np.uint64:
    """Hash a (group_id, attribute_value) pair into a uint64.

    Uses xxhash with ``group_id`` as the seed so that identical values
    under different groups produce different hashes.  The seed is
    interpreted as an unsigned 64-bit integer (via bitmask) so that
    sign-extension of int64 values does not affect the result.
    """
    seed = int(group_id) & 0xFFFF_FFFF_FFFF_FFFF
    return xxhash.xxh64(str(value), seed=seed).intdigest()


def _hash_row_u64(row: list[int]) -> np.uint64:
    """Hash a sequence of attribute values into a uint64 via xxhash streaming."""
    hasher = xxhash.xxh64()
    for v in row:
        hasher.update(str(v))
    return hasher.intdigest()


class GroupIndexHash(GroupIndexBase):
    """Group index with xxhash-based group assignment.

    Group assignments are computed by hashing attribute-value combinations
    with xxhash instead of the multiplicative ``old_idx * count + val``
    scheme.  This avoids integer-overflow risks and removes the need for
    post-hoc compactification -- there are no gaps because every hash value
    present in the index corresponds to an actual group.

    Raw hash values are stored directly in the ``index`` array (as int64
    for compatibility with the base-class contract).  The
    ``get_disorder_score`` method sorts objects by their hash, iterates
    groups sequentially, and sums per-group disorder contributions
    **without ever materialising the full ``n_groups x n_values``
    distribution matrix**.
    """

    @classmethod
    def from_data(
        cls,
        x: npt.NDArray[np.int64],
        x_counts: npt.NDArray[np.int64],
        attrs: rght.IndexListLike | None = None,
    ):
        """Split objects into groups by hashing attribute values.

        Each row (object) is hashed via xxhash streaming over its
        attribute values.  Raw hash values are stored directly as
        ``index``.
        """
        if attrs is None:
            attrs = range(x.shape[1])
        unified_attrs = unify_index_list(attrs)
        if len(unified_attrs) == 0:
            return cls.create_uniform(size=len(x))

        n = len(x)
        hashes = np.zeros(n, dtype=np.uint64)
        for i in range(n):
            hashes[i] = _hash_row_u64([x[i, a] for a in unified_attrs])

        n_groups = len(np.unique(hashes))
        return cls(index=hashes.view(np.int64), n_groups=n_groups)

    def split(
        self,
        values: npt.NDArray[np.int64],
        values_count: int,
        compress: bool = False,
    ):
        """Split groups by hashing (old_group_id, attribute_value) pairs.

        Each object's new group is determined by a seeded xxhash of the
        attribute value, where the current group hash serves as the seed.
        Raw hashes are stored directly -- no mapping to sequential IDs.
        """
        self._check_values(values)

        n = self.n_objs
        new_hashes = np.zeros(n, dtype=np.uint64)
        for i in range(n):
            new_hashes[i] = _hash_pair_u64(self.index[i], values[i])

        n_groups = len(np.unique(new_hashes))
        return type(self)(index=new_hashes.view(np.int64), n_groups=n_groups)

    def compress(self):
        """No-op -- no gaps to remove when using raw hash values."""
        return type(self)(index=self.index.copy(), n_groups=self.n_groups)

    def get_distribution(
        self,
        values: npt.NDArray[np.int64],
        values_count: int,
    ) -> npt.NDArray[np.int64]:
        """Build the full distribution matrix via sort-then-scan.

        Because ``self.index`` stores raw (non-sequential) hash values it
        cannot be used for random-access writes.  Objects are sorted by
        group hash and sequential row indices are assigned on the fly.

        Prefer ``get_disorder_score`` for the streaming (low-memory) code
        path.
        """
        self._check_values(values)

        n = self.n_objs
        if n == 0:
            return np.zeros((0, values_count), dtype=np.int64)

        order = np.argsort(self.index)
        sorted_values = values[order]
        sorted_groups = self.index[order]

        result = np.zeros((self.n_groups, values_count), dtype=np.int64)

        i = 0
        row = 0
        while i < n:
            j = i + 1
            while j < n and sorted_groups[j] == sorted_groups[i]:
                j += 1

            group_vals = sorted_values[i:j]
            result[row] = np.bincount(group_vals, minlength=values_count)
            row += 1
            i = j

        return result

    def get_disorder_score(
        self,
        values: npt.NDArray[np.int64],
        values_count: int,
        disorder_fun: rght.DisorderMeasure,
    ) -> rght.DisorderMeasureReturnType:
        """Compute disorder score via streaming per-group decomposition.

        Objects are sorted by group hash, then each contiguous group is
        processed independently.  For every group a 1xV distribution row
        is built and passed to ``disorder_fun`` with ``n_elements =
        n_objs`` (the TOTAL number of objects).  Because all built-in
        disorder measures internally weight each group by ``group_size /
        n_objs``, summing the per-group results yields the exact total
        score.

        This method **never** allocates the full ``n_groups x n_values``
        matrix, making it memory-friendly when the number of groups is
        large.
        """
        self._check_values(values)

        n = self.n_objs
        if n == 0:
            return disorder_fun(
                np.zeros((0, values_count), dtype=np.int64),
                n,
            )

        order = np.argsort(self.index)
        sorted_values = values[order]
        sorted_groups = self.index[order]

        total: float = 0.0
        i = 0
        while i < n:
            j = i + 1
            while j < n and sorted_groups[j] == sorted_groups[i]:
                j += 1

            group_vals = sorted_values[i:j]
            counts = np.bincount(group_vals, minlength=values_count)
            per_group_row = counts.reshape(1, -1)

            total += disorder_fun(per_group_row, n)

            i = j

        return total
