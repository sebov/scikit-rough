"""Group index subpackage.

Provides the :class:`GroupIndexProtocol` interface and six concrete
implementations:

- :class:`GroupIndexNumba` -- numba-accelerated (default, exposed as
  :class:`GroupIndex` for backward compatibility)
- :class:`GroupIndexPure` -- pure numpy, no numba dependency
- :class:`GroupIndexHash` -- xxhash-based, streaming disorder computation
- :class:`GroupIndexHashNumba` -- numba-jitted boost::hash_combine,
  streaming disorder
- :class:`GroupIndexDict` -- dict-based groups, on-the-fly compactification
- :class:`GroupIndexDictNumba` -- dict-based split, numba-jitted disorder
"""

from skrough.structs.group_index._dict import GroupIndexDict
from skrough.structs.group_index._dict_numba import GroupIndexDictNumba
from skrough.structs.group_index._hash import GroupIndexHash
from skrough.structs.group_index._hash_numba import GroupIndexHashNumba
from skrough.structs.group_index._numba import GroupIndexNumba as GroupIndex
from skrough.structs.group_index._numba import GroupIndexNumba
from skrough.structs.group_index._protocol import GroupIndexProtocol
from skrough.structs.group_index._pure import GroupIndexPure

__all__ = [
    "GroupIndex",
    "GroupIndexDict",
    "GroupIndexDictNumba",
    "GroupIndexHash",
    "GroupIndexHashNumba",
    "GroupIndexNumba",
    "GroupIndexProtocol",
    "GroupIndexPure",
    "resolve_group_index_class",
    "GROUP_INDEX_BY_NAME",
]

GROUP_INDEX_BY_NAME: dict[str, type[GroupIndexProtocol]] = {
    "numba": GroupIndexNumba,
    "pure": GroupIndexPure,
    "hash": GroupIndexHash,
    "hash_numba": GroupIndexHashNumba,
    "dict": GroupIndexDict,
    "dict_numba": GroupIndexDictNumba,
}
"""String names for built-in :class:`GroupIndexProtocol` implementations."""


def resolve_group_index_class(
    group_index: str | type[GroupIndexProtocol] | None,
) -> type[GroupIndexProtocol]:
    """Resolve a string name or class to a concrete GroupIndex class.

    Args:
        group_index: One of the keys in :data:`GROUP_INDEX_BY_NAME`,
            a class satisfying :class:`GroupIndexProtocol`, or :obj:`None`.

    Returns:
        A concrete :class:`GroupIndexProtocol` subclass.

    Raises:
        ValueError: If *group_index* is a string not present in
            :data:`GROUP_INDEX_BY_NAME`.
    """
    if group_index is None:
        return GroupIndex
    if isinstance(group_index, str):
        try:
            return GROUP_INDEX_BY_NAME[group_index]
        except KeyError:
            raise ValueError(
                f"Unknown group_index name {group_index!r}. "
                f"Choose from: {list(GROUP_INDEX_BY_NAME)}."
            ) from None
    return group_index
