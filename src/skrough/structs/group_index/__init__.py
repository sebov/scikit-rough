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
]
