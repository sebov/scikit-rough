import numpy as np

import skrough.typing as rght


def unify_index_list(items: rght.IndexListLike) -> rght.IndexList:
    """Unify index-like input.

    Unify index-like input, i.e., :class:`numpy.ndarray` or a sequence of integers
    indexing objects/attributes/elements to a common form of
    :class:`~skrough.typing.IndexList`.

    Args:
        items: Index-like input to be unified into a common form of
        :class:`~skrough.typing.IndexList`.

    Returns:
        Unified index list.

    Examples:
        >>> unify_index_list(np.arange(5))
        array([0, 1, 2, 3, 4])
        >>> unify_index_list([1,2,3])
        array([1, 2, 3])
        >>> unify_index_list([])
        array([], dtype=int64)
    """
    return np.asarray(items, dtype=np.int64)
