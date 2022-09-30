"""Helper functions for :mod:`skrough.algorithms.meta` subpackage."""

import logging
from typing import Callable, List, Optional, Sequence, TypeVar

import skrough.typing as rght
from skrough.logs import log_start_end

logger = logging.getLogger(__name__)


T = TypeVar("T", bound=Callable)


@log_start_end(logger)
def normalize_sequence(
    items: Optional[rght.OneOrSequence[T]],
    optional: bool,
) -> List[T]:
    """Normalize a sequence of items to a list.

    The function normalizes input items to a list form. The input ``items`` can be given
    as a single element, a sequence or :obj:`None` (optionally) and a list is returned
    that corresponds to the given input, i.e., respectively, a list containing the
    single element, the input list itself or an empty list (optionally).

    The function is instrumented by ``optional`` argument which controls the function's
    behavior for :obj:`None` passed as ``items`` argument. For ``optional is True`` the
    function will return an empty list, while for ``optional is False`` the function
    will raise a ``ValueError`` exception.

    Args:
        items: Items that should be normalized to a list.
        optional: Controls the function's behavior for :obj:`None` passed as ``items``,
            i.e., for ``optional is True`` the function will return an empty list,
            otherwise it will raise a ``ValueError`` exception.

    Raises:
        ValueError: When ``optional is True`` and :obj:`None` is given as ``items``.

    Returns:
        A list that corresponds to the input ``items``.
    """
    if optional is False and not items:
        raise ValueError("Hooks argument should not be empty.")
    result: List[T]
    if items is None:
        result = []
    elif not isinstance(items, Sequence):
        result = [items]
    else:
        result = list(items)
    return result
