import logging
from typing import Callable, List, Optional, Sequence, TypeVar

import skrough.typing as rght
from skrough.logs import log_start_end

logger = logging.getLogger(__name__)


T = TypeVar("T", bound=Callable)


@log_start_end(logger)
def normalize_hook_sequence(
    hooks: Optional[rght.OneOrSequence[T]],
    optional: bool,
) -> List[T]:
    if optional is False and not hooks:
        raise ValueError("Hooks argument should not be empty.")
    result: List[T]
    if hooks is None:
        result = []
    elif not isinstance(hooks, Sequence):
        result = [hooks]
    else:
        result = list(hooks)
    return result
