import logging
from typing import Callable, Optional, Sequence, TypeVar

import skrough.typing as rght
from skrough.logs import log_start_end
from skrough.structs.state import ProcessingState

logger = logging.getLogger(__name__)


T = TypeVar("T", bound=Callable)


@log_start_end(logger)
def normalize_hook_sequence(
    hooks: Optional[rght.OneOrSequence[T]],
    optional: bool,
) -> Sequence[T]:
    if optional is False and not hooks:
        raise ValueError("Hooks argument should not be empty.")
    result: Sequence[T]
    if hooks is None:
        result = []
    elif not isinstance(hooks, Sequence):
        result = [hooks]
    else:
        result = hooks
    return result


@log_start_end(logger)
def aggregate_chain_process_elements_hooks(
    hooks: Optional[rght.OneOrSequence[rght.ProcessElementsHook]],
) -> rght.ProcessElementsFunction:
    normalized_hooks = normalize_hook_sequence(hooks, optional=True)

    @log_start_end(logger)
    def _process_elements_function(
        state: ProcessingState,
        elements: rght.Elements,
    ) -> rght.Elements:
        result = elements
        for hook in normalized_hooks:
            result = hook(state, result)
        return result

    return _process_elements_function
