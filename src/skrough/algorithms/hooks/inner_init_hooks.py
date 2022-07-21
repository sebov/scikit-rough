import logging

import skrough.typing as rght
from skrough.algorithms.hooks.names import HOOKS_EMPTY_ITERATIONS_COUNT
from skrough.logs import log_start_end
from skrough.structs.state import ProcessingState

logger = logging.getLogger(__name__)


@log_start_end(logger)
def inner_init_hook_note_empty(
    state: ProcessingState,
    elements: rght.Elements,
) -> rght.Elements:
    if len(elements) == 0:
        value = state.values.get(HOOKS_EMPTY_ITERATIONS_COUNT, 0) + 1
    else:
        value = 0
    state.values[HOOKS_EMPTY_ITERATIONS_COUNT] = value
    return elements
