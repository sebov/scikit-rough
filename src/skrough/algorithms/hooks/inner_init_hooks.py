import logging

import skrough.typing as rght
from skrough.logs import log_start_end
from skrough.structs.state import ProcessingState

logger = logging.getLogger(__name__)


@log_start_end(logger)
def inner_init_hook_consecutive_empty_iterations_count(
    state: ProcessingState,
    elements: rght.Elements,
) -> rght.Elements:
    if len(elements) == 0:
        if state.is_set_values_consecutive_empty_iterations_count():
            value = state.get_values_consecutive_empty_iterations_count()
        else:
            value = 0
        value += 1
    else:
        value = 0
    state.set_values_consecutive_empty_iterations_count(value)
    return elements
