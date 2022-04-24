import logging

import skrough.typing as rght
from skrough.algorithms.hooks.names import EMPTY_SELECTED_ATTRS_COUNT
from skrough.logs import log_start_end
from skrough.structs.state import GrowShrinkState

logger = logging.getLogger(__name__)


@log_start_end(logger)
def grow_inner_init_note_empty(
    state: GrowShrinkState,
    input_attrs: rght.Elements,
) -> rght.Elements:
    if len(input_attrs) == 0:
        value = state.values.get(EMPTY_SELECTED_ATTRS_COUNT, 0) + 1
    else:
        value = 0
    state.values[EMPTY_SELECTED_ATTRS_COUNT] = value
    return input_attrs
