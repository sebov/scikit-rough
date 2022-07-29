import logging

import skrough.typing as rght
from skrough.algorithms.hooks.names import (
    HOOKS_DATA_X,
    HOOKS_DATA_X_COUNTS,
    HOOKS_GROUP_INDEX,
    HOOKS_RESULT_ATTRS,
)
from skrough.logs import log_start_end
from skrough.structs.group_index import GroupIndex
from skrough.structs.state import ProcessingState

logger = logging.getLogger(__name__)


@log_start_end(logger)
def inner_process_hook_add_first_attr(
    state: ProcessingState,
    elements: rght.Elements,
) -> rght.Elements:
    if len(elements) > 0:
        attr = elements[0]
        elements = elements[1:]
        state.values[HOOKS_RESULT_ATTRS].append(attr)
        group_index: GroupIndex = state.values[HOOKS_GROUP_INDEX]
        state.values[HOOKS_GROUP_INDEX] = group_index.split(
            values=state.values[HOOKS_DATA_X][:, attr],
            values_count=state.values[HOOKS_DATA_X_COUNTS][attr],
        )
    return elements
