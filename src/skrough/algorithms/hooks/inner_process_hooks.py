import logging

import skrough.typing as rght
from skrough.algorithms.key_names import (
    VALUES_GROUP_INDEX,
    VALUES_RESULT_ATTRS,
    VALUES_X,
    VALUES_X_COUNTS,
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
        state.values[VALUES_RESULT_ATTRS].append(attr)
        group_index: GroupIndex = state.values[VALUES_GROUP_INDEX]
        state.values[VALUES_GROUP_INDEX] = group_index.split(
            values=state.values[VALUES_X][:, attr],
            values_count=state.values[VALUES_X_COUNTS][attr],
            compress=True,
        )
    return elements
