import logging

import numpy as np

import skrough.typing as rght
from skrough.algorithms.hooks.names import (
    HOOKS_DATA_X,
    HOOKS_DATA_X_COUNTS,
    HOOKS_GROUP_INDEX,
    HOOKS_RESULT_ATTRS,
)
from skrough.logs import log_start_end
from skrough.structs.state import GrowShrinkState

logger = logging.getLogger(__name__)


@log_start_end(logger)
def inner_process_hook_add_attr(
    state: GrowShrinkState,
    elements: rght.Elements,
) -> rght.Elements:
    if len(elements) > 0:
        attr = np.take(elements, 0)
        elements = np.delete(elements, 0)
        state.values[HOOKS_RESULT_ATTRS].append(attr)
        state.values[HOOKS_GROUP_INDEX] = state.values[HOOKS_GROUP_INDEX].split(
            state.values[HOOKS_DATA_X][:, attr],
            state.values[HOOKS_DATA_X_COUNTS][attr],
        )
    return elements
