import logging

import skrough.typing as rght
from skrough.algorithms.hooks.helpers import check_if_below_approx_value_threshold
from skrough.algorithms.key_names import (
    VALUES_GROUP_INDEX,
    VALUES_RESULT_ATTRS,
    VALUES_RESULT_OBJS,
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


def inner_process_hook_discard_first_attr_approx_threshold(
    state: ProcessingState,
    elements: rght.Elements,
) -> rght.Elements:
    attr = elements[0]
    elements = elements[1:]
    attrs_to_try = [a for a in state.values[VALUES_RESULT_ATTRS] if a != attr]
    x = state.values[VALUES_X]
    x_counts = state.values[VALUES_X_COUNTS]
    if VALUES_RESULT_OBJS in state.values:
        x = x[state.values[VALUES_RESULT_OBJS]]
    group_index = GroupIndex.from_data(
        x=x,
        x_counts=x_counts,
        attrs=attrs_to_try,
    )
    if check_if_below_approx_value_threshold(state, group_index):
        state.values[VALUES_RESULT_ATTRS] = attrs_to_try
    return elements
