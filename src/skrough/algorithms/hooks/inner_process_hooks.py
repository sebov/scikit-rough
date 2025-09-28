import logging

import skrough.typing as rght
from skrough.algorithms.hooks.helpers import check_if_below_approx_threshold
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
        state.get_values_result_attrs().append(int(attr))
        group_index = state.get_group_index()
        state.set_group_index(
            group_index.split(
                values=state.get_values_x()[:, attr],
                values_count=int(state.get_values_x_counts()[attr]),
                compress=True,
            )
        )
    return elements


def inner_process_hook_discard_first_attr_approx_threshold(
    state: ProcessingState,
    elements: rght.Elements,
) -> rght.Elements:
    attr = elements[0]
    elements = elements[1:]
    attrs_to_try = [a for a in state.get_values_result_attrs() if a != attr]
    x = state.get_values_x()
    x_counts = state.get_values_x_counts()
    if state.is_set_values_result_objs():
        x = x[state.get_values_result_objs()]
    group_index = GroupIndex.from_data(
        x=x,
        x_counts=x_counts,
        attrs=attrs_to_try,
    )
    if check_if_below_approx_threshold(state, group_index):
        state.set_values_result_attrs(attrs_to_try)
    return elements
