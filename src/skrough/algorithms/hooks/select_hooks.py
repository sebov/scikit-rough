import logging

import numpy as np

import skrough.typing as rght
from skrough.logs import log_start_end
from skrough.structs.group_index import GroupIndex
from skrough.structs.state import ProcessingState

logger = logging.getLogger(__name__)


@log_start_end(logger)
def select_hook_attrs_disorder_score_based(
    state: ProcessingState,
    elements: rght.Elements,
) -> rght.Elements:
    group_index: GroupIndex = state.get_values_group_index()
    scores = np.fromiter(
        (
            group_index.get_disorder_score_after_split(
                split_values=state.get_values_x()[:, i],
                split_values_count=int(state.get_values_x_counts()[i]),
                values=state.get_values_y(),
                values_count=state.get_values_y_count(),
                disorder_fun=state.get_config_disorder_fun(),
            )
            for i in elements
        ),
        dtype=float,
    )
    # find indices for which the scores are the lowest
    attrs_count = state.get_config_select_attrs_disorder_score_based_max_count()
    selected_attrs_idx = np.argsort(scores)[:attrs_count]
    return np.asarray(elements)[selected_attrs_idx]
