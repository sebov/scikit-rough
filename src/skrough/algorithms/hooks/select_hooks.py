import logging

import numpy as np

import skrough.typing as rght
from skrough.algorithms.key_names import (
    CONFIG_DISORDER_FUN,
    CONFIG_SELECT_ATTRS_DISORDER_SCORE_BASED_MAX_COUNT,
)
from skrough.logs import log_start_end
from skrough.structs.group_index import GroupIndex
from skrough.structs.state import ProcessingState

logger = logging.getLogger(__name__)


@log_start_end(logger)
def select_hook_attrs_disorder_score_based(
    state: ProcessingState,
    elements: rght.Elements,
) -> rght.Elements:
    group_index: GroupIndex = state.get_group_index()
    scores = np.fromiter(
        (
            group_index.get_disorder_score_after_split(
                split_values=state.get_values_x()[:, i],
                split_values_count=int(state.get_values_x_counts()[i]),
                values=state.get_values_y(),
                values_count=state.get_values_y_count(),
                disorder_fun=state.config[CONFIG_DISORDER_FUN],
            )
            for i in elements
        ),
        dtype=float,
    )
    # find indices for which the scores are the lowest
    attrs_count = state.config[CONFIG_SELECT_ATTRS_DISORDER_SCORE_BASED_MAX_COUNT]
    selected_attrs_idx = np.argsort(scores)[:attrs_count]
    return np.asarray(elements)[selected_attrs_idx]
