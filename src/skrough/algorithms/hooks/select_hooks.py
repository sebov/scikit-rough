import logging

import numpy as np

import skrough.typing as rght
from skrough.algorithms.key_names import (
    CONFIG_CHAOS_FUN,
    CONFIG_SELECT_ATTRS_CHAOS_SCORE_BASED_MAX_COUNT,
    VALUES_GROUP_INDEX,
    VALUES_X,
    VALUES_X_COUNTS,
    VALUES_Y,
    VALUES_Y_COUNT,
)
from skrough.logs import log_start_end
from skrough.structs.group_index import GroupIndex
from skrough.structs.state import ProcessingState

logger = logging.getLogger(__name__)


@log_start_end(logger)
def select_hook_attrs_chaos_score_based(
    state: ProcessingState,
    elements: rght.Elements,
) -> rght.Elements:
    group_index: GroupIndex = state.values[VALUES_GROUP_INDEX]
    scores = np.fromiter(
        (
            group_index.get_chaos_score_after_split(
                split_values=state.values[VALUES_X][:, i],
                split_values_count=state.values[VALUES_X_COUNTS][i],
                values=state.values[VALUES_Y],
                values_count=state.values[VALUES_Y_COUNT],
                chaos_fun=state.config[CONFIG_CHAOS_FUN],
            )
            for i in elements
        ),
        dtype=float,
    )
    # find indices for which the scores are the lowest
    attrs_count = state.config[CONFIG_SELECT_ATTRS_CHAOS_SCORE_BASED_MAX_COUNT]
    selected_attrs_idx = np.argsort(scores)[:attrs_count]
    return np.asarray(elements)[selected_attrs_idx]
