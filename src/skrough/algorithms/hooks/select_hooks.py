import logging

import numpy as np

import skrough.typing as rght
from skrough.algorithms.hooks.names import (
    HOOKS_CHAOS_FUN,
    HOOKS_DATA_X,
    HOOKS_DATA_X_COUNTS,
    HOOKS_DATA_Y,
    HOOKS_DATA_Y_COUNT,
    HOOKS_GROUP_INDEX,
    HOOKS_SELECT_ATTRS_CHAOS_SCORE_BASED_MAX_COUNT,
    HOOKS_SELECT_RANDOM_MAX_COUNT,
)
from skrough.logs import log_start_end
from skrough.structs.group_index import GroupIndex
from skrough.structs.state import ProcessingState

logger = logging.getLogger(__name__)


@log_start_end(logger)
def select_hook_random(
    state: ProcessingState,
    elements: rght.Elements,
) -> rght.Elements:
    return state.rng.choice(
        elements,
        min(len(elements), state.config[HOOKS_SELECT_RANDOM_MAX_COUNT]),
        replace=False,
    )


@log_start_end(logger)
def select_hook_attrs_chaos_score_based(
    state: ProcessingState,
    elements: rght.Elements,
) -> rght.Elements:
    group_index: GroupIndex = state.values[HOOKS_GROUP_INDEX]
    scores = np.fromiter(
        (
            group_index.get_chaos_score_after_split(
                split_values=state.values[HOOKS_DATA_X][:, i],
                split_values_count=state.values[HOOKS_DATA_X_COUNTS][i],
                values=state.values[HOOKS_DATA_Y],
                values_count=state.values[HOOKS_DATA_Y_COUNT],
                chaos_fun=state.config[HOOKS_CHAOS_FUN],
            )
            for i in elements
        ),
        dtype=float,
    )
    # find indices for which the scores are the lowest
    attrs_count = state.config[HOOKS_SELECT_ATTRS_CHAOS_SCORE_BASED_MAX_COUNT]
    selected_attrs_idx = np.argsort(scores)[:attrs_count]
    return np.asarray(elements)[selected_attrs_idx]
