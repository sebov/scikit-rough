import logging

import numpy as np

from skrough.algorithms.hooks.names import (
    APPROX_THRESHOLD,
    BASE_CHAOS_SCORE,
    CHAOS_FUN,
    EMPTY_SELECTED_ATTRS_COUNT,
    EMPTY_SELECTED_ATTRS_MAX_COUNT,
    RESULT_ATTRS,
    RESULT_ATTRS_MAX_COUNT,
    SINGLE_GROUP_INDEX,
)
from skrough.chaos_score import get_chaos_score_for_group_index
from skrough.logs import log_start_end
from skrough.structs.state import GrowShrinkState

logger = logging.getLogger(__name__)


@log_start_end(logger)
def grow_stop_approx_threshold(
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    state: GrowShrinkState,
) -> bool:
    chaos_fun = state.config[CHAOS_FUN]
    base_chaos_score = state.values[BASE_CHAOS_SCORE]
    approx_threshold = state.values[APPROX_THRESHOLD]
    current_chaos_score = get_chaos_score_for_group_index(
        state.values[SINGLE_GROUP_INDEX], len(x), y, y_count, chaos_fun
    )
    current_dependency_in_data = base_chaos_score - current_chaos_score
    logger.debug("current_chaos_score = %f", current_chaos_score)
    logger.debug("current_dependency_in_data = %f", current_dependency_in_data)
    logger.debug("approx_threshold = %f", approx_threshold)
    return current_dependency_in_data >= approx_threshold


@log_start_end(logger)
def grow_stop_count(
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    state: GrowShrinkState,
) -> bool:
    return len(state.values[RESULT_ATTRS]) >= state.config[RESULT_ATTRS_MAX_COUNT]


@log_start_end(logger)
def grow_stop_empty_add_attrs(
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    state: GrowShrinkState,
) -> bool:
    return (
        state.values.get(EMPTY_SELECTED_ATTRS_COUNT, 0)
        >= state.config[EMPTY_SELECTED_ATTRS_MAX_COUNT]
    )
