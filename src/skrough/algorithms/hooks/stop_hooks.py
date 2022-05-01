import logging

from skrough.algorithms.hooks.names import (
    HOOKS_APPROX_THRESHOLD,
    HOOKS_BASE_CHAOS_SCORE,
    HOOKS_CHAOS_FUN,
    HOOKS_DATA_Y,
    HOOKS_DATA_Y_COUNT,
    HOOKS_EMPTY_ITERATIONS_COUNT,
    HOOKS_EMPTY_ITERATIONS_MAX_COUNT,
    HOOKS_GROUP_INDEX,
    HOOKS_RESULT_ATTRS,
    HOOKS_RESULT_ATTRS_MAX_COUNT,
)
from skrough.chaos_score import get_chaos_score_for_group_index
from skrough.logs import log_start_end
from skrough.structs.state import GrowShrinkState

logger = logging.getLogger(__name__)


@log_start_end(logger)
def stop_hook_approx_threshold(
    state: GrowShrinkState,
) -> bool:
    current_chaos_score = get_chaos_score_for_group_index(
        state.values[HOOKS_GROUP_INDEX],
        state.values[HOOKS_DATA_Y],
        state.values[HOOKS_DATA_Y_COUNT],
        state.config[HOOKS_CHAOS_FUN],
    )
    base_chaos_score = state.values[HOOKS_BASE_CHAOS_SCORE]
    approx_threshold = state.values[HOOKS_APPROX_THRESHOLD]
    current_dependency_in_data = base_chaos_score - current_chaos_score
    logger.debug("current_chaos_score = %f", current_chaos_score)
    logger.debug("current_dependency_in_data = %f", current_dependency_in_data)
    logger.debug("approx_threshold = %f", approx_threshold)
    return current_dependency_in_data >= approx_threshold


@log_start_end(logger)
def stop_hook_attrs_count(
    state: GrowShrinkState,
) -> bool:
    return (
        len(state.values[HOOKS_RESULT_ATTRS])
        >= state.config[HOOKS_RESULT_ATTRS_MAX_COUNT]
    )


@log_start_end(logger)
def stop_hook_empty_iterations(
    state: GrowShrinkState,
) -> bool:
    return (
        state.values.get(HOOKS_EMPTY_ITERATIONS_COUNT, 0)
        >= state.config[HOOKS_EMPTY_ITERATIONS_MAX_COUNT]
    )
