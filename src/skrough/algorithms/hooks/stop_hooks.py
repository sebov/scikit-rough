import logging

from skrough.algorithms.hooks.names import (
    HOOKS_APPROX_CHAOS_SCORE_VALUE_THRESHOLD,
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
    approx_chaos_score_value_threshold = state.values[
        HOOKS_APPROX_CHAOS_SCORE_VALUE_THRESHOLD
    ]
    logger.debug("current_chaos_score = %f", current_chaos_score)
    logger.debug(
        "approx_chaos_score_value_threshold = %f", approx_chaos_score_value_threshold
    )
    return current_chaos_score <= approx_chaos_score_value_threshold


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
