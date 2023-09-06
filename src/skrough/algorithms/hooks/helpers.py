import logging

from skrough.algorithms.key_names import (
    CONFIG_CHAOS_FUN,
    VALUES_CHAOS_SCORE_APPROX_THRESHOLD,
    VALUES_RESULT_OBJS,
    VALUES_Y,
    VALUES_Y_COUNT,
)
from skrough.logs import log_start_end
from skrough.structs.group_index import GroupIndex
from skrough.structs.state import ProcessingState

logger = logging.getLogger(__name__)


@log_start_end(logger)
def check_if_below_approx_value_threshold(
    state: ProcessingState,
    group_index: GroupIndex,
) -> bool:
    values = state.values[VALUES_Y]
    values_count = state.values[VALUES_Y_COUNT]
    if VALUES_RESULT_OBJS in state.values:
        values = values[state.values[VALUES_RESULT_OBJS]]
    current_chaos_score = group_index.get_chaos_score(
        values=values,
        values_count=values_count,
        chaos_fun=state.config[CONFIG_CHAOS_FUN],
    )
    approx_chaos_score_value_threshold = state.values[
        VALUES_CHAOS_SCORE_APPROX_THRESHOLD
    ]
    logger.debug("current_chaos_score = %f", current_chaos_score)
    logger.debug(
        "approx_chaos_score_value_threshold = %f", approx_chaos_score_value_threshold
    )
    return bool(current_chaos_score <= approx_chaos_score_value_threshold)
