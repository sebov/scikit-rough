import logging

from skrough.algorithms.key_names import (
    CONFIG_DISORDER_FUN,
    VALUES_DISORDER_SCORE_APPROX_THRESHOLD,
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
    current_disorder_score = group_index.get_disorder_score(
        values=values,
        values_count=values_count,
        disorder_fun=state.config[CONFIG_DISORDER_FUN],
    )
    approx_disorder_score_value_threshold = state.values[
        VALUES_DISORDER_SCORE_APPROX_THRESHOLD
    ]
    logger.debug("current_disorder_score = %f", current_disorder_score)
    logger.debug(
        "approx_disorder_score_value_threshold = %f",
        approx_disorder_score_value_threshold,
    )
    return bool(current_disorder_score <= approx_disorder_score_value_threshold)
