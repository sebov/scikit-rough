import logging

from skrough.algorithms.key_names import (
    CONFIG_DISORDER_FUN,
    VALUES_DISORDER_SCORE_APPROX_THRESHOLD,
)
from skrough.logs import log_start_end
from skrough.structs.group_index import GroupIndex
from skrough.structs.state import ProcessingState

logger = logging.getLogger(__name__)


@log_start_end(logger)
def check_if_below_approx_threshold(
    state: ProcessingState,
    group_index: GroupIndex,
) -> bool:
    values = state.get_values_y()
    values_count = state.get_values_y_count()
    if state.is_set_values_result_objs():
        values = values[state.get_values_result_objs()]
    current_disorder_score = group_index.get_disorder_score(
        values=values,
        values_count=values_count,
        disorder_fun=state.config[CONFIG_DISORDER_FUN],
    )
    approx_disorder_score_threshold = state.values[
        VALUES_DISORDER_SCORE_APPROX_THRESHOLD
    ]
    logger.debug("current_disorder_score = %f", current_disorder_score)
    logger.debug(
        "approx_disorder_score_value_threshold = %f",
        approx_disorder_score_threshold,
    )
    return bool(current_disorder_score <= approx_disorder_score_threshold)
