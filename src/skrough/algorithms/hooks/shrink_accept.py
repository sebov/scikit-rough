import logging

import numpy as np

from skrough.algorithms.hooks.names import APPROX_THRESHOLD, BASE_CHAOS_SCORE, CHAOS_FUN
from skrough.chaos_score import get_chaos_score_for_group_index
from skrough.logs import log_start_end
from skrough.structs.group_index import GroupIndex
from skrough.structs.state import GrowShrinkState

logger = logging.getLogger(__name__)


@log_start_end(logger)
def shrink_accept_group_index_approx_threshold(
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    state: GrowShrinkState,
    group_index_to_check: GroupIndex,
) -> bool:
    chaos_fun = state.config[CHAOS_FUN]
    base_chaos_score = state.values[BASE_CHAOS_SCORE]
    approx_threshold = state.values[APPROX_THRESHOLD]
    current_chaos_score = get_chaos_score_for_group_index(
        group_index_to_check, len(x), y, y_count, chaos_fun
    )
    current_dependency_in_data = base_chaos_score - current_chaos_score
    return current_dependency_in_data >= approx_threshold
