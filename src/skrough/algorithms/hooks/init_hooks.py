import logging

from skrough.algorithms.hooks.names import (
    CONFIG_CHAOS_FUN,
    CONFIG_EPSILON,
    INPUT_X,
    INPUT_Y,
    VALUES_CHAOS_SCORE_APPROX_THRESHOLD,
    VALUES_CHAOS_SCORE_BASE,
    VALUES_CHAOS_SCORE_TOTAL,
    VALUES_GROUP_INDEX,
    VALUES_RESULT_ATTRS,
    VALUES_RESULT_OBJS,
    VALUES_X,
    VALUES_X_COUNTS,
    VALUES_Y,
    VALUES_Y_COUNT,
)
from skrough.chaos_score import get_chaos_score_stats
from skrough.dataprep import prepare_factorized_array, prepare_factorized_vector
from skrough.logs import log_start_end
from skrough.structs.group_index import GroupIndex
from skrough.structs.state import ProcessingState

logger = logging.getLogger(__name__)


@log_start_end(logger)
def init_hook_factorize_data_x_y(
    state: ProcessingState,
) -> None:
    x, x_counts = prepare_factorized_array(state.input_data[INPUT_X])
    y, y_count = prepare_factorized_vector(state.input_data[INPUT_Y])
    state.values[VALUES_X] = x
    state.values[VALUES_X_COUNTS] = x_counts
    state.values[VALUES_Y] = y
    state.values[VALUES_Y_COUNT] = y_count


@log_start_end(logger)
def init_hook_single_group_index(
    state: ProcessingState,
) -> None:
    group_index = GroupIndex.create_uniform(len(state.values[VALUES_X]))
    state.values[VALUES_GROUP_INDEX] = group_index


@log_start_end(logger)
def init_hook_result_objs_empty(
    state: ProcessingState,
) -> None:
    state.values[VALUES_RESULT_OBJS] = []


@log_start_end(logger)
def init_hook_result_attrs_empty(
    state: ProcessingState,
) -> None:
    state.values[VALUES_RESULT_ATTRS] = []


@log_start_end(logger)
def init_hook_approx_threshold(
    state: ProcessingState,
) -> None:
    chaos_stats = get_chaos_score_stats(
        x=state.values[VALUES_X],
        x_counts=state.values[VALUES_X_COUNTS],
        y=state.values[VALUES_Y],
        y_count=state.values[VALUES_Y_COUNT],
        chaos_fun=state.config[CONFIG_CHAOS_FUN],
        epsilon=state.config[CONFIG_EPSILON],
    )
    state.values.update(
        {
            VALUES_CHAOS_SCORE_BASE: chaos_stats.base,
            VALUES_CHAOS_SCORE_TOTAL: chaos_stats.total,
            VALUES_CHAOS_SCORE_APPROX_THRESHOLD: chaos_stats.approx_threshold,
        }
    )
