import logging

from skrough.algorithms.hooks.names import (
    HOOKS_CHAOS_FUN,
    HOOKS_CHAOS_SCORE_APPROX_THRESHOLD,
    HOOKS_CHAOS_SCORE_BASE,
    HOOKS_CHAOS_SCORE_TOTAL,
    HOOKS_DATA_X,
    HOOKS_DATA_X_COUNTS,
    HOOKS_DATA_Y,
    HOOKS_DATA_Y_COUNT,
    HOOKS_EPSILON,
    HOOKS_GROUP_INDEX,
    HOOKS_INPUT_X,
    HOOKS_INPUT_Y,
    HOOKS_RESULT_ATTRS,
    HOOKS_RESULT_OBJS,
)
from skrough.chaos_score import get_chaos_score_stats
from skrough.dataprep import prepare_factorized_array, prepare_factorized_vector
from skrough.logs import log_start_end
from skrough.structs.group_index import GroupIndex
from skrough.structs.state import ProcessingState

logger = logging.getLogger(__name__)


@log_start_end(logger)
def init_state_hook_factorize_data_x_y(
    state: ProcessingState,
) -> None:
    x, x_counts = prepare_factorized_array(state.input_data[HOOKS_INPUT_X])
    y, y_count = prepare_factorized_vector(state.input_data[HOOKS_INPUT_Y])
    state.values[HOOKS_DATA_X] = x
    state.values[HOOKS_DATA_X_COUNTS] = x_counts
    state.values[HOOKS_DATA_Y] = y
    state.values[HOOKS_DATA_Y_COUNT] = y_count


@log_start_end(logger)
def init_state_hook_single_group_index(
    state: ProcessingState,
) -> None:
    group_index = GroupIndex.create_uniform(len(state.values[HOOKS_DATA_X]))
    state.values[HOOKS_GROUP_INDEX] = group_index


@log_start_end(logger)
def init_state_hook_result_objs_empty(
    state: ProcessingState,
) -> None:
    state.values[HOOKS_RESULT_OBJS] = []


@log_start_end(logger)
def init_state_hook_result_attrs_empty(
    state: ProcessingState,
) -> None:
    state.values[HOOKS_RESULT_ATTRS] = []


@log_start_end(logger)
def init_state_hook_approx_threshold(
    state: ProcessingState,
) -> None:
    chaos_stats = get_chaos_score_stats(
        x=state.values[HOOKS_DATA_X],
        x_counts=state.values[HOOKS_DATA_X_COUNTS],
        y=state.values[HOOKS_DATA_Y],
        y_count=state.values[HOOKS_DATA_Y_COUNT],
        chaos_fun=state.config[HOOKS_CHAOS_FUN],
        epsilon=state.config[HOOKS_EPSILON],
    )
    state.values.update(
        {
            HOOKS_CHAOS_SCORE_BASE: chaos_stats.base,
            HOOKS_CHAOS_SCORE_TOTAL: chaos_stats.total,
            HOOKS_CHAOS_SCORE_APPROX_THRESHOLD: chaos_stats.approx_threshold,
        }
    )
