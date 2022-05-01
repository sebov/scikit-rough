import logging

import numpy as np

from skrough.algorithms.hooks.names import (
    HOOKS_APPROX_THRESHOLD,
    HOOKS_BASE_CHAOS_SCORE,
    HOOKS_CHAOS_FUN,
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
    HOOKS_TOTAL_CHAOS_SCORE,
)
from skrough.chaos_score import get_chaos_stats
from skrough.const import APPROX_THRESHOLD, BASE_CHAOS_SCORE, TOTAL_CHAOS_SCORE
from skrough.dataprep import prepare_factorized_x
from skrough.logs import log_start_end
from skrough.structs.group_index import GroupIndex
from skrough.structs.state import GrowShrinkState

logger = logging.getLogger(__name__)


@log_start_end(logger)
def init_state_hook_factorize_data_x_y(
    state: GrowShrinkState,
):
    # TODO: refactor
    x, x_counts = prepare_factorized_x(state.input[HOOKS_INPUT_X])
    y, y_count = prepare_factorized_x(
        np.expand_dims(state.input[HOOKS_INPUT_Y], axis=1)
    )
    state.values[HOOKS_DATA_X] = x
    state.values[HOOKS_DATA_X_COUNTS] = x_counts
    state.values[HOOKS_DATA_Y] = np.squeeze(y, axis=1)
    state.values[HOOKS_DATA_Y_COUNT] = y_count[0]


@log_start_end(logger)
def init_state_hook_single_group_index(
    state: GrowShrinkState,
):
    group_index = GroupIndex.create_one_group(len(state.values[HOOKS_DATA_X]))
    state.values[HOOKS_GROUP_INDEX] = group_index


@log_start_end(logger)
def init_state_hook_result_objs_empty(
    state: GrowShrinkState,
):
    state.values[HOOKS_RESULT_OBJS] = []


@log_start_end(logger)
def init_state_hook_result_attrs_empty(
    state: GrowShrinkState,
):
    state.values[HOOKS_RESULT_ATTRS] = []


@log_start_end(logger)
def init_state_hook_approx_threshold(
    state: GrowShrinkState,
) -> None:
    chaos_stats = get_chaos_stats(
        state.values[HOOKS_DATA_X],
        state.values[HOOKS_DATA_X_COUNTS],
        state.values[HOOKS_DATA_Y],
        state.values[HOOKS_DATA_Y_COUNT],
        state.config[HOOKS_CHAOS_FUN],
        state.config[HOOKS_EPSILON],
    )
    # TODO: handle typing of chaos_stats
    state.values.update(
        {
            HOOKS_BASE_CHAOS_SCORE: chaos_stats[BASE_CHAOS_SCORE],
            HOOKS_TOTAL_CHAOS_SCORE: chaos_stats[TOTAL_CHAOS_SCORE],
            HOOKS_APPROX_THRESHOLD: chaos_stats[APPROX_THRESHOLD],
        }
    )
