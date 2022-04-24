import logging

import numpy as np

from skrough.algorithms.hooks.names import (
    APPROX_THRESHOLD,
    BASE_CHAOS_SCORE,
    DATA_X,
    DATA_X_COUNTS,
    DATA_Y,
    DATA_Y_COUNT,
    INPUT_X,
    INPUT_Y,
    RESULT_ATTRS,
    RESULT_OBJS,
    SINGLE_GROUP_INDEX,
)
from skrough.chaos_score import get_chaos_score
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
    x, x_counts = prepare_factorized_x(state.input[INPUT_X])
    y, y_count = prepare_factorized_x(np.expand_dims(state.input[INPUT_Y], axis=1))
    state.values[DATA_X] = x
    state.values[DATA_X_COUNTS] = x_counts
    state.values[DATA_Y] = np.squeeze(y, axis=1)
    state.values[DATA_Y_COUNT] = y_count[0]


@log_start_end(logger)
def init_state_hook_single_group_index(
    state: GrowShrinkState,
):
    group_index = GroupIndex.create_one_group(len(state.values[DATA_X]))
    state.values[SINGLE_GROUP_INDEX] = group_index


@log_start_end(logger)
def init_state_hook_result_objs_empty(
    state: GrowShrinkState,
):
    state.values[RESULT_OBJS] = []


@log_start_end(logger)
def init_state_hook_result_attrs_empty(
    state: GrowShrinkState,
):
    state.values[RESULT_ATTRS] = []


@log_start_end(logger)
def init_state_hook_approx_threshold(
    state: GrowShrinkState,
) -> None:
    chaos_fun = state.config["chaos_fun"]
    epsilon = state.config["epsilon"]
    x_counts = state.values[DATA_X_COUNTS]
    y_count = state.values[DATA_Y_COUNT]

    # compute base chaos score
    base_chaos_score = get_chaos_score(
        state.values[DATA_X],
        x_counts,
        state.values[DATA_Y],
        y_count,
        [],
        chaos_fun=chaos_fun,
    )
    logger.debug("base_chaos_score = %f", base_chaos_score)

    # compute total chaos score
    total_chaos_score = get_chaos_score(
        state.values[DATA_X],
        x_counts,
        state.values[DATA_Y],
        y_count,
        range(state.values[DATA_X].shape[1]),
        chaos_fun=chaos_fun,
    )
    logger.debug("total_chaos_score = %f", total_chaos_score)

    total_dependency_in_data = base_chaos_score - total_chaos_score
    logger.debug("total_dependency_in_data = %f", total_dependency_in_data)

    approx_threshold = (1 - epsilon) * total_dependency_in_data - np.finfo(float).eps
    logger.debug("approx_threshold = %f", approx_threshold)

    state.values.update(
        {
            BASE_CHAOS_SCORE: base_chaos_score,
            APPROX_THRESHOLD: approx_threshold,
        }
    )
