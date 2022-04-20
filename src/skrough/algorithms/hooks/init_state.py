import logging

import numpy as np

from skrough.algorithms.hooks.names import (
    APPROX_THRESHOLD,
    BASE_CHAOS_SCORE,
    RESULT_ATTRS,
    RESULT_OBJS,
    SINGLE_GROUP_INDEX,
)
from skrough.chaos_score import get_chaos_score
from skrough.logs import log_start_end
from skrough.structs.group_index import GroupIndex
from skrough.structs.state import GrowShrinkState

logger = logging.getLogger(__name__)


@log_start_end(logger)
def init_state_single_group_index(
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    state: GrowShrinkState,
):
    group_index = GroupIndex.create_one_group(len(x))
    state.values[SINGLE_GROUP_INDEX] = group_index


@log_start_end(logger)
def init_state_result_objs_empty(
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    state: GrowShrinkState,
):
    state.values[RESULT_OBJS] = []


@log_start_end(logger)
def init_state_result_attrs_empty(
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    state: GrowShrinkState,
):
    state.values[RESULT_ATTRS] = []


@log_start_end(logger)
def init_state_approx_threshold(
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    state: GrowShrinkState,
) -> None:
    chaos_fun = state.config["chaos_fun"]
    epsilon = state.config["epsilon"]

    # compute base chaos score
    base_chaos_score = get_chaos_score(
        x,
        x_counts,
        y,
        y_count,
        [],
        chaos_fun=chaos_fun,
    )
    logger.debug("base_chaos_score = %f", base_chaos_score)

    # compute total chaos score
    total_chaos_score = get_chaos_score(
        x,
        x_counts,
        y,
        y_count,
        range(x.shape[1]),
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
