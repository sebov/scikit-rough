import logging

import numpy as np

import skrough.typing as rght
from skrough.algorithms.hooks.names import (
    CHAOS_FUN,
    DATA_X,
    DATA_X_COUNTS,
    DATA_Y,
    DATA_Y_COUNT,
    SELECT_ATTRS_GAIN_BASED_COUNT,
    SELECT_RANDOM_COUNT,
    SINGLE_GROUP_INDEX,
)
from skrough.algorithms.hooks.utils import split_groups_and_compute_chaos_score
from skrough.logs import log_start_end
from skrough.structs.state import GrowShrinkState

logger = logging.getLogger(__name__)


@log_start_end(logger)
def select_hook_random(
    state: GrowShrinkState,
    elements: rght.Elements,
) -> rght.Elements:
    return state.rng.choice(elements, state.config[SELECT_RANDOM_COUNT])


@log_start_end(logger)
def select_hook_grow_attrs_gain_based(
    state: GrowShrinkState,
    attr_elements: rght.Elements,
) -> rght.Elements:
    chaos_fun = state.config[CHAOS_FUN]
    attrs_count = state.config[SELECT_ATTRS_GAIN_BASED_COUNT]
    x_counts = state.values[DATA_X_COUNTS]
    y_count = state.values[DATA_Y_COUNT]
    scores = np.fromiter(
        (
            split_groups_and_compute_chaos_score(
                state.values[SINGLE_GROUP_INDEX],
                state.values[DATA_X][:, i],
                x_counts[i],
                state.values[DATA_Y],
                y_count,
                chaos_fun,
            )
            for i in attr_elements
        ),
        dtype=float,
    )
    # find indices for which the scores are the lowest
    selected_attrs_idx = np.argsort(scores)[:attrs_count]
    return attr_elements[selected_attrs_idx]
