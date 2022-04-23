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
    SELECT_ATTRS_RANDOM_COUNT,
    SINGLE_GROUP_INDEX,
)
from skrough.algorithms.hooks.utils import split_groups_and_compute_chaos_score
from skrough.logs import log_start_end
from skrough.structs.state import GrowShrinkState

logger = logging.getLogger(__name__)


@log_start_end(logger)
def grow_select_attrs_random(
    state: GrowShrinkState,
    input_attrs: rght.GSElements,
) -> rght.GSElements:
    return state.rng.choice(input_attrs, state.config[SELECT_ATTRS_RANDOM_COUNT])


@log_start_end(logger)
def grow_select_attrs_gain_based(
    state: GrowShrinkState,
    input_attrs: rght.GSElements,
) -> rght.GSElements:
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
            for i in input_attrs
        ),
        dtype=float,
    )
    # find indices for which the scores are the lowest
    selected_attrs_idx = np.argsort(scores)[:attrs_count]
    return input_attrs[selected_attrs_idx]
