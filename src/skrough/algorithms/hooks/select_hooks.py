import logging

import numpy as np

import skrough.typing as rght
from skrough.algorithms.hooks.names import (
    HOOKS_CHAOS_FUN,
    HOOKS_DATA_X,
    HOOKS_DATA_X_COUNTS,
    HOOKS_DATA_Y,
    HOOKS_DATA_Y_COUNT,
    HOOKS_GROUP_INDEX,
    HOOKS_SELECT_ATTRS_GAIN_BASED_COUNT,
    HOOKS_SELECT_RANDOM_COUNT,
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
    return state.rng.choice(elements, state.config[HOOKS_SELECT_RANDOM_COUNT])


@log_start_end(logger)
def select_hook_grow_attrs_gain_based(
    state: GrowShrinkState,
    attr_elements: rght.Elements,
) -> rght.Elements:
    scores = np.fromiter(
        (
            split_groups_and_compute_chaos_score(
                state.values[HOOKS_GROUP_INDEX],
                state.values[HOOKS_DATA_X][:, i],
                state.values[HOOKS_DATA_X_COUNTS][i],
                state.values[HOOKS_DATA_Y],
                state.values[HOOKS_DATA_Y_COUNT],
                state.config[HOOKS_CHAOS_FUN],
            )
            for i in attr_elements
        ),
        dtype=float,
    )
    # find indices for which the scores are the lowest
    attrs_count = state.config[HOOKS_SELECT_ATTRS_GAIN_BASED_COUNT]
    selected_attrs_idx = np.argsort(scores)[:attrs_count]
    return attr_elements[selected_attrs_idx]
