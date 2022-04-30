import logging

import numpy as np

import skrough.typing as rght
from skrough.algorithms.hooks.names import (
    HOOKS_DATA_X,
    HOOKS_DATA_X_COUNTS,
    HOOKS_DATA_Y,
    HOOKS_DATA_Y_COUNT,
    HOOKS_GROW_POST_SELECT_ATTRS_DAAR_ALLOWED_RANDOMNESS,
    HOOKS_GROW_POST_SELECT_ATTRS_DAAR_N_OF_PROBES,
    HOOKS_GROW_POST_SELECT_ATTRS_DAAR_SMOOTHING_PARAMETER,
    HOOKS_SINGLE_GROUP_INDEX,
)
from skrough.algorithms.hooks.utils import split_groups_and_compute_chaos_score
from skrough.logs import log_start_end
from skrough.structs.group_index import GroupIndex
from skrough.structs.state import GrowShrinkState

logger = logging.getLogger(__name__)


DEFAULT_DAAR_SMOOTHING_PARAMETER = 1


@log_start_end(logger)
def _check_if_better_than_shuffled(
    group_index: GroupIndex,
    attr_values: np.ndarray,
    attr_count: int,
    y: np.ndarray,
    y_count: int,
    n_of_probes: int,
    smoothing_parameter: float,
    allowed_randomness: float,
    chaos_fun: rght.ChaosMeasure,
    rng: np.random.Generator,
) -> bool:
    attr_chaos_score = split_groups_and_compute_chaos_score(
        group_index,
        attr_values,
        attr_count,
        y,
        y_count,
        chaos_fun,
    )
    attr_is_better_count = 0
    for _ in range(n_of_probes):
        attr_values_shuffled = rng.permutation(attr_values)
        shuffled_chaos_score = split_groups_and_compute_chaos_score(
            group_index,
            attr_values_shuffled,
            attr_count,
            y,
            y_count,
            chaos_fun,
        )
        attr_is_better_count += int(attr_chaos_score < shuffled_chaos_score)

    smoothing_dims = 2  # binomial distribution, i.e., better/worse
    attr_probe_score = (attr_is_better_count + smoothing_parameter) / (
        n_of_probes + smoothing_parameter * smoothing_dims
    )
    logger.debug("attr_probe_score == %f", attr_probe_score)
    return attr_probe_score >= (1 - allowed_randomness)


@log_start_end(logger)
def grow_verify_attrs_daar(
    state: GrowShrinkState,
    input_attrs: rght.Elements,
) -> rght.Elements:
    daar_n_of_probes = state.config[HOOKS_GROW_POST_SELECT_ATTRS_DAAR_N_OF_PROBES]
    logger.debug("Param daar_n_of_probes == %d", daar_n_of_probes)
    daar_smoothing_parameter = state.config.get(
        HOOKS_GROW_POST_SELECT_ATTRS_DAAR_SMOOTHING_PARAMETER,
        DEFAULT_DAAR_SMOOTHING_PARAMETER,
    )
    logger.debug("Param daar_smoothing_parameter == %f", daar_smoothing_parameter)
    daar_allowed_randomness = state.config[
        HOOKS_GROW_POST_SELECT_ATTRS_DAAR_ALLOWED_RANDOMNESS
    ]
    logger.debug("Param daar_allowed_randomness == %f", daar_allowed_randomness)
    chaos_fun = state.config["chaos_fun"]
    x_counts = state.values[HOOKS_DATA_X_COUNTS]
    y_count = state.values[HOOKS_DATA_Y_COUNT]
    result = []
    for input_attr in input_attrs:
        logger.debug("Check if attr <%d> is better than shuffled", input_attr)
        if _check_if_better_than_shuffled(
            state.values[HOOKS_SINGLE_GROUP_INDEX],
            state.values[HOOKS_DATA_X][:, input_attr],
            x_counts[input_attr],
            state.values[HOOKS_DATA_Y],
            y_count,
            daar_n_of_probes,
            daar_smoothing_parameter,
            daar_allowed_randomness,
            chaos_fun,
            state.rng,
        ):
            logger.debug(
                "Attr <%d> is better than shuffled with respect to allowed_randomness",
                input_attr,
            )
            result.append(input_attr)
    return np.asarray(result)
