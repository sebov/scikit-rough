import logging

import numpy as np

import skrough.typing as rght
from skrough.algorithms.hooks.names import (
    HOOKS_GROW_POST_SELECT_ATTRS_DAAR_ALLOWED_RANDOMNESS,
    HOOKS_GROW_POST_SELECT_ATTRS_DAAR_N_OF_PROBES,
    VALUES_GROUP_INDEX,
    VALUES_X,
    VALUES_X_COUNTS,
    VALUES_Y,
    VALUES_Y_COUNT,
)
from skrough.attrs_checks import check_if_attr_better_than_shuffled
from skrough.logs import log_start_end
from skrough.structs.state import ProcessingState

logger = logging.getLogger(__name__)


DEFAULT_DAAR_SMOOTHING_PARAMETER = 1


@log_start_end(logger)
def grow_verify_attrs_daar(
    state: ProcessingState,
    input_attrs: rght.Elements,
) -> rght.Elements:
    daar_n_of_probes = state.config[HOOKS_GROW_POST_SELECT_ATTRS_DAAR_N_OF_PROBES]
    logger.debug("Param daar_n_of_probes == %d", daar_n_of_probes)
    daar_allowed_randomness = state.config[
        HOOKS_GROW_POST_SELECT_ATTRS_DAAR_ALLOWED_RANDOMNESS
    ]
    logger.debug("Param daar_allowed_randomness == %f", daar_allowed_randomness)
    chaos_fun = state.config["chaos_fun"]
    x_counts = state.values[VALUES_X_COUNTS]
    y_count = state.values[VALUES_Y_COUNT]
    result = []
    for input_attr in input_attrs:
        logger.debug("Check if attr <%d> is better than shuffled", input_attr)
        if check_if_attr_better_than_shuffled(
            group_index=state.values[VALUES_GROUP_INDEX],
            attr_values=state.values[VALUES_X][:, input_attr],
            attr_values_count=x_counts[input_attr],
            values=state.values[VALUES_Y],
            values_count=y_count,
            n_of_probes=daar_n_of_probes,
            allowed_randomness=daar_allowed_randomness,
            chaos_fun=chaos_fun,
            rng=state.rng,
        ):
            logger.debug(
                "Attr <%d> is better than shuffled with respect to allowed_randomness",
                input_attr,
            )
            result.append(input_attr)
    return np.asarray(result)
