import logging

import numpy as np

import skrough.typing as rght
from skrough.algorithms.key_names import (
    CONFIG_CHAOS_FUN,
    CONFIG_DAAR_ALLOWED_RANDOMNESS,
    CONFIG_DAAR_FAST,
    CONFIG_DAAR_PROBES_COUNT,
    CONFIG_DAAR_SMOOTHING_PARAMETER,
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
DEFAULT_DAAR_FAST = False


@log_start_end(logger)
def filter_hook_attrs_first_daar(
    state: ProcessingState,
    elements: rght.Elements,
) -> rght.Elements:
    daar_allowed_randomness = state.config[CONFIG_DAAR_ALLOWED_RANDOMNESS]
    logger.debug("Param daar_allowed_randomness == %f", daar_allowed_randomness)
    daar_fast = state.config.get(CONFIG_DAAR_FAST, DEFAULT_DAAR_FAST)
    logger.debug("Param daar_fast == %f", daar_fast)
    daar_probes_count = state.config[CONFIG_DAAR_PROBES_COUNT]
    logger.debug("Param daar_probes_count == %d", daar_probes_count)
    daar_smoothing_parameter = state.config.get(
        CONFIG_DAAR_SMOOTHING_PARAMETER, DEFAULT_DAAR_SMOOTHING_PARAMETER
    )
    logger.debug("Param daar_smoothing_parameter == %f", daar_smoothing_parameter)
    chaos_fun = state.config[CONFIG_CHAOS_FUN]

    group_index = state.values[VALUES_GROUP_INDEX]
    x = state.values[VALUES_X]
    x_counts = state.values[VALUES_X_COUNTS]
    y = state.values[VALUES_Y]
    y_count = state.values[VALUES_Y_COUNT]
    result = []
    for attr in elements:
        logger.debug("Check if attr <%d> is better than shuffled", attr)
        if check_if_attr_better_than_shuffled(
            group_index=group_index,
            attr_values=x[:, attr],
            attr_values_count=x_counts[attr],
            values=y,
            values_count=y_count,
            allowed_randomness=daar_allowed_randomness,
            probes_count=daar_probes_count,
            smoothing_parameter=daar_smoothing_parameter,
            fast=daar_fast,
            chaos_fun=chaos_fun,
            rng=state.rng,
        ):
            logger.debug(
                "Attr <%d> is better than shuffled with respect to allowed_randomness",
                attr,
            )
            result.append(attr)
            break  # in this version we finish whenever the first one is found
    return np.asarray(result)
