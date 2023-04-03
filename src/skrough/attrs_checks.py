import logging

import numpy as np

import skrough.typing as rght
from skrough.logs import log_start_end
from skrough.structs.group_index import GroupIndex

logger = logging.getLogger(__name__)


DEFAULT_SMOOTHING_PARAMETER = 1


@log_start_end(logger)
def check_if_attr_better_than_shuffled(
    group_index: GroupIndex,
    attr_values: np.ndarray,
    attr_values_count: int,
    values: np.ndarray,
    values_count: int,
    probes_count: int,
    allowed_randomness: float,
    chaos_fun: rght.ChaosMeasure,
    rng: np.random.Generator,
    smoothing_parameter: float = DEFAULT_SMOOTHING_PARAMETER,
) -> bool:
    attr_chaos_score = group_index.get_chaos_score_after_split(
        attr_values,
        attr_values_count,
        values,
        values_count,
        chaos_fun,
    )
    attr_is_better_count = 0
    for _ in range(probes_count):
        attr_values_shuffled = rng.permutation(attr_values)
        shuffled_chaos_score = group_index.get_chaos_score_after_split(
            attr_values_shuffled,
            attr_values_count,
            values,
            values_count,
            chaos_fun,
        )
        attr_is_better_count += int(attr_chaos_score < shuffled_chaos_score)

    smoothing_dims = 2  # binomial distribution, i.e., better/worse
    attr_probe_score = (attr_is_better_count + smoothing_parameter) / (
        probes_count + smoothing_parameter * smoothing_dims
    )
    logger.debug("attr_probe_score == %f", attr_probe_score)
    logger.debug("allowed_randomness == %f", allowed_randomness)
    logger.debug("attr_probe_threshold == %f", (1 - allowed_randomness))

    return attr_probe_score >= (1 - allowed_randomness)
