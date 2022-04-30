import logging
from typing import Optional, Sequence

import numpy as np

import skrough.typing as rght
from skrough.const import APPROX_THRESHOLD, BASE_CHAOS_SCORE, TOTAL_CHAOS_SCORE
from skrough.distributions import get_dec_distribution
from skrough.logs import log_start_end
from skrough.structs.group_index import GroupIndex

logger = logging.getLogger(__name__)


@log_start_end(logger)
def get_chaos_score_for_group_index(
    group_index: GroupIndex,
    n_objects: int,
    y: np.ndarray,
    y_count: int,
    chaos_fun: rght.ChaosMeasure,
) -> rght.ChaosMeasureReturnType:
    """
    Compute chaos score for the given grouping of objects (into equivalence classes).
    """
    distribution = get_dec_distribution(group_index, y, y_count)
    return chaos_fun(distribution, n_objects)


@log_start_end(logger)
def get_chaos_score(
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    attrs: Sequence[int],
    chaos_fun: rght.ChaosMeasure,
) -> rght.ChaosMeasureReturnType:
    """
    Compute chaos score for the grouping (equivalence classes) induced by the given
    subset of attributes.
    """
    group_index = GroupIndex.create_from_data(x, x_counts, attrs)
    result = get_chaos_score_for_group_index(group_index, len(x), y, y_count, chaos_fun)
    return result


@log_start_end(logger)
def get_chaos_stats(
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    chaos_fun: rght.ChaosMeasure,
    epsilon: Optional[float] = None,
):
    # TODO: handle typing of the result
    result = {}

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
    result[BASE_CHAOS_SCORE] = base_chaos_score

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
    result[TOTAL_CHAOS_SCORE] = total_chaos_score

    delta_dependency = base_chaos_score - total_chaos_score
    logger.debug("total_dependency_in_data = %f", delta_dependency)

    if epsilon is not None:
        approx_threshold = (1 - epsilon) * delta_dependency - np.finfo(float).eps
        logger.debug("approx_threshold = %f", approx_threshold)
        result[APPROX_THRESHOLD] = approx_threshold

    return result
