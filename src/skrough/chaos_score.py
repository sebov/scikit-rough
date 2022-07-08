import logging
from typing import List, Optional, Sequence, Set

import numpy as np
from attrs import define

import skrough.typing as rght
from skrough.logs import log_start_end
from skrough.structs.group_index import GroupIndex

logger = logging.getLogger(__name__)


@log_start_end(logger)
def get_chaos_score_for_group_index(
    group_index: GroupIndex,
    y: np.ndarray,
    y_count: int,
    chaos_fun: rght.ChaosMeasure,
) -> rght.ChaosMeasureReturnType:
    """
    Compute chaos score for the given grouping of objects (into equivalence classes).
    """
    distribution = group_index.get_distribution(y, y_count)
    return chaos_fun(distribution, group_index.size)


@log_start_end(logger)
def get_chaos_score(
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    attrs: rght.AttrsLike,
    chaos_fun: rght.ChaosMeasure,
) -> rght.ChaosMeasureReturnType:
    """
    Compute chaos score for the grouping (equivalence classes) induced by the given
    subset of attributes.
    """
    group_index = GroupIndex.create_from_data(x, x_counts, attrs)
    result = get_chaos_score_for_group_index(group_index, y, y_count, chaos_fun)
    return result


@define
class ChaosScoreStats:
    base: float
    total: float
    for_increment_attrs: Optional[List[float]] = None
    approx_threshold: Optional[float] = None


@log_start_end(logger)
def get_chaos_score_stats(
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    chaos_fun: rght.ChaosMeasure,
    increment_attrs: Optional[Sequence[rght.AttrsLike]] = None,
    epsilon: Optional[float] = None,
) -> ChaosScoreStats:
    group_index = GroupIndex.create_uniform(len(x))

    # compute base chaos score
    base_chaos_score = get_chaos_score_for_group_index(
        group_index,
        y,
        y_count,
        chaos_fun,
    )

    increment_attrs_chaos_score = None
    attrs_added: Set[int] = set()
    if increment_attrs is not None:
        increment_attrs_chaos_score = []
        for attrs in increment_attrs:
            attrs_to_add = set(attrs) - attrs_added
            for attr in attrs_to_add:
                group_index = group_index.split(x[:, attr], x_counts[attr])
            attrs_added = attrs_added.union(attrs_to_add)
            chaos_score = get_chaos_score_for_group_index(
                group_index,
                y,
                y_count,
                chaos_fun,
            )
            increment_attrs_chaos_score.append(chaos_score)

    # add remaining attrs
    attrs_other = set(range(x.shape[1])) - attrs_added
    for attr in attrs_other:
        group_index = group_index.split(x[:, attr], x_counts[attr])

    # compute total chaos score
    total_chaos_score = get_chaos_score_for_group_index(
        group_index,
        y,
        y_count,
        chaos_fun,
    )

    approx_threshold = None
    if epsilon is not None:
        delta_dependency = base_chaos_score - total_chaos_score
        approx_threshold = (1 - epsilon) * delta_dependency - np.finfo(float).eps

    result = ChaosScoreStats(
        base=base_chaos_score,
        total=total_chaos_score,
        for_increment_attrs=increment_attrs_chaos_score,
        approx_threshold=approx_threshold,
    )
    logger.debug("chaos_stats = %s", result)
    return result
