# from __future__ import annotations

import numpy as np

import skrough.distributions
import skrough.group_index
import skrough.typing as rght
from skrough.containers import GroupIndex


def get_chaos_score_for_group_index(
    group_index: GroupIndex,
    n_objects: int,
    y: np.ndarray,
    y_count: int,
    chaos_fun: rght.ChaosMeasure,
):
    """
    Compute chaos score for the given grouping of objects (into equivalence classes)
    """
    distribution = skrough.distributions.get_dec_distribution(group_index, y, y_count)
    return chaos_fun(distribution, n_objects)


def get_chaos_score(
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    attrs,
    chaos_fun: rght.ChaosMeasure,
):
    """
    Compute chaos score for the grouping (equivalence classes) induced by the given
    subset of attributes
    """
    group_index = skrough.group_index.batch_split_into_groups(x, x_counts, attrs)
    result = get_chaos_score_for_group_index(group_index, len(x), y, y_count, chaos_fun)
    return result
