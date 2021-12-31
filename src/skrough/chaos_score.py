# from __future__ import annotations

import typing

import numpy as np

import skrough.distributions
import skrough.group_index


def _compute_chaos_score(
    group_index: np.ndarray,
    n_groups: int,
    n_objects: int,
    yy: np.ndarray,
    yy_count_distinct: int,
    chaos_fun: typing.Callable,
):
    """
    Compute chaos score for the given grouping of objects (into equivalence classes)
    """
    distribution = skrough.distributions.get_dec_distribution(
        group_index, n_groups, yy, yy_count_distinct
    )
    return chaos_fun(distribution, n_objects)


def get_chaos_score(
    xx,
    xx_count_distinct,
    yy,
    yy_count_distinct,
    attrs,
    chaos_fun,
    _batch_split_into_groups_fun=skrough.group_index.batch_split_into_groups,
    _compute_chaos_score_fun=_compute_chaos_score,
):
    """
    Compute chaos score for the grouping (equivalence classes) induced by the given subset of attributes
    """
    group_index, n_groups = _batch_split_into_groups_fun(xx, xx_count_distinct, attrs)
    result = _compute_chaos_score_fun(
        group_index, n_groups, len(xx), yy, yy_count_distinct, chaos_fun
    )
    return result
