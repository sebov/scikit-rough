# from __future__ import annotations

import typing

import numpy as np

import skrough.distributions
import skrough.group_index


def compute_chaos_score_for_group_index(
    group_index: np.ndarray,
    n_groups: int,
    n_objects: int,
    yy: np.ndarray,
    yy_count: int,
    chaos_fun: typing.Callable,
):
    """
    Compute chaos score for the given grouping of objects (into equivalence classes)
    """
    distribution = skrough.distributions.get_dec_distribution(
        group_index, n_groups, yy, yy_count
    )
    return chaos_fun(distribution, n_objects)


def compute_chaos_score(
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    attrs,
    chaos_fun: typing.Callable,
):
    """
    Compute chaos score for the grouping (equivalence classes) induced by the given
    subset of attributes
    """
    group_index, n_groups = skrough.group_index.batch_split_into_groups(
        x, x_counts, attrs
    )
    result = compute_chaos_score_for_group_index(
        group_index, n_groups, len(x), y, y_count, chaos_fun
    )
    return result
