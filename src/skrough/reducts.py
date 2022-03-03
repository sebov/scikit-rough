from __future__ import annotations

import typing

import numpy as np
import sklearn.utils

import skrough as rgh
import skrough.typing as rght
from skrough.containers import GroupIndex

# greedy_heuristic_reduct.py


# TODO: handle data consistency === chaos
# if check_data_consistency:
#     # by default check_data_consistency should be True
#     # but for now we do not handle this
#     raise NotImplementedError("check_data_consistency==True not implemented")
# else:
#     pass

# TODO: should check sklearn.utils.validation.check_is_fitted(self, ["x", "y"])

# TODO: handle homogeneous groups
# def remove_homogenous_groups(self, group_index, n_groups, y, y_count_distinct):
#     distribution = compute_dec_distribution(group_index, n_groups,
#                                             y, y_count_distinct)
#     groups_homogeneity = compute_homogeneity(distribution)


def split_groups_and_compute_chaos_score(
    group_index: GroupIndex,
    attr: int,
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    chaos_fun: rght.ChaosMeasure,
):
    tmp_group_index = rgh.group_index.split_groups(
        group_index, x[:, attr], x_counts[attr]
    )
    return rgh.chaos_score.get_chaos_score_for_group_index(
        tmp_group_index, len(x), y, y_count, chaos_fun
    )


# def split_groups_and_compute_chaos_score_2(
#     self, group_index, n_groups, attr_values, attr_count, y, y_count, chaos_fun
# ):
#     tmp_group_index, tmp_n_groups = rgh.group_index.split_groups(
#         group_index, n_groups, attr_values, attr_count
#     )
#     return rgh.chaos_score.compute_chaos_score_for_group_index(
#         tmp_group_index, tmp_n_groups, len(attr_values), y, y_count, chaos_fun
#     )


def get_best_attr(
    group_index: GroupIndex,
    candidate_attrs: typing.Sequence[int],
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    chaos_fun: rght.ChaosMeasure,
):
    scores = np.fromiter(
        (
            split_groups_and_compute_chaos_score(
                group_index, i, x, x_counts, y, y_count, chaos_fun
            )
            for i in candidate_attrs
        ),
        dtype=float,
    )
    return candidate_attrs[scores.argmin()]


def get_reduct_greedy_heuristic(
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    chaos_fun: rght.ChaosMeasure,
    epsilon: float = 0.0,
    n_candidate_attrs: int | None = None,
    random_state: rght.Seed = None,
):
    random_state = sklearn.utils.check_random_state(random_state)

    # TODO: check params, e.g., epsilon, n_candidate_attrs

    # init group_index
    # TODO:
    group_index = GroupIndex.create_one_group(len(x))

    # compute base chaos score
    base_chaos_score = rgh.chaos_score.get_chaos_score_for_group_index(
        group_index, len(x), y, y_count, chaos_fun
    )

    # compute total chaos score
    total_chaos_score = 0

    total_dependency_in_data = base_chaos_score - total_chaos_score
    approx_threshold = (1 - epsilon) * total_dependency_in_data - np.finfo(float).eps

    result_attrs: list[int] = []
    while True:
        current_chaos_score = rgh.chaos_score.get_chaos_score_for_group_index(
            group_index, len(x), y, y_count, chaos_fun
        )
        current_dependency_in_data = base_chaos_score - current_chaos_score
        if current_dependency_in_data >= approx_threshold:
            break
        candidate_attrs = np.delete(np.arange(x.shape[1]), result_attrs)
        if n_candidate_attrs is not None:
            # TODO: introduce random_state usage
            candidate_attrs = np.random.choice(
                candidate_attrs,
                np.min([len(candidate_attrs), n_candidate_attrs]),
                replace=False,
            )
        best_attr = get_best_attr(
            group_index, candidate_attrs, x, x_counts, y, y_count, chaos_fun
        )
        result_attrs.append(best_attr)
        group_index = rgh.group_index.split_groups(
            group_index,
            x[:, best_attr],
            x_counts[best_attr],
        )

    # TODO: add reduction phase

    return rgh.containers.Reduct(attrs=result_attrs)
