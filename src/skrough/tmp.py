from typing import List, Optional, Sequence

import numpy as np

import skrough.typing as rght
from skrough.chaos_score import get_chaos_score, get_chaos_score_for_group_index
from skrough.structs.group_index import GroupIndex
from skrough.structs.reduct import Reduct

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
    tmp_group_index = group_index.split(x[:, attr], x_counts[attr])
    return get_chaos_score_for_group_index(
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
    candidate_attrs: Sequence[int],
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    chaos_fun: rght.ChaosMeasure,
) -> int:
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


def reduction_phase(
    xx,
    xx_count_distinct,
    yy,
    yy_count_distinct,
    # group_index,
    # n_groups,
    chaos_fun,
    attrs: List[int],
) -> List[int]:
    before_reduction_chaos_score = get_chaos_score(
        xx, xx_count_distinct, yy, yy_count_distinct, attrs, chaos_fun
    )
    result_attrs_reduction = set(attrs)
    for i in reversed(attrs):
        attrs_to_try = result_attrs_reduction - {i}
        current_chaos_score = get_chaos_score(
            xx, xx_count_distinct, yy, yy_count_distinct, list(attrs_to_try), chaos_fun
        )
        if current_chaos_score <= before_reduction_chaos_score:
            result_attrs_reduction = attrs_to_try
    return sorted(result_attrs_reduction)


def get_reduct_greedy_heuristic(
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    chaos_fun: rght.ChaosMeasure,
    epsilon: float = 0.0,
    n_candidate_attrs: Optional[int] = None,
    seed: rght.Seed = None,
) -> Reduct:
    rng = np.random.default_rng(seed)

    # TODO: check params, e.g., epsilon, n_candidate_attrs

    # init group_index
    group_index = GroupIndex.create_one_group(len(x))

    # compute base chaos score
    base_chaos_score = get_chaos_score_for_group_index(
        group_index, len(x), y, y_count, chaos_fun
    )

    # compute total chaos score
    total_chaos_score = 0

    total_dependency_in_data = base_chaos_score - total_chaos_score
    approx_threshold = (1 - epsilon) * total_dependency_in_data - np.finfo(float).eps

    result_attrs: List[int] = []
    while True:
        current_chaos_score = get_chaos_score_for_group_index(
            group_index, len(x), y, y_count, chaos_fun
        )
        current_dependency_in_data = base_chaos_score - current_chaos_score
        if current_dependency_in_data >= approx_threshold:
            break
        candidate_attrs: np.ndarray = np.delete(np.arange(x.shape[1]), result_attrs)
        if n_candidate_attrs is not None:
            candidate_attrs = rng.choice(
                candidate_attrs,
                np.min([len(candidate_attrs), n_candidate_attrs]),
                replace=False,
            )
        best_attr = get_best_attr(
            group_index, candidate_attrs.tolist(), x, x_counts, y, y_count, chaos_fun
        )
        result_attrs.append(best_attr)
        group_index = group_index.split(
            x[:, best_attr],
            x_counts[best_attr],
        )

    result_attrs = reduction_phase(
        x,
        x_counts,
        y,
        y_count,
        chaos_fun,
        result_attrs,
    )

    return Reduct(attrs=result_attrs)
