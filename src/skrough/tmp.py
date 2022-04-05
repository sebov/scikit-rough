from typing import List

# import skrough.typing as rght
from skrough.chaos_score import get_chaos_score

# import numpy as np

# from skrough.chaos_score import get_chaos_score_for_group_index
# from skrough.structs.group_index import GroupIndex
# from skrough.structs.reduct import Reduct

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


# def split_groups_and_compute_chaos_score_2(
#     self, group_index, n_groups, attr_values, attr_count, y, y_count, chaos_fun
# ):
#     tmp_group_index, tmp_n_groups = rgh.group_index.split_groups(
#         group_index, n_groups, attr_values, attr_count
#     )
#     return rgh.chaos_score.compute_chaos_score_for_group_index(
#         tmp_group_index, tmp_n_groups, len(attr_values), y, y_count, chaos_fun
#     )


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
